#include "SlidingProblem.hh"


std::pair<Real, size_t> SlidingProblem::feasibleStepLength(const Eigen::VectorXd &vars, const Eigen::VectorXd &step) const {
    BENCHMARK_SCOPED_TIMER_SECTION timer("computeFeasibleStepSize");
    Real alpha = std::numeric_limits<Real>::max();
    size_t blocking_idx = std::numeric_limits<size_t>::max();

    for (size_t i = 0; i < m_boundConstraints.size(); ++i) {
        Real len = m_boundConstraints[i].feasibleStepLength(vars, step);
        if (len < alpha) { alpha = len; blocking_idx = i; }
    }

    if (m_options.hasCollisions) { 
        Eigen::VectorXd nodalVars = m_rods.extractNodalDoFs(vars.head(numVars()));
        Eigen::VectorXd nodalStep = m_rods.extractNodalDoFs(step.head(numVars()));
        const Eigen::MatrixXd V0 = Eigen::Map<Eigen::Matrix<Real, Eigen::Dynamic, 3, Eigen::RowMajor>>(nodalVars.data(), m_rods.numVertices(), 3);
        const Eigen::MatrixXd V1 = V0 + Eigen::Map<Eigen::Matrix<Real, Eigen::Dynamic, 3, Eigen::RowMajor>>(nodalStep.data(), m_rods.numVertices(), 3);
        alpha = compute_collision_free_stepsize(m_collisionMesh, V0, V1, ipc::BroadPhaseMethod::HASH_GRID, m_options.dHat, m_options.Wang2021MaxIter);
    }

    if (m_options.printIterInfo) {
        std::string alphaStr = (alpha == std::numeric_limits<Real>::max()) ? "+inf" : std::to_string(alpha);
        std::cout << "Feasible alpha = " << alphaStr << std::endl;
    }

    return std::make_pair(alpha, blocking_idx);
}

void SlidingProblem::m_iterationCallback(size_t i) {
    m_rods.updateSourceFrame(); 
    m_rods.updateRotationParametrizations();
        
    updateCachedVars();
    updateCharacteristicLength();
    updateCachedSparsityPattern();

    BENCHMARK_START_TIMER("CustomCallback");
    if (m_customCallback) m_customCallback(*this, i);
    BENCHMARK_STOP_TIMER("CustomCallback");
}

void minimize_twist(PeriodicRod &pr, bool verbose) {
    // Get the underlying ElasticRod object and
    // minimize twisting energy wrt theta (from RodLinkage.cc)
    ElasticRod &rod = pr.rod;
    const size_t ne = rod.numEdges();
    auto pts = rod.deformedPoints();
    auto ths = rod.thetas();

    // First, remove any unnecessary twist stored in the rod by rotating the second endpoint
    // by an integer multiple of 2PI (leaving d2 unchanged).
    Real rodRefTwist = 0;
    const auto &dc = rod.deformedConfiguration();
    for (size_t j = 1; j < ne; ++j)
        rodRefTwist += dc.referenceTwist[j];
    const size_t lastEdge = ne - 1;
    Real desiredTheta = ths[0] - rodRefTwist;
    // Probably could be implemented with an fmod...
    while (ths[lastEdge] - desiredTheta >  M_PI) ths[lastEdge] -= 2 * M_PI;
    while (ths[lastEdge] - desiredTheta < -M_PI) ths[lastEdge] += 2 * M_PI;

    if (verbose) {
        std::cout << "rodRefTwist: "         << rodRefTwist        << std::endl;
        std::cout << "desiredTheta: "        << desiredTheta       << std::endl;
        std::cout << "old last edge theta: " << dc.theta(lastEdge) << std::endl;
        std::cout << "new last edge theta: " << ths[lastEdge]      << std::endl;
    }
    
    rod.setDeformedConfiguration(pts, ths);

    auto H = rod.hessThetaEnergyTwist();
    auto g = rod.gradEnergyTwist();
    std::vector<Real> rhs(ne);
    for (size_t j = 0; j < ne; ++j)
        rhs[j] = -g.gradTheta(j);

    H.fixVariable(0, 0); // lock rotation in the first edge
    auto thetaStep = H.solve(rhs);

    for (size_t j = 0; j < ths.size(); ++j)
        ths[j] += thetaStep[j];

    rod.setDeformedConfiguration(pts, ths);
}

void spread_twist_preserving_link(PeriodicRod &pr, bool verbose) { 
    // Minimize twisting energy wrt theta while keeping 
    // the first and the last edges of the underlying ElasticRod fixed
    if (verbose) {
        std::cout << "Initial total twist: " << pr.twist() << std::endl; 
        std::cout << "Initial twisting energy: " << pr.energyTwist() << std::endl; 
    }

    ElasticRod &rod = pr.rod;
    const size_t ne = rod.numEdges();
    auto pts = rod.deformedPoints();
    auto ths = rod.thetas();

    auto H = rod.hessThetaEnergyTwist();
    auto g = rod.gradEnergyTwist();
    std::vector<Real> rhs(ne);
    for (size_t j = 0; j < ne; ++j)
        rhs[j] = -g.gradTheta(j);

    H.fixVariable(0,                   0.0);
    H.fixVariable(pr.rod.numEdges()-1, 0.0);
    auto thetaStep = H.solve(rhs);

    for (size_t j = 0; j < ths.size(); ++j)
        ths[j] += thetaStep[j];

    rod.setDeformedConfiguration(pts, ths);

    if (verbose) {
        std::cout << "Final total twist: " << pr.twist() << std::endl; 
        std::cout << "Final twisting energy: " << pr.energyTwist() << std::endl; 
    }
}

ConvergenceReport compute_equilibrium(
    PeriodicRodList rods,
    const SlidingProblemOptions &problemOptions,
    const NewtonOptimizerOptions &optimizerOptions, 
    std::vector<size_t> fixedVars, 
    const Eigen::VectorXd &externalForces,
    const SlidingProblem::SoftConstraintsList &softConstraints,
    CallbackFunction customCallback,
    Real hessianShift
    ) {
    std::unique_ptr<SlidingProblem> problem = std::make_unique<SlidingProblem>(rods, problemOptions);
    problem->addFixedVariables(fixedVars);
    if (externalForces.size() > 0) {
        assert((size_t)externalForces.size() == rods.numDoF());
        problem->external_forces = externalForces;
    }
    problem->addSoftConstraints(softConstraints);
    problem->hessianShift = hessianShift;
    if (customCallback)
        problem->setCustomIterationCallback(customCallback);
    std::unique_ptr<NewtonOptimizer> optimizer = std::make_unique<NewtonOptimizer>(std::move(problem));
    optimizer->options = optimizerOptions;
    return optimizer->optimize();
}
