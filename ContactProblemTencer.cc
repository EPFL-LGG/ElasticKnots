#include "ContactProblemTencer.hh"


std::pair<Real, size_t> ContactProblemTencer::feasibleStepLength(const Eigen::VectorXd &vars, const Eigen::VectorXd &step) const {
    BENCHMARK_SCOPED_TIMER_SECTION timer("computeFeasibleStepSize");
    // std::cout << "feasible step length " << std::endl;
    // std::cout << "feasible step length " << std::endl;
    Real alpha = std::numeric_limits<Real>::max();
    size_t blocking_idx = std::numeric_limits<size_t>::max();

    for (size_t i = 0; i < m_boundConstraints.size(); ++i) {
        Real len = m_boundConstraints[i].feasibleStepLength(vars, step);
        if (len < alpha) { alpha = len; blocking_idx = i; }
    }

    if (m_options.hasCollisions) { 
        // std::cout << "feasible step length 2" << std::endl;
        // std::cout << "feasible step length 2" << std::endl;
        Eigen::VectorXd nodalVars = m_tencer.extractNodalDoFs(vars.head(numVars()));
        Eigen::VectorXd nodalStep = m_tencer.extractNodalDoFs(step.head(numVars()));
        //  std::cout << "feasible step length 4" << std::endl;
        //  std::cout << "feasible step length 4" << std::endl;
        const Eigen::MatrixXd V0 = Eigen::Map<Eigen::Matrix<Real, Eigen::Dynamic, 3, Eigen::RowMajor>>(nodalVars.data(), m_tencer.numVertices() + m_tencer.num_spring_free_vertices(), 3);
        const Eigen::MatrixXd V1 = V0 + Eigen::Map<Eigen::Matrix<Real, Eigen::Dynamic, 3, Eigen::RowMajor>>(nodalStep.data(), m_tencer.numVertices() + m_tencer.num_spring_free_vertices(), 3);
        // std::cout << "V0" << V0 << std::endl;
        // std::cout << "V1" << V1 << std::endl;
        alpha = compute_collision_free_stepsize(m_collisionMesh, V0, V1, ipc::BroadPhaseMethod::HASH_GRID, m_options.dHat, m_options.Wang2021MaxIter);
        // std::cout << "feasible step length 3" << std::endl;
        // std::cout << "feasible step length 3" << std::endl;
    }

    if (m_options.printIterInfo) {
        std::string alphaStr = (alpha == std::numeric_limits<Real>::max()) ? "+inf" : std::to_string(alpha);
        std::cout << "Feasible alpha = " << alphaStr << std::endl;
    }

    // std::cout << "feasible step length 1" << std::endl;
    // std::cout << "feasible step length 1" << std::endl;

    return std::make_pair(alpha, blocking_idx);
}

void ContactProblemTencer::m_iterationCallback(size_t i) {
    m_tencer.updateSourceFrame(); 
    m_tencer.updateRotationParametrizations();
        
    updateCachedVars();
    updateCharacteristicLength();
    updateCachedSparsityPattern();

    BENCHMARK_START_TIMER("CustomCallback");
    if (m_customCallback) m_customCallback(*this, i);
    BENCHMARK_STOP_TIMER("CustomCallback");
}

ConvergenceReport compute_equilibrium(
    ContactTencer& tencer,
    const ContactProblemOptions &problemOptions,
    const NewtonOptimizerOptions &optimizerOptions, 
    std::vector<size_t> fixedVars, 
    const Eigen::VectorXd &externalForces,
    CallbackFunction customCallback,
    double hessianShift
    ) {
    std::unique_ptr<ContactProblemTencer> problem = std::make_unique<ContactProblemTencer>(tencer, problemOptions);
    problem->addFixedVariables(fixedVars);
    if (externalForces.size() > 0) {
        assert((size_t)externalForces.size() == tencer.numDefoVars());
        problem->external_forces = externalForces;
    }
    problem->hessianShift = hessianShift;
    if (customCallback)
        problem->setCustomIterationCallback(customCallback);
    std::unique_ptr<NewtonOptimizer> optimizer = std::make_unique<NewtonOptimizer>(std::move(problem));
    optimizer->options = optimizerOptions;
    // std::cout << "optimize" << std::endl;
    // std::cout << "optimize" << std::endl;
    auto report = optimizer->optimize();
    // std::cout << "optimize ok" << std::endl;
    // std::cout << "optimize ok" << std::endl;
    return report;
}
