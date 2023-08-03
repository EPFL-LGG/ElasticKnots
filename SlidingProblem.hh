
#ifndef SLIDING_PROBLEM_HH
#define SLIDING_PROBLEM_HH

#include <utility>
#include <algorithm>
#include <Eigen/Core>
#include "PeriodicRodList.hh"
#include "SoftConstraint.hh"
#include <tight_inclusion/ccd.hpp>
#include <ipc/ipc.hpp>
#include <MeshFEM/newton_optimizer/newton_optimizer.hh>

#include <functional>

using CallbackFunction = std::function<void(NewtonProblem &, size_t)>;

struct SlidingProblemOptions {  // TODO KNOTS: rename
    bool hasCollisions = true;               // Detect collision and prevent penetration
    bool printIterInfo = false;              // Print information at each iteration
    size_t minContactEdgeDist = 1;           // The minimum distance (in indices) for two edges to be included in the contact set. Note: values > 1 break guarantees of topology preservation.
    size_t Wang2021MaxIter = 1e6;            // Maximum number of iterations for Tight-Inclusion CCD [Wang et al. 2021]
    bool projectContactHessianPSD = false;   // Project each individual contacts' Hessian to make them positive semi-definite before assembling the problem Hessian
    Real contactStiffness = 1.0;             // Relative stiffness of the contact constraints
    Real dHat = 1e-7;                        // Distance at which the barrier eneregy is clamped to zero (repulsion only for distance < dHat, see [Li et al. 2020])
};

struct SlidingProblem : public NewtonProblem {
    using SoftConstraintsList = std::vector<std::shared_ptr<SoftConstraint>>;

    SlidingProblem(
        PeriodicRodList &rods, 
        SlidingProblemOptions options = SlidingProblemOptions()
        ) : m_rods(rods), m_options(options) {

        updateCachedVars();
        updateCachedSparsityPattern();
        updateCharacteristicLength();

        rods.updateSourceFrame();

        if (m_options.hasCollisions) {
            // Initialize CollisionMesh with the connectivity information required by IPC
            Eigen::MatrixXd vertices(m_rods.numVertices(), 3);
            Eigen::MatrixXi edges(m_rods.numEdges(), 2);
            Eigen::MatrixXi faces(0, 3);  // no faces
            for (size_t ri = 0; ri < m_rods.size(); ri++) {
                const size_t fni = m_rods.firstGlobalNodeIndexInRod(ri);
                const size_t nvr = m_rods.numVerticesInRod(ri);
                for (size_t i = 0; i < nvr; i++) {
                    size_t ni = fni + i;
                    vertices.row(ni) = m_rods.getNode(ni);
                    edges(ni, 0) = ni;
                    edges(ni, 1) = ni + 1;
                }
                edges(fni + nvr - 1, 1) = fni;  // overwrite second node in last edge
            }
            m_collisionMesh = ipc::CollisionMesh(vertices, edges, faces);

            updateConstraintSet();
            updateCachedSparsityPattern();
        }

        // Compute minimium edge rest length and check its compatibility with the constraint barrier thickness
        Real min_rl = std::numeric_limits<Real>::infinity();
        for (size_t ri = 0; ri < m_rods.size(); ri++) {
            const std::vector<Real> rli = m_rods[ri]->restLengths();
            Real min_rli = *std::min_element(rli.begin(), rli.end());
            if (min_rli < min_rl)
                min_rl = min_rli;
        }
        if (min_rl < m_options.dHat)
            std::cerr << "WARNING: The minimum edge rest length is smaller than the constraint barrier thickness. "
                         "The simulations will continue, but the result might be non-physical due to spurious contact forces between neighboring edges. "
                         "Consider increasing corss-section radius or using a coarser polyline. "
                         "Increasing the minContactEdgeDist parameter would remove non-physical contact forces at the expense of topology preservation guarantees." << std::endl;
    }

    virtual void setVars(const Eigen::VectorXd &vars) override {
        m_rods.setDoFs(vars.head(numVars()));
        if (m_options.hasCollisions)
            updateConstraintSet();
        m_cachedVars = vars;
    }
    virtual const Eigen::VectorXd getVars() const override { return m_cachedVars; }
    virtual size_t numVars() const override { return m_rods.numDoF(); }
    size_t numRods() const { return m_rods.size(); }
    size_t numIPCConstraints() const { return m_constraintSet.size(); }
    
    virtual bool hasCollisions() const { return m_options.hasCollisions; };

    void addSoftConstraint(const std::shared_ptr<SoftConstraint> &sc) { m_softConstraints.push_back(sc); }
    void addSoftConstraints(const std::vector<std::shared_ptr<SoftConstraint>> &scList) { for (auto &sc : scList) m_softConstraints.push_back(sc); }

    virtual Real energy() const override { 
        Real e = m_rods.energy();
        e += contactEnergy();
        e += externalPotentialEnergy();
        for (auto &sc : m_softConstraints)
            sc->energy(m_rods, e);
        return e;
    }

    virtual Eigen::VectorXd gradient(bool freshIterate = false) const override {
        Eigen::VectorXd result = Eigen::VectorXd::Zero(numVars());
        result.head(numVars()) = m_rods.gradient(freshIterate);        // rods
        if (m_options.hasCollisions) {
            Eigen::VectorXd bpGrad = m_options.contactStiffness * compute_barrier_potential_gradient(m_collisionMesh, m_rods.deformedPointsMatrix(), m_constraintSet, m_options.dHat);
            for (size_t ri = 0; ri < m_rods.size(); ri++)
                result.segment(m_rods.firstGlobalDofIndexInRod(ri), 3*m_rods.numVerticesInRod(ri)) += bpGrad.segment(3*m_rods.firstGlobalNodeIndexInRod(ri), 3*m_rods.numVerticesInRod(ri));
        }
        if (external_forces.size() > 0) {                              // external potential energy
            assert((size_t)external_forces.size() == numVars());
            result.head(numVars()) -= external_forces;
        }
        for (auto &sc : m_softConstraints)                             // soft constraints
            sc->gradient(m_rods, result);
        return result;
    }

    Real contactEnergy() const {
        Real energy = 0;
        if (m_options.hasCollisions)
            energy = m_options.contactStiffness * compute_barrier_potential(m_collisionMesh, m_rods.deformedPointsMatrix(), m_constraintSet, m_options.dHat);
        return energy;
    }

    Eigen::MatrixXd contactForces() const {
        Eigen::MatrixXd result = Eigen::MatrixXd::Zero(m_rods.numVertices(), 3);
        if (m_options.hasCollisions) {
            Eigen::VectorXd bpGrad = m_options.contactStiffness * compute_barrier_potential_gradient(m_collisionMesh, m_rods.deformedPointsMatrix(), m_constraintSet, m_options.dHat);
            for (size_t i = 0; i < m_rods.numEdges(); i++)
                result.row(i) = - bpGrad.segment(3*i, 3);
        }
        return result;
    }

    Real externalPotentialEnergy() const {
        if (external_forces.size() == 0) return 0.0;
        auto x = m_rods.getDoFs();
        if (external_forces.size() != x.size()) throw std::runtime_error("Invalid external force vector");
        return -external_forces.dot(x);
    }

    virtual SuiteSparseMatrix hessianSparsityPattern() const override { return m_hessianSparsity; }

    virtual std::pair<Real, size_t> feasibleStepLength(const Eigen::VectorXd &vars, const Eigen::VectorXd &step) const override;

    // "Physical" distance of a step relative to some characteristic lengthscale of the problem.
    // (Useful for determining reasonable step lengths to take when the Newton step is not possible.)
    // Note: overridden since the estimation of velocity only needs the DER dofs and not the material variables   
    virtual Real characteristicDistance(const Eigen::VectorXd &d) const override {
        return m_rods.approxLinfVelocity(d.head(numVars())) / m_characteristicLength;
    }

    virtual void writeIterateFiles(size_t /*it*/)                   const override { if (writeIterates) { assert(false); } }
    virtual void writeDebugFiles(const std::string &/*errorName*/)  const override { assert(false); }
    virtual void customIterateReport(ConvergenceReport &/*report*/) const override {  }
    void setCustomIterationCallback(const CallbackFunction &cb) { m_customCallback = cb; }

    // The external generalized forces acting on each degree of freedom.
    // For position variables, these are true forces, while for other degrees of freedom
    // these act as a one-form computing the work done by a perturbation to the
    // degrees of freedom.
    // When this vector is empty, no forces are applied.
    // These forces can be used to apply gravity, custom actuation torques, or any
    // other loading scenario.
    Eigen::VectorXd external_forces;

    void updateConstraintSet() {
        BENCHMARK_START_TIMER_SECTION("Build constraint set");
        m_constraintSet.build(m_collisionMesh, m_rods.deformedPointsMatrix(), m_options.dHat);
        BENCHMARK_STOP_TIMER_SECTION("Build constraint set");

        clearConstraintsBetweenNeighboringEdges();
    }

    void clearConstraintsBetweenNeighboringEdges() {
        BENCHMARK_SCOPED_TIMER_SECTION timer("Remove constraints on neighboring edges");

        // minContactEdgeDist controls the neighborhood size in indices units.
        // The default value is 1, which excluded adjacent edges from the contact set; 
        // values > 1 should be used for fine meshes (i.e. when edges' length is comparable or smaller 
        // than the radius of the collision mesh), but break guarantees of no false negatives.
        size_t mdi = m_options.minContactEdgeDist;
        const Eigen::MatrixXi &E = m_collisionMesh.edges();
        const Eigen::MatrixXi &F = m_collisionMesh.faces();
        auto &vv_const = m_constraintSet.vv_constraints;
        auto &ev_const = m_constraintSet.ev_constraints;
        auto &ee_const = m_constraintSet.ee_constraints;

        for (int i = 0; i < int(vv_const.size()); i++) {
            const auto &vertex_indices = vv_const[i].vertex_indices(E, F);
            if (m_rods.elementsAreNeighbors(vertex_indices[0], vertex_indices[1], mdi)) {
                vv_const.erase(vv_const.begin() + i);
                i--;
            }
        }
        for (int i = 0; i < int(ev_const.size()); i++) {
            const auto &vertex_indices = ev_const[i].vertex_indices(E, F);
            if (m_rods.elementsAreNeighbors(vertex_indices[0], vertex_indices[1], mdi) || 
                m_rods.elementsAreNeighbors(vertex_indices[0], vertex_indices[2], mdi)) {
                ev_const.erase(ev_const.begin() + i);
                i--;
            }
        }
        for (int i = 0; i < int(ee_const.size()); i++) {
            const auto &vertex_indices = ee_const[i].vertex_indices(E, F);
            if (m_rods.elementsAreNeighbors(vertex_indices[0], vertex_indices[2], mdi) || 
                m_rods.elementsAreNeighbors(vertex_indices[0], vertex_indices[3], mdi) ||
                m_rods.elementsAreNeighbors(vertex_indices[1], vertex_indices[2], mdi) || 
                m_rods.elementsAreNeighbors(vertex_indices[1], vertex_indices[3], mdi)) {
                ee_const.erase(ee_const.begin() + i);
                i--;
            }
        }
    }

    virtual void m_iterationCallback(size_t i) override;

    // Use the diagonal of the bounding box as characteristic length 
    // (instead of the length of the rod used by EquilibriumProblem)
    void updateCharacteristicLength() { 
        Pt3 bbMin = Eigen::Vector3d::Ones()*std::numeric_limits<Real>::max();
        Pt3 bbMax = Eigen::Vector3d::Ones()*std::numeric_limits<Real>::min();
        const std::vector<Pt3> &pts = m_rods.deformedPoints();
        const size_t nv = m_rods.numVertices();
        for (size_t i = 0; i < nv; i++) {
            bbMin = bbMin.cwiseMin(pts[i]);
            bbMax = bbMax.cwiseMax(pts[i]);
        }
        m_characteristicLength = (bbMax - bbMin).norm();
    }

    void updateCachedVars() {
        m_cachedVars.resize(numVars());
        m_cachedVars.head(m_rods.numDoF()) = m_rods.getDoFs();
    }

    void updateCachedSparsityPattern() {
        BENCHMARK_SCOPED_TIMER_SECTION timer("updateCachedSparsityPattern");

        // Compute the constant part of the sparsity pattern only once.
        if (m_rodHessianSparsity.nnz() == 0) {
            m_rodHessianSparsity = m_rods.hessianSparsityPattern();
            m_hessianSparsity = m_rodHessianSparsity;
        }

        if (m_options.hasCollisions) {
            BENCHMARK_SCOPED_TIMER_SECTION contactsTimer("updateCachedSparsityPattern_contacts");
            TripletMatrix<Triplet<Real>> triplets(numVars(), numVars());
            const Real dummy = 1;  // adding zero in a sparse triplet matrix skips the entry
            auto set3x3Block = [&](size_t a, size_t b) {
                size_t i = std::min(a, b);
                size_t j = std::max(a, b);
                for (size_t ii = i; ii < i+3; ii++) {
                    for (size_t jj = j; jj < j+3; jj++) {
                        if (jj < ii) continue;
                        triplets.addNZ(ii, jj, dummy);
                    }
                }
            };
            const Eigen::MatrixXi &E = m_collisionMesh.edges();
            const Eigen::MatrixXi &F = m_collisionMesh.faces();
            for (size_t i = 0; i < m_constraintSet.size(); i++) {
                const auto &vertex_indices = m_constraintSet[i].vertex_indices(E, F);
                for (auto i : vertex_indices) {
                    if (i != -1) {  // vertex_indices always has size 4; if e.g. the constraint is edge-vertex, the last index will be -1
                        const size_t dofi = m_rods.globalDofIndexFromGlobalNodeIndex(i);
                        for (auto j : vertex_indices) {
                            if (j != -1) {
                                const size_t dofj = m_rods.globalDofIndexFromGlobalNodeIndex(j);
                                set3x3Block(dofi, dofj);
                            }
                        }
                    }
                }
            }
            SuiteSparseMatrix hspIPC(triplets.m, triplets.n);
            if (triplets.nz.size() != 0)   // otherwise, hspIPC will be safely left empty
                hspIPC.setFromTMatrix(triplets);
            hspIPC.fill(0.0);
            hspIPC.symmetry_mode = SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE;

            if (!m_contactHessianSparsity.sparsityPatternsMatch(hspIPC)) {
                m_contactHessianSparsity = std::move(hspIPC);
                m_hessianSparsity = SuiteSparseMatrix::addWithDistinctSparsityPattern(m_rodHessianSparsity, m_contactHessianSparsity);
                m_sparsityPatternFactorizationUpToDate = false;
            }
        }
    }

    virtual void m_evalHessian(SuiteSparseMatrix &result, bool projectionMask) const override {
        BENCHMARK_SCOPED_TIMER_SECTION timer("m_evalHessian");
        
        result = m_hessianSparsity;
        BENCHMARK_START_TIMER("m_evalHessian_rod");
        m_rods.hessian(result);
        BENCHMARK_STOP_TIMER("m_evalHessian_rod");
        BENCHMARK_START_TIMER("m_evalHessian_contacts");
        if (m_options.hasCollisions) {
            const bool projectIPCHessian = projectionMask && m_options.projectContactHessianPSD;
            Eigen::SparseMatrix<double> IPCHessianEigen = m_options.contactStiffness * compute_barrier_potential_hessian(m_collisionMesh, m_rods.deformedPointsMatrix(), m_constraintSet, m_options.dHat, projectIPCHessian);

            // Convert Eigen::SparseMatrix into TripletMatrix; convert dofs from nodes-only to with-theta-vars.
            auto to_upper_triangular_triplet_matrix = [&](Eigen::SparseMatrix<double> & M){
                TripletMatrix<Triplet<Real>> triplet_matrix(numVars(), numVars());
                for (int i = 0; i < M.outerSize(); i++) {
                    for (typename Eigen::SparseMatrix<double>::InnerIterator it(M, i); it; ++it) {
                        const size_t node_row = size_t(floor(it.row()/3));
                        const size_t node_col = size_t(floor(it.col()/3));
                        const size_t row = m_rods.globalDofIndexFromGlobalNodeIndex(node_row) + it.row() % 3;  // adapt dof index to DER extended variables (with thetas)
                        const size_t col = m_rods.globalDofIndexFromGlobalNodeIndex(node_col) + it.col() % 3;
                        if (col < row) continue;
                        triplet_matrix.addNZ(row, col, it.value());
                    }
                }
                return triplet_matrix;
            };

            SuiteSparseMatrix IPCHessian;
            TripletMatrix<Triplet<Real>> IPCHessianTriplet = to_upper_triangular_triplet_matrix(IPCHessianEigen);
            if (IPCHessianTriplet.nz.size() != 0)  // otherwise, IPCHessian will be safely left empty
                IPCHessian.setFromTMatrix(IPCHessianTriplet);
            IPCHessian.symmetry_mode = SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE;
            result.addWithSubSparsity(IPCHessian);
        }
        BENCHMARK_STOP_TIMER("m_evalHessian_contacts");
        BENCHMARK_START_TIMER("m_evalHessian_softConstraints");
        for (auto &sc : m_softConstraints)
            sc->hessian(m_rods, result);
        BENCHMARK_STOP_TIMER("m_evalHessian_softConstraints");

        if (hessianShift != 0.0)
            result.addScaledIdentity(hessianShift);
    }

    virtual void m_evalMetric(SuiteSparseMatrix &result) const override {
        result.setZero();
        SuiteSparseMatrix rodsMassMatrix = m_rods.hessianSparsityPattern();
        m_rods.massMatrix(rodsMassMatrix, /* updated source; evaluated at the same time as the Hessian */ true, /* useLumped = */ true);
        result.addWithSubSparsity(rodsMassMatrix);
    }

    TripletMatrix<Triplet<Real>> hessian() const {
        SuiteSparseMatrix H = hessianSparsityPattern();
        m_evalHessian(H, /* projectionMask */ true);
        TripletMatrix<Triplet<Real>> Htrip = H.getTripletMatrix();
        Htrip.symmetry_mode = TripletMatrix<Triplet<Real>>::SymmetryMode::UPPER_TRIANGLE;
        return Htrip;
    }

    Real hessianShift = 0.0;  // Multiple of the identity to add to the Hessian on each evaluation.

    mutable SuiteSparseMatrix m_hessianSparsity, m_rodHessianSparsity, m_contactHessianSparsity;
    Real m_characteristicLength = 1.0;
    CallbackFunction m_customCallback;

    PeriodicRodList &m_rods;
    Eigen::VectorXd m_cachedVars;    // [r1, ..., rn], where ri = [x1, y1, z1, ..., xk, yk, zk, th1, ..., thk-1].
    SoftConstraintsList m_softConstraints;
    SlidingProblemOptions m_options;

    // ipc-toolkit
    ipc::Constraints m_constraintSet;
    ipc::CollisionMesh m_collisionMesh;  // Mesh connectivity
};


void minimize_twist(PeriodicRod &rod, bool verbose = false);
void spread_twist_preserving_link(PeriodicRod &pr, bool verbose = false);


ConvergenceReport compute_equilibrium(
    PeriodicRodList rods,
    const SlidingProblemOptions &problemOptions = SlidingProblemOptions(),
    const NewtonOptimizerOptions &optimizerOptions = NewtonOptimizerOptions(), 
    std::vector<size_t> fixedVars = std::vector<size_t>(), 
    const Eigen::VectorXd &externalForces = Eigen::VectorXd(),
    const SlidingProblem::SoftConstraintsList &softConstraints = SlidingProblem::SoftConstraintsList(),
    CallbackFunction customCallback = nullptr,
    double hessianShift = 0.0
);

#endif /* end of include guard: SLIDING_PROBLEM_HH */
