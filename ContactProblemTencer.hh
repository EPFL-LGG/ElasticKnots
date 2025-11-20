
#ifndef CONTACT_PROBLEM_TENCER_HH
#define CONTACT_PROBLEM_TENCER_HH

#include <utility>
#include <algorithm>
#include <Eigen/Core>
#include "ContactTencer.hh"
#include "ContactProblem.hh"
#include "SoftConstraint.hh"
#include <tight_inclusion/ccd.hpp>
#include <ipc/ipc.hpp>
#include <MeshFEM/newton_optimizer/newton_optimizer.hh>

#include <functional>

using CallbackFunction = std::function<void(NewtonProblem &, size_t)>;


struct ContactProblemTencer : public NewtonProblem {
    using SoftConstraintsList = std::vector<std::shared_ptr<SoftConstraint>>;

    ContactProblemTencer(
        ContactTencer &tencer, 
        ContactProblemOptions options = ContactProblemOptions()
        ) : m_tencer(tencer), m_options(options) {

        updateCachedVars();
        updateCachedSparsityPattern();
        updateCharacteristicLength();

        m_tencer.updateSourceFrame();

        // TODO : add spring collisions
        if (m_options.hasCollisions) {
            // Initialize CollisionMesh with the connectivity information required by IPC
            Eigen::MatrixXd vertices(m_tencer.numVertices() + m_tencer.num_spring_free_vertices(), 3);
            Eigen::MatrixXi edges(m_tencer.numEdges() + m_tencer.num_spring_edges(), 2);
            Eigen::MatrixXi faces(0, 3);  // no faces
            for (size_t ri = 0; ri < m_tencer.closed_rods.size(); ri++) {
                const size_t fni = m_tencer.closed_rods.firstGlobalNodeIndexInRod(ri);
                const size_t nvr = m_tencer.closed_rods.numVerticesInRod(ri);
                for (size_t i = 0; i < nvr; i++) {
                    size_t ni = fni + i;
                    vertices.row(ni) = m_tencer.closed_rods.getNode(ni);
                    edges(ni, 0) = ni;
                    edges(ni, 1) = ni + 1;
                }
                edges(fni + nvr - 1, 1) = fni;  // overwrite second node in last edge
            }
            size_t vertex_idx = m_tencer.numVertices();
            size_t edge_idx = m_tencer.numEdges();
            auto av = m_tencer.get_attachment_vertices();
            size_t current_spring_vertex_idx = 0;
            size_t prev_spring_vertex_idx = 0;
            for (size_t si = 0; si < m_tencer.num_springs(); ++si){
                int attachment_idx = 0;
                for (size_t j = 0; j < m_tencer.springs[si].get_num_points(); ++j){
                    if (av[si].spring_vertices[attachment_idx] == j){
                        current_spring_vertex_idx = m_tencer.closed_rods.firstGlobalNodeIndexInRod(av[si].rod_idx[attachment_idx]) + av[si].rod_vertices[attachment_idx];
                        attachment_idx ++;
                    }
                    else{
                        current_spring_vertex_idx = vertex_idx;
                        vertices.row(vertex_idx) = m_tencer.springs[si].positions[j];
                        vertex_idx ++;
                    }
                    if (j > 0){
                        edges(edge_idx, 0) = prev_spring_vertex_idx;
                        edges(edge_idx, 1) = current_spring_vertex_idx;
                        edge_idx ++;
                    }
                    prev_spring_vertex_idx = current_spring_vertex_idx;
                }
            }
            m_collisionMesh = ipc::CollisionMesh(vertices, edges, faces);

            // // debug 
            // std::cout << "vertices" << std::endl;
            // for (size_t i = 0; i < m_tencer.numVertices() + m_tencer.num_spring_free_vertices(); ++i){
            //     for (int j = 0; j < 3; ++j){
            //         std::cout << vertices(i,j) << " " ;
            //     }
            //     std::cout << std::endl;
            // }
            // std::cout << "edges" << std::endl;
            // for (size_t i = 0; i < m_tencer.numEdges() + m_tencer.num_spring_edges(); ++i){
            //     for (int j = 0; j < 2; ++j){
            //         std::cout << edges(i,j) << " " ;
            //     }
            //     std::cout << std::endl;
            // }
            // m_tencer.print_neighbors(m_options.minContactEdgeDist);

            updateConstraintSet();
            updateCachedSparsityPattern();
        }

        // Compute minimium edge rest length and check its compatibility with the constraint barrier thickness
        Real min_rl = std::numeric_limits<Real>::infinity();
        for (size_t ri = 0; ri < m_tencer.closed_rods.size(); ri++) {
            const std::vector<Real> rli = m_tencer.closed_rods[ri]->restLengths();
            Real min_rli = *std::min_element(rli.begin(), rli.end());
            if (min_rli < min_rl)
                min_rl = min_rli;
        }
        if (min_rl < m_options.dHat)
            std::cerr << "WARNING: The minimum edge rest length is smaller than the constraint barrier thickness.\n"
                         "The simulations will continue, but the result might be non-physical due to spurious contact forces between neighboring edges.\n"
                         "Consider decreasing the cross-section radius or using a coarser polyline.\n"
                         "Increasing the minContactEdgeDist parameter would remove non-physical contact forces, too, but only at the expense of topology preservation guarantees.\n" << std::endl;
    }

    ContactTencer get_tencer_copy() {return ContactTencer(m_tencer);}
    virtual void setVars(const Eigen::VectorXd &vars) override {
        m_tencer.setDefoVars(vars.head(numVars()));
        if (m_options.hasCollisions)
            updateConstraintSet();
        m_cachedVars = vars;
    }
    virtual const Eigen::VectorXd getVars() const override { return m_cachedVars; }
    virtual size_t numVars() const override { return m_tencer.numDefoVars(); }
    size_t numRods() const { return m_tencer.closed_rods.size(); }
    size_t numIPCConstraints() const { return m_constraintSet.size(); }
    
    virtual bool hasCollisions() const { return m_options.hasCollisions; };

    virtual Real energy() const override { 
        Real e = m_tencer.energy();
        e += contactEnergy();
        e += externalPotentialEnergy();
        return e;
    }

    virtual Eigen::VectorXd gradient(bool freshIterate = false) const override {
        Eigen::VectorXd result = Eigen::VectorXd::Zero(numVars());
        result.head(numVars()) = m_tencer.gradient(freshIterate);        // rods
        if (m_options.hasCollisions) {
            Eigen::VectorXd bpGrad = m_options.contactStiffness * compute_barrier_potential_gradient(m_collisionMesh, m_tencer.deformedPointsMatrix(), m_constraintSet, m_options.dHat);
            for (size_t ri = 0; ri < m_tencer.closed_rods.size(); ri++)
                result.segment(m_tencer.firstGlobalDofIndexInRod(ri), 3*m_tencer.numVerticesInRod(ri)) += bpGrad.segment(3*m_tencer.firstGlobalNodeIndexInRod(ri), 3*m_tencer.numVerticesInRod(ri));
            result.tail(m_tencer.num_spring_free_vertices()*3) += bpGrad.tail(m_tencer.num_spring_free_vertices()*3);
        }
        if (external_forces.size() > 0) {                              // external potential energy
            assert((size_t)external_forces.size() == numVars());
            result.head(numVars()) -= external_forces;
        }
        return result;
    }

    Real contactEnergy() const {
        Real energy = 0;
        if (m_options.hasCollisions)
            energy = m_options.contactStiffness * compute_barrier_potential(m_collisionMesh, m_tencer.deformedPointsMatrix(), m_constraintSet, m_options.dHat);
        return energy;
    }

    Eigen::MatrixXd contactForces() const {
        size_t n = m_tencer.numVertices() + m_tencer.num_spring_free_vertices();
        Eigen::MatrixXd result = Eigen::MatrixXd::Zero(n, 3);
        if (m_options.hasCollisions) {
            Eigen::VectorXd bpGrad = m_options.contactStiffness * compute_barrier_potential_gradient(m_collisionMesh, m_tencer.deformedPointsMatrix(), m_constraintSet, m_options.dHat);
            for (size_t i = 0; i < n; i++)
                result.row(i) = - bpGrad.segment(3*i, 3);
        }
        return result;
    }

    Real externalPotentialEnergy() const {
        if (external_forces.size() == 0) return 0.0;
        auto x = m_tencer.getDefoVars();
        if (external_forces.size() != x.size()) throw std::runtime_error("Invalid external force vector");
        return -external_forces.dot(x);
    }

    virtual SuiteSparseMatrix hessianSparsityPattern() const override { return m_hessianSparsity; }

    virtual std::pair<Real, size_t> feasibleStepLength(const Eigen::VectorXd &vars, const Eigen::VectorXd &step) const override;

    // "Physical" distance of a step relative to some characteristic lengthscale of the problem.
    // (Useful for determining reasonable step lengths to take when the Newton step is not possible.)
    // Note: overridden since the estimation of velocity only needs the DER dofs and not the material variables   
    virtual Real characteristicDistance(const Eigen::VectorXd &d) const override {
        return m_tencer.approxLinfVelocity(d.head(numVars())) / m_characteristicLength;
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
        m_constraintSet.build(m_collisionMesh, m_tencer.deformedPointsMatrix(), m_options.dHat);
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
            if (m_tencer.elementsAreNeighbors(vertex_indices[0], vertex_indices[1], mdi)) {
                vv_const.erase(vv_const.begin() + i);
                i--;
            }
        }
        for (int i = 0; i < int(ev_const.size()); i++) {
            const auto &vertex_indices = ev_const[i].vertex_indices(E, F);
            if (m_tencer.elementsAreNeighbors(vertex_indices[0], vertex_indices[1], mdi) || 
                m_tencer.elementsAreNeighbors(vertex_indices[0], vertex_indices[2], mdi)) {
                ev_const.erase(ev_const.begin() + i);
                i--;
            }
        }
        for (int i = 0; i < int(ee_const.size()); i++) {
            const auto &vertex_indices = ee_const[i].vertex_indices(E, F);
            if (m_tencer.elementsAreNeighbors(vertex_indices[0], vertex_indices[2], mdi) || 
                m_tencer.elementsAreNeighbors(vertex_indices[0], vertex_indices[3], mdi) ||
                m_tencer.elementsAreNeighbors(vertex_indices[1], vertex_indices[2], mdi) || 
                m_tencer.elementsAreNeighbors(vertex_indices[1], vertex_indices[3], mdi)) {
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
        const std::vector<Pt3> &pts = m_tencer.deformedPoints();
        const size_t nv = m_tencer.numVertices();
        for (size_t i = 0; i < nv; i++) {
            bbMin = bbMin.cwiseMin(pts[i]);
            bbMax = bbMax.cwiseMax(pts[i]);
        }
        m_characteristicLength = (bbMax - bbMin).norm();
    }

    void updateCachedVars() {
        m_cachedVars.resize(numVars());
        m_cachedVars.head(m_tencer.numDefoVars()) = m_tencer.getDefoVars();
    }

    void print_constraint_set(){
        const Eigen::MatrixXi &E = m_collisionMesh.edges();
        const Eigen::MatrixXi &F = m_collisionMesh.faces();
        for (size_t i = 0; i < m_constraintSet.size(); i++) {
            const auto &vertex_indices = m_constraintSet[i].vertex_indices(E, F);
            for (auto i : vertex_indices) {
                std::cout << i << " ";
            }
            std::cout << std::endl;
        }
    }

    void updateCachedSparsityPattern() {
        BENCHMARK_SCOPED_TIMER_SECTION timer("updateCachedSparsityPattern");

        // Compute the constant part of the sparsity pattern only once.
        if (m_rodHessianSparsity.nnz() == 0) {
            m_rodHessianSparsity = m_tencer.hessianSparsityPattern();
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
                        const size_t dofi = m_tencer.globalDofIndexFromGlobalNodeIndex(i);
                        for (auto j : vertex_indices) {
                            if (j != -1) {
                                const size_t dofj = m_tencer.globalDofIndexFromGlobalNodeIndex(j);
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
        // std::cout << "m_evalHessian" << std::endl;
        // std::cout << "m_evalHessian" << std::endl;
        BENCHMARK_SCOPED_TIMER_SECTION timer("m_evalHessian");
        
        result = m_hessianSparsity;
        BENCHMARK_START_TIMER("m_evalHessian_rod");
        m_tencer.hessian(result);
        BENCHMARK_STOP_TIMER("m_evalHessian_rod");
        BENCHMARK_START_TIMER("m_evalHessian_contacts");
        if (m_options.hasCollisions) {
            const bool projectIPCHessian = projectionMask && m_options.projectContactHessianPSD;
            Eigen::SparseMatrix<double> IPCHessianEigen = m_options.contactStiffness * compute_barrier_potential_hessian(m_collisionMesh, m_tencer.deformedPointsMatrix(), m_constraintSet, m_options.dHat, projectIPCHessian);

            // Convert Eigen::SparseMatrix into TripletMatrix; convert dofs from nodes-only to with-theta-vars.
            auto to_upper_triangular_triplet_matrix = [&](Eigen::SparseMatrix<double> & M){
                TripletMatrix<Triplet<Real>> triplet_matrix(numVars(), numVars());
                for (int i = 0; i < M.outerSize(); i++) {
                    for (typename Eigen::SparseMatrix<double>::InnerIterator it(M, i); it; ++it) {
                        const size_t node_row = size_t(floor(it.row()/3));
                        const size_t node_col = size_t(floor(it.col()/3));
                        const size_t row = m_tencer.globalDofIndexFromGlobalNodeIndex(node_row) + it.row() % 3;  // adapt dof index to DER extended variables (with thetas)
                        const size_t col = m_tencer.globalDofIndexFromGlobalNodeIndex(node_col) + it.col() % 3;
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

        if (hessianShift != 0.0)
            result.addScaledIdentity(hessianShift);

        // std::cout << "m_evalHessian ok" << std::endl;
        // std::cout << "m_evalHessian ok" << std::endl;
    }

    virtual void m_evalMetric(SuiteSparseMatrix &result) const override {
        result.setZero();
        SuiteSparseMatrix rodsMassMatrix = m_tencer.hessianSparsityPattern();
        m_tencer.massMatrix(rodsMassMatrix, /* updated source; evaluated at the same time as the Hessian */ true, /* useLumped = */ true);
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

    ContactTencer& m_tencer;
    Eigen::VectorXd m_cachedVars;    // [r1, ..., rn], where ri = [x1, y1, z1, ..., xk, yk, zk, th1, ..., thk-1].
    ContactProblemOptions m_options;

    // ipc-toolkit
    ipc::Constraints m_constraintSet;
    ipc::CollisionMesh m_collisionMesh;  // Mesh connectivity
};



ConvergenceReport compute_equilibrium(
    ContactTencer& tencer,
    const ContactProblemOptions &problemOptions = ContactProblemOptions(),
    const NewtonOptimizerOptions &optimizerOptions = NewtonOptimizerOptions(), 
    std::vector<size_t> fixedVars = std::vector<size_t>(), 
    const Eigen::VectorXd &externalForces = Eigen::VectorXd(),
    CallbackFunction customCallback = nullptr,
    double hessianShift = 0.0
);

#endif /* end of include guard: SLIDING_PROBLEM_HH */
