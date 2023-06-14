#ifndef PERIODIC_ROD_LIST_HH
#define PERIODIC_ROD_LIST_HH

#include <ElasticRods/PeriodicRod.hh>
#include "SparseMatricesUtilities.hh"

using Pt3 = Pt3_T<Real>;

struct PeriodicRodList {
    using EnergyType = typename PeriodicRod::EnergyType;

    PeriodicRodList(PeriodicRod &rod) {
        m_rods.push_back(std::make_shared<PeriodicRod>(rod));
        initializeCounters();
    }

    PeriodicRodList(const std::vector<PeriodicRod> &rods) {
        for (auto r : rods)
            m_rods.push_back(std::make_shared<PeriodicRod>(r));
        assert(rodsHaveCircularCrossSection());
        assert(rodsHaveSameRadius());
        initializeCounters();
    }

    bool rodsHaveCircularCrossSection() const {
        for (size_t i = 0; i < numRods(); i++) {
            Real r = m_rods[i]->crossSectionHeight(0) / 2;  // assume the cross-section is uniform along the rod
            Real a = m_rods[i]->crossSectionArea(0);
            if (abs(M_PI * r * r - a) / a > 1e-4)
                return false;
        }
        return true;
    }

    bool rodsHaveSameRadius() const {
        assert(numRods() > 0);
        Real r = m_rods[0]->crossSectionHeight(0);
        bool haveSameRadius = true;
        for (size_t i = 1; i < numRods(); i++)
            haveSameRadius = haveSameRadius && (r == m_rods[0]->crossSectionHeight(0));   // assume the cross-section is uniform along the rod
        return haveSameRadius;
    }

    size_t size() const { return m_rods.size(); }
    size_t numRods() const { return size(); }  // alias
    auto begin() const { return m_rods.begin(); }
    auto begin()       { return m_rods.begin(); }
    auto end()   const { return m_rods.end(); }
    auto end()         { return m_rods.end(); }
    std::shared_ptr<PeriodicRod> operator[](int i) const { if (i < 0 || i > (int)numRods()-1) throw std::runtime_error("rod " + std::to_string(i) + " does not exist"); return m_rods[i]; }
    std::shared_ptr<PeriodicRod>     getRod(int i) const { return operator[](i); }  // alias
    std::vector<std::shared_ptr<PeriodicRod>> &getRods() { return m_rods; }
    
    size_t numDoF() const { return m_numDofs; }
    size_t numVertices(bool countGhost = false) const { return countGhost ? m_numElems + 2*numRods() : m_numElems; }
    size_t numEdges   (bool countGhost = false) const { return countGhost ? m_numElems +   numRods() : m_numElems; }

    std::vector<size_t> numDoFPerRod()      const { return m_numDofsPerRod; }
    std::vector<size_t> numVerticesPerRod() const { return m_numElemsPerRod; }
    std::vector<size_t> numEdgesPerRod()    const { return m_numElemsPerRod; }

    size_t numDoFInRod(size_t i)      const { return m_numDofsPerRod[i]; }
    size_t numVerticesInRod(size_t i) const { return m_numElemsPerRod[i]; }
    size_t numEdgesInRod(size_t i)    const { return m_numElemsPerRod[i]; }

    Real crossSectionRadius() const {
        assert(rodsHaveCircularCrossSection());
        assert(rodsHaveSameRadius());
        return m_rods[0]->rod.material(0).crossSectionHeight / 2;
    }

    bool elementsAreNeighbors(int i, int j, int d = 1) const {
        BENCHMARK_SCOPED_TIMER_SECTION("elementsAreNeighbors");
        size_t ri = rodIndexFromGlobalNodeIndex(i);
        size_t rj = rodIndexFromGlobalNodeIndex(j);
        if (ri != rj)
            return false;
        return m_rods[ri]->elementsAreNeighbors(i, j, d);
    }

    bool isValidContactEdge(size_t ei) const { return localEdgeIndex(ei) < m_numElemsPerRod[rodIndexFromGlobalEdgeIndex(ei)]; }
    bool isFirstEdge       (size_t ei) const { return localEdgeIndex(ei) == 0; }
    bool isLastEdge        (size_t ei) const { return localEdgeIndex(ei) == m_numElemsPerRod[rodIndexFromGlobalEdgeIndex(ei)]-1; }

    size_t nextNodeIndex(size_t i) const {
        size_t ri = rodIndexFromGlobalNodeIndex(i);
        size_t nvi = numVerticesInRod(ri);
        return (i+1) % nvi;
    }
    
    size_t numCumulatedTwistVars(size_t ri) const {
        size_t to = 0;
        for (size_t i = 0; i < ri; i++)
            to += m_rods[i]->numEdges() + 1;  // thetas + totalOpeningAngle
        return to;
    }

    // Remove twist variables (thetas and total opening angle) from the vector of spatial dofs
    Eigen::VectorXd extractNodalDoFs(const Eigen::VectorXd &spatialVars) const {
        assert((size_t)spatialVars.size() == numDoF());
        Eigen::VectorXd nodalVars(3*numVertices());
        size_t gi = 0;  // global index accounting also for theta vars
        size_t gj = 0;  // global index accounting only for nodal dofs
        for (const auto &r : m_rods) {
            size_t nRodDofs = r->numDoF();
            size_t nNodalRodDofs = 3*r->numVertices();
            nodalVars.segment(gj, nNodalRodDofs) = spatialVars.segment(gi, nNodalRodDofs);
            gi += nRodDofs;
            gj += nNodalRodDofs;
        }
        assert(gi == numDoF());
        assert(gj == 3*numVertices());
        return nodalVars;
    }

    Pt3 getNode(size_t gi) const {
        assert(gi < numVertices());
        const size_t ri = rodIndexFromGlobalNodeIndex(gi);
        const size_t li = localNodeIndex(gi);
        return m_rods[ri]->getNode(li);
    }

    Eigen::VectorXd getDoFs() const {
        Eigen::VectorXd dofs(numDoF());
        size_t gvi = 0; // global index
        for (const auto &r : m_rods) {
            Eigen::VectorXd rodDofs = r->getDoFs();  // TODO: use segment...
            for (size_t i = 0; i < r->numDoF(); i++)
                dofs[gvi + i] = rodDofs[i];
            gvi += r->numDoF();
        }
        assert(gvi == numDoF());
        return dofs;
    }

    void setDoFs(const Eigen::VectorXd &x) {
        assert((size_t)x.size() == numDoF());
        size_t gvi = 0; // global index
        for (const auto &r : m_rods) {
            size_t nRodDofs = r->numDoF();
            r->setDoFs(x.head(gvi + nRodDofs).tail(nRodDofs));  // TODO: use segment...
            gvi += nRodDofs;
        }
    }
    
    Real energy(EnergyType etype = EnergyType::Full) const {
        Real energy = 0;
        for (const auto &r : m_rods)
            energy += r->energy(etype);
        return energy;
    }

    Eigen::VectorXd gradient(bool freshIterate = false) const {  // TODO: improve efficiency
        Eigen::VectorXd result = Eigen::VectorXd::Zero(numDoF());
        size_t gi = 0;  // global index
        for (size_t i = 0; i < numRods(); i++) {
            result.segment(gi, m_rods[i]->numDoF()) = m_rods[i]->gradient(freshIterate);
            gi += m_rods[i]->numDoF();
        }
        return result;
    }

    void hessian(SuiteSparseMatrix &result) const {
        size_t gi = 0;  // global index
        for (const auto &r : m_rods) {
            SuiteSparseMatrix rodHessian = r->hessianSparsityPattern();
            r->hessian(rodHessian);
            extendSparseMatrixSouthEast(rodHessian, numDoF() - gi);
            result.addWithDistinctSparsityPattern(rodHessian, 1.0, gi, 0, std::numeric_limits<int>::max());
            gi += r->numDoF();
        }
    }

    SuiteSparseMatrix hessianSparsityPattern() const {
        SuiteSparseMatrix result(numDoF(), numDoF());
        size_t gi = 0;  // global index
        for (const auto &r : m_rods) {
            SuiteSparseMatrix rodSparsity = r->hessianSparsityPattern();
            extendSparseMatrixSouthEast(rodSparsity, numDoF() - gi);
            result.addWithDistinctSparsityPattern(rodSparsity, 1.0, gi, 0, std::numeric_limits<int>::max());
            gi += r->numDoF();
        }
        result.symmetry_mode = SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE;
        return result;
    }

    void massMatrix(SuiteSparseMatrix &result, bool updatedSource, bool useLumped) const {
        size_t gi = 0;  // global index
        for (const auto &r : m_rods) {
            SuiteSparseMatrix rodMassMatrix = r->hessianSparsityPattern();
            r->massMatrix(rodMassMatrix, updatedSource, useLumped);
            result.addWithSubSparsity(rodMassMatrix, 1.0, gi, 0, std::numeric_limits<int>::max());
            gi += r->numDoF();
        }
    }

    std::vector<Pt3> deformedPoints() const {
        std::vector<Pt3> globPts;
        globPts.reserve(numVertices());
        for (const auto &r : m_rods) {
            const std::vector<Pt3> &rodPts = r->deformedPoints();
            globPts.insert(globPts.end(), rodPts.begin(), rodPts.end());
        }
        return globPts;
    }

    Eigen::MatrixXd deformedPointsMatrix() const {
        std::vector<Pt3> stdvectorOfPts = deformedPoints();
        return Eigen::Map<Eigen::Matrix<Real, 3, Eigen::Dynamic>>(stdvectorOfPts[0].data(), 3, numVertices()).transpose();
    }

    std::vector<Real> deformedLengths() const {
        std::vector<Real> globLen;
        globLen.reserve(numEdges());
        for (const auto &r : m_rods) {
            const std::vector<Real> &len = r->deformedLengths();
            globLen.insert(globLen.end(), len.begin(), len.end());
        }
        return globLen;
    }

    std::vector<Real> restLengths() const {
        std::vector<Real> globLen;
        globLen.reserve(numEdges());
        for (const auto &r : m_rods) {
            const std::vector<Real> &len = r->restLengths();
            globLen.insert(globLen.end(), len.begin(), len.end());
        }
        return globLen;
    }

    Real characteristicLength() const {
        Real cl = std::numeric_limits<Real>::min();
        for (const auto &r : m_rods) {
            Real clr = r->characteristicLength();
            if (clr > cl)
                cl = clr;
        }
        return cl;
    }

    Real approxLinfVelocity(const Eigen::VectorXd &x) const {
        Real maxvel = std::numeric_limits<int>::min();
        size_t gi = 0;  // global index
        for (const auto &r : m_rods) {
            size_t nRodDofs = r->numDoF();
            maxvel = std::max(maxvel, r->approxLinfVelocity(x.head(gi + nRodDofs).tail(nRodDofs)));
            gi += nRodDofs;
        }
        return maxvel;
    }

    void updateSourceFrame()              { for(auto &r : m_rods) r->updateSourceFrame(); }
    void updateRotationParametrizations() { for(auto &r : m_rods) r->updateRotationParametrizations(); }

    // Visualise the mesh of the underlying ElasticRod (two additional nodes, one additional edge)
    void visualizationGeometry(std::vector<MeshIO::IOVertex > &vertices,
                               std::vector<MeshIO::IOElement> &quads,
                               const bool averagedMaterialFrames = false) const {
        for (const auto &pr : m_rods)
            pr->rod.visualizationGeometry(vertices, quads, averagedMaterialFrames);
    }

    void saveVisualizationGeometry(const std::string &path, const bool averagedMaterialFrames = false) const {
        std::vector<MeshIO::IOVertex > vertices;
        std::vector<MeshIO::IOElement> quads;
        visualizationGeometry(vertices, quads, averagedMaterialFrames);
        MeshIO::save(path, vertices, quads, MeshIO::FMT_GUESS, MeshIO::MESH_QUAD);
    }

    template<typename Derived>
    Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Derived::ColsAtCompileTime>
    visualizationField(const std::vector<Derived> &perRodFields) const {
        if (perRodFields.size() != numRods()) throw std::runtime_error("Invalid per-rod-field size");
        using FieldStorage = Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Derived::ColsAtCompileTime>;
        std::vector<FieldStorage> perRodVisualizationFields;
        perRodVisualizationFields.reserve(numRods());
        int fullSize = 0;
        const int cols = perRodFields.at(0).cols();
        for (size_t ri = 0; ri < numRods(); ++ri) {
            if (cols != perRodFields[ri].cols()) throw std::runtime_error("Mixed field types forbidden.");
            perRodVisualizationFields.push_back(m_rods[ri]->rod.visualizationField(perRodFields[ri]));
            fullSize += perRodVisualizationFields.back().rows();
        }
        FieldStorage result(fullSize, cols);
        int offset = 0;
        for (const auto &vf : perRodVisualizationFields) {
            result.block(offset, 0, vf.rows(), cols) = vf;
            offset += vf.rows();
        }
        assert(offset == fullSize);
        return result;
    }

    // Provide a single vector containing the fields for all the rods stored in sequence
    // Note: the field should already contain duplicates to be visualized on duplicated edges or nodes of each PeriodicRod
    template<typename Derived>
    Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Derived::ColsAtCompileTime>
    visualizationField(const Derived &field) const {
        size_t gi = 0; // global index
        std::vector<Derived> perRodFields(numRods());
        if ((size_t)field.size() == numEdges(/*countGhost*/true)) {
            for (size_t ri = 0; ri < numRods(); ri++) {
                perRodFields[ri] = field.segment(gi, m_rods[ri]->numEdges(/*countGhost*/true));
                gi += m_rods[ri]->numEdges(/*countGhost*/true);
            }
            for (const auto &f : perRodFields)
                if (f[0] != f[f.size()-1])
                    throw std::runtime_error("Values on first and last edges must match.");
            return visualizationField(perRodFields);
        }
        if ((size_t)field.size() == numVertices(/*countGhost*/true)) {
            for (size_t ri = 0; ri < numRods(); ri++) {
                perRodFields[ri] = field.segment(gi, m_rods[ri]->numVertices(/*countGhost*/true));
                gi += m_rods[ri]->numVertices(/*countGhost*/true);
            }
            for (const auto &f : perRodFields) 
                if (f[0] != f[f.size()-2] || f[1] != f[f.size()-1])
                    throw std::runtime_error("Values on first and second nodes must match second-last and last ones.");
            return visualizationField(perRodFields);
        }
        else
            throw std::runtime_error("Invalid field size (note that the field should already contain duplicates to be visualized on ghost edges or nodes).");
    }

    bool dofIsNode (size_t dof) const { 
        assert(dof < numDoF());
        size_t ri = rodIndexFromGlobalDofIndex(dof);
        size_t firstThetaDof = m_firstDofPerRod[ri] + m_rods[ri]->thetaOffset();
        return dof < firstThetaDof; 
    }
    bool dofIsTwist(size_t dof) const { 
        assert(dof < numDoF());
        size_t ri = rodIndexFromGlobalDofIndex(dof);
        size_t twistDof = m_firstDofPerRod[ri] + m_rods[ri]->numDoF() - 1;
        return dof == twistDof;
    }
    bool dofIsTheta(size_t dof) const { return !dofIsNode(dof) && !dofIsTwist(dof); }

    // Node and dof indexing.
    size_t globalNodeIndex      (size_t li , size_t ri ) const { assert(li < m_numElemsPerRod[ri]); return globalIndex(li, ri, m_firstElemPerRod); }
    size_t globalDofIndex       (size_t dof, size_t ri ) const { assert(dof < m_numDofsPerRod[ri]); return globalIndex(dof, ri, m_firstDofPerRod); }
    size_t localNodeIndex                   (size_t gi ) const { assert(gi < m_numElems); return localIndex(gi, m_firstElemPerRod); }
    size_t localDofIndex                    (size_t dof) const { assert(dof < m_numDofs); return localIndex(dof, m_firstDofPerRod); }
    size_t firstGlobalNodeIndexInRod        (size_t ri ) const { assert(ri < numRods()); return m_firstElemPerRod[ri]; }
    size_t firstGlobalDofIndexInRod         (size_t ri ) const { assert(ri < numRods()); return m_firstDofPerRod[ri]; }
    size_t rodIndexFromGlobalNodeIndex      (size_t gi ) const { assert(gi < m_numElems); return rodIndexFromGlobalIndex(gi, m_firstElemPerRod); }
    size_t rodIndexFromGlobalDofIndex       (size_t dof) const { assert(dof < m_numDofs); return rodIndexFromGlobalIndex(dof, m_firstDofPerRod); }
    size_t globalDofIndexFromGlobalNodeIndex(size_t gni) const { assert(gni < numVertices()); return 3*gni + numCumulatedTwistVars(rodIndexFromGlobalNodeIndex(gni)); }
    size_t globalNodeIndexFromGlobalDofIndex(size_t dof) const { assert(dofIsNode(dof)); return floor((dof - numCumulatedTwistVars(rodIndexFromGlobalDofIndex(dof))) / 3); }

    // Edge indexing. In periodic rods, edge and node indexing coincide: functions for edges are then just aliases.
    size_t globalEdgeIndex       (size_t li, size_t ri) const { return globalNodeIndex(li, ri); }
    size_t localEdgeIndex                   (size_t gi) const { return localNodeIndex(gi); }
    size_t firstGlobalEdgeIndexInRod        (size_t ri) const { return firstGlobalNodeIndexInRod(ri); }
    size_t rodIndexFromGlobalEdgeIndex      (size_t gi) const { return rodIndexFromGlobalNodeIndex(gi); }
    size_t globalDofIndexFromGlobalEdgeIndex(size_t gi) const { return globalDofIndexFromGlobalNodeIndex(gi); }

private:

    size_t localIndex(size_t gi,              const std::vector<size_t> &firstIndexPerRod) const { return gi - firstIndexPerRod[rodIndexFromGlobalIndex(gi, firstIndexPerRod)]; }
    size_t globalIndex(size_t li, size_t ri,  const std::vector<size_t> &firstIndexPerRod) const { return li + firstIndexPerRod[ri]; }
    size_t rodIndexFromGlobalIndex(size_t gi, const std::vector<size_t> &firstIndexPerRod) const { return std::upper_bound(firstIndexPerRod.begin(), firstIndexPerRod.end(), gi) - firstIndexPerRod.begin() - 1; }

    void initializeCounters() {

        size_t nr = numRods();
        m_numDofsPerRod.resize(nr);
        m_numElemsPerRod.resize(nr);
        m_firstDofPerRod.resize(nr);
        m_firstElemPerRod.resize(nr);

        size_t gi = 0; // global index
        for (size_t ri = 0; ri < nr; ri++) {
            const auto &r = m_rods[ri];
            m_numDofsPerRod[ri]  = r->numDoF();
            m_numElemsPerRod[ri] = r->numElems();
            gi += r->numVertices();
        }
        m_numDofs  = std::accumulate(m_numDofsPerRod .begin(), m_numDofsPerRod .end(), 0);
        m_numElems = std::accumulate(m_numElemsPerRod.begin(), m_numElemsPerRod.end(), 0);

        m_firstDofPerRod[0] = 0;
        m_firstElemPerRod[0] = 0;
        std::partial_sum(m_numDofsPerRod .begin(), m_numDofsPerRod .end() - 1, m_firstDofPerRod .begin() + 1);
        std::partial_sum(m_numElemsPerRod.begin(), m_numElemsPerRod.end() - 1, m_firstElemPerRod.begin() + 1);
    }

    std::vector<std::shared_ptr<PeriodicRod>> m_rods;

    size_t m_numDofs;
    std::vector<size_t> m_numDofsPerRod;
    std::vector<size_t> m_firstDofPerRod;

    size_t m_numElems;  // #nodes == #edges in periodic rods
    std::vector<size_t> m_numElemsPerRod;
    std::vector<size_t> m_firstElemPerRod;
};

#endif /* end of include guard: PERIODIC_ROD_LIST_HH */
