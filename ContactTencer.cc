#include "ContactTencer.hh"

ContactTencer::ContactTencer (const std::vector<PeriodicRod> &rods, const std::vector<Spring> &spr, const std::vector<SpringAttachments> &sav):
    closed_rods(rods),
    springs(spr),
    m_attachment_vertices(sav) {
        int rodNumDefoVars = closed_rods.numDoF();
        m_num_spring_free_vertices = 0;
        // std::vector<int> 
        for (size_t i = 0; i < springs.size(); ++i){
            m_num_spring_free_vertices += springs[i].get_num_points() - sav[i].num_attached_points();
            int attachment_idx = 0;
            for (size_t j = 0; j < springs[i].get_num_points(); j++){
                if (m_attachment_vertices[i].spring_vertices[attachment_idx] == j){attachment_idx ++;}
                else{
                    m_spring_free_vertices_spring_idx.push_back(i);
                    m_spring_free_vertices_vertex_idx.push_back(j);
                }
            }

        }
        m_numDefoVars = rodNumDefoVars + m_num_spring_free_vertices * 3;

        // std::cout << "m_numDefoVars " << m_numDefoVars << std::endl;
        // for (size_t i = 0; i < m_spring_free_vertices_vertex_idx.size(); ++i){
        //     std::cout << m_spring_free_vertices_spring_idx[i] << " ";
        // }
        // std::cout << std::endl;
        // for (size_t i = 0; i < m_spring_free_vertices_vertex_idx.size(); ++i){
        //     std::cout << m_spring_free_vertices_vertex_idx[i] << " ";
        // }
        // std::cout << std::endl;
        // std::cout << "neighbors" << std::endl;
        // for (size_t i = 0; i < numVertices() + m_num_spring_free_vertices; ++i){
        //     for (size_t j = 0; j < numVertices() + m_num_spring_free_vertices; ++j){
        //         if (elementsAreNeighbors(i,j,1)) std::cout << i << " " << j << std::endl;
        //     }
        // }

        update_spring_attached_coords();
    }

void ContactTencer::print_neighbors(int d){
    std::cout << "neighbors" << std::endl;
    for (size_t i = 0; i < numVertices() + m_num_spring_free_vertices; ++i){
        for (size_t j = 0; j < numVertices() + m_num_spring_free_vertices; ++j){
            if (elementsAreNeighbors(i,j,d)) std::cout << i << " " << j << std::endl;
        }
    }
}

Eigen::Matrix<Real, 3, 1> ContactTencer::find_vertex_coords(size_t rod_idx, int num_vertex){
    // std::cout << "find_vertex_coords " << std::endl;
    // std::cout << "rod_idx " << rod_idx << std::endl;
    // std::cout << "num_vertex " << num_vertex << std::endl;
    VecX vars = closed_rods[rod_idx]->getDoFs();
    Vec3 coords = vars.segment(3*num_vertex,3);
    return coords;
}

void ContactTencer::update_spring_attached_coords(){
    for (size_t i = 0; i < springs.size(); ++i){
        // std::cout << "i " << i << std::endl;
        int attachment_idx = 0;
        for (size_t j = 0; j < springs[i].get_num_points(); j++){
            // std::cout << "j " << j << std::endl;
            // std::cout << "attachment_idx " << attachment_idx << std::endl;
            // std::cout << "m_attachment_vertices.size() " << m_attachment_vertices.size() <<std::endl;
            // std::cout << "m_attachment_vertices[i].spring_vertices.size() " << m_attachment_vertices[i].spring_vertices.size() <<std::endl;
            // std::cout << "m_attachment_vertices[i].spring_vertices[attachment_idx] " << m_attachment_vertices[i].spring_vertices[attachment_idx] << std::endl;
            if (m_attachment_vertices[i].spring_vertices[attachment_idx] == j){
                springs[i].positions[j] = find_vertex_coords(m_attachment_vertices[i].rod_idx[attachment_idx], m_attachment_vertices[i].rod_vertices[attachment_idx]);
                attachment_idx ++;
            }
        }
    }
}


Eigen::Matrix<Real, Eigen::Dynamic, 1> ContactTencer::getDefoVars() const{
    VecX defoVars(numDefoVars());
    int defo_var_idx = closed_rods.numDoF();
    // Rod defo vars
    defoVars.head(defo_var_idx) = closed_rods.getDoFs();
    // Spring free positions defo vars
    for (size_t i = 0; i < springs.size(); ++i){
        int attachment_idx = 0;
        for (size_t j = 0; j < springs[i].get_num_points(); j++){
            if (m_attachment_vertices[i].spring_vertices[attachment_idx] == j){
                attachment_idx ++;
            }
            else{
                defoVars.segment(defo_var_idx,3) = springs[i].positions[j];
                defo_var_idx += 3;
            }
        }
    }
    return defoVars;
}

void ContactTencer::setDefoVars(const Eigen::Ref<const VecX> &vars){
    int defo_var_idx = closed_rods.numDoF();
    // Rod defo vars
    closed_rods.setDoFs(vars.head(defo_var_idx));
    // Spring free positions defo vars
    for (size_t i = 0; i < springs.size(); ++i){
        int attachment_idx = 0;
        for (size_t j = 0; j < springs[i].get_num_points(); j++){
            if (m_attachment_vertices[i].spring_vertices[attachment_idx] == j){
                springs[i].positions[j] = find_vertex_coords(m_attachment_vertices[i].rod_idx[attachment_idx], m_attachment_vertices[i].rod_vertices[attachment_idx]);
                attachment_idx ++;
            }
            else{
                springs[i].positions[j] = vars.segment(defo_var_idx,3);
                defo_var_idx += 3;
            }
        }
    }
}

size_t ContactTencer::num_spring_edges() const {
    size_t res = 0;
    for (size_t i = 0; i < springs.size(); ++i){
        res += springs[i].get_num_points() - 1;
    }
    return res;
}

std::vector<Pt3> ContactTencer::deformedPoints() const {
    std::vector<Pt3> stdvectorOfPts = closed_rods.deformedPoints();
    size_t idx = closed_rods.numDoF();
    VecX defo_vars = getDefoVars();
    while (idx < numDefoVars()){
        stdvectorOfPts.push_back(defo_vars.segment(idx,3));
        idx += 3;
    }
    return stdvectorOfPts;
}

Eigen::MatrixXd ContactTencer::deformedPointsMatrix() const{
    std::vector<Pt3> stdvectorOfPts = deformedPoints();
    return Eigen::Map<Eigen::Matrix<Real, 3, Eigen::Dynamic>>(stdvectorOfPts[0].data(), 3, numVertices()+num_spring_free_vertices()).transpose();
}

size_t ContactTencer::globalDofIndexFromGlobalNodeIndex(size_t gni) const{
    if (gni < numVertices()) return closed_rods.globalDofIndexFromGlobalNodeIndex(gni);
    return closed_rods.numDoF() + 3 * (gni - numVertices());
}

Real ContactTencer::energy(TencerEnergyType ke,EnergyType e) const{
    switch (ke){
    case TencerEnergyType::Elastic:
        return elastic_energy(e);
    case TencerEnergyType::Springs:
        return springs_energy();
    case TencerEnergyType::Full:
        return elastic_energy(e) + springs_energy();
    default:
        throw std::runtime_error("Not implemented for this tencer energy type");
    }
}

Eigen::VectorXd ContactTencer::extractNodalDoFs(const Eigen::VectorXd &spatialVars) const {
    // std::cout << "extractNodalDoFs" << std::endl;
    // std::cout << "extractNodalDoFs" << std::endl;
    // std::cout << spatialVars.size() << std::endl;
    // std::cout << spatialVars.size() << std::endl;
    // std::cout << numDefoVars() << std::endl;
    // std::cout << numDefoVars() << std::endl;
    // std::cout << 3 * numVertices() + 3 * m_num_spring_free_vertices << std::endl;
    // std::cout << 3 * numVertices() + 3 * m_num_spring_free_vertices << std::endl;
    VecX result(3 * numVertices() + 3 * m_num_spring_free_vertices);
    // std::cout << "extractNodalDoFs 1" << std::endl;
    // std::cout << "extractNodalDoFs 1" << std::endl;
    // std::cout << closed_rods.extractNodalDoFs(spatialVars.head(closed_rods.numDoF())).size() << std::endl;
    // std::cout << closed_rods.extractNodalDoFs(spatialVars.head(closed_rods.numDoF())).size() << std::endl;
    result.head(3 * numVertices()) = closed_rods.extractNodalDoFs(spatialVars.head(closed_rods.numDoF()));
    // std::cout << "extractNodalDoFs 2" << std::endl;
    // std::cout << "extractNodalDoFs 2" << std::endl;
    result.tail(3 * m_num_spring_free_vertices) = spatialVars.tail(3 * m_num_spring_free_vertices);
    // std::cout << result << std::endl;
    // std::cout << "extractNodalDoFs ok" << std::endl;
    // std::cout << "extractNodalDoFs ok" << std::endl;
    return result;
}

Real ContactTencer::springs_energy() const{
    Real energy = 0;
    for (const auto spring : springs){
        energy += spring.energy();
    }
    return energy;
}

Eigen::Matrix<Real, Eigen::Dynamic, 1>  ContactTencer::gradient(TencerEnergyType ke, bool updatedParametrization) const {
    switch (ke){
    case TencerEnergyType::Elastic: return rods_gradient(updatedParametrization);
    case TencerEnergyType::Springs:
        return springs_gradient();
    case TencerEnergyType::Full:
        return rods_gradient(updatedParametrization) + springs_gradient();
    default:
        throw std::runtime_error("Not implemented for this tencer energy type");
    }
}

Eigen::Matrix<Real, Eigen::Dynamic, 1>  ContactTencer::rods_gradient(bool updatedParametrization) const {
    VecX grad = VecX::Zero(numDefoVars());
    grad.head(closed_rods.numDoF()) = closed_rods.gradient(updatedParametrization);
    return grad;
}

Eigen::Matrix<Real, Eigen::Dynamic, 1>  ContactTencer::springs_gradient() const {
    VecX result = VecX::Zero(numDefoVars());
    int defo_var_idx = closed_rods.numDoF();
    for (size_t i = 0; i < springs.size(); ++i){
        int attachment_idx = 0;
        VecX part_g = springs[i].dE_dx();
        for (size_t j = 0; j < springs[i].get_num_points(); j++){
            if (m_attachment_vertices[i].spring_vertices[attachment_idx] == j){
                size_t idx = closed_rods.firstGlobalDofIndexInRod(m_attachment_vertices[i].rod_idx[attachment_idx]) + m_attachment_vertices[i].rod_vertices[attachment_idx] * 3;
                result.segment(idx,3) += part_g.segment(3*j,3);
                attachment_idx ++;
            }
            else{
                result.segment(defo_var_idx,3) += part_g.segment(3*j,3);
                defo_var_idx += 3;
            }
        }
    }
    return result;
}

void ContactTencer::hessian(CSCMat &Hout, TencerEnergyType ke, EnergyType e) const{
    switch (ke){
    case TencerEnergyType::Elastic:
        rods_hessian(Hout, e);
        break;
    case TencerEnergyType::Springs:
        return springs_hessian(Hout);
    case TencerEnergyType::Full:{
        rods_hessian(Hout, e);
        springs_hessian(Hout);
        break;
    }
    default:
        throw std::runtime_error("Not implemented for this tencer energy type");
    }
}

void ContactTencer::rods_hessian(CSCMat &Hout, EnergyType e) const{
    SuiteSparseMatrix rodHessian = closed_rods.hessianSparsityPattern();
    closed_rods.hessian(rodHessian);
    extendSparseMatrixSouthEast(rodHessian, numDefoVars());
    Hout.addWithDistinctSparsityPattern(rodHessian, 1.0, 0, 0, std::numeric_limits<int>::max());
}

void ContactTencer::springs_hessian(CSCMat &Hout) const{
    int defo_var_idxA = closed_rods.numDoF();
    int defo_var_idxB = closed_rods.numDoF();
    size_t idxA;
    size_t idxB;
    for (size_t i = 0; i < springs.size(); ++i){
        int attachment_idxA = 0;
        int save_defo_var_idxA = defo_var_idxA;
        MatX h = springs[i].d2E_dx2();
        for (size_t j = 0; j < springs[i].get_num_points(); j++){
            defo_var_idxB = save_defo_var_idxA;
            if (m_attachment_vertices[i].spring_vertices[attachment_idxA] == j){
                idxA = closed_rods.firstGlobalDofIndexInRod(m_attachment_vertices[i].rod_idx[attachment_idxA]) + m_attachment_vertices[i].rod_vertices[attachment_idxA] * 3;
                attachment_idxA ++;
            }
            else{
                idxA = defo_var_idxA;
                defo_var_idxA += 3;
            }
            int attachment_idxB = 0;
            for (size_t k = 0; k < springs[i].get_num_points(); k++){
                if (m_attachment_vertices[i].spring_vertices[attachment_idxB] == k){
                    idxB = closed_rods.firstGlobalDofIndexInRod(m_attachment_vertices[i].rod_idx[attachment_idxB]) + m_attachment_vertices[i].rod_vertices[attachment_idxB] * 3;
                    attachment_idxB ++;
                }
                else{
                    idxB = defo_var_idxB;
                    defo_var_idxB += 3;
                }
                for (int l = 0; l < 3; ++l){
                    for (int m = 0; m < 3; ++m){
                        if (idxA+l <= idxB + m){
                            Hout.addNZ(idxA+l,idxB+m,h(3*j+l,3*k+m));
                        }
                    }
                }
            }
        }
    }
}

CSCMatrix<SuiteSparse_long, Real> ContactTencer::get_hessian(TencerEnergyType ke, EnergyType e) const {
    CSCMat Hsp_csc = hessianSparsityPattern(ke,0.0);
    hessian(Hsp_csc,ke,e);
    return Hsp_csc;
}

CSCMatrix<SuiteSparse_long, Real> ContactTencer::hessianSparsityPattern(TencerEnergyType ke, Real val) const{
    switch (ke){
    case TencerEnergyType::Elastic:
        return rods_hessianSparsityPattern();
    case TencerEnergyType::Springs:
        return springs_hessianSparsityPattern();
    case TencerEnergyType::Full:{
        CSCMat res = rods_hessianSparsityPattern();
        res.addWithDistinctSparsityPattern(springs_hessianSparsityPattern());
        return res;
    }
    default:
        throw std::runtime_error("Not implemented for this tencer energy type");
    }
}

CSCMatrix<SuiteSparse_long, Real> ContactTencer::rods_hessianSparsityPattern(Real val) const {
    SuiteSparseMatrix rodHsp = closed_rods.hessianSparsityPattern(val);
    extendSparseMatrixSouthEast(rodHsp, numDefoVars());
    return rodHsp;
}

CSCMatrix<SuiteSparse_long, Real> ContactTencer::springs_hessianSparsityPattern(Real val) const {
    int n = numDefoVars();
    TripletMatrix<Triplet<Real>> Hsp(n, n);
    Hsp.symmetry_mode = TripletMatrix<Triplet<Real>>::SymmetryMode::UPPER_TRIANGLE;
    int defo_var_idxA = closed_rods.numDoF();
    int defo_var_idxB = closed_rods.numDoF();
    size_t idxA;
    size_t idxB;
    for (size_t i = 0; i < springs.size(); ++i){
        // std::cout << "i " << i << std::endl;
        int attachment_idxA = 0;
        int save_defo_var_idxA = defo_var_idxA;
        MatX h = springs[i].d2E_dx2();
        for (size_t j = 0; j < springs[i].get_num_points(); j++){
            // std::cout << "j " << j << std::endl;
            // std::cout << "attachment_idxA " << attachment_idxA << std::endl;
            // std::cout << "defo_var_idxA " << defo_var_idxA << std::endl;
            defo_var_idxB = save_defo_var_idxA;
            if (m_attachment_vertices[i].spring_vertices[attachment_idxA] == j){
                idxA = closed_rods.firstGlobalDofIndexInRod(m_attachment_vertices[i].rod_idx[attachment_idxA]) + m_attachment_vertices[i].rod_vertices[attachment_idxA] * 3;
                attachment_idxA ++;
            }
            else{
                idxA = defo_var_idxA;
                defo_var_idxA += 3;
            }
            // std::cout << "idxA " << idxA << std::endl;
            int attachment_idxB = 0;
            for (size_t k = 0; k < springs[i].get_num_points(); k++){
                // std::cout << "k " << k << std::endl;
                // std::cout << "attachment_idxB " << attachment_idxB << std::endl;
                // std::cout << "defo_var_idxB " << defo_var_idxB << std::endl;
                if (m_attachment_vertices[i].spring_vertices[attachment_idxB] == k){
                    idxB = closed_rods.firstGlobalDofIndexInRod(m_attachment_vertices[i].rod_idx[attachment_idxB]) + m_attachment_vertices[i].rod_vertices[attachment_idxB] * 3;
                    attachment_idxB ++;
                }
                else{
                    idxB = defo_var_idxB;
                    defo_var_idxB += 3;
                }
                // std::cout << "idxB " << idxB << std::endl;
                for (int l = 0; l < 3; ++l){
                    for (int m = 0; m < 3; ++m){
                        if (idxA+l <= idxB + m){
                            Hsp.addNZ(idxA+l,idxB+m,1.0);
                        }
                    }
                }
            }
        }
    }
    CSCMat Hsp_csc(Hsp);
    Hsp_csc.fill(val);
    return Hsp_csc;
}

bool ContactTencer::elementsAreNeighbors(int i, int j, int d) const {
    if (i < numVertices() && j < numVertices() && closed_rods.elementsAreNeighbors(i,j,d)) return true; // both are neighboring rod vertices
    if (i < numVertices() && j < numVertices()){ // Non-neighboring rod vertices may be neighbors because of a spring
        for (size_t si = 0; si < num_springs(); ++si){
            int attachment_idx = 0;
            for (int vi = 0; vi < springs[si].get_num_points(); ++vi) {
                if (m_attachment_vertices[si].spring_vertices[attachment_idx] == vi){
                    if (m_attachment_vertices[si].rod_vertices[attachment_idx] == i){
                        int attachment_idx2 = 0;
                        for (int pi = 0; pi < springs[si].get_num_points(); ++pi){
                            if (m_attachment_vertices[si].spring_vertices[attachment_idx2] == pi){
                                if (m_attachment_vertices[si].rod_vertices[attachment_idx2] == j) {return std::abs(pi - vi) <= d;}
                                attachment_idx2++;
                            }
                        }
                    }
                    attachment_idx++;
                }
            }
        }
        return false;
    }
    if (i >= numVertices() && j >= numVertices()) { // if i and j and both spring free points
        // std::cout << "test " << i << " " << j << std::endl;
        int ii = i - numVertices();
        int jj = j - numVertices();
        // std::cout << m_spring_free_vertices_spring_idx[ii] << " " << m_spring_free_vertices_spring_idx[jj] << std::endl;
        if (m_spring_free_vertices_spring_idx[ii] != m_spring_free_vertices_spring_idx[jj]) return false;
        // std::cout << "***" << std::endl;
        return std::abs(m_spring_free_vertices_vertex_idx[ii] - m_spring_free_vertices_vertex_idx[jj]) <= d;
    }
    // else: one is a spring free point and one is a rod vertex index
    int ii = std::max(i,j) - numVertices();
    int jj = std::min(i,j);
    int si = m_spring_free_vertices_spring_idx[ii];
    int vi = m_spring_free_vertices_vertex_idx[ii];
    int attachment_idx = 0;
    for (int pi = 0; pi < springs[si].get_num_points(); ++pi){
        if (m_attachment_vertices[si].spring_vertices[attachment_idx] == pi){
            if (m_attachment_vertices[si].rod_vertices[attachment_idx] == jj) {return std::abs(pi - vi) <= d;}
            attachment_idx++;
        }
    }
    return false;
}

void ContactTencer::massMatrix(SuiteSparseMatrix &result, bool updatedSource, bool useLumped) const{
    // SuiteSparseMatrix rodMassMatrix = closed_rods.hessianSparsityPattern();
    // closed_rods.massMatrix(result, updatedSource, useLumped);
    // result.addWithSubSparsity(rodMassMatrix, 1.0, 0, 0, std::numeric_limits<int>::max());
    result.setIdentity(true);
}