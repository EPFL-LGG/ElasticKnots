#include "Spring.hh"

template <typename Real_>
Real_ Spring_T<Real_>::get_total_length() const{
    Real_ total_length = 0;
    for (size_t i = 1; i < positions.size(); ++i){
        total_length += (positions[i] - positions[i-1]).norm();
    }
    return total_length;
}

template <typename Real_>
Real_ Spring_T<Real_>::energy() const{
    Real_ l = get_total_length() - rest_length;
    return 0.5 * stiffness * l * l + min_length_energy() + regularization_energy();
}



template <typename Real_>
Real_ Spring_T<Real_>::dE_dk() const{
    Real_ l = get_total_length() - rest_length;
    return 0.5 * l * l;
}

template <typename Real_>
Eigen::Matrix<Real_, Eigen::Dynamic, 1> Spring_T<Real_>::dE_dx() const{
    size_t n = positions.size();
    VecX_T res = VecX_T::Zero(3 * n);
    Real_ l = get_total_length() - rest_length;
    for (size_t i = 0; i < n; ++i){
        res.segment(3*i,3) = stiffness * l * dterm(i);
    }
    return res + min_length_gradient() + regularization_gradient();
}

template <typename Real_>
Eigen::Matrix<Real_, 3, 1> Spring_T<Real_>::dterm(size_t i) const{
    size_t n = positions.size();
    if (i == 0){
        Vec3_T d = (positions[0] - positions[1]);
        Real_ s = d.norm();     
        return d / s;
    }
    if (i == n-1){
        Vec3_T d = (positions[n-1] - positions[n-2]);
        Real_ s = d.norm();
        return d / s;
    }
    Vec3_T d_minus = (positions[i] - positions[i-1]);
    Vec3_T d_plus = (positions[i] - positions[i+1]);
    Real_ s_plus = d_plus.norm();
    Real_ s_minus = d_minus.norm();
    return d_plus / s_plus + d_minus / s_minus;
}

template <typename Real_>
Eigen::Matrix<Real_, 3, 3> Spring_T<Real_>::d2term(size_t i) const{
    if (i == 0 || i >= positions.size()){
        return Mat3_T::Zero();
    }
    Vec3_T d = (positions[i] - positions[i-1]);
    Real_ n = d.norm();
    return - d*d.transpose() / (n*n*n) + 1/n * Mat3_T::Identity();
}

template <typename Real_>
Real_ Spring_T<Real_>::min_length_energy() const{
    Real_ length;
    Real_ energy = 0;
    for (size_t i = 1; i < positions.size(); ++i){
        length = (positions[i] - positions[i-1]).norm();
        if (length < min_segment_length){
            Real_ d = length - min_segment_length;
            energy += - d * d* log(length / min_segment_length);
        }
    }
    return energy;
}

template <typename Real_>
Eigen::Matrix<Real_, Eigen::Dynamic, 1> Spring_T<Real_>::min_length_gradient() const{
    size_t n = positions.size();
    VecX_T g = VecX_T::Zero(3*n);
    Vec3_T diff;
    Real_ length;
    Real_ d;
    for (size_t i = 0; i < n; ++i){
        if (i < n-1){
            diff = positions[i] - positions[i+1];
            length = diff.norm();
            if (length < min_segment_length){
                d = min_segment_length - length;
                g.segment(3*i,3) += 2 * diff * d / length * log(length / min_segment_length);
                g.segment(3*i,3) -= diff * d * d / length / length;
            }
        }
        if (i > 0){
            diff = positions[i] - positions[i-1];
            length = diff.norm();
            if (length < min_segment_length){
                d = min_segment_length - length;
                g.segment(3*i,3) += 2 * diff * d / length * log(length / min_segment_length);
                g.segment(3*i,3) -= diff * d * d / length / length;
            }
        }
    }
    return g;
}

template <typename Real_>
Eigen::Matrix<Real_, 3, 3> Spring_T<Real_>::min_length_hessian_helper(Vec3_T u, Real_ d) const{
    Real_ diff = d - min_segment_length;
    Mat3_T res = Mat3_T::Zero();
    Real_ ln = log(d/min_segment_length);
    for (int i = 0; i < 3; ++i){
        res(i,i) += - 2 * diff / d * ln;
        res(i,i) += - diff * diff / d / d;
        for (int j = 0; j < 3; ++j){
            res(i,j) += (-2*ln - 4*diff/d + 2*diff*ln/d + 2*diff*diff/d/d) * u(i) * u(j) / d / d;
        }
    }
    return res;
}

template <typename Real_>
Eigen::Matrix<Real_, -1, -1> Spring_T<Real_>::min_length_hessian() const{
    int n = positions.size();
    MatX_T res = MatX_T::Zero(3*n,3*n);
    Vec3_T diff;
    Real_ length;
    for (int i = 0; i < n; ++i){
        if (i < n-1){
            diff = positions[i] - positions[i+1];
            length = diff.norm();
            if (length < min_segment_length){
                Mat3_T h = min_length_hessian_helper(diff,length);
                res.block(3*i,3*i,3,3) += h;
                res.block(3*i,3*i+3,3,3) -= h;
            }
        }
        if (i > 0){
            diff = positions[i] - positions[i-1];
            length = diff.norm();
            if (length < min_segment_length){
                Mat3_T h = min_length_hessian_helper(diff,length);
                res.block(3*i,3*i,3,3) += h;
                res.block(3*i,3*i-3,3,3) -= h;
            }
        }
    }
    return res;
}

template <typename Real_>
Real_ Spring_T<Real_>::regularization_energy() const{
    size_t num_edges = positions.size() - 1;
    Real_ target_length = rest_length / num_edges;
    Real_ res = 0;
    for (size_t i = 1; i < positions.size(); ++i){
        Real_ diff = (positions[i] - positions[i-1]).norm() - target_length;
        res += 0.5 * diff * diff;
    }
    return res * regularization_weight;
}

template <typename Real_>
Eigen::Matrix<Real_, Eigen::Dynamic, 1> Spring_T<Real_>::regularization_gradient() const{
    size_t n = positions.size();
    size_t num_edges = n - 1;
    Real_ target_length = rest_length / num_edges;
    VecX_T g = VecX_T::Zero(3*n);
    for (size_t i = 1; i < n; ++i){
        Vec3_T d = positions[i] - positions[i-1];
        Real_ dist = d.norm();
        g.segment(3*i,3) += regularization_weight * (1 - target_length/dist) * d;
        g.segment(3*(i-1),3) -= regularization_weight * (1 - target_length/dist) * d;
    }
    return g;
}

template <typename Real_>
Eigen::Matrix<Real_, -1, -1> Spring_T<Real_>::regularization_hessian() const {
    size_t n = positions.size();
    MatX_T res = MatX_T::Zero(3*n,3*n);
    for (size_t i = 1; i < n; ++i){
        res.block(3*(i-1),3*(i-1),6,6) += regularization_hessian_helper(i);
    }
    return res;
}

template <typename Real_>
Eigen::Matrix<Real_, -1, -1> Spring_T<Real_>::regularization_hessian_helper(size_t i) const {
    Eigen::Matrix<Real_, 6, 6> values = Eigen::Matrix<Real_, 6, 6>::Zero();
    size_t n = positions.size();
    size_t num_edges = n - 1;
    Real_ target_length = rest_length / num_edges;
    Vec3_T d = positions[i-1] - positions[i];
    Real_ dist = d.norm();
    Real_ L3 = dist*dist*dist;
    values.block(0,0,3,3) =  d * d.transpose();
    values.block(3,0,3,3) = -d * d.transpose();
    values.block(0,3,3,3) = -d * d.transpose();
    values.block(3,3,3,3) =  d * d.transpose();
    values *= (regularization_weight * target_length / L3);
    for (int i = 0; i < 6; i++) { 
        values(i, i) -= regularization_weight * target_length / dist; 
        values(i, (i + 3) % 6) += regularization_weight * target_length / dist; 
    }
    for (int i = 0; i < 6; i++) { 
        values(i, i) += regularization_weight; 
        values(i, (i + 3) % 6) -= regularization_weight; 
    }
    return values;
}



template <typename Real_>
Eigen::Matrix<Real_, -1, -1> Spring_T<Real_>::d2E_dx2()  const{
    int n = positions.size();
    MatX_T res = MatX_T::Zero(3*n,3*n);
    Real_ L = get_total_length() - rest_length;
    for (int i = 0; i < n; ++i){
        Vec3_T di = dterm(i);
        for (int j = 0; j < i-1; ++j){
            Vec3_T dj = dterm(j);
            for (int k = 0; k < 3; ++k){
                for (int l = 0; l < 3; ++l){
                    res(3*i+k,3*j+l) = stiffness * di[k] * dj[l];
                }
            }
        }
        for (int j = i+2; j < n; ++j){
            Vec3_T dj = dterm(j);
            for (int k = 0; k < 3; ++k){
                for (int l = 0; l < 3; ++l){
                    res(3*i+k,3*j+l) = stiffness * di[k] * dj[l];
                }
            }
        }
        // case j = i
        Mat3_T d2i = d2term(i);
        Mat3_T d2i1 = d2term(i+1);
        for (int k = 0; k < 3; ++k){
            for (int l = 0; l < 3; ++l){
                res(3*i+k,3*i+l) = stiffness * di[k] * di[l] + stiffness * L * (d2i(k,l) + d2i1(k,l));
            }
        }
        // case j = i-1
        if (i >= 1){
            Vec3_T dim1 = dterm(i-1);
            for (int k = 0; k < 3; ++k){
                for (int l = 0; l < 3; ++l){
                    res(3*i+k,3*i-3+l) = stiffness * di[k] * dim1[l] - stiffness * L * d2i(k,l);
                }
            }
        }
        // case j = i+1
        if (i < n-1){
            Vec3_T di1 = dterm(i+1);
            for (int k = 0; k < 3; ++k){
                for (int l = 0; l < 3; ++l){
                    res(3*i+k,3*i+3+l) = stiffness * di[k] * di1[l] - stiffness * L * d2i1(k,l);
                }
            }
        }
    }
    // return min_length_hessian();
    return res + min_length_hessian() + regularization_hessian();
}

template <typename Real_>
Eigen::Matrix<Real_, -1, 1> Spring_T<Real_>::d2E_dxdk()  const{
    size_t n = positions.size();
    VecX_T res = VecX_T::Zero(3 * n + 1);
    Real_ l = get_total_length() - rest_length;
    for (size_t i = 0; i < n; ++i){
        res.segment(3*i,3) =  l * dterm(i);
    }
    return res;
}

template <typename Real_>
Eigen::Matrix<Real_, -1, 1> Spring_T<Real_>::dE_dxk()  const{
    size_t n = positions.size();
    VecX_T res = VecX_T::Zero(3 * n + 1);
    res.head(3*n) = dE_dx();
    res[3*n] = dE_dk();
    return res;
}

template struct Spring_T<Real>;
template struct Spring_T<ADReal>;