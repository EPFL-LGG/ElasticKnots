
#ifndef SOFT_CONSTRAINT_HH
#define SOFT_CONSTRAINT_HH

#include <Eigen/Core>

inline Real sign(const Real &x){ return (Real(0) < x) - (x < Real(0)); };

struct SoftConstraint {
    virtual void energy  (const PeriodicRodList &rods, Real &e)              const = 0;
    virtual void gradient(const PeriodicRodList &rods, Eigen::VectorXd &g)   const = 0;
    virtual void hessian (const PeriodicRodList &rods, SuiteSparseMatrix &h) const = 0;
    virtual ~SoftConstraint() { }
};

struct FlatnessConstraint : SoftConstraint {

    FlatnessConstraint(Real stiffness, const Eigen::Vector3d &n, const Eigen::Vector3d &center, Real upper_d, Real lower_d)  
        : stiffness(stiffness), n(n.normalized()), center(center), upper_d(upper_d), lower_d(lower_d)  {  }

    // FlatnessConstraint(Real stiffness, const Eigen::Vector3d &n)
    //     : FlatnessConstraint(stiffness, n, {0.0, 0.0, 0.0})

    void energy(const PeriodicRodList &rods, Real &e) const override {
        for (size_t i = 0; i < rods.numVertices(); i++) {
            Real n_dot_xi = n.dot(rods.getNode(i) - center);
            // Real distance = std::abs(n_dot_xi);
            // Real out_band = std::max(distance - upper_d, lower_d - distance);
            if      (n_dot_xi >   upper_d)
                e += 0.5 * stiffness * (  n_dot_xi - upper_d)*(  n_dot_xi - upper_d);
            else if (n_dot_xi < - lower_d)
                e += 0.5 * stiffness * (- n_dot_xi - lower_d)*(- n_dot_xi - lower_d);
        }
    }

    void gradient(const PeriodicRodList &rods, Eigen::VectorXd &g) const override {
        for (size_t i = 0; i < rods.numVertices(); i++) {
            size_t dof = rods.globalDofIndexFromGlobalNodeIndex(i);
            Real n_dot_xi = n.dot(rods.getNode(i) - center);

            if      (n_dot_xi >   upper_d)
                g.segment<3>(dof) += stiffness * (n_dot_xi - upper_d) * n;
            else if (n_dot_xi < - lower_d)
                g.segment<3>(dof) += stiffness * (n_dot_xi + lower_d) * n;
        }
    }

    void hessian(const PeriodicRodList &rods, SuiteSparseMatrix &h) const override {
        TripletMatrix<Triplet<Real>> triplets(rods.numDoF(), rods.numDoF());

        auto set3x3Block = [&](size_t i, size_t j, const Eigen::Matrix3d &block) {
            assert(i <= j);
            for (size_t ii = i; ii < i+3; ii++) {
                for (size_t jj = j; jj < j+3; jj++) {
                    if (jj < ii) continue;
                    triplets.addNZ(ii, jj, block(ii-i, jj-j));
                }
            }
        };
    
        const Eigen::Matrix3d nOuter = n * n.transpose();
        for (size_t i = 0; i < rods.numVertices(); i++) {
            Real n_dot_xi = n.dot(rods.getNode(i) - center);
            if (n_dot_xi > upper_d || n_dot_xi < - lower_d) {
                size_t dof = rods.globalDofIndexFromGlobalNodeIndex(i);
                set3x3Block(dof, dof, stiffness * nOuter);
            }
        }
        SuiteSparseMatrix flatnessHessian;
        if (triplets.nz.size() != 0)  // otherwise, flatnessHessian will be safely left empty
            flatnessHessian.setFromTMatrix(triplets);
        flatnessHessian.symmetry_mode = SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE;
        h.addWithSubSparsity(flatnessHessian);
    }

    Real stiffness;
    Eigen::Vector3d n;
    Real upper_d;
    Real lower_d;
    Eigen::Vector3d center;
};


struct VolumeConstraint : SoftConstraint {

    VolumeConstraint(Real stiffness, const Eigen::Vector3d &aspectRatio)  // TODO: generalize to randomly oriented ellipsoids (now the metric is axis-aligned)
        : stiffness(stiffness), aspectRatio(aspectRatio) {  }

    VolumeConstraint(Real stiffness)
        : stiffness(stiffness), aspectRatio({1.0, 1.0, 1.0}) {  }

    void energy(const PeriodicRodList &rods, Real &e) const override {
        for (size_t i = 0; i < rods.numVertices(); i++) {
            const Eigen::Vector3d & xi = rods.getNode(i);
            for (size_t d = 0; d < 3; d++)
                e += 0.5 * stiffness * aspectRatio[d]*xi[d]*xi[d];
        }
    }

    void gradient(const PeriodicRodList &rods, Eigen::VectorXd &g) const override {
        for (size_t i = 0; i < rods.numVertices(); i++) {
            const Eigen::Vector3d & xi = rods.getNode(i);
            size_t dof = rods.globalDofIndexFromGlobalNodeIndex(i);
            for (size_t d = 0; d < 3; d++)
                g(dof+d) += stiffness * aspectRatio[d]*xi[d];
        }
    }

    void hessian(const PeriodicRodList &rods, SuiteSparseMatrix &h) const override {
        TripletMatrix<Triplet<Real>> triplets(rods.numDoF(), rods.numDoF());

        const Real val = stiffness;
        for (size_t i = 0; i < rods.numVertices(); i++) {
            size_t dof = rods.globalDofIndexFromGlobalNodeIndex(i);
            triplets.addNZ(dof,   dof,   val*aspectRatio[0]);
            triplets.addNZ(dof+1, dof+1, val*aspectRatio[1]);
            triplets.addNZ(dof+2, dof+2, val*aspectRatio[2]);
        }
        SuiteSparseMatrix volumeHessian;
        if (triplets.nz.size() != 0)  // otherwise, volumeHessian will be safely left empty
            volumeHessian.setFromTMatrix(triplets);
        volumeHessian.symmetry_mode = SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE;
        h.addWithSubSparsity(volumeHessian);
    }

    Real stiffness;
    Eigen::Vector3d aspectRatio;
};


struct SphericalShellConstraint : SoftConstraint {

    SphericalShellConstraint(Real stiffness, const Eigen::Vector3d &center, Real upper_d, Real lower_d)
        : stiffness(stiffness), center(center), upper_d(upper_d), lower_d(lower_d) {  }

    SphericalShellConstraint(Real stiffness)
        : stiffness(stiffness), center({0.0, 0.0, 0.0}), upper_d(std::numeric_limits<int>::max()), lower_d(0.0) {  }

    void energy(const PeriodicRodList &rods, Real &e) const override {
        for (size_t i = 0; i < rods.numVertices(); i++) {
            const Eigen::Vector3d & xi = rods.getNode(i);
            Real d = (xi - center).norm();
            if (d > upper_d)
                e += 0.5 * stiffness * (d - upper_d)*(d - upper_d);
            else if (d < lower_d)
                e += 0.5 * stiffness * (lower_d - d)*(lower_d - d);
        }
    }

    void gradient(const PeriodicRodList &rods, Eigen::VectorXd &g) const override {
        for (size_t i = 0; i < rods.numVertices(); i++) {
            const Eigen::Vector3d & xi = rods.getNode(i);
            size_t dof = rods.globalDofIndexFromGlobalNodeIndex(i);
            Real d = (xi - center).norm();

            if (d > upper_d)
                g.segment<3>(dof) += stiffness * (d - upper_d) * (xi - center).normalized();
            else if (d < lower_d)
                g.segment<3>(dof) += stiffness * (d - lower_d) * (xi - center).normalized();
        }
    }

    void hessian(const PeriodicRodList &rods, SuiteSparseMatrix &h) const override {
        TripletMatrix<Triplet<Real>> triplets(rods.numDoF(), rods.numDoF());
    
        auto set3x3Block = [&](size_t i, size_t j, const Eigen::Matrix3d &block) {
            assert(i <= j);
            for (size_t ii = i; ii < i+3; ii++) {
                for (size_t jj = j; jj < j+3; jj++) {
                    if (jj < ii) continue;
                    triplets.addNZ(ii, jj, block(ii-i, jj-j));
                }
            }
        };

        for (size_t i = 0; i < rods.numVertices(); i++) {
            const Eigen::Vector3d & xi = rods.getNode(i);
            Real d = (xi - center).norm();

            // if (d > upper_d) {
            //     const Eigen::Vector3d & n = (xi - center).normalized()
            //     Eigen::Matrix3d dn_dxi = (Eigen::Matrix3d::Identity() - n*n.transpose()) / d;
            // }
            // else if (d < lower_d) {
            //     const Eigen::Vector3d & n = (xi - center).normalized()
            //     Eigen::Matrix3d dn_dxi = (Eigen::Matrix3d::Identity() - n*n.transpose()) / d;
            // }

            if (d > upper_d || d < lower_d) {
                const Eigen::Vector3d n = (xi - center).normalized();
                const Eigen::Matrix3d dn_dxi = (Eigen::Matrix3d::Identity() - n*n.transpose()) / d;
                Real d_hat = d > upper_d ? upper_d : lower_d;
                size_t dof = rods.globalDofIndexFromGlobalNodeIndex(i);
                set3x3Block(dof, dof, stiffness * (Eigen::Matrix3d::Identity() - d_hat * dn_dxi));
            }
        }
        SuiteSparseMatrix sphereShellHessian;
        if (triplets.nz.size() != 0)  // otherwise, sphereShellHessian will be safely left empty
            sphereShellHessian.setFromTMatrix(triplets);
        sphereShellHessian.symmetry_mode = SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE;
        h.addWithSubSparsity(sphereShellHessian);
    }

    // void gradient(const PeriodicRodList &rods, Eigen::VectorXd &g) const override {
    //     for (size_t i = 0; i < rods.numVertices(); i++) {
    //         const Eigen::Vector3d & xi = rods.getNode(i);
    //         size_t dof = rods.globalDofIndexFromGlobalNodeIndex(i);
    //         for (size_t d = 0; d < 3; d++)
    //             g(dof+d) += stiffness * 2 * aspectRatio[d]*xi[d];
    //     }
    // }

    // void hessian(const PeriodicRodList &rods, SuiteSparseMatrix &h) const override {
    //     TripletMatrix<Triplet<Real>> triplets(rods.numDoF(), rods.numDoF());

    //     const Real val = 2*stiffness;
    //     for (size_t i = 0; i < rods.numVertices(); i++) {
    //         size_t dof = rods.globalDofIndexFromGlobalNodeIndex(i);
    //         triplets.addNZ(dof,   dof,   val*aspectRatio[0]);
    //         triplets.addNZ(dof+1, dof+1, val*aspectRatio[1]);
    //         triplets.addNZ(dof+2, dof+2, val*aspectRatio[2]);
    //     }
    //     SuiteSparseMatrix volumeHessian;
    //     volumeHessian.setFromTMatrix(triplets);
    //     volumeHessian.symmetry_mode = SuiteSparseMatrix::SymmetryMode::UPPER_TRIANGLE;
    //     h.addWithSubSparsity(volumeHessian);
    // }

    Real stiffness;
    Eigen::Vector3d aspectRatio;
    Real upper_d;
    Real lower_d;
    Eigen::Vector3d center;
};

#endif /* end of include guard: SOFT_CONSTRAINT_HH */

