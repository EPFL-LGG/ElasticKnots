#ifndef ONESIDEDQUADRATIC_HH
#define ONESIDEDQUADRATIC_HH
#include <ElasticRods/PeriodicRod.hh>

template <typename Real_>
struct  OneSidedQuadratic_T{
    Real k;
    Real epsilon;
    OneSidedQuadratic_T(Real kk, Real ee) : k(kk), epsilon(ee) {}
    OneSidedQuadratic_T() : k(1), epsilon(1e-6) {}

    Real_ q(const Real_ x) const{
        if (x < -epsilon) return 0;
        if (x > epsilon) return k*(x*x + epsilon*epsilon/3);
        Real_ x2 = x*x;
        Real_ x3 = x2*x;
        return k*(x3/(6*epsilon) + x2/2 + epsilon*x/2 + epsilon*epsilon/6);
    }

    Real_ dq_dx(const Real_ x) const{
        if (x < -epsilon) return 0;
        if (x > epsilon) return k*2*x;
        Real_ x2 = x*x;
        return k*(x2/(2*epsilon) + x + epsilon/2);
    }

    Real_ d2q_dx2(const Real_ x) const{
        if (x < -epsilon) return 0;
        if (x > epsilon) return 2*k;
        return k*(x/epsilon + 1);
    }

    Real get_k(){return k;}
    void set_k(Real kk){k = kk;}
    Real get_eps(){return epsilon;}
    void set_eps(Real ee){epsilon = ee;}


};

#endif

