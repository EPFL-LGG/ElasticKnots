#ifndef Spring_HH
#define Spring_HH
#include <ElasticRods/PeriodicRod.hh>
#include "helper.hh"



template <typename Real_>
struct  Spring_T {
    

    using Vec3_T = Eigen::Matrix<Real_, 3, 1>;
    using VecX_T  = Eigen::Matrix<Real_, Eigen::Dynamic, 1>;
    using CSCMat_T = CSCMatrix<SuiteSparse_long, Real_>;
    using Mat3_T = Eigen::Matrix<Real_, 3, 3>;
    using MatX_T = Eigen::Matrix<Real_, -1, -1>;

    enum class CompressionType {
        Compression,
        NoCompression
    };

    std::vector<Vec3_T> positions;
    double stiffness;
    double rest_length;
    double min_segment_length = 1e-3;
    double regularization_weight;
    CompressionType compression_type;
    Real compression_tolerance;

    Spring_T(std::vector<Vec3_T> &p, double s, double l = 0, CompressionType c = CompressionType::Compression, Real tol = 1e-3) : positions(p), stiffness(s), rest_length(l), regularization_weight(1e-5), compression_type(c), compression_tolerance(tol) {}

    // Copy constructor converting from another floating point type (e.g., double to autodiff)
    template<typename Real_2>
    Spring_T(const Spring_T<Real_2> &sp){ set(sp); }

    // Copy constructor
    Spring_T(const Spring_T &sp){ set(sp); }

    // Copy assignment converting from another floating point type (e.g., double to autodiff)
    template<typename Real_2>
    Spring_T<Real_>& operator=(const Spring_T<Real_2> &sp) { 
        set(sp);
        return *this;
    }

    // Copy assignment
    Spring_T& operator=(const Spring_T &sp) { 
        set(sp);
        return *this;
    }

    ~Spring_T() = default;

    std::string mangledName() const {
        return "Spring" + autodiffOrNotString<Real_>() + ">"; 
    }

    template<typename Real_2>
    void set(const Spring_T<Real_2> &sp){
        stiffness = sp.stiffness;
        rest_length = sp.rest_length;
        min_segment_length = sp.min_segment_length;
        regularization_weight = sp.regularization_weight;
        castStdADVector(sp.positions,positions);
        compression_type = sp.compression_type;
        compression_tolerance = sp.compression_tolerance;
    }

    Real_ energy() const;
    
    // Derivatives 

    VecX_T dE_dx() const;
    Real_ dE_dk() const;
    VecX_T dE_dxk() const;
    MatX_T d2E_dx2()  const;
    VecX_T d2E_dxdk() const;

    Vec3_T dterm(size_t i) const;
    Mat3_T d2term(size_t i) const;

    Real_ min_length_energy() const;
    VecX_T min_length_gradient() const;
    MatX_T min_length_hessian() const;
    Mat3_T min_length_hessian_helper(Vec3_T u, Real_ d) const;

    Real_ regularization_energy() const;
    VecX_T regularization_gradient() const;
    MatX_T regularization_hessian() const;
    MatX_T regularization_hessian_helper(size_t i) const;

    Real_ Q(Real_ x) const;
    Real_ dQ_dx(Real_ x) const;
    Real_ d2Q_dx2(Real_ x) const;
    
    size_t get_num_points() const {return positions.size();}
    Vec3_T get_point_coords(size_t i) const {return positions[i];}
    std::vector<Vec3_T> get_coords() const {return positions;}
    double get_rest_length() const {return rest_length;}
    double get_stiffness() const {return stiffness;}
    double get_regularization_weight() const {return regularization_weight;}
    Real_ get_total_length() const;
    void set_coords(std::vector<Vec3_T> &c){positions = c;}
    void set_stiffness(double s){stiffness = s;}
    bool is_compressed() {return get_total_length() < rest_length;}

};




#endif