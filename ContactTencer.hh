#ifndef ContactTencer_HH
#define ContactTencer_HH

#include "PeriodicRodList.hh"
#include "Spring.hh"



struct SpringAttachments {
    using VeciX  = Eigen::Matrix<int, Eigen::Dynamic, 1>;
    SpringAttachments(VeciX r, VeciX v, VeciX s) : rod_idx(r), rod_vertices(v), spring_vertices(s) {}
    SpringAttachments(const SpringAttachments &sp) : rod_idx(sp.rod_idx), rod_vertices(sp.rod_vertices), spring_vertices(sp.spring_vertices) {}

    VeciX rod_idx;
    VeciX rod_vertices;
    VeciX spring_vertices;

    // Comparison operator
    bool operator==(const SpringAttachments &sav) const {
        if (rod_idx.size() != sav.rod_idx.size()) return false;
        for (int i = 0; i < rod_idx.size(); ++i){
            if (rod_idx[i] != sav.rod_idx[i] or rod_vertices[i] != sav.rod_vertices[i] or spring_vertices[i] != sav.spring_vertices[i]){
                return false;
            }
        }
        return true;
    }

    int num_attached_points() const {return rod_idx.size();}
};



struct ContactTencer {

    using Vec3 = Eigen::Matrix<Real, 3, 1>;
    using VecX  = Eigen::Matrix<Real, Eigen::Dynamic, 1>;
    using CSCMat = CSCMatrix<SuiteSparse_long, Real>;
    using MatX = Eigen::Matrix<Real, -1, -1>;

    using PR = PeriodicRod_T<Real>;
    using EnergyType = typename PR::EnergyType;


    using Spring = Spring_T<Real>;

    enum class TencerEnergyType {Full, Elastic, Springs};

    PeriodicRodList closed_rods;
    std::vector<Spring> springs;

    // for gravity
    static constexpr size_t N   = 3; 
    Real rho = 1.0;
    

    // Constructor from rod list and springs
    ContactTencer(const std::vector<PeriodicRod> &rods, const std::vector<Spring> &spr, const std::vector<SpringAttachments> &sav);

    // Copy another tencer.
    void set(const ContactTencer &k) {
        
        // copy rods
        closed_rods = PeriodicRodList(k.closed_rods);

        // copy springs
        castStdADVector(k.springs,springs);

        m_numDefoVars = k.numDefoVars();
        m_attachment_vertices = k.get_attachment_vertices();

        update_spring_attached_coords();
    }

    // Copy constructor
    ContactTencer(const ContactTencer &k) { set(k); }

    // Copy assignment
    ContactTencer& operator=(const ContactTencer &r) { 
        set(r); 
        return *this;
    }

    std::string mangledName() const {
        return "ContactTencer"; 
    }






    // Getters : rods 
    PeriodicRodList get_closed_rods() const {return PeriodicRodList(closed_rods);}
    size_t num_closed_rods() const {return closed_rods.size();}

    // Getters : springs
    std::vector<Spring> get_springs() const {return springs;}
    Spring get_spring(int i) const {return springs[i];}
    size_t num_springs() const {return springs.size();}
    std::vector<SpringAttachments> get_attachment_vertices() const {return m_attachment_vertices;}
    size_t num_spring_free_vertices() const {return m_num_spring_free_vertices;}
    size_t num_spring_edges() const;
    
    
    // Getters and setters : defo and rest variables
    size_t numDefoVars() const {return m_numDefoVars;}
    VecX getDefoVars() const; 
    void setDefoVars(const Eigen::Ref<const VecX> &vars);
    Eigen::MatrixXd deformedPointsMatrix() const; 
    std::vector<Pt3> deformedPoints() const; 
    
    size_t numVertices() const {return closed_rods.numVertices();}
    size_t numEdges() const {return closed_rods.numEdges();}

    size_t firstGlobalDofIndexInRod (size_t ri ) const {return closed_rods.firstGlobalDofIndexInRod(ri);}
    size_t firstGlobalNodeIndexInRod(size_t ri ) const { return closed_rods.firstGlobalNodeIndexInRod(ri);}
    size_t globalDofIndexFromGlobalNodeIndex(size_t gni) const; 
    size_t numVerticesInRod(size_t i) const {return closed_rods.numVerticesInRod(i);}
    // Remove twist variables (thetas and total opening angle) from the vector of spatial dofs
    Eigen::VectorXd extractNodalDoFs(const Eigen::VectorXd &spatialVars) const;

    Vec3 find_vertex_coords(size_t rod_idx, int num_vertex);
    void update_spring_attached_coords();

    void print_neighbors(int d);
   
    // Energies, gradients, hessians
    
    Real energy() const {return energy(TencerEnergyType::Full);};
    Real energy(TencerEnergyType ke,EnergyType e = EnergyType::Full) const;
    Real elastic_energy(EnergyType etype = EnergyType::Full) const {return closed_rods.energy(etype);}
    Real springs_energy() const;
    
    VecX gradient(bool updatedParametrization = false) const {return gradient(TencerEnergyType::Full,updatedParametrization);}
    VecX gradient(TencerEnergyType ke, bool updatedParametrization = false) const;
    VecX rods_gradient(bool updatedParametrization = false) const;
    VecX springs_gradient() const;
    
    void hessian(CSCMat &Hout, TencerEnergyType ke = TencerEnergyType::Full, EnergyType e = EnergyType::Full) const; // TODO
    void rods_hessian(CSCMat &Hout, EnergyType e = EnergyType::Full) const;
    void springs_hessian(CSCMat &Hout) const;
    CSCMat get_hessian(TencerEnergyType ke = TencerEnergyType::Full, EnergyType e = EnergyType::Full) const;
    
    CSCMat hessianSparsityPattern(Real val = 0.0) const {return hessianSparsityPattern(TencerEnergyType::Full, val);}
    CSCMat hessianSparsityPattern(TencerEnergyType ke, Real val = 0.0) const;
    CSCMat rods_hessianSparsityPattern(Real val = 0.0) const;
    CSCMat springs_hessianSparsityPattern(Real val = 0.0) const;

    Real approxLinfVelocity(const Eigen::VectorXd &x) const {return closed_rods.approxLinfVelocity(x);}

    bool elementsAreNeighbors(int i, int j, int d = 1) const;

    




    // Additional methods required by compute_equilibrium
    void updateSourceFrame()              { closed_rods.updateSourceFrame();}
    void updateRotationParametrizations() { closed_rods.updateRotationParametrizations();}
    // Update our parametrization of the system's DoFs for comptibility with EquilibriumSolver (MeshFEM)
    void updateParametrization() {updateSourceFrame(); updateRotationParametrizations();}



    void massMatrix(SuiteSparseMatrix &result, bool updatedSource, bool useLumped) const;// {return closed_rods.massMatrix(result,updatedSource,useLumped);}

    
    
    

protected:
    

    size_t m_numDefoVars;                                           // Total number of Defo variables for the ContactTencer
    size_t m_num_spring_free_vertices;

    std::vector<SpringAttachments> m_attachment_vertices;           // Guided springs attachment vertices
    std::vector<int> m_spring_free_vertices_spring_idx;
    std::vector<int> m_spring_free_vertices_vertex_idx;

};




#endif
