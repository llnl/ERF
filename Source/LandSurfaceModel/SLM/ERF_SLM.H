#ifndef ERF_SLM_H
#define ERF_SLM_H

#include <string>
#include <vector>
#include <memory>

#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_TableData.H>
#include <AMReX_ParmParse.H>

#include <ERF_NullSurf.H>
#include <ERF_Constants.H>
#include <ERF_IndexDefines.H>
#include <ERF_DataStruct.H>

namespace LsmVar_SLM {
   enum {
      // independent variables
      theta = 0,
      // inputs from ERF
      tref,            // temperature at reference level
      qref,            // specific humidity at ref
      pref,            // pressure at reference level
      dref,            // density at reference level
      uref,            // zonal wind speed at ref
      vref,            // meridional wind speed at ref
      precipref,       // precipitation rate at ref
      swdsvisxyref,    // downwelling SW radiation, direct visible flux
      swdsnirxyref,    // downwelling SW radiation, direct near-IR flux
      swdsvisdxyref,   // downwelling SW radiation, diffuse visible flux
      swdsnirdxyref,   // downwelling SW radiation, diffuse near-IR flux
      lwref,           // downwelling LW radiation
      coszrsxy,        // zenith angle
      tsurf,           // surface temperature
      // model variables
      tv,              // vegetation temperature
      mv,              // vegetation moisture storage
      soilt,           // soil temperature
      soilw,           // soil wetness
      // soil parameters
      sand,            // sand percentage
      clay,            // clay percentage
      s_depth,         // thickness of each soil layer in meters
      node_z,          // depth of the center of each soil layer
      interface_z,     // depth of the soil layer interface
      soilw_nudge,     // nudging of soilw
      soilt_nudge,     // nudging of soilt
      sw_over_sat,     // soilw over saturation
      soil_relax_hgt,  // depth dependent soil relaxation function
      // vegetation
      rootF,           // fraction of total root length of a vegetation in each soil layer
      w_s_FC,          // wetness at field capacity
      w_s_WP,          // wetness at wilting point
      // soil parameters calculated from sand/clay
      sst_capa,        // heat capacity for soil solids
      sst_cond,        // heat conductivity of soil solids
      poro_soil,       // saturation moisture content, porosity
      theta_FC,        // volumetric moisture content at field capacity
      theta_WP,        // volumetric moisture content at wilting point
      m_pot_sat,       // moisture potential at saturation
      Bconst,          // B constant
      ks,              // hydraulic conductivity at saturation [m/s]
      // model outputs
      flbu,            // surface u-momentum
      flbv,            // surface v-momentum
      flbq,            // surface vapor flux (latent heat flux)
      flbt,            // surface heat flux
      prsfc,           // precipitation rate reaching soil
      NumVars
  };
}

/* See Chen & Dudhia (2001) https://doi.org/10.1175/1520-0493(2001)129<0569:CAALSH>2.0.CO;2 */
class SLM : public NullSurf {

    using FabPtr = std::shared_ptr<amrex::MultiFab>;

public:
    // Constructor
    SLM () {}

    // Destructor
    virtual ~SLM () = default;

    // Set thermo and grid properties
    void
    Define (SolverChoice& /*sc*/) override
    {
        // NOTE: We should parse things from sc here,
        //       but they are hard coded because this
        //       is a demonstration for now.
    }

    // Initialize data structures
    void
    Init (const amrex::MultiFab& cons_in,
          const amrex::MultiFab& u_in,
          const amrex::MultiFab& v_in,
          const amrex::Geometry& geom,
          const amrex::Real& dt) override;

    // Wrapper to do all the updating
    void
    Advance (const amrex::Real& dt) override
    {
        m_dt = dt;

        this->init_slm_vars();

        this->ComputeFluxes();
        this->AdvanceSLM();
        this->ComputeTsurf();
    }

    // Compute surface temperature
    void
    ComputeTsurf ();

    // Compute diffusive fluxes
    void
    ComputeFluxes ();

    // Advance the lsm state vars
    void
    AdvanceSLM ();

    // Get state vars from lsm class
    amrex::MultiFab*
    Lsm_Data_Ptr (const int& varIdx) override
    {
        AMREX_ALWAYS_ASSERT(varIdx < SLM::m_lsm_size && varIdx >= 0);
        int lsmIdx = LsmVarMap[varIdx];
        AMREX_ALWAYS_ASSERT(lsmIdx < LsmVar_SLM::NumVars && lsmIdx>=0);
        return lsm_fab_vars[lsmIdx].get();
    }

    // Get flux vars from lsm class
    amrex::MultiFab*
    Lsm_Flux_Ptr (const int& varIdx) override
    {
        AMREX_ALWAYS_ASSERT(varIdx < SLM::m_lsm_size && varIdx >= 0);
        int lsmIdx = LsmVarMap[varIdx];
        AMREX_ALWAYS_ASSERT(lsmIdx < LsmVar_SLM::NumVars && lsmIdx>=0);
        return lsm_fab_flux[lsmIdx].get();
    }

    // Get lsm geometry
    amrex::Geometry
    Lsm_Geom ( ) override { return m_lsm_geom; }

    // Get number of vars lsm class contains
    int
    Lsm_Data_Size () override { return SLM::m_lsm_size; }

    // Get variable names
    std::string
    Lsm_VarName (const int& varIdx) override
    {
        AMREX_ALWAYS_ASSERT(varIdx < SLM::m_lsm_size && varIdx >= 0);
        return LsmVarName[varIdx];
    }

    void
    Copy_State_to_Lsm (const amrex::MultiFab& cons_in, const amrex::MultiFab& u_in, const amrex::MultiFab& v_in) override;

    void
    Copy_Lsm_to_State (amrex::MultiFab& cons_in) override;

    void
    Update_Lsm_Vars (amrex::MultiFab& cons_in, amrex::MultiFab& u_in, amrex::MultiFab& v_in) override
    {
        this->Copy_State_to_Lsm(cons_in, u_in, v_in);
        //this->Diagnose();
        //this->Compute_Coefficients();
    }

    void
    Update_State_Vars (amrex::MultiFab& cons_in) override
    {
        this->Copy_Lsm_to_State(cons_in);
    }

    void
    set_flux_inputs(const amrex::MultiFab* sw_lw_fluxes_in,
                    const amrex::MultiFab* zenith_in) override;

    void
    set_precip_input(const amrex::MultiFab* precip_in) override;

    void
    set_terrain_inputs(const amrex::Vector<std::unique_ptr<amrex::MultiFab>>& sst_in,
                       const amrex::Vector<std::unique_ptr<amrex::iMultiFab>>& lmask_in) override;

    static std::vector<std::vector<amrex::Real>>
    read_cols(const std::string& fname, const int skip_nlines = 1);

  private:
    void init_from_file();

    void slm_init();

    void init_landtype();

    void init_soil_tw();

    void vege_root_init();

    void init_slm_vars();

    void radiative_fluxes(const int i, const int j);

    void transfer_coeff(const int i, const int j);

    void resistances(const int i, const int j);

    void vapor_fluxes(const int i, const int j);

    void soil_water(const int i, const int j);

    void soil_temperature(const amrex::MFIter &mfi);

    amrex::Real fh_calc(const amrex::Real &t, const amrex::Real &mps,
                        const amrex::Real &sw, const amrex::Real &B);

    // number of lsm variables (theta)
    int m_lsm_size = 14;

    // LsmVar map (state indices -> LsmVar enum)
    amrex::Vector<int> LsmVarMap;

    // Lsm varnames
    amrex::Vector<std::string> LsmVarName;

    // geometry for atmosphere
    amrex::Geometry m_geom;

    // geometry for lsm
    amrex::Geometry m_lsm_geom;

    // timestep
    amrex::Real m_dt;

    // domain klo-1 or lsm khi
    int khi_lsm;

    int klo_lsm;

    // constants
    // amrex::Real m_fac_cond;
    // amrex::Real m_fac_fus;
    // amrex::Real m_fac_sub;
    // amrex::Real m_gOcp;

    // independent variables
    amrex::Array<FabPtr, LsmVar_SLM::NumVars> lsm_fab_vars;

    // flux array for conjugate transfer
    amrex::Array<FabPtr, LsmVar_SLM::NumVars> lsm_fab_flux;

    // Vars that should be parsed
    // ==========================
    // Number of grid points in z
    int m_nz_lsm = 5;

    // Size of grid spacing in z
    amrex::Real m_dz_lsm = 0.1; // 3 [m] below surface

    // Specific heat
    amrex::Real m_cp_soil =  1.26e6; // [J/m^3 K]

    // Conductivity
    amrex::Real m_k_soil = 0.2; // dry soil [W/m K]

    // Theta Dirichlet value at lowest point below surface
    amrex::Real m_theta_dir = 400.0; // [K]

    // Thermal diffusivity
    amrex::Real m_d_soil = m_k_soil / m_cp_soil; // [m^2/s]


    // Input variables
    // TODO: move these to inputs parsing
    int landtype0 = 1; // landtype over whole domain

    amrex::Real LAI0 = 1.0; // default leaf area index
    amrex::Real clay0 = 50.0; // uniform clay content (%)
    amrex::Real sand0 = 50.0; // uniform sand content (%)
    amrex::Real sw0 = 0.0; // uniform soil wetness (fraction)
    amrex::Real st0 = 300.0; // uniform soil temperature (K)

    amrex::Real tabs_s = 0.0; // surface temperature (K)
    amrex::Real t00 = 300.0; // constant offset for sstxy

    //   TODO: implement reading these input files
    bool readlandtype = false; // read landtype map from file
    bool readLAI = false; // read LAI file
    std::string landtypefile;
    std::string LAIfile;

    bool dosoilwnudging = false;
    bool dosoiltnudging = false;
    amrex::Real tausoil = 86400.0;

    amrex::Real mws_mx0 = 50.0; // maximum puddle water storage (mm)

    amrex::Real tfriz = 273.15;

    amrex::Real pres0 = p_0; // reference surface pressure [Pa]

    amrex::Real cp_water = 4.1796e6; // heat capacity of water [J/m^3K]

    // Model variables
    //   NOTE: these MultiFabs are defined over the grid only in X and Y (no Z component)
    amrex::MultiFab t_canop; // canopy leaf temperature
    amrex::MultiFab mw; // water storage on leaves, kg/m^2 = mm of water
    amrex::MultiFab mws; // water storage in puddle, kg/m^2 = mm of water
    amrex::MultiFab t_skin; // skin temperature, surface level viewed from atmosphere

    amrex::MultiFab t_cas; // canopy air space temperature
    amrex::MultiFab q_cas; // canopy air space specific humidity

    amrex::TableData<amrex::Real, 1> soilw_inc;

    // TODO: placeholder landmask array - should probably use the one defined by ERF
    amrex::iMultiFab landmask;
    amrex::iMultiFab landtype; // IGBP landtype (1 to 16)
    amrex::MultiFab LAI; // leaf area index

    amrex::MultiFab sstxy;

    // Vegetation parameter lists
    amrex::iMultiFab vegetype; // vegetation type (0 - baresoil, non-zero for vegetated)
    amrex::MultiFab vege_YES; // 0-baresoil, 1-vegetated
    amrex::MultiFab cp_vege; // heat capacity of vegetation
    amrex::MultiFab z0_sfc; // surface level roughness: for vegetated z0_sfc=z0_vege, for non-vegetated, z0_sfc=z0_soil
    amrex::MultiFab Khai_L; // leaf distribution factor
    amrex::MultiFab phi_1;
    amrex::MultiFab phi_2;
    amrex::MultiFab IR_emis_vege; // vegetation emissivity for infrared radiation
    amrex::MultiFab IR_emis_soil; // soil emissivity for infrared radiation
    amrex::MultiFab ztop; // height of vegetation
    amrex::MultiFab disp_hgt; // displacement height
    amrex::MultiFab Rgl; // parameter for stomatal resistance
    amrex::MultiFab Rc_min; // minimum stomatal resistance
    amrex::MultiFab hs_rc; // parameter for stomatal resistance
    amrex::MultiFab rootL; // root length
    amrex::MultiFab root_a; // coef a for root density
    amrex::MultiFab root_b; // coef b for root density
    amrex::MultiFab precip_extinc; // precipitation extinction coef
    amrex::MultiFab mw_mx; // max of mw
    amrex::MultiFab mws_mx; // max of mws
    amrex::MultiFab BAI; // Basal area index, sq.feet/acre

    amrex::TableData<amrex::Real, 1> alpha;
    amrex::TableData<amrex::Real, 1> beta;

    amrex::MultiFab tau_soil; // soil nudging time in seconds, when dosoiltnudging or dosoilwnudging=true

    amrex::MultiFab soilw_inc_tot;

    amrex::MultiFab ustar, tstar;

    amrex::Real mom_trans_coef; // surface momentum transfer coef.
    amrex::Real heat_trans_coef; // surface heat transfer coef.
    amrex::Real xsi; // M-O stability parameter
    amrex::Real RiB; // bulk Richardson number

    // Sensible heat fluxes
    amrex::MultiFab shf_canop; // flux transfer between canopy and air space [W/m^2]
    amrex::MultiFab shf_soil; // flux transfer between soil surface and air space [W/m^2]
    amrex::MultiFab shf_air; // flux transfer between air space and the reference level [W/m^2]

    // Latent heat fluxes
    amrex::MultiFab lhf_canop;
    amrex::MultiFab lhf_soil;
    amrex::MultiFab lhf_air;

    // Albedos
    amrex::MultiFab albedovis_v, albedonir_v; // visible and NIR albedos for vegetation
    amrex::MultiFab albedovis_s, albedonir_s; // visible and NIR albedos for soil


    // Resistances
    //   TODO: in the fortran version, these are scalar quantities but are updated over i,j?
    amrex::MultiFab r_a; // aerodynamic resistance between surface and ref level
    amrex::MultiFab r_b; // leaf boundary layer resistance
    amrex::MultiFab r_c; // stomatal resistance
    amrex::MultiFab r_d; // undercanopy aerodynamic resistance
    amrex::Real r_soil = 0.0;
    amrex::Real r_litter = 0.0; // not used currently


    amrex::MultiFab net_rad_canopy; // net absorbed shortwave radiation by canopy (net_rad(1))
    amrex::MultiFab wet_canop; // wetted portion of canopy

    // Variables related to setting initial conditions (for testing purposes only)
    bool set_from_file = false;
    int num_ref_inputs = 1;
    std::string ref_sounding_file, ref_flux_file, ref_sst_file;
};
#endif
