#ifndef ERF_NULLSURF_H
#define ERF_NULLSURF_H

#include <AMReX_MultiFabUtil.H>
#include <AMReX_Geometry.H>
#include <ERF_DataStruct.H>

class NullSurf {

 public:
    NullSurf () {}

    virtual ~NullSurf () = default;

    virtual
    void
    Define (SolverChoice& /*sc*/) { }

    // u_in and v_in might not be needed in all LSM models (e.g MM5), maybe refactor?
    virtual
    void Init (const amrex::MultiFab& /*cons_in*/,
               const amrex::MultiFab& /*u_in*/,
               const amrex::MultiFab& /*v_in*/,
               const amrex::Geometry& /*geom*/,
               const amrex::Real& /*dt_advance*/) { }

    virtual
    void
    Advance (const amrex::Real& /*dt_advance*/) { }

    virtual
    void
    Update_Lsm_Vars (amrex::MultiFab& /*cons_in*/,
                     amrex::MultiFab& /*u_in*/,
                     amrex::MultiFab& /*v_in*/) { }

    virtual
    void
    Update_State_Vars (amrex::MultiFab& /*cons_in*/) { }

    virtual
    void
    Copy_State_to_Lsm (const amrex::MultiFab& /*cons_in*/,
                       const amrex::MultiFab& /*u_in*/,
                       const amrex::MultiFab& /*v_in*/) { }

    virtual
    void
    Copy_Lsm_to_State (amrex::MultiFab& /*cons_in*/) { }

    virtual
    amrex::MultiFab*
    Lsm_Data_Ptr (const int& /*varIdx*/ ) { return nullptr; }

    virtual
    amrex::MultiFab*
    Lsm_Flux_Ptr (const int& /*varIdx*/ ) { return nullptr; }

    virtual
    amrex::Geometry
    Lsm_Geom ( ) { return m_lsm_geom; }

    virtual
    int
    Lsm_Data_Size () { return NullSurf::m_lsm_size; }

    virtual
    std::string
    Lsm_VarName (const int& /*varIdx*/)
    {
        return varname;
    }

    virtual
    void
    set_flux_inputs(const amrex::MultiFab* sw_lw_fluxes_in,
                    const amrex::MultiFab* zenith_in) { }

    virtual
    void
    set_precip_input(const amrex::MultiFab* precip_in) { }

 private:
    int m_lsm_size = 1;
    amrex::Geometry m_lsm_geom;
    std::string varname = " ";
};
#endif
