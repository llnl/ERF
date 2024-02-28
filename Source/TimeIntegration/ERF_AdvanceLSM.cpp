#include <ERF.H>

using namespace amrex;

void ERF::advance_lsm (int lev,
                       MultiFab& cons,
                       MultiFab& u_in,
                       MultiFab& v_in,
                       const Real& dt_advance)
{
    if (solverChoice.lsm_type != LandSurfaceType::None) {
        lsm.Update_Lsm_Vars_Lev(lev, cons, u_in, v_in);
#ifdef ERF_USE_RRTMGP
        lsm.set_LSM_flux_inputs(lev, sw_lw_fluxes[lev].get(), solar_zenith[lev].get());
#endif
        lsm.set_LSM_precip_input(lev, precip[lev].get());
        lsm.Advance(lev, dt_advance);
        lsm.Update_State_Vars_Lev(lev, cons);
    }
}
