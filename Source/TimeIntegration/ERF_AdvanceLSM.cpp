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
        lsm.Advance(lev, dt_advance);
        lsm.Update_State_Vars_Lev(lev, cons);
    }
}
