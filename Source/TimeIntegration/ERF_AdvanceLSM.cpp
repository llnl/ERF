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
        //lsm.set_LSM_flux_inputs(lev, sw_lw_fluxes[lev].get(), solar_zenith[lev].get());
#endif
        const bool use_moist = solverChoice.moisture_type != MoistureType::None && solverChoice.moisture_type != MoistureType::Kessler_NoRain;
        int rain_comp = 8;
        if (use_moist) {
            if(solverChoice.moisture_type == MoistureType::Kessler) {
                rain_comp = 4;
            } else {
                rain_comp = 8;
            }

            // qmoist[8] is total rain accumulation in mm over entire simulation
            // Convert to a rate of mm/s for SLM
            for ( MFIter mfi(*precip[lev], TileNoZ()); mfi.isValid(); ++mfi) {
                const auto& box2d = mfi.tilebox();
                auto precip_array = precip[lev]->array(mfi);
                auto rain_accum_array   = qmoist[lev][rain_comp]->const_array(mfi);
                ParallelFor(box2d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    precip_array(i, j, k) = std::max(0.0, (rain_accum_array(i, j, k) - precip_array(i, j, k)) / dt_advance);
                });
            }

            lsm.set_LSM_precip_input(lev, precip[lev].get());
        }

        lsm.set_LSM_terrain_inputs(lev, sst_lev, lmask_lev);
        lsm.Advance(lev, dt_advance);
        lsm.Update_State_Vars_Lev(lev, cons);

        if (use_moist) {
            // copy current rain_accum into precip used to compute the rate for the next timestep
            for ( MFIter mfi(*precip[lev], TileNoZ()); mfi.isValid(); ++mfi) {
                const auto& box2d = mfi.tilebox();
                auto precip_array = precip[lev]->array(mfi);
                auto rain_accum_array   = qmoist[lev][rain_comp]->const_array(mfi);
                ParallelFor(box2d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    precip_array(i, j, k) = rain_accum_array(i, j, k);
                });
            }
        }
    }
}
