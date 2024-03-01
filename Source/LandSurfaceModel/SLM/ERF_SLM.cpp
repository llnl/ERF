#include <ERF_SLM.H>
#include "EOS.H"
#include "TileNoZ.H"

using namespace amrex;

/* Initialize lsm data structures */
void
SLM::Init (const MultiFab& cons_in,
           const MultiFab& u_in,
           const MultiFab& v_in,
           const Geometry& geom,
           const Real& dt)
{
    m_dt = dt;
    m_geom = geom;

    Box domain = geom.Domain();
    khi_lsm    = domain.smallEnd(2) - 1; // index of z_r
    klo_lsm    = khi_lsm - m_nz_lsm + 1;

    LsmVarMap.resize(m_lsm_size);
    LsmVarMap = {LsmVar_SLM::theta};

    LsmVarName.resize(m_lsm_size);
    LsmVarName = {"theta"};

    // NOTE: All boxes in ba extend from zlo to zhi, so this transform is valid.
    //       If that were to change, the dm and new ba are no longer valid and
    //       direct copying between lsm data/flux vars cannot be done in a parfor.

    // Set box array for lsm data
    IntVect ng(0,0,1);
    BoxArray ba = cons_in.boxArray();
    DistributionMapping dm = cons_in.DistributionMap();
    BoxList bl_lsm = ba.boxList();
    for (auto& b : bl_lsm) {
        b.setBig(2, khi_lsm);                  // First point below the surface
        b.setSmall(2, klo_lsm);                // Last point below the surface
    }
    BoxArray ba_lsm(std::move(bl_lsm));

    // Set up lsm geometry
    const RealBox& dom_rb = m_geom.ProbDomain();
    const Real*    dom_dx = m_geom.CellSize();
    RealBox lsm_rb = dom_rb;
    Real lsm_dx[AMREX_SPACEDIM] = {AMREX_D_DECL(dom_dx[0],dom_dx[1],m_dz_lsm)};
    Real lsm_z_hi = dom_rb.lo(2); // z_r
    Real lsm_z_lo = lsm_z_hi - Real(m_nz_lsm)*lsm_dx[2]; // bottom Z of last soil layer: z_s
    lsm_rb.setHi(2,lsm_z_hi); lsm_rb.setLo(2,lsm_z_lo);
    m_lsm_geom.define( ba_lsm.minimalBox(), lsm_rb, m_geom.Coord(), m_geom.isPeriodic() );

    // Create the data and fluxes
    for (auto ivar = 0; ivar < LsmVar_SLM::NumVars; ++ivar) {
        // State vars are CC
        Real theta_0 = m_theta_dir;
        lsm_fab_vars[ivar] = std::make_shared<MultiFab>(ba_lsm, dm, 1, ng);
        lsm_fab_vars[ivar]->setVal(0.0);

        // Fluxes are nodal in z
        lsm_fab_flux[ivar] = std::make_shared<MultiFab>(convert(ba_lsm, IntVect(0,0,1)), dm, 1, IntVect(0,0,0));
        lsm_fab_flux[ivar]->setVal(0.);
    }

    // Create local 2D data
    BoxList bl_lsm_2d = ba_lsm.boxList();
    for (auto& b : bl_lsm_2d) {
        b.setRange(2, 0, 1);
    }
    BoxArray ba_lsm_2d(std::move(bl_lsm_2d));
    IntVect ng_2d(0, 0, 0);

    // TODO: Placeholder landmask array - fix!
    landmask.define(ba_lsm_2d, dm, 1, ng_2d);
    landmask.setVal(1);

    landtype.define(ba_lsm_2d, dm, 1, ng_2d);
    landtype.setVal(landtype0);

    LAI.define(ba_lsm_2d, dm, 1, ng_2d);
    LAI.setVal(LAI0);

    t_canop.define(ba_lsm_2d, dm, 1, ng_2d);
    mw.define(ba_lsm_2d, dm, 1, ng_2d);
    mws.define(ba_lsm_2d, dm, 1, ng_2d);
    t_skin.define(ba_lsm_2d, dm, 1, ng_2d);
    t_cas.define(ba_lsm_2d, dm, 1, ng_2d);
    q_cas.define(ba_lsm_2d, dm, 1, ng_2d);

    vegetype.define(ba_lsm_2d, dm, 1, ng_2d);
    cp_vege.define(ba_lsm_2d, dm, 1, ng_2d);
    z0_sfc.define(ba_lsm_2d, dm, 1, ng_2d);
    Khai_L.define(ba_lsm_2d, dm, 1, ng_2d);
    phi_1.define(ba_lsm_2d, dm, 1, ng_2d);
    phi_2.define(ba_lsm_2d, dm, 1, ng_2d);
    IR_emis_vege.define(ba_lsm_2d, dm, 1, ng_2d);
    IR_emis_soil.define(ba_lsm_2d, dm, 1, ng_2d);
    ztop.define(ba_lsm_2d, dm, 1, ng_2d);
    disp_hgt.define(ba_lsm_2d, dm, 1, ng_2d);
    Rgl.define(ba_lsm_2d, dm, 1, ng_2d);
    Rc_min.define(ba_lsm_2d, dm, 1, ng_2d);
    hs_rc.define(ba_lsm_2d, dm, 1, ng_2d);
    rootL.define(ba_lsm_2d, dm, 1, ng_2d);
    root_a.define(ba_lsm_2d, dm, 1, ng_2d);
    root_b.define(ba_lsm_2d, dm, 1, ng_2d);
    precip_extinc.define(ba_lsm_2d, dm, 1, ng_2d);
    mw_mx.define(ba_lsm_2d, dm, 1, ng_2d);

    mws_mx.define(ba_lsm_2d, dm, 1, ng_2d);
    mws_mx.setVal(mws_mx0);

    BAI.define(ba_lsm_2d, dm, 1, ng_2d);

    tau_soil.define(ba_lsm_2d, dm, 1, ng_2d);
    soilw_inc_tot.define(ba_lsm_2d, dm, 1, ng_2d);
    ustar.define(ba_lsm_2d, dm, 1, ng_2d);
    tstar.define(ba_lsm_2d, dm, 1, ng_2d);

    shf_canop.define(ba_lsm_2d, dm, 1, ng_2d);
    shf_soil.define(ba_lsm_2d, dm, 1, ng_2d);
    shf_air.define(ba_lsm_2d, dm, 1, ng_2d);
    lhf_canop.define(ba_lsm_2d, dm, 1, ng_2d);
    lhf_soil.define(ba_lsm_2d, dm, 1, ng_2d);
    lhf_air.define(ba_lsm_2d, dm, 1, ng_2d);

    albedovis_v.define(ba_lsm_2d, dm, 1, ng_2d);
    albedonir_v.define(ba_lsm_2d, dm, 1, ng_2d);
    albedovis_s.define(ba_lsm_2d, dm, 1, ng_2d);
    albedonir_s.define(ba_lsm_2d, dm, 1, ng_2d);

    // Initialize 1D arrays
    soilw_inc.resize({klo_lsm},  {khi_lsm});
    alpha.resize({klo_lsm},  {khi_lsm});
    beta.resize({klo_lsm},  {khi_lsm});


    slm_init();
}

/**
 * Initialize SLM
 */
void SLM::slm_init()
{

    // TODO: implement slm_setparm() function here

    // Read land_type data from file, otherwise landtype is uniformly set to landtype0
    if (readlandtype && landtype0 == 0)
    {
        // TODO: read input land type file
    }

    // validate the landtype flag
    const int landtype_min = landtype.min(0);
    const int landtype_max = landtype.max(0);
    if (landtype_min < 0 || landtype_max > 16)
    {
        amrex::Abort("SLM: landtype values are outside of valid 0-16 range! min = " +
                     std::to_string(landtype_min) + ", max = " + std::to_string(landtype_max));
    }

    if (readLAI && LAI0 == 0.0)
    {
        // TODO: read input LAI file
    }

    // validate LAI - check that they are within [0,10]
    const double lai_min = LAI.min(0);
    const double lai_max = LAI.max(0);
    if (lai_min < 0.0 || lai_max > 10.0)
    {
      amrex::Abort("SLM: LAI values are outside of valid [0-10] range! min = " +
                   std::to_string(lai_min) + ", max = " + std::to_string(lai_max));
    }

    // Sets model properties based on the landtype
    init_landtype();

    // TODO: VEGYES and set minimum LAI

    for ( MFIter mfi(landtype, TileNoZ()); mfi.isValid(); ++mfi) {
        const auto& box = mfi.tilebox();
        auto IR_emis_vege_arr = IR_emis_vege.array(mfi);
        auto phi_1_arr = phi_1.array(mfi);
        auto phi_2_arr = phi_2.array(mfi);
        auto precip_extinc_arr = precip_extinc.array(mfi);
        auto mw_mx_arr = mw_mx.array(mfi);
        auto LAI_arr = LAI.array(mfi);
        auto Khai_L_arr = Khai_L.array(mfi);

        ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int)
        {
            IR_emis_vege_arr(i, j, 0) = 0.97 * (1.0 - std::exp(-1.0 * LAI_arr(i, j, 0)));
            phi_1_arr(i, j, 0) = 0.5 - 0.633 * Khai_L_arr(i, j, 0) - 0.33 * (std::pow(Khai_L_arr(i, j, 0), 2));
            phi_2_arr(i, j, 0) = 0.877 * (1.0 - 2.0 * phi_1_arr(i, j, 0));
            precip_extinc_arr(i, j, 0) = phi_1_arr(i, j, 0) + phi_2_arr(i, j, 0);
            mw_mx_arr(i, j, 0) = 0.1 * LAI_arr(i, j, 0);
        });
    }

    // Initialize soil parameters
    init_soil_tw();

    IR_emis_soil.setVal(0.98);

    auto theta = lsm_fab_vars[LsmVar_SLM::theta];
    for ( MFIter mfi(*theta, TileNoZ()); mfi.isValid(); ++mfi) {
        const auto& box3d = mfi.tilebox();

        auto landmask_arr = landmask.const_array(mfi);

        auto sand_arr = lsm_fab_vars[LsmVar_SLM::sand]->const_array(mfi);
        auto clay_arr = lsm_fab_vars[LsmVar_SLM::clay]->const_array(mfi);

        auto w_s_FC_arr = lsm_fab_vars[LsmVar_SLM::w_s_FC]->array(mfi);
        auto w_s_WP_arr = lsm_fab_vars[LsmVar_SLM::w_s_WP]->array(mfi);
        auto sst_capa_arr = lsm_fab_vars[LsmVar_SLM::sst_capa]->array(mfi);
        auto sst_cond_arr = lsm_fab_vars[LsmVar_SLM::sst_cond]->array(mfi);
        auto poro_soil_arr = lsm_fab_vars[LsmVar_SLM::poro_soil]->array(mfi);
        auto theta_FC_arr = lsm_fab_vars[LsmVar_SLM::theta_FC]->array(mfi);
        auto theta_WP_arr = lsm_fab_vars[LsmVar_SLM::theta_WP]->array(mfi);
        auto m_pot_sat_arr = lsm_fab_vars[LsmVar_SLM::m_pot_sat]->array(mfi);
        auto Bconst_arr = lsm_fab_vars[LsmVar_SLM::Bconst]->array(mfi);
        auto ks_arr = lsm_fab_vars[LsmVar_SLM::ks]->array(mfi);

        ParallelFor(box3d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            if (landmask_arr(i, j, 0) == 1)
            {
                const amrex::Real sand_per = sand_arr(i, j, k); // sand percentage

                // soil solids thermal conductivity (Johansen 1975)
                //  quartz (= SAND content) thermal conductivity = 7.7 W/mK
                const amrex::Real mineral_tcond = sand_per > 20.0 ? 2.0 : 3.0; // thermal conductivity of other minerals [W/mK]
                sst_cond_arr(i, j, k) = std::pow(7.7, (sand_per * 0.01)) * std::pow(mineral_tcond, (1.0 - (sand_per * 0.01)));

                // Calculated following Cosby et al. 1984 ( hydraulic properties)
                // hydraulic conductivity at satuation , mm/s
                ks_arr(i, j, k) = std::pow(10.0, (0.0153*sand_per) - 0.884) * (25.4 / 3600.0); // [mm/s] from [inch/hr]

                // constant B
                Bconst_arr(i, j, k) = 0.159 * clay_arr(i, j, k) + 2.91;

                // porosity (or saturation volumetric water content)
                poro_soil_arr(i, j, k) = -0.00126 * sand_per + 0.489; // volume/volume

                // moisture potential at saturation, [mm]
                m_pot_sat_arr(i, j, k) = std::min(-150.0, -10.0*(std::pow(10.0, 1.88 - 0.0131*sand_per))); // [mm] from [cm]

                // soil heat capacity, [J/m^3/K]
                //  Following de Vries(1963) using SAND=34% CLAY=63%
                sst_capa_arr(i, j, k) = 1.0e6 * (2.128*sand_per + 2.385*clay_arr(i, j, k)) / (sand_per + clay_arr(i, j, k));

                // volumetric moisture content at field capacity
                // field capacity is assumed to be the occasion when hydraulic conductivity is 0.1mm/d
                theta_FC_arr(i, j, k) = poro_soil_arr(i, j, k) * std::pow((0.1 / 86400.0 / ks_arr(i, j, k)), 1.0 / (2.0 * Bconst_arr(i, j, k) + 3.0));

                // volumetric moisture content at wilting point
                theta_WP_arr(i, j, k) = poro_soil_arr(i, j, k) * std::pow((-150000.0 / m_pot_sat_arr(i, j, k)), (-1.0 / Bconst_arr(i, j, k)));

                // soil wetness at field capacity
                w_s_FC_arr(i, j, k) = theta_FC_arr(i, j, k) / poro_soil_arr(i, j, k);

                // soil wetness at wilting point
                w_s_WP_arr(i, j, k) = theta_WP_arr(i, j, k) / poro_soil_arr(i, j, k);
            }
        });
    }

    // Calculate fraction of root in each soil layer
    vege_root_init();

    // TODO:
    // soilt_obs == soilt
    // soilw_obs == soilw
}

/**
 * Helper function to set properties according to the IGBP class
 */
void SLM::init_landtype()
{
    for ( MFIter mfi(landtype, TileNoZ()); mfi.isValid(); ++mfi) {
        const auto& box = mfi.tilebox();

        auto landtype_arr = landtype.const_array(mfi);
        auto landmask_arr = landmask.array(mfi);
        auto LAI_arr = LAI.array(mfi);

        auto albedovis_v_arr = albedovis_v.array(mfi);
        auto albedonir_v_arr = albedonir_v.array(mfi);
        auto albedovis_s_arr = albedovis_s.array(mfi);
        auto albedonir_s_arr = albedonir_s.array(mfi);
        auto ztop_arr = ztop.array(mfi);
        auto disp_hgt_arr = disp_hgt.array(mfi);
        auto z0_sfc_arr = z0_sfc.array(mfi);
        auto Khai_L_arr = Khai_L.array(mfi);
        auto rootL_arr = rootL.array(mfi);
        auto root_a_arr = root_a.array(mfi);
        auto root_b_arr = root_b.array(mfi);
        auto Rc_min_arr = Rc_min.array(mfi);
        auto Rgl_arr = Rgl.array(mfi);
        auto hs_rc_arr = hs_rc.array(mfi);
        auto BAI_arr = BAI.array(mfi);
        auto vegetype_arr = vegetype.array(mfi);

        ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int)
        {
            switch (landtype_arr(i, j, 0))
            {
                case 0: // water
                    // TODO: should this landmask be ERF's lmask_lev variable?
                    landmask_arr(i, j, 0) = 0;
                    vegetype_arr(i, j, 0) = 0;
                    break;
                case 1: // evergreen needleaf forest
                    albedovis_v_arr(i, j, 0) = 0.04;
                    albedonir_v_arr(i, j, 0) = 0.20;
                    albedovis_s_arr(i, j, 0) = 0.19;
                    albedonir_s_arr(i, j, 0) = 0.38;
                    ztop_arr(i, j, 0) = 20.;
                    disp_hgt_arr(i, j, 0) = 0.68 * ztop_arr(i, j, 0);
                    z0_sfc_arr(i, j, 0) = 1.09;
                    Khai_L_arr(i, j, 0) = 0.25;
                    rootL_arr(i, j, 0) = 1.5;
                    root_a_arr(i, j, 0) = 5.558;
                    root_b_arr(i, j, 0) = 2.614;
                    Rc_min_arr(i, j, 0) = 100.;
                    Rgl_arr(i, j, 0) = 30.;
                    hs_rc_arr(i, j, 0) = 54.53;
                    BAI_arr(i, j, 0) = 100.;
                    vegetype_arr(i, j, 0) = 1;
                    break;
                default:
                    // TODO: check if AMReX::Abort can be called inside ParFor?
                    amrex::Abort("landtype invalid for i = " + std::to_string(i) + " j = " + std::to_string(j));
            }

            if (vegetype_arr(i, j, 0) == 1 && LAI_arr(i, j, 0) == 0.0)
            {
                amrex::Abort("LAI is not set for vegetated land point at i = " + std::to_string(i) + " j = " + std::to_string(j));
            }
        });
    }
}

/**
 * Reads soil input file, initializes soil wetness and temperature
 */
void SLM::init_soil_tw()
{
    // TODO - read from input file

    auto theta = lsm_fab_vars[LsmVar_SLM::theta];
    for ( MFIter mfi(*theta, TileNoZ()); mfi.isValid(); ++mfi) {
        const auto& box3d = mfi.tilebox();

        auto landmask_arr = landmask.const_array(mfi);

        auto soilt_arr = lsm_fab_vars[LsmVar_SLM::soilt]->array(mfi);
        auto soilw_arr = lsm_fab_vars[LsmVar_SLM::soilw]->array(mfi);

        auto sand_arr = lsm_fab_vars[LsmVar_SLM::sand]->array(mfi);
        auto clay_arr = lsm_fab_vars[LsmVar_SLM::clay]->array(mfi);

        auto s_depth_arr = lsm_fab_vars[LsmVar_SLM::s_depth]->array(mfi);
        auto soil_relax_hgt_arr = lsm_fab_vars[LsmVar_SLM::soil_relax_hgt]->array(mfi);

        auto tau_soil_arr = tau_soil.array(mfi);

        ParallelFor(box3d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            if (landmask_arr(i, j, 0) == 1)
            {
                tau_soil_arr(i, j, 0) = tausoil;

                // TODO: replace this with input from file
                s_depth_arr(i, j, k) = m_dz_lsm;
                clay_arr(i, j, k) = clay0;
                sand_arr(i, j, k) = sand0;
                soilt_arr(i, j, k) = st0;
                soilw_arr(i, j, k) = sw0;

                soil_relax_hgt_arr(i, j, k) = 0.0;

                // TODO: nrestart conditional here
                // TODO: sstxy - should be ERF surface temp array?
                //sstxy()
            } else {
                //sstxy()
            }
        });
    }

    // TODO: is it worth storing s_depth_mm, or should we always convert?
    // s_depth_mm[] = s_depth[] * 1000.0

    // Calculate node_z (depth of the center of each soil layer)
    // TODO: this can probably be done natively via AMReX?
    for ( MFIter mfi(landtype, TileNoZ()); mfi.isValid(); ++mfi) {
        const auto& box = mfi.tilebox();

        auto s_depth_arr = lsm_fab_vars[LsmVar_SLM::s_depth]->const_array(mfi);

        auto node_z_arr = lsm_fab_vars[LsmVar_SLM::node_z]->array(mfi);
        auto interface_z_arr = lsm_fab_vars[LsmVar_SLM::interface_z]->array(mfi);

        ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int)
        {
            // k = 0 = reference level
            // k = -1 = top of soil layer = khi_lsm
            // k = -m_nz_lsm = klo_lsm

            node_z_arr(i, j, khi_lsm) = 0.5 * s_depth_arr(i, j, khi_lsm);
            for (int k = khi_lsm - 1; k >= klo_lsm; k--)
            {
                node_z_arr(i, j, k) = 0.5 * s_depth_arr(i, j, k);
                for (int kk = khi_lsm; kk >= k + 1; kk--)
                {
                    node_z_arr(i, j, k) = node_z_arr(i, j, k) + s_depth_arr(i, j, kk);
                }
            }

            for (int k = khi_lsm; k >= klo_lsm; k--)
            {
                interface_z_arr(i, j, k) = node_z_arr(i, j, k) + 0.5*s_depth_arr(i, j, k);
            }
        });
    }
}

/**
 * Assigns root fraction in each soil layer
 * Fraction of total root in each soil layer is determined based on the soil
 * depth and vegetation root parameters
 */
void SLM::vege_root_init()
{
    for ( MFIter mfi(landtype, TileNoZ()); mfi.isValid(); ++mfi) {
        const auto& box = mfi.tilebox();

        auto landmask_arr = landmask.const_array(mfi);

        auto interface_z_arr = lsm_fab_vars[LsmVar_SLM::interface_z]->const_array(mfi);
        auto rootF_arr = lsm_fab_vars[LsmVar_SLM::rootF]->array(mfi);
        auto rootL_arr = rootL.const_array(mfi);

        auto root_a_arr = root_a.const_array(mfi);
        auto root_b_arr = root_b.const_array(mfi);

        ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int)
        {
            if (landmask_arr(i, j, 0) == 1)
            {
                int nrootind = khi_lsm;
                for (int k = klo_lsm; k <= khi_lsm - 1; k++)
                {
                    if (interface_z_arr(i, j, k) >= rootL_arr(i, j, 0) && interface_z_arr(i, j, k + 1) <= rootL_arr(i, j, 0))
                    {
                        nrootind = k;
                    }
                }

                rootF_arr(i, j, nrootind) = 1.0 - 0.5 * (std::exp(-1.0 * root_a_arr(i, j, 0) * rootL_arr(i, j, 0)) +
                                                         std::exp(-1.0 * root_b_arr(i, j, 0) * rootL_arr(i, j, 0)));

                const amrex::Real tot_root_density = rootF_arr(i, j, nrootind);
                for (int k = khi_lsm; k >= nrootind + 1; k--)
                {
                    rootF_arr(i, j, k) = 1.0 - 0.5 * (std::exp(-1.0 * root_a_arr(i, j, 0) * interface_z_arr(i, j, k)) +
                                                      std::exp(-1.0 * root_b_arr(i, j, 0) * interface_z_arr(i, j, k)));
                }

                if (nrootind < khi_lsm)
                {
                    for (int k = nrootind; k <= khi_lsm - 1; k++)
                    {
                        rootF_arr(i, j, k) = rootF_arr(i, j, k) - rootF_arr(i, j, k + 1);
                    }
                }

                // to ensure total root density equals 1
                amrex::Real new_root_density = 0.0;
                for (int k = khi_lsm; k >= nrootind; k--)
                {
                    rootF_arr(i, j, k) = rootF_arr(i, j, k) / tot_root_density;
                    new_root_density += rootF_arr(i, j, k);
                }

                AMREX_ASSERT_WITH_MESSAGE(std::abs(new_root_density - 1.0) < 1.0e-12, "total root density should equal 1.0");
            }
        });
    }
}

/* Extrapolate surface temperature and store in ghost cell */
void
SLM::ComputeTsurf ()
{
    // Expose for GPU copy
    int khi = khi_lsm;

    for ( MFIter mfi(*(lsm_fab_vars[LsmVar_SLM::theta])); mfi.isValid(); ++mfi) {
        auto box2d = mfi.tilebox(); box2d.makeSlab(2,khi);

        auto theta_array = lsm_fab_vars[LsmVar_SLM::theta]->array(mfi);

        ParallelFor( box2d, [=] AMREX_GPU_DEVICE (int i, int j, int )
        {
            theta_array(i,j,khi+1) = 1.5*theta_array(i,j,khi) - 0.5*theta_array(i,j,khi-1);
        });
    }
}

/* Compute the diffusive fluxes */
void
SLM::ComputeFluxes ()
{
    // Expose for GPU copy
    int khi = khi_lsm;
    Real Dsoil = m_d_soil;
    Real dzInv = m_lsm_geom.InvCellSize(2);

    for ( MFIter mfi(*(lsm_fab_flux[LsmVar_SLM::theta])); mfi.isValid(); ++mfi) {
        auto box3d = mfi.tilebox();

        // Do not overwrite the flux at the top (comes from MOST BC)
        if (box3d.bigEnd(2) == khi+1) box3d.setBig(2,khi);

        auto theta_array = lsm_fab_vars[LsmVar_SLM::theta]->array(mfi);
        auto theta_flux  = lsm_fab_flux[LsmVar_SLM::theta]->array(mfi);

        ParallelFor( box3d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            theta_flux(i,j,k) = Dsoil * ( theta_array(i,j,k) - theta_array(i,j,k-1) ) * dzInv;
        });
    }
}

/* Advance the solution with a simple explicit update (should use tridiagonal solve) */
void
SLM::AdvanceSLM ()
{
    // Expose for GPU copy
    Real dt = m_dt;
    Real dzInv = m_lsm_geom.InvCellSize(2);

    for ( MFIter mfi(*(lsm_fab_vars[LsmVar_SLM::theta])); mfi.isValid(); ++mfi) {
        auto box3d = mfi.tilebox();

        auto theta_array = lsm_fab_vars[LsmVar_SLM::theta]->array(mfi);
        auto theta_flux  = lsm_fab_flux[LsmVar_SLM::theta]->array(mfi);

        ParallelFor( box3d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            theta_array(i,j,k) += dt * ( theta_flux(i,j,k+1) - theta_flux(i,j,k) ) * dzInv;
        });
    }
}

void SLM::Copy_State_to_Lsm(const MultiFab& cons_in, const MultiFab& u_in, const MultiFab& v_in)
{
    int khi = khi_lsm;

    auto theta = lsm_fab_vars[LsmVar_SLM::theta];

    // Get the temperature, density, pressure at the reference level
    for ( MFIter mfi(*theta, TileNoZ()); mfi.isValid(); ++mfi) {
        const auto& box3d = mfi.tilebox();

        // Create a box with the same i,j bounds, but only at z = 0
        amrex::Box b2d = box3d;
        b2d.setRange(2, 0);

        auto states_array = cons_in.array(mfi);
        auto u_array = u_in.array(mfi);
        auto v_array = v_in.array(mfi);

        auto tref_array  = lsm_fab_vars[LsmVar_SLM::tref]->array(mfi);
        auto rho_array   = lsm_fab_vars[LsmVar_SLM::dref]->array(mfi);
        auto pres_array  = lsm_fab_vars[LsmVar_SLM::pref]->array(mfi);

        auto slm_u       = lsm_fab_vars[LsmVar_SLM::uref]->array(mfi);
        auto slm_v       = lsm_fab_vars[LsmVar_SLM::vref]->array(mfi);

        ParallelFor(b2d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            const Real qv = states_array(i,j,k,RhoQ1_comp)/states_array(i,j,k,Rho_comp);
            rho_array(i,j,k)   = states_array(i,j,k,Rho_comp);
            pres_array(i,j,k)  = getPgivenRTh(states_array(i,j,k,RhoTheta_comp), qv)/100.;
            tref_array(i,j,k)  = getTgivenRandRTh(states_array(i,j,k,Rho_comp),
                                                  states_array(i,j,k,RhoTheta_comp),
                                                  qv);

            slm_u(i,j,k) = u_array(i,j,k);
            slm_v(i,j,k) = v_array(i,j,k);
        });
    }
}

void
SLM::set_flux_inputs(const amrex::MultiFab* sw_lw_fluxes_in,
                     const amrex::MultiFab* zenith_in)
{
    auto theta = lsm_fab_vars[LsmVar_SLM::theta];

    for ( MFIter mfi(*theta, TileNoZ()); mfi.isValid(); ++mfi) {
        const auto& box3d = mfi.tilebox();

        // Create a box with the same i,j bounds, but only at z = 0
        amrex::Box b2d = box3d;
        b2d.setRange(2, 0);

        auto sw_lw_fluxes_arr = sw_lw_fluxes_in->const_array(mfi);
        auto zenith_array     = zenith_in->const_array(mfi);

        auto slm_dir_sw_vis  = lsm_fab_vars[LsmVar_SLM::swdsvisxyref]->array(mfi);
        auto slm_dir_sw_nir  = lsm_fab_vars[LsmVar_SLM::swdsnirxyref]->array(mfi);
        auto slm_diff_sw_vis = lsm_fab_vars[LsmVar_SLM::swdsvisdxyref]->array(mfi);
        auto slm_diff_sw_nir = lsm_fab_vars[LsmVar_SLM::swdsnirdxyref]->array(mfi);

        auto slm_lw          = lsm_fab_vars[LsmVar_SLM::lwref]->array(mfi);
        auto slm_zenith      = lsm_fab_vars[LsmVar_SLM::coszrsxy]->array(mfi);

        ParallelFor(b2d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            slm_dir_sw_vis(i, j, k) = sw_lw_fluxes_arr(i, j, k, 0);
            slm_dir_sw_nir(i, j, k) = sw_lw_fluxes_arr(i, j, k, 1);

            slm_diff_sw_vis(i, j, k) = sw_lw_fluxes_arr(i, j, k, 2);
            slm_diff_sw_nir(i, j, k) = sw_lw_fluxes_arr(i, j, k, 3);

            slm_lw(i, j, k) = sw_lw_fluxes_arr(i, j, k, 4);
            slm_zenith(i, j, k) = zenith_array(i, j, k, 0);
        });
    }
}

void
SLM::set_precip_input(const amrex::MultiFab* precip_in)
{
    auto theta = lsm_fab_vars[LsmVar_SLM::theta];

    for ( MFIter mfi(*theta, TileNoZ()); mfi.isValid(); ++mfi) {
        const auto& box3d = mfi.tilebox();

        // Create a box with the same i,j bounds, but only at z = 0
        amrex::Box b2d = box3d;
        b2d.setRange(2, 0);

        auto precip_array = precip_in->const_array(mfi);

        auto slm_precip   = lsm_fab_vars[LsmVar_SLM::precipref]->array(mfi);

        ParallelFor(b2d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            slm_precip(i, j, k) = precip_array(i, j, k, 0);
        });
    }
}


void SLM::Copy_Lsm_to_State(MultiFab& cons_in)
{
    for ( amrex::MFIter mfi(cons_in,amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
        const auto& box3d = mfi.tilebox();

        auto states_arr = cons_in.array(mfi);

        auto tref_array  = lsm_fab_vars[LsmVar_SLM::tref]->array(mfi);
        auto rho_array   = lsm_fab_vars[LsmVar_SLM::dref]->array(mfi);
        auto pres_array  = lsm_fab_vars[LsmVar_SLM::pref]->array(mfi);

        // get potential total density, temperature, qt, qp
        ParallelFor( box3d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            // TODO
            /*
            states_arr(i,j,k,RhoTheta_comp) = rho_arr(i,j,k)*theta_arr(i,j,k);
            states_arr(i,j,k,RhoQ1_comp)    = rho_arr(i,j,k)*qv_arr(i,j,k);
            states_arr(i,j,k,RhoQ2_comp)    = rho_arr(i,j,k)*qc_arr(i,j,k);
            states_arr(i,j,k,RhoQ3_comp)    = rho_arr(i,j,k)*qp_arr(i,j,k);
            */
        });
    }

    // Fill interior ghost cells and periodic boundaries
    cons_in.FillBoundary(m_geom.periodicity());
}