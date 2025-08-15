/**
 *  Wrapper for advancing the solution with the slow RHS in the absence of acoustic substepping
 */
    auto no_substep_fun = [&](Vector<MultiFab>& S_sum,
                              Vector<MultiFab>& S_old,
                              Vector<MultiFab>& F_slow,
                              const Real time_for_fp, const Real slow_dt,
                              const int nrk)
    {
        BL_PROFILE("no_substep_fun");
        amrex::ignore_unused(nrk);
        int n_data = IntVars::NumTypes;

        const auto& dxInv = fine_geom.InvCellSizeArray();

        const amrex::GpuArray<int, IntVars::NumTypes> scomp_fast = {0,0,0,0};
        const amrex::GpuArray<int, IntVars::NumTypes> ncomp_fast = {2,1,1,1};

        if (verbose) amrex::Print() << " No-substepping time integration at level " << level
                                    << std::setprecision(timeprecision)
                                    << " to " << time_for_fp
                                    << " with dt = " << slow_dt << std::endl;

        // Update S_sum = S_stage only for the fast variables
#ifdef _OPENMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        {
            for ( MFIter mfi(S_sum[IntVars::cons],TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const Box bx = mfi.tilebox();
                Box tbx = mfi.nodaltilebox(0);
                Box tby = mfi.nodaltilebox(1);
                Box tbz = mfi.nodaltilebox(2);

                Vector<Array4<Real> >  ssum_h(n_data);
                Vector<Array4<Real> >  sold_h(n_data);
                Vector<Array4<Real> > fslow_h(n_data);

                for (int i = 0; i < n_data; ++i) {
                     ssum_h[i]  = S_sum[i].array(mfi);
                     sold_h[i]  = S_old[i].array(mfi);
                    fslow_h[i] = F_slow[i].array(mfi);
                }

                Gpu::AsyncVector<Array4<Real> >  sold_d(n_data);
                Gpu::AsyncVector<Array4<Real> >  ssum_d(n_data);
                Gpu::AsyncVector<Array4<Real> > fslow_d(n_data);

                Gpu::copy(Gpu::hostToDevice,  sold_h.begin(),  sold_h.end(),  sold_d.begin());
                Gpu::copy(Gpu::hostToDevice,  ssum_h.begin(),  ssum_h.end(),  ssum_d.begin());
                Gpu::copy(Gpu::hostToDevice, fslow_h.begin(), fslow_h.end(), fslow_d.begin());

                Array4<Real>*  sold =  sold_d.dataPtr();
                Array4<Real>*  ssum =  ssum_d.dataPtr();
                Array4<Real>* fslow = fslow_d.dataPtr();

                // Moving terrain
                if ( solverChoice.terrain_type == TerrainType::MovingFittedMesh )
                {
                    const Array4<const Real>& dJ_old =     detJ_cc[level]->const_array(mfi);
                    const Array4<const Real>& dJ_new = detJ_cc_new[level]->const_array(mfi);

                    const Array4<const Real>& z_nd_old =     z_phys_nd[level]->const_array(mfi);
                    const Array4<const Real>& z_nd_new = z_phys_nd_new[level]->const_array(mfi);

                    const Array4<const Real>& mf_u = mapfac[level][MapFacType::u_x]->const_array(mfi);
                    const Array4<const Real>& mf_v = mapfac[level][MapFacType::v_y]->const_array(mfi);

                    const Array4<Real      >& z_t_arr  =  z_t_rk[level]->array(mfi);

                    // We have already scaled the slow source term to have the extra factor of dJ
                    ParallelFor(bx, ncomp_fast[IntVars::cons],
                    [=] AMREX_GPU_DEVICE (int i, int j, int k, int nn) {
                        const int n = scomp_fast[IntVars::cons] + nn;
                        ssum[IntVars::cons](i,j,k,n) = dJ_old(i,j,k) *  sold[IntVars::cons](i,j,k,n)
                                                          + slow_dt * fslow[IntVars::cons](i,j,k,n);
                        ssum[IntVars::cons](i,j,k,n) /= dJ_new(i,j,k);
                    });

                    // We have already scaled the slow source term to have the extra factor of dJ
                    ParallelFor(tbx, tby,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                        Real h_zeta_old = Compute_h_zeta_AtIface(i, j, k, dxInv, z_nd_old);
                        Real h_zeta_new = Compute_h_zeta_AtIface(i, j, k, dxInv, z_nd_new);
                        ssum[IntVars::xmom](i,j,k) = ( h_zeta_old *  sold[IntVars::xmom](i,j,k)
                                                       + slow_dt * fslow[IntVars::xmom](i,j,k) ) / h_zeta_new;
                    },
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                        Real h_zeta_old = Compute_h_zeta_AtJface(i, j, k, dxInv, z_nd_old);
                        Real h_zeta_new = Compute_h_zeta_AtJface(i, j, k, dxInv, z_nd_new);
                        ssum[IntVars::ymom](i,j,k) = ( h_zeta_old *  sold[IntVars::ymom](i,j,k)
                                                       + slow_dt * fslow[IntVars::ymom](i,j,k) ) / h_zeta_new;
                    });
                    ParallelFor(tbz,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                        if (k == 0) {
                            // Here we take advantage of the fact that moving terrain has a slip wall
                            // so we can just use the new value at (i,j,0).
                            Real rho_on_face =  ssum[IntVars::cons](i,j,k,Rho_comp);
                            ssum[IntVars::zmom](i,j,k) = WFromOmega(i,j,k,rho_on_face*z_t_arr(i,j,k),
                                                                    ssum[IntVars::xmom], ssum[IntVars::ymom],
                                                                    mf_u, mf_v, z_nd_new,dxInv);
                        } else {
                            Real dJ_old_kface = 0.5 * (dJ_old(i,j,k) + dJ_old(i,j,k-1));
                            Real dJ_new_kface = 0.5 * (dJ_new(i,j,k) + dJ_new(i,j,k-1));
                            ssum[IntVars::zmom](i,j,k) = ( dJ_old_kface *  sold[IntVars::zmom](i,j,k)
                                                             + slow_dt * fslow[IntVars::zmom](i,j,k) ) / dJ_new_kface;
                        }
                    });

                } else { // Fixed or no terrain
                    const Array4<const Real>& dJ_old = detJ_cc[level]->const_array(mfi);
                    ParallelFor(bx, ncomp_fast[IntVars::cons],
                    [=] AMREX_GPU_DEVICE (int i, int j, int k, int nn) {
                        const int n = scomp_fast[IntVars::cons] + nn;
                        if (dJ_old(i,j,k) > 0.0) {
                            ssum[IntVars::cons](i,j,k,n) = sold[IntVars::cons](i,j,k,n) + slow_dt *
                               ( fslow[IntVars::cons](i,j,k,n) );
                        }
                    });

                    // Commenting out the update is a HACK while developing the EB capability
                    if (solverChoice.terrain_type == TerrainType::EB)
                    {
                        const Array4<const Real>& vfrac_u = (get_eb(level).get_u_const_factory())->getVolFrac().const_array(mfi);
                        const Array4<const Real>& vfrac_v = (get_eb(level).get_v_const_factory())->getVolFrac().const_array(mfi);
                        const Array4<const Real>& vfrac_w = (get_eb(level).get_w_const_factory())->getVolFrac().const_array(mfi);

                        ParallelFor(tbx, tby, tbz,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                            if (vfrac_u(i,j,k) > 0.0) {
                                // ssum[IntVars::xmom](i,j,k) = sold[IntVars::xmom](i,j,k);
                                ssum[IntVars::xmom](i,j,k) = sold[IntVars::xmom](i,j,k)
                                                           + slow_dt * fslow[IntVars::xmom](i,j,k);
                            } else {
                                ssum[IntVars::xmom](i,j,k) = sold[IntVars::xmom](i,j,k);
                            }
                        },
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                            if (vfrac_v(i,j,k) > 0.0) {
                                // ssum[IntVars::ymom](i,j,k) = sold[IntVars::ymom](i,j,k);
                                ssum[IntVars::ymom](i,j,k) = sold[IntVars::ymom](i,j,k)
                                                           + slow_dt * fslow[IntVars::ymom](i,j,k);
                            } else {
                                ssum[IntVars::ymom](i,j,k) = sold[IntVars::ymom](i,j,k);
                            }
                        },
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                            if (vfrac_w(i,j,k) > 0.0) {
                                // ssum[IntVars::zmom](i,j,k) = sold[IntVars::zmom](i,j,k);
                                ssum[IntVars::zmom](i,j,k) = sold[IntVars::zmom](i,j,k)
                                                           + slow_dt * fslow[IntVars::zmom](i,j,k);
                            } else {
                                ssum[IntVars::zmom](i,j,k) = sold[IntVars::zmom](i,j,k);
                            }
                        });

                    } else {
                        ParallelFor(tbx, tby, tbz,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                            ssum[IntVars::xmom](i,j,k) = sold[IntVars::xmom](i,j,k)
                                                       + slow_dt * fslow[IntVars::xmom](i,j,k);
                        },
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                            ssum[IntVars::ymom](i,j,k) = sold[IntVars::ymom](i,j,k)
                                                       + slow_dt * fslow[IntVars::ymom](i,j,k);
                        },
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                            ssum[IntVars::zmom](i,j,k) = sold[IntVars::zmom](i,j,k)
                                                       + slow_dt * fslow[IntVars::zmom](i,j,k);
                        });
                    } // no EB
                } // not moving terrain
            } // mfi

            if (solverChoice.terrain_type == TerrainType::EB)
            {
                // Redistribute cons states (cell-centered)

                MultiFab dUdt_tmp(ba, dm, F_slow[IntVars::cons].nComp(), F_slow[IntVars::cons].nGrow(), MFInfo(), EBFactory(level));
                dUdt_tmp.setVal(0.);
                MultiFab::Copy(dUdt_tmp, F_slow[IntVars::cons], 0, 0, F_slow[IntVars::cons].nComp(), F_slow[IntVars::cons].nGrow());
                dUdt_tmp.FillBoundary(fine_geom.periodicity());
                dUdt_tmp.setDomainBndry(1.234e10, 0, ncomp_fast[IntVars::cons], fine_geom);

                BCRec const* bc_ptr_d = domain_bcs_type_d.data();
                S_old[IntVars::cons].FillBoundary(fine_geom.periodicity());
                S_old[IntVars::cons].setDomainBndry(1.234e10, 0, ncomp_fast[IntVars::cons], fine_geom);

                // Update F_slow by Redistribution.
                redistribute_term ( ncomp_fast[IntVars::cons], fine_geom, F_slow[IntVars::cons], dUdt_tmp,
                                    S_old[IntVars::cons], EBFactory(level), bc_ptr_d, slow_dt);

                // Redistribute momentum states (face-centered)

                Vector<MultiFab> dUdt_mom(AMREX_SPACEDIM);
                for (int i = 0; i < AMREX_SPACEDIM; ++i) {
                    dUdt_mom[i].define(F_slow[1+i].boxArray(), F_slow[1+i].DistributionMap(), F_slow[1+i].nComp(), F_slow[1+i].nGrow(), MFInfo());
                    dUdt_mom[i].setVal(0.);
                    MultiFab::Copy(dUdt_mom[i], F_slow[1+i], 0, 0, F_slow[1+i].nComp(), F_slow[1+i].nGrow());
                    dUdt_mom[i].FillBoundary(fine_geom.periodicity());
                    dUdt_mom[i].setDomainBndry(1.234e10, 0, ncomp_fast[1+i], fine_geom);
                }

                for (int i = 0; i < AMREX_SPACEDIM; ++i) {
                    S_old[1+i].FillBoundary(fine_geom.periodicity());
                    S_old[1+i].setDomainBndry(1.234e10, 0, ncomp_fast[1+i], fine_geom);
                }

                // Update F_slow by Redistribution.
                redistribute_term( ncomp_fast[IntVars::xmom], fine_geom, F_slow[IntVars::xmom], dUdt_mom[0],
                                   S_old[IntVars::xmom], *(get_eb(level).get_u_const_factory()), bc_ptr_d, slow_dt, IntVars::xmom);
                redistribute_term( ncomp_fast[IntVars::ymom], fine_geom, F_slow[IntVars::ymom], dUdt_mom[1],
                                   S_old[IntVars::ymom], *(get_eb(level).get_v_const_factory()), bc_ptr_d, slow_dt, IntVars::ymom);
                redistribute_term( ncomp_fast[IntVars::zmom], fine_geom, F_slow[IntVars::zmom], dUdt_mom[2],
                                   S_old[IntVars::zmom], *(get_eb(level).get_w_const_factory()), bc_ptr_d, slow_dt, IntVars::zmom);

                // Update state using the updated F_slow. (NOTE: redistribute_term returns RHS not state variables.)

                for ( MFIter mfi(S_sum[IntVars::cons],TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {
                    const Box bx = mfi.tilebox();
                    Box tbx = mfi.nodaltilebox(0);
                    Box tby = mfi.nodaltilebox(1);
                    Box tbz = mfi.nodaltilebox(2);

                    Vector<Array4<Real> >  ssum_h(n_data);
                    Vector<Array4<Real> >  sold_h(n_data);
                    Vector<Array4<Real> > fslow_h(n_data);

                    for (int i = 0; i < n_data; ++i) {
                         ssum_h[i] =  S_sum[i].array(mfi);
                         sold_h[i] =  S_old[i].array(mfi);
                        fslow_h[i] = F_slow[i].array(mfi);
                    }

                    Gpu::AsyncVector<Array4<Real> >  sold_d(n_data);
                    Gpu::AsyncVector<Array4<Real> >  ssum_d(n_data);
                    Gpu::AsyncVector<Array4<Real> > fslow_d(n_data);

                    Gpu::copy(Gpu::hostToDevice,  sold_h.begin(),  sold_h.end(),  sold_d.begin());
                    Gpu::copy(Gpu::hostToDevice,  ssum_h.begin(),  ssum_h.end(),  ssum_d.begin());
                    Gpu::copy(Gpu::hostToDevice, fslow_h.begin(), fslow_h.end(), fslow_d.begin());

                    Array4<Real>*  sold =  sold_d.dataPtr();
                    Array4<Real>*  ssum =  ssum_d.dataPtr();
                    Array4<Real>* fslow = fslow_d.dataPtr();

                    // const Array4<const Real>& vfrac_c = detJ_cc[level]->const_array(mfi);
                    const Array4<const Real>& vfrac_c = (get_eb(level).get_const_factory())->getVolFrac().const_array(mfi);

                    ParallelFor(bx, ncomp_fast[IntVars::cons], [=] AMREX_GPU_DEVICE (int i, int j, int k, int nn)
                    {
                        const int n = scomp_fast[IntVars::cons] + nn;
                        if (vfrac_c(i,j,k) > 0.0) {
                            ssum[IntVars::cons](i,j,k,n) = sold[IntVars::cons](i,j,k,n) + slow_dt *
                                ( fslow[IntVars::cons](i,j,k,n) );
                        }
                    });

                    const Array4<const Real>& vfrac_u = (get_eb(level).get_u_const_factory())->getVolFrac().const_array(mfi);
                    const Array4<const Real>& vfrac_v = (get_eb(level).get_v_const_factory())->getVolFrac().const_array(mfi);
                    const Array4<const Real>& vfrac_w = (get_eb(level).get_w_const_factory())->getVolFrac().const_array(mfi);

                    ParallelFor(tbx, tby, tbz,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                        if (vfrac_u(i,j,k) > 0.0) {
                            ssum[IntVars::xmom](i,j,k) = sold[IntVars::xmom](i,j,k)
                            + slow_dt * fslow[IntVars::xmom](i,j,k);
                        }
                    },
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                        if (vfrac_v(i,j,k) > 0.0) {
                            ssum[IntVars::ymom](i,j,k) = sold[IntVars::ymom](i,j,k)
                                                    + slow_dt * fslow[IntVars::ymom](i,j,k);
                        }
                    },
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                        if (vfrac_w(i,j,k) > 0.0) {
                            ssum[IntVars::zmom](i,j,k) = sold[IntVars::zmom](i,j,k)
                                                    + slow_dt * fslow[IntVars::zmom](i,j,k);
                        }
                    });
                } // MFIter

            } // EB

        } // omp

        // Even if we update all the conserved variables we don't need
        // to fillpatch the slow ones every acoustic substep
        apply_bcs(S_sum, time_for_fp, S_sum[IntVars::cons].nGrow(), S_sum[IntVars::xmom].nGrow(),
                  fast_only=true, vel_and_mom_synced=false);

        if (solverChoice.anelastic[level]) {
            bool have_tb = (thin_xforce[0] || thin_yforce[0] || thin_zforce[0]);
            if (have_tb) {
                project_velocity_tb(level, slow_dt, S_sum);
            } else {
                project_momenta(level, slow_dt, S_sum);
            }
        }

        // SK ********************************************************************************
        // if ( ( std::fmod(time_for_fp, 5.0) < 0.5*slow_dt ) &&nrk == 2) {
        if ( nrk == 2) {
            for ( MFIter mfi(S_sum[IntVars::cons],TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const Box bx = mfi.tilebox();
                Box tbx = mfi.nodaltilebox(0);
                Box tby = mfi.nodaltilebox(1);
                Box tbz = mfi.nodaltilebox(2);

                const int ilo = bx.smallEnd(0);
                const int ihi = bx.bigEnd(0);
                const int khi = bx.bigEnd(2);

                const int idom_hi = fine_geom.Domain().bigEnd(0);
                const int jdom_hi = fine_geom.Domain().bigEnd(1);
                const int icen = (idom_hi+1)/2;
                const int jcen = (jdom_hi+1)/2;

                int k110lo = 0;
                int k300lo = 0;
                int k500lo = 0;

                Real z110lo = 0; // low height
                Real z300lo = 0;
                Real z500lo = 0;

                Real z110hi = 0;
                Real z300hi = 0;
                Real z500hi = 0;

                Vector<Real> dz(3, 0.0);

                if (solverChoice.terrain_type == TerrainType::StaticFittedMesh) {
                    if (ilo<icen && icen<ihi) {
                        const Array4<const Real>& z_nd_old = z_phys_nd[level]->const_array(mfi);
                        std::vector<bool> exceed{false, false, false};
                        for (int k=0; k<=khi+1; ++k) {
                            Real z = z_nd_old(icen,jcen,k);
                            if (!exceed[0] && z > 110.0) {
                                k110lo = k-1;
                                z110lo = z_nd_old(icen,jcen,k-1);
                                z110hi = z_nd_old(icen,jcen,k);
                                dz[0] = z110hi - z110lo;
                                exceed[0] = true;
                            }
                            if (!exceed[1] && z > 300.0) {
                                k300lo = k-1;
                                z300lo = z_nd_old(icen,jcen,k-1);
                                z300hi = z_nd_old(icen,jcen,k);
                                dz[1] = z300hi - z300lo;
                                exceed[1] = true;
                            }
                            if (!exceed[2] && z > 500.0) {
                                k500lo = k-1;
                                z500lo = z_nd_old(icen,jcen,k-1);
                                z500hi = z_nd_old(icen,jcen,k);
                                dz[2] = z500hi - z500lo;
                                exceed[2] = true;
                            }
                            // std::cout << "SK: no_substep_fun: k = " << k <<", z = " << z <<", k110lo = " << k110lo <<", k300lo = " << k300lo <<", k500lo = " << k500lo << std::endl;
                        }
                    }

                } else if (solverChoice.terrain_type == TerrainType::EB) {

                    const Real* dx = fine_geom.CellSize();
                    dz.assign(3, dx[2]);

                    k110lo = std::floor(110.0/dz[0] - 0.5); // low index
                    k300lo = std::floor(300.0/dz[1] - 0.5);
                    k500lo = std::floor(500.0/dz[2] - 0.5);

                    z110lo = (0.5+k110lo)*dz[0]; // low height
                    z300lo = (0.5+k300lo)*dz[1];
                    z500lo = (0.5+k500lo)*dz[2];

                    z110hi = z110lo + dz[0]; // high height
                    z300hi = z300lo + dz[1];
                    z500hi = z500lo + dz[2];
                }

                const int k110hi = k110lo + 1; // high index
                const int k300hi = k300lo + 1;
                const int k500hi = k500lo + 1;

                const Real w110hi = (110.0 - z110lo)/dz[0];
                const Real w300hi = (300.0 - z300lo)/dz[1];
                const Real w500hi = (500.0 - z500lo)/dz[2];

                const Real w110lo = 1.0 - w110hi;
                const Real w300lo = 1.0 - w300hi;
                const Real w500lo = 1.0 - w500hi;

                Vector<Array4<Real>> ssum_h(n_data);
                for (int i = 0; i < n_data; ++i) {
                     ssum_h[i] = S_sum[i].array(mfi);
                }
                Gpu::AsyncVector<Array4<Real>> ssum_d(n_data);
                Gpu::copy(Gpu::hostToDevice, ssum_h.begin(), ssum_h.end(), ssum_d.begin());
                Array4<Real>* ssum = ssum_d.dataPtr();

                Gpu::DeviceScalar<Real> u110_d(0.0);
                Gpu::DeviceScalar<Real> u300_d(0.0);
                Gpu::DeviceScalar<Real> u500_d(0.0);
                Real* u110_d_ptr = u110_d.dataPtr();
                Real* u300_d_ptr = u300_d.dataPtr();
                Real* u500_d_ptr = u500_d.dataPtr();

                if (ilo<icen && icen<ihi) {
                    amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE (int /*dummy*/) {

                        Real u110_lo = ssum[IntVars::xmom](icen,jcen,k110lo) * 2.0 / 
                                    ( ssum[IntVars::cons](icen  ,jcen,k110lo,Rho_comp) 
                                    + ssum[IntVars::cons](icen-1,jcen,k110lo,Rho_comp) );
                        Real u110_hi = ssum[IntVars::xmom](icen,jcen,k110hi) * 2.0 /
                                    ( ssum[IntVars::cons](icen  ,jcen,k110hi,Rho_comp) 
                                    + ssum[IntVars::cons](icen-1,jcen,k110hi,Rho_comp) );
                        Real u110 = w110lo * u110_lo + w110hi * u110_hi;

                        Real u300_lo = ssum[IntVars::xmom](icen,jcen,k300lo) * 2.0 /
                                    ( ssum[IntVars::cons](icen  ,jcen,k300lo,Rho_comp) 
                                    + ssum[IntVars::cons](icen-1,jcen,k300lo,Rho_comp) );
                        Real u300_hi = ssum[IntVars::xmom](icen,jcen,k300hi) * 2.0 /
                                    ( ssum[IntVars::cons](icen  ,jcen,k300hi,Rho_comp) 
                                    + ssum[IntVars::cons](icen-1,jcen,k300hi,Rho_comp) );
                        Real u300 = w300lo * u300_lo + w300hi * u300_hi;

                        Real u500_lo = ssum[IntVars::xmom](icen,jcen,k500lo) * 2.0 /
                                    ( ssum[IntVars::cons](icen  ,jcen,k500lo,Rho_comp) 
                                    + ssum[IntVars::cons](icen-1,jcen,k500lo,Rho_comp) );
                        Real u500_hi = ssum[IntVars::xmom](icen,jcen,k500hi) * 2.0 /
                                    ( ssum[IntVars::cons](icen  ,jcen,k500hi,Rho_comp) 
                                    + ssum[IntVars::cons](icen-1,jcen,k500hi,Rho_comp) );
                        Real u500 = w500lo * u500_lo + w500hi * u500_hi;

                        Gpu::Atomic::Add(u110_d_ptr,u110);
                        Gpu::Atomic::Add(u300_d_ptr,u300);
                        Gpu::Atomic::Add(u500_d_ptr,u500);
                    });                        

                    // copy back to host
                    Real u110 = u110_d.dataValue();
                    Real u300 = u300_d.dataValue();
                    Real u500 = u500_d.dataValue();

                    // Gpu::copy(Gpu::deviceToHost, u110_d.dataPtr(), u110_d.dataPtr()+1, &h_total_power);

                    std::string terrain_type_str;
                    switch (solverChoice.terrain_type) {
                        case TerrainType::EB:
                            terrain_type_str = "EB";
                            break;
                        case TerrainType::StaticFittedMesh:
                            terrain_type_str = "FittedMesh";
                            break;
                        default:
                            terrain_type_str = "unknown";
                            break;
                    }

                    std::string filename = "u_values_" + terrain_type_str + ".txt";
                    static bool first_write = true;
                    std::ofstream out;

                    if (first_write) {
                        out.open(filename, std::ios::out); // overwrite (initialize)
                        first_write = false;
                    } else {
                        out.open(filename, std::ios::app); // append mode
                    }
                    out << time_for_fp << ", " << u110 << ", " << u300 << ", " << u500 << std::endl;
                    out.close();

                }
            } // MFIter
        } // nrk
        // SK ********************************************************************************

    };
