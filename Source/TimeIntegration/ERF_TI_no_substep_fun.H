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
        if ( ( std::fmod(time_for_fp, 0.001) < slow_dt ) &&nrk == 2) {
        // if ( nrk == 2) {
            for ( MFIter mfi(S_sum[IntVars::cons],TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const Box bx = mfi.tilebox();
                Box tbx = mfi.nodaltilebox(0);
                Box tby = mfi.nodaltilebox(1);
                Box tbz = mfi.nodaltilebox(2);

                const int ilo = bx.smallEnd(0);
                const int ihi = bx.bigEnd(0);
                const int jlo = bx.smallEnd(1);
                const int jhi = bx.bigEnd(1);
                const int khi = bx.bigEnd(2);

                const int idom_hi = fine_geom.Domain().bigEnd(0);
                const int jdom_hi = fine_geom.Domain().bigEnd(1);
                const int icen = (idom_hi+1)/2;
                const int jcen = (jdom_hi+1)/2;

                int kh1lo = 0;
                int kh2lo = 0;
                int kh3lo = 0;

                Real zh1lo = 0; // low height
                Real zh2lo = 0;
                Real zh3lo = 0;

                // Vector<Real> dz(3, 0.0);
                Real dz[3] = {0.0,0.0,0.0};

                if (solverChoice.terrain_type == TerrainType::StaticFittedMesh) {
                    if (ilo<=icen && icen<ihi && jlo<=jcen && jcen<jhi) {
                        const Array4<const Real>& z_nd_old = z_phys_nd[level]->const_array(mfi);
#ifdef AMREX_USE_GPU
                        amrex::Gpu::DeviceScalar<int> d_kh1(0), d_kh2(0), d_kh3(0);
                        amrex::Gpu::DeviceScalar<Real> d_zh1(0.0), d_zh2(0.0), d_zh3(0.0);
                        int* k_lo_ptr[3] = {d_kh1.dataPtr(), d_kh2.dataPtr(), d_kh3.dataPtr()};
                        Real* z_lo_ptr[3] = {d_zh1.dataPtr(), d_zh2.dataPtr(), d_zh3.dataPtr()};

                        amrex::Gpu::DeviceScalar<Real> d_dz[3] = {amrex::Gpu::DeviceScalar<Real>(0.0),
                                                                  amrex::Gpu::DeviceScalar<Real>(0.0),
                                                                  amrex::Gpu::DeviceScalar<Real>(0.0)};
                        Real* dz_ptr[3] = {d_dz[0].dataPtr(), d_dz[1].dataPtr(), d_dz[2].dataPtr()};

                        amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE (int /*dummy*/) {
                            bool exceed0 = false;
                            bool exceed1 = false;
                            bool exceed2 = false;
                            for (int k=0; k<=khi+1; ++k) {
                                Real z = z_nd_old(icen,jcen,k);
                                if (!exceed0 && z > 0.55) {
                                    *k_lo_ptr[0] = k-1;
                                    *z_lo_ptr[0] = z_nd_old(icen,jcen,k-1);
                                    *dz_ptr[0] = z_nd_old(icen,jcen,k) - z_nd_old(icen,jcen,k-1);
                                    exceed0 = true;
                                }
                                if (!exceed1 && z > 5.0) {
                                    *k_lo_ptr[1] = k-1;
                                    *z_lo_ptr[1] = z_nd_old(icen,jcen,k-1);
                                    *dz_ptr[1] = z_nd_old(icen,jcen,k) - z_nd_old(icen,jcen,k-1);
                                    exceed1 = true;
                                }
                                if (!exceed2 && z > 9.0) {
                                    *k_lo_ptr[2] = k-1;
                                    *z_lo_ptr[2] = z_nd_old(icen,jcen,k-1);
                                    *dz_ptr[2] = z_nd_old(icen,jcen,k) - z_nd_old(icen,jcen,k-1);
                                    exceed2 = true;
                                }
                            }
                        });
                        kh1lo = d_kh1.dataValue(); zh1lo = d_zh1.dataValue();
                        kh2lo = d_kh2.dataValue(); zh2lo = d_zh2.dataValue();
                        kh3lo = d_kh3.dataValue(); zh3lo = d_zh3.dataValue();

                        dz[0] = d_dz[0].dataValue();
                        dz[1] = d_dz[1].dataValue();
                        dz[2] = d_dz[2].dataValue();
#else
                        // CPU
                        bool e0=false, e1=false, e2=false;
                        for (int k=0;k<=khi+1;++k) {
                            Real z = z_nd_old(icen,jcen,k);
                            if(!e0 && z>0.55) { kh1lo=k-1; zh1lo=z_nd_old(icen,jcen,k-1); e0=true; }
                            if(!e1 && z>5.0) { kh2lo=k-1; zh2lo=z_nd_old(icen,jcen,k-1); e1=true; }
                            if(!e2 && z>9.0) { kh3lo=k-1; zh3lo=z_nd_old(icen,jcen,k-1); e2=true; }
                        }
                        dz[0] = z_nd_old(icen,jcen,kh1lo+1) - zh1lo;
                        dz[1] = z_nd_old(icen,jcen,kh2lo+1) - zh2lo;
                        dz[2] = z_nd_old(icen,jcen,kh3lo+1) - zh3lo;
#endif
                    }

                } else if (solverChoice.terrain_type == TerrainType::EB) {

                    const Real* dx = fine_geom.CellSize();
                    // dz.assign(3, dx[2]);
                    dz[0] = dx[2];
                    dz[1] = dx[2];
                    dz[2] = dx[2];

                    kh1lo = std::floor(0.55/dz[0] - 0.5); // low index
                    kh2lo = std::floor(5.0/dz[1] - 0.5);
                    kh3lo = std::floor(9.0/dz[2] - 0.5);

                    zh1lo = (0.5+kh1lo)*dz[0]; // low height
                    zh2lo = (0.5+kh2lo)*dz[1];
                    zh3lo = (0.5+kh3lo)*dz[2];
                }

                const int kh1hi = kh1lo + 1; // high index
                const int kh2hi = kh2lo + 1;
                const int kh3hi = kh3lo + 1;

                const Real wh1hi = (0.55 - zh1lo)/dz[0];
                const Real wh2hi = (5.0 - zh2lo)/dz[1];
                const Real wh3hi = (9.0 - zh3lo)/dz[2];

                const Real wh1lo = 1.0 - wh1hi;
                const Real wh2lo = 1.0 - wh2hi;
                const Real wh3lo = 1.0 - wh3hi;

                Vector<Array4<Real>> ssum_h(n_data);
                for (int i = 0; i < n_data; ++i) {
                     ssum_h[i] = S_sum[i].array(mfi);
                }
                Gpu::AsyncVector<Array4<Real>> ssum_d(n_data);
                Gpu::copy(Gpu::hostToDevice, ssum_h.begin(), ssum_h.end(), ssum_d.begin());
                Array4<Real>* ssum = ssum_d.dataPtr();

                Gpu::DeviceScalar<Real> uh1_d(0.0);
                Gpu::DeviceScalar<Real> uh2_d(0.0);
                Gpu::DeviceScalar<Real> uh3_d(0.0);
                Real* uh1_d_ptr = uh1_d.dataPtr();
                Real* uh2_d_ptr = uh2_d.dataPtr();
                Real* uh3_d_ptr = uh3_d.dataPtr();

                if (ilo<=icen && icen<ihi && jlo<=jcen && jcen<jhi) {
                    amrex::ParallelFor(1, [=] AMREX_GPU_DEVICE (int /*dummy*/) {

                        Real uh1_lo = ssum[IntVars::xmom](icen,jcen,kh1lo) * 2.0 / 
                                    ( ssum[IntVars::cons](icen  ,jcen,kh1lo,Rho_comp) 
                                    + ssum[IntVars::cons](icen-1,jcen,kh1lo,Rho_comp) );
                        Real uh1_hi = ssum[IntVars::xmom](icen,jcen,kh1hi) * 2.0 /
                                    ( ssum[IntVars::cons](icen  ,jcen,kh1hi,Rho_comp) 
                                    + ssum[IntVars::cons](icen-1,jcen,kh1hi,Rho_comp) );
                        Real uh1 = wh1lo * uh1_lo + wh1hi * uh1_hi;

                        Real uh2_lo = ssum[IntVars::xmom](icen,jcen,kh2lo) * 2.0 /
                                    ( ssum[IntVars::cons](icen  ,jcen,kh2lo,Rho_comp) 
                                    + ssum[IntVars::cons](icen-1,jcen,kh2lo,Rho_comp) );
                        Real uh2_hi = ssum[IntVars::xmom](icen,jcen,kh2hi) * 2.0 /
                                    ( ssum[IntVars::cons](icen  ,jcen,kh2hi,Rho_comp) 
                                    + ssum[IntVars::cons](icen-1,jcen,kh2hi,Rho_comp) );
                        Real uh2 = wh2lo * uh2_lo + wh2hi * uh2_hi;

                        Real uh3_lo = ssum[IntVars::xmom](icen,jcen,kh3lo) * 2.0 /
                                    ( ssum[IntVars::cons](icen  ,jcen,kh3lo,Rho_comp) 
                                    + ssum[IntVars::cons](icen-1,jcen,kh3lo,Rho_comp) );
                        Real uh3_hi = ssum[IntVars::xmom](icen,jcen,kh3hi) * 2.0 /
                                    ( ssum[IntVars::cons](icen  ,jcen,kh3hi,Rho_comp) 
                                    + ssum[IntVars::cons](icen-1,jcen,kh3hi,Rho_comp) );
                        Real uh3 = wh3lo * uh3_lo + wh3hi * uh3_hi;

                        Gpu::Atomic::Add(uh1_d_ptr,uh1);
                        Gpu::Atomic::Add(uh2_d_ptr,uh2);
                        Gpu::Atomic::Add(uh3_d_ptr,uh3);
                    });                        

                    // copy back to host
                    Real uh1 = uh1_d.dataValue();
                    Real uh2 = uh2_d.dataValue();
                    Real uh3 = uh3_d.dataValue();

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
                    out << time_for_fp << ", " << uh1 << ", " << uh2 << ", " << uh3 << std::endl;
                    out.close();

                }
            } // MFIter
        } // nrk
        // SK ********************************************************************************

    };
