#include "ERF_SLM.H"

#include "ERF_NCInterface.H"

using namespace amrex;

#ifdef ERF_USE_NETCDF
void
SLM::writeSLM_NetCDF(const MultiFab& mf, const Vector<std::string>& varnames, const amrex::Real time,
                     const std::string plot_prefix, const int level_step)
{
    std::string plotfilename = plot_prefix + ".nc";

    ncutils::NCFile ncf =
        (first_step)
        ? ncutils::NCFile::create_par(plotfilename, NC_CLOBBER | NC_NETCDF4)
        : ncutils::NCFile::open_par(plotfilename, NC_WRITE);

    if (first_step)
    {
        writeNCHeader(ncf);

        // Write out 3D variables that are constant in time
        for (const int var : const_vars)
        {
            amrex::Print() << " Writing CONSTANT 3D SLM MF '" << LsmVarName_Full[var] << "'" << std::endl;
            writeMFtoNC(ncf, lsm_fab_vars[var].get(), LsmVarName_Full[var], -1.0);
        }

        // Write out 2D variables that are constant in time
        writeMFtoNC(ncf, &LAI, "LAI", -1.0);
        writeMFtoNC(ncf, &BAI, "BAI", -1.0);
        writeMFtoNC(ncf, &ztop, "ztop", -1.0);
    }

    // Write out SLM 3D fields
    for (int var = 0; var < LsmVar_SLM::NumVars; ++var) {
        if (const_vars.find(var) != const_vars.end()) continue;
        //writeMFtoNC(ncf, lsm_fab_vars[var].get(), LsmVarName_Full[var], time, true); // write ghost cells
        writeMFtoNC(ncf, lsm_fab_vars[var].get(), LsmVarName_Full[var], time, false); // write ghost cells
    }

    // Write out SLM 2D fields
    // TODO: fix this
    //  2D fields are combined into the single mf with varnames components
    AMREX_ALWAYS_ASSERT(mf.nComp() == varnames.size());
    IntVect ng(0, 0, 0);
    for (int var = 0; var < mf.nComp(); ++var)
    {
        MultiFab fab(mf.boxArray(), mf.DistributionMap(), 1, ng);
        MultiFab::Copy(fab, mf, var, 0, 1, 0);
        writeMFtoNC(ncf, &fab, varnames[var], time);
    }

    writeMFtoNC(ncf, &cp_vege, "cp_vege", time);
    writeMFtoNC(ncf, &z0_sfc, "z0_sfc", time);

    //writeMFtoNC(ncf, &Khai_L, "Khai_L", -1.0);

    //writeMFtoNC(ncf, &ustar, "ustar", time); // ustar and tstar already saved in 2D fab array above

    // additional slm outputs
    for (int var = 0; var < slm_diag.nComp(); ++var)
    {
        MultiFab fab(slm_diag.boxArray(), slm_diag.DistributionMap(), 1, ng);
        MultiFab::Copy(fab, slm_diag, var, 0, 1, 0);
        writeMFtoNC(ncf, &fab, diag_names[var], time);
    }

    ncf.close();
}

void
SLM::writeNCHeader(ncutils::NCFile &nc_file)
{
    // TODO
}

void SLM::writeMFtoNC(ncutils::NCFile &nc_file, const MultiFab* mf,
                      const std::string name, Real time, bool write_ghost)
{
    IntVect ngrow = mf->nGrowVect();
    int num_times;

    if (!nc_file.has_var(name))
    {
        nc_file.enter_def_mode();

        // create time dimension in file
        // TODO: fix this - move time dim to NCInterface
        if (time > -1.0 && !nc_file.has_dim("time"))
        {
            nc_file.def_dim("time", NC_UNLIMITED);
            nc_file.def_var("time", ncutils::NCDType::Real, {"time"});
        }

        IntVect box_size = mf->boxArray().minimalBox().size();
        if (write_ghost)
        {
            box_size += 2 * ngrow;
        }

        bool add_dims = true;
        // Check if file has x and y dimensions
        if (nc_file.has_dim("x") && nc_file.has_dim("y"))
        {
            // Check that the dimension bounds match
            size_t x_size = nc_file.dim("x").len();
            size_t y_size = nc_file.dim("y").len();

            if (x_size == box_size[0] && y_size == box_size[1])
            {
                // reuse these dimensions for this dataset
                add_dims = false;
            }
        } else {
            // if the x and y dimensions don't exist, create them
            nc_file.def_dim("x", box_size[0]);
            nc_file.def_dim("y", box_size[1]);
            nc_file.def_dim("z", box_size[2]);
            add_dims = false;
        }

        std::vector<std::string> dim_names;
        if (add_dims) {
            dim_names = {name + "_z", name + "_y", name + "_x"}; // col major
            nc_file.def_dim(name + "_x", box_size[0]);
            nc_file.def_dim(name + "_y", box_size[1]);
            nc_file.def_dim(name + "_z", box_size[2]);
        } else {
            dim_names = {"z", "y", "x"}; // col major
        }

        // add a dimension for MF components
        if (mf->nComp() > 1)
        {
            nc_file.def_dim(name + "_nComp", mf->nComp());
            dim_names.emplace(dim_names.begin(), name + "_nComp");
        }

        if (time > -1.0)
        {
            dim_names.emplace(dim_names.begin(), "time");
        }

        nc_file.def_var(name, ncutils::NCDType::Real, dim_names);
        nc_file.exit_def_mode();

        if (write_ghost)
        {
            nc_file.var(name).put_attr("growVect", std::vector<int>(ngrow.begin(), ngrow.end()));
        }
    }

    // Find the current time index in the file
    // This is needed to avoid the time dimension from growing each time a MF is written to file
    // TODO: fix this - move time dim to NCInterface
    int time_index = -1;
    if (time > -1.0)
    {
        size_t time_dim_len = nc_file.dim("time").len();
        size_t time_var_len = nc_file.var("time").shape()[0];

        Real last_time = -1.0;
        if (time_dim_len > 0)
        {
            std::vector<size_t> start = {time_dim_len - 1};
            std::vector<size_t> count = {1};
            nc_file.var("time").get(&last_time, start, count);
        }

        AMREX_ALWAYS_ASSERT(time_dim_len == time_var_len);

        if (last_time != time)
        {
            nc_file.var("time").par_access(NC_COLLECTIVE);
            nc_file.var("time").put(&time, {static_cast<size_t>(std::max(0, static_cast<int>(time_var_len)))}, {1});

            time_dim_len = nc_file.dim("time").len();
            time_var_len = nc_file.var("time").shape()[0];
        }
        time_index = std::max(0, static_cast<int>(time_dim_len - 1));
    }

    const MultiFab *data = mf;
    std::unique_ptr<MultiFab> mf_tmp;
    if (!write_ghost && ngrow != 0)
    {
        mf_tmp = std::make_unique<MultiFab>(mf->boxArray(), mf->DistributionMap(), mf->nComp(), 0, MFInfo(), mf->Factory());
        MultiFab::Copy(*mf_tmp, *mf, 0, 0, mf->nComp(), 0);
        data = mf_tmp.get();
    }

    // TODO: fix - this is required since file opened each time step, so the variable access prop is reset
    nc_file.var(name).par_access(NC_COLLECTIVE);

    amrex::Arena* Arena_Used = amrex::The_Arena();
#ifdef AMREX_USE_GPU
    Arena_Used = amrex::The_Pinned_Arena();
#endif

    for(MFIter mfi(*data); mfi.isValid(); ++mfi)
    {
        const auto &box = mfi.fabbox();
        FArrayBox tmp(box,  data->nComp(), Arena_Used);
        for(int comp = 0; comp < data->nComp(); ++comp)
        {
            //const auto *dataPtr = data->get(mfi).dataPtr(comp);
            const auto *dataPtr = tmp.dataPtr(comp);
            AMREX_ALWAYS_ASSERT(dataPtr != nullptr);

            // TODO: this assumes z always start at 0.. handle box.smallEnd()[2] better for 2D/3D MFs
            std::vector<size_t> starts = {0,
                                          static_cast<unsigned long>(box.smallEnd()[1]),
                                          static_cast<unsigned long>(box.smallEnd()[0])};
            std::vector<size_t> counts = {static_cast<unsigned long>(box.length()[2]),
                                          static_cast<unsigned long>(box.length()[1]),
                                          static_cast<unsigned long>(box.length()[0])};

            if (data->nComp() > 1)
            {
                starts.emplace(starts.begin(), comp);
                counts.emplace(counts.begin(), 1);
            }

            if (time > -1.0)
            {
                starts.emplace(starts.begin(), time_index);
                counts.emplace(counts.begin(), 1);
            }

            auto mf_arr = data->array(mfi, comp);
            auto tmp_arr = tmp.array(comp);
            ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                tmp_arr(i, j, k) = mf_arr(i, j, k);
            });

            nc_file.var(name).put(dataPtr, starts, counts);
        }
    }
}

void
SLM::writeMFVecToNC(ncutils::NCFile &nc_file,
                    const Vector<const MultiFab*> &mf_vec,
                    const Vector<std::string> &mf_names,
                    const Geometry& geom) const
{
    // Helper function to write a vector of MF to a NetCDF file
    // it is assumed that all MFs in mf_vec share geom
}
#endif
