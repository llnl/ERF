#include "prob.H"

using namespace amrex;

std::unique_ptr<ProblemBase>
amrex_probinit(const amrex_real* problo, const amrex_real* probhi)
{
    return std::make_unique<Problem>(problo, probhi);
}

Problem::Problem(const amrex::Real* /*problo*/, const amrex::Real* /*probhi*/)
{
    // Parse params
    ParmParse pp("prob");
    pp.query("SLM_num_ref_inputs", parms.SLM_num_ref_inputs);
    pp.query("SLM_ref_sounding_file", parms.SLM_ref_sounding_file);
    pp.query("SLM_ref_flux_file", parms.SLM_ref_flux_file);
    pp.query("SLM_ref_sst_file", parms.SLM_ref_sst_file);

    init_base_parms(parms.rho_0, parms.T_0);
}

void Problem::read_SLM_inputs()
{
    // Reads the SLM input sounding file assuming the following fields:
    //  t[day], pres0[mb], tabs[K], q[g/kg], uvel[m/s], vvel[m/s]
    sounding = read_cols(parms.SLM_ref_sounding_file, 1);

    // Reads the SLM input flux file assuming the following fields:
    //  t[day], swdn[W/m2], lwdn[W/m2], swup[W/m2], lwup[W/m2]
    fluxes = read_cols(parms.SLM_ref_flux_file, 1);

    // Reads the SLM input SST file assuming the following fields:
    // t[day], sst[K], precip[mm/s]
    sst = read_cols(parms.SLM_ref_sst_file, 1);
}

/**
 * Read columns of data from a file, returning each column in a vector.
 */
std::vector<std::vector<amrex::Real>> Problem::read_cols(const std::string &fname, const int skip_nlines)
{
    std::ifstream ifs(fname);
    if (!ifs.is_open())
    {
        amrex::Error("Error opening input file " + fname);
    }

    std::vector<std::vector<amrex::Real>> datasets;
    std::string line;
    int i = 0;
    int ncols = -1;

    while (std::getline(ifs, line))
    {
        i++;
        if (i <= skip_nlines) continue;

        std::istringstream iss(line);

        amrex::Real tmp;
        // Get the number of columns in the file
        if (ncols == -1) {
            int j = 0;
            while (iss >> tmp) {
                datasets.push_back(std::vector<amrex::Real>());
                j+= 1;
            }

            ncols = j;
            iss = std::istringstream(line);

            amrex::Print() << "-> got " << std::to_string(j) << " columns\n";
        }

        int j = 0;
        while (iss >> tmp) {
            // verify each line has the same number of columns
            if (j > ncols) {
              amrex::Error(
                "Error reading file '" + fname + "': expected line " +
                std::to_string(i) + " to have " + std::to_string(ncols) +
                " columns, but got " + std::to_string(j));
            }
            datasets[j].push_back(tmp);
            j+= 1;
        }
    }

    ifs.close();

    return datasets;
}

void
Problem::init_custom_pert(
    const Box& bx,
    const Box& xbx,
    const Box& ybx,
    const Box& zbx,
    Array4<Real      > const& state,
    Array4<Real      > const& x_vel,
    Array4<Real      > const& y_vel,
    Array4<Real      > const& z_vel,
    Array4<Real      > const& r_hse,
    Array4<Real      > const& p_hse,
    Array4<Real const> const& z_nd,
    Array4<Real const> const& z_cc,
    GeometryData const& geomdata,
    Array4<Real const> const& /*mf_m*/,
    Array4<Real const> const& /*mf_u*/,
    Array4<Real const> const& /*mf_v*/,
    const SolverChoice& sc)
{
    const bool use_moisture = (sc.moisture_type != MoistureType::None);

    read_SLM_inputs();

    ParallelFor(bx, [=, parms=parms] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        // Geometry
        const Real* prob_lo = geomdata.ProbLo();
        const Real* prob_hi = geomdata.ProbHi();
        const Real* dx = geomdata.CellSize();
        const Real x = prob_lo[0] + (i + 0.5) * dx[0];
        const Real y = prob_lo[1] + (j + 0.5) * dx[1];
        const Real z = prob_lo[2] + (k + 0.5) * dx[2];

        const Real qv = sounding[3][0] / 1000.0; // qv converted to kg/kg
        const Real pres = sounding[1][0] * 100.0; // pressure converted from mbar to Pa
        const Real theta = getThgivenPandT(sounding[2][0], pres, R_d / Cp_d);
        const Real rho = getRhogivenThetaPress(theta, pres, R_d / Cp_d, qv);

        state(i, j, k, Rho_comp) = rho;
        state(i, j, k, RhoTheta_comp) = rho * theta;
        state(i, j, k, RhoScalar_comp) = 0.0;

        state(i, j, k, RhoKE_comp) = 0.0;
        state(i, j, k, RhoQKE_comp) = 0.0;

        if (use_moisture) {
            state(i, j, k, RhoQ1_comp) = qv;
            state(i, j, k, RhoQ2_comp) = 0.0;
        }
    });

    amrex::ParallelFor(xbx, [=, parms=parms] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        const auto *const prob_hi  = geomdata.ProbHi();
        const auto *const dx       = geomdata.CellSize();
        const Real z = (k + 0.5) * dx[2];
        x_vel(i, j, k) = sounding[4][0];
    });

    amrex::ParallelFor(ybx, [=, parms=parms] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        const auto *const prob_hi  = geomdata.ProbHi();
        const auto *const dx       = geomdata.CellSize();
        const Real z = (k + 0.5) * dx[2];
        y_vel(i, j, k) = sounding[5][0];
    });

    amrex::ParallelFor(zbx, [=, parms=parms] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
        z_vel(i, j, k) = parms.w_0;
    });

}