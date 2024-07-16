# ==================== Includes ====================
#nprocs() == 1 && addprocs(8)
using Pkg
path = joinpath(abspath(@__DIR__),"..")
println("activating: ", path)
Pkg.activate(path)
using LadderDGA
using Plots

include("helper_functions.jl")

cfg_test_01 = "/home/julisn/Hamburg/ED_data/tsc_test/U2.0/U2.0_b20.0_mu1.0.toml" #joinpath(@__DIR__, "../test/test_data/config_b1u2.toml")
cfg_test_02 = "/home/julisn/Hamburg/ED_data/tsc_test/U2.0/AlDGA_U2.0_b20.0_mu1.0.toml"#joinpath(@__DIR__, "../test/test_data/config_AlDGA_example.toml")

wp, mP, sP, env, kGridsStr = readConfig(cfg_test_01);
lDGAhelper = setup_LDGA(kGridsStr[1], mP, sP, env);

# ====================== lDGA ======================
bubble     = calc_bubble(:DMFT, lDGAhelper);
χm, γm = calc_χγ(:m, lDGAhelper, bubble);
χd, γd = calc_χγ(:d, lDGAhelper, bubble);
λ₀ = calc_λ0(bubble, lDGAhelper)

# ===================== AlDGA ======================
# AlDGAhelper_01, χm_01, γm_01, χd_01, γd_01, G_ladder_01, Σ_ladder_01, converged_01, it_01, λm = run_AlDGA_convergence(cfg_test_01; eps=1e-12, maxit=100)
AlDGAhelper_02, χm_02, γm_02, χd_02, γd_02, G_ladder_02, Σ_ladder_02, converged_02, it_02, λm = run_AlDGA_convergence(cfg_test_02; eps=1e-12, maxit=100)


# ==================== Results =====================
res_m      = λm_correction(χm, γm, χd, γd, λ₀, lDGAhelper); print(res_m)
res_dm     = λdm_correction(χm, γm, χd, γd, λ₀, lDGAhelper); print(res_dm)
res_m_sc   = λm_sc_correction(χm, γm, χd, γd, λ₀, lDGAhelper); print(res_m_sc)
res_dm_sc  = λdm_sc_correction(χm, γm, χd, γd, λ₀, lDGAhelper; validation_threshold=1e-6); print(res_dm_sc)
res_m_tsc  = λm_tsc_correction(χm, γm, χd, γd, λ₀, lDGAhelper; max_steps_sc = 200, validation_threshold=1e-6); print(res_m_tsc)
res_dm_tsc = λdm_tsc_correction(χm, γm, χd, γd, λ₀, lDGAhelper; max_steps_sc = 200, validation_threshold=1e-6); print(res_dm_tsc)

p_sigma = plot(- ((2 .* 0:(size(Σ_ladder_02,2)-2) .+ 1) .* π ./ AlDGAhelper_02.mP.β ) .* imag(kintegrate(AlDGAhelper_02.kG, Σ_ladder_02,1)[1,:]).parent)

xr = -sP.n_iω:sP.n_iω
p = plot(xr,real(kintegrate(AlDGAhelper_01.kG, χm_01, 1)[1,:]), ylabel="χ_m", xlims=(-10,10), markershape=:auto, label="AsymptlDGA", markersize=10)
xr2 = -AlDGAhelper_02.sP.n_iω:AlDGAhelper_02.sP.n_iω
plot!(xr2,real(kintegrate(AlDGAhelper_02.kG, χm_02, 1)[1,:]), markershape=:auto, label="AsymptlDGA (RPA start)", markersize=8)
plot!(xr,real(kintegrate(lDGAhelper.kG, χm, 1)[1,:]), markershape=:auto, label="DMFT", markersize=6)
χm_pl_val = chi_loc(χm, res_m_tsc, res_m_tsc.λm)
χm_pl_val2 = chi_loc(χm, res_dm_tsc, res_dm_tsc.λm)
plot!(xr,χm_pl_val, markershape=:auto, label="lDGA^tsc_m")
plot!(xr,χm_pl_val2, markershape=:auto, label="lDGA^tsc_dm")
