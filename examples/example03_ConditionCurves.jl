using Plots, LaTeXStrings
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using LadderDGA
using JLD2

cfg_file = ARGS[1]

wp, mP, sP, env, kGridsStr = readConfig(cfg_file);
lDGAhelper = setup_LDGA(kGridsStr[1], mP, sP, env, silent=true);
bubble     = calc_bubble(:DMFT, lDGAhelper);
χm, γm = calc_χγ(:m, lDGAhelper, bubble);
χd, γd = calc_χγ(:d, lDGAhelper, bubble);
λ₀ = calc_λ0(bubble, lDGAhelper);

#xr_m, fr_m = LadderDGA.LambdaCorrection.PPCond_curve(χm,γm,χd,γd,λ₀, lDGAhelper)
res_m_ntc      = λm_correction(χm, γm, χd, γd, λ₀, lDGAhelper, tc=false)
res_m      = λm_correction(χm, γm, χd, γd, λ₀, lDGAhelper, tc=true)
res_dm     = λdm_correction(χm, γm, χd, γd, λ₀, lDGAhelper; tc=true)
#res_dm_ntc = λdm_correction(χm, γm, χd, γd, λ₀, lDGAhelper; tc=false)


#xr_m, fr_m = LadderDGA.LambdaCorrection.PPCond_curve(χm,γm,χd,γd,λ₀, lDGAhelper)
#xr_dm, fr_dm = LadderDGA.LambdaCorrection.EPotCond_curve(χm,γm,χd,γd,λ₀, lDGAhelper; maxit=50)
se = calc_Σ(χm, γm, χd, γd, λ₀, lDGAhelper, tc=true)
se_ntc = calc_Σ(χm, γm, χd, γd, λ₀, lDGAhelper, tc=false)


#=χm_sum = sum_kω(lDGAhelper.kG, χm, λ = res_dm.λm)
χd_sum = sum_kω(lDGAhelper.kG, χd, λ = res_dm.λd)
PP_p1  = lDGAhelper.mP.n / 2 * (1 - lDGAhelper.mP.n / 2)
PP_p2  = real(χd_sum + χm_sum) / 2
println(PP_p1, " vs ", PP_p2)
results_E_rhs_c1 = lDGAhelper.mP.n/2*(1-lDGAhelper.mP.n/2)
results_λm, results_λd, results_n, results_mu, results_E_kin_1, results_E_pot_1, results_E_pot_2, results_E_lhs_c1 = LadderDGA.LambdaCorrection.λdm_correction_curve(χm, γm, χd, γd, λ₀, lDGAhelper; λm_max=30.0, λd_max=30.0, λd_samples=500, tc=false)
res_dm = LadderDGA.LambdaCorrection.λdm_correction(χm, γm, χd, γd, λ₀, lDGAhelper; νmax=100,fit_μ=true, verbose=true, tc=false);
=#