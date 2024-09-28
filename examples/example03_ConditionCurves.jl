using Plots, LaTeXStrings
using Plots.PlotMeasures
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using LadderDGA
using JLD2

cfg_file = ARGS[1]
plot_flag = true

wp, mP, sP, env, kGridsStr = readConfig(cfg_file);
lDGAhelper = setup_LDGA(kGridsStr[1], mP, sP, env, silent=true);
bubble     = calc_bubble(:DMFT, lDGAhelper);
χm, γm = calc_χγ(:m, lDGAhelper, bubble);
χd, γd = calc_χγ(:d, lDGAhelper, bubble);
λ₀ = calc_λ0(bubble, lDGAhelper);

#xr_m, fr_m = LadderDGA.LambdaCorrection.PPCond_curve(χm,γm,χd,γd,λ₀, lDGAhelper)
#res_m_ntc      = λm_correction(χm, γm, χd, γd, λ₀, lDGAhelper, tc=:plain)
#res_m      = λm_correction(χm, γm, χd, γd, λ₀, lDGAhelper)
#res_dm     = λdm_correction(χm, γm, χd, γd, λ₀, lDGAhelper)
#res_dm_ntc = λdm_correction(χm, γm, χd, γd, λ₀, lDGAhelper; tc=false)
#res_dm_sc  = λdm_sc_correction(χm, γm, χd, γd, λ₀, lDGAhelper; tc=true)

#xr_dmsc, fr_dmsc = LadderDGA.LambdaCorrection.EPotCond_sc_curve(χm,γm,χd,γd,λ₀, lDGAhelper, maxit=60, λmax=30.0,sc_conv_abs=1e-7, mixing=0.3, verbose=true)
#xr_m, fr_m = LadderDGA.LambdaCorrection.PPCond_curve(χm,γm,χd,γd,λ₀, lDGAhelper, λmax=30.0)
#xr_dm, fr_dm = LadderDGA.LambdaCorrection.EPotCond_curve(χm,γm,χd,γd,λ₀, lDGAhelper; maxit=120, λmax=30.0)
#xr_dmtsc, fr_dmtsc = LadderDGA.LambdaCorrection.EPotCond_sc_curve(χm,γm,χd,γd,λ₀, lDGAhelper; method=:tsc, maxit=20, λmax=30.0,sc_conv_abs=1e-7, mixing=0.3, verbose=true)

χm_sum = sum_kω(lDGAhelper.kG, χm, λ = res_dm.λm)
χd_sum = sum_kω(lDGAhelper.kG, χd, λ = res_dm.λd)
PP_p1  = lDGAhelper.mP.n / 2 * (1 - lDGAhelper.mP.n / 2)
PP_p2  = real(χd_sum + χm_sum) / 2
println(PP_p1, " vs ", PP_p2)
results_E_rhs_c1 = lDGAhelper.mP.n/2*(1-lDGAhelper.mP.n/2)
#results_λm, results_λd, results_n, results_mu, results_E_kin_1, results_E_pot_1, results_E_pot_2, results_E_lhs_c1 = LadderDGA.LambdaCorrection.λdm_correction_curve(χm, γm, χd, γd, λ₀, lDGAhelper; λm_max=30.0, λd_max=30.0, λd_samples=500, tc=false)
xr_lm, fr_lm = LadderDGA.LambdaCorrection.λm_of_λd_curve(χm,γm,χd, γd,λ₀, lDGAhelper, maxit=2000, λmax=30.0)


if plot_flag
    λd_min = LadderDGA.LambdaCorrection.get_λ_min(χd)
    λm_min = LadderDGA.LambdaCorrection.get_λ_min(χm)

    #p0 = heatmap(λd_range, λm_range, PP_diff, xlabel=L"\lambda_\mathrm{d}", ylabel=L"\lambda_\mathrm{m}", clims=(-0.1,0.1),xtickfontsize=12,ytickfontsize=12,xguidefontsize=20,yguidefontsize=20,legendfontsize=12, right_margin=[10mm 0mm])
	#p1 = plot(xr_m, fr_m, ylims=(-1,1), xlims=(-1,5), linewidth=2, xlabel=L"\lambda_\mathrm{m}", label=L"\mathrm{PP}^{(2)} - \mathrm{PP}^{(1)}", title=L"\mathrm{lD}\Gamma\mathrm{A}_\mathrm{m}",xtickfontsize=12,ytickfontsize=12,xguidefontsize=20,yguidefontsize=20,legendfontsize=12)
	#vline!([λm_min], label=L"\lambda_\mathrm{m,min}", linewidth=2.0)

    #label=L"\mathrm{lD}\Gamma\mathrm{A}_{\mathrm{dm}}", 
    p1 = plot(xr_m, fr_m, linewidth=1.5, ylims=(-1.0,1.0), xlims=(λm_min - 0.01, 0), label=L"\mathrm{PP}^{(1)}- \mathrm{PP}^{(2)}", xlabel=L"\lambda_\mathrm{m}",xtickfontsize=12,ytickfontsize=12,xguidefontsize=20,yguidefontsize=20,legendfontsize=12, left_margin=[5mm 0mm])
    vline!(p1, [λm_min], label=L"\lambda_\mathrm{m,min}", linewidth=2.0, linestyle=:dash)
    hline!(p1, [0.0], label=L"\mathrm{solution}", linewidth=2.0, linestyle=:dash)
    savefig(p1, "PP_curve.pdf")

    p2 = plot(xr_lm, fr_lm, linewidth=1.5, ylims=(-0.1,0.1), xlabel=L"\lambda_\mathrm{d}", ylabel=L"\lambda_\mathrm{m}",xtickfontsize=12,ytickfontsize=12,xguidefontsize=20,yguidefontsize=20,legendfontsize=12, left_margin=[5mm 0mm])
    vline!(p2, [λd_min], label=L"\lambda_\mathrm{d,min}", linewidth=2.0, linestyle=:dash)
    savefig(p1, "lm_of_ld.pdf")

    p3 = plot(xr_dm, fr_dm, linewidth=1.5, ylims=(-0.5,0.2), label=L"\mathrm{lD}\Gamma\mathrm{A}_{\mathrm{dm}}", ylabel=L"E^{(1)}_\mathrm{pot} - E^{(2)}_\mathrm{pot}", xlabel=L"\lambda_\mathrm{d}",xtickfontsize=12,ytickfontsize=12,xguidefontsize=20,yguidefontsize=20,legendfontsize=12, left_margin=[5mm 0mm])
    plot!(p3, xr_dmsc, fr_dmsc, linewidth=1.5, label=L"\mathrm{lD}\Gamma\mathrm{A}_{\mathrm{dm,sc}}")
    vline!(p3, [λd_min], label=L"\lambda_\mathrm{d,min}", linewidth=2.0, linestyle=:dash)
    hline!(p3, [0.0], label=L"\mathrm{solution}", linewidth=2.0, linestyle=:dash)
    savefig(p3, "EPotDiffCurve.pdf")
end