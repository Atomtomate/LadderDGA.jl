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
h = lDGAhelper;
#=
xr_msc, fr_dmsc = LadderDGA.LambdaCorrection.EPotCond_sc_curve(χm,γm,χd,γd,λ₀, lDGAhelper, maxit=60, sc_conv_abs=1e-7, mixing=0.3, verbose=true)



ltest_01 = 10.184341906626997
χ_λ!(χd, ltest_01)
rhs,PP_p1 = LadderDGA.LambdaCorrection.λm_rhs(χm, χd, h)
mtest_01  = LadderDGA.LambdaCorrection.λm_correction_val(χm, rhs, h)
reset!(χd)

conv_01, μ_it_01, G_ladder_it_01, Σ_ladder_it_01, tr_01 = run_sc(χm, γm, χd, γd, λ₀, mtest_01, ltest_01, h; mixing=0.3, trace=true, maxit=2000) 
Epot_2_01 = EPot_p2(χm, χd, mtest_01, ltest_01, h.mP.n, h.mP.U, h.kG)
Ekin_1_01, Epot_1_01 = calc_E(G_ladder_it_01, Σ_ladder_it_01, μ_it_01, h.kG, h.mP)
println(Epot_1_01 - Epot_2_01)

ltest_02 = 10.803581222044905
χ_λ!(χd, ltest_02)
rhs,PP_p1 = LadderDGA.LambdaCorrection.λm_rhs(χm, χd, h)
mtest_02  = LadderDGA.LambdaCorrection.λm_correction_val(χm, rhs, h)
reset!(χd)

conv_02, μ_it_02, G_ladder_it_02, Σ_ladder_it_02, tr_02 = run_sc(χm, γm, χd, γd, λ₀, mtest_02, ltest_02, h; mixing=0.3, trace=true, maxit=2000) 
Epot_2_02 = EPot_p2(χm, χd, mtest_02, ltest_02, h.mP.n, h.mP.U, h.kG)
Ekin_1_02, Epot_1_02 = calc_E(G_ladder_it_02, Σ_ladder_it_02, μ_it_02, h.kG, h.mP)
println(Epot_1_02 - Epot_2_02)

#running λm=0.09234749091504138, λd=10.803581222044905
#-> converged = true, ΔEPot = 0.011088248675441559 (0.10113594679345783 - 0.09004769811801627). μ = 0.9999074426768697

#running λm=0.09244332736112838, λd=10.184341906626997
#-> converged = true, ΔEPot = -0.003352143113469125 (0.08800046950187854 - 0.09135261261534766). μ = 0.9694525504747822
=#

#conv_tsc_02, λm_tsc, λd_tsc, μ_it_tsc_02, G_ladder_it_tsc_02, Σ_ladder_it_tsc_02, tr_tsc_02 = LadderDGA.LambdaCorrection.run_tsc(χm, γm, χd, γd, λ₀, h; mixing=0.3, trace=true, maxit=150, tc=true) 
#Epot_2_tsc_02 = EPot_p2(χm, χd, λm_tsc, λd_tsc, h.mP.n, h.mP.U, h.kG)
#Ekin_1_tsc_02, Epot_1_tsc_02 = calc_E(G_ladder_it_tsc_02, Σ_ladder_it_tsc_02, μ_it_tsc_02, h.kG, h.mP)
#println(Epot_1_tsc_02 - Epot_2_tsc_02)

using OffsetArrays
h = lDGAhelper;
ωn2_tail = LadderDGA.ω2_tail(χm)
Nq, Nω = size(χm)
fft_νGrid= h.sP.fft_range

Kνωq_pre    = Vector{ComplexF64}(undef, length(h.kG.kMult))
G_ladder_it = OffsetArray(Matrix{ComplexF64}(undef, Nq, length(fft_νGrid)), 1:Nq, fft_νGrid) 
G_ladder_bak = similar(G_ladder_it)
Σ_ladder_it = OffsetArray(Matrix{ComplexF64}(undef, Nq, 100), 1:Nq, 0:100-1)
iν = LadderDGA.iν_array(h.mP.β, collect(axes(Σ_ladder_it, 2)))
tc_factor = (true ? LadderDGA.tail_factor(h.mP.U, h.mP.β, h.mP.n, h.Σ_loc, iν) : 0.0 ./ iν)

λd = 0.0;
tr_tsc_01 = []
conv_tsc_01, λm_tsc_01, λd_tsc_01, μ_it_tsc_01 = LadderDGA.LambdaCorrection.run_tsc!(G_ladder_it, Σ_ladder_it, G_ladder_bak, Kνωq_pre, tc_factor, 
                χm, γm, χd, γd, λ₀, h;
                maxit=150, mixing=0.3, mixing_start_it=10, conv_abs=1e-8, trace=tr_tsc_01, verbose=true)