using Pkg
Pkg.activate(@__DIR__)
using LadderDGA

cfg_file = "/home/julian/Hamburg/ED_data/asympt_tests/b1.0_mu1.0_tp0.toml"

wp, mP, sP, env, kGridsStr = readConfig(cfg_file);
# Σ_ladder_list = []
# Σ_ladder_λm_list = []
# Σ_ladder_λdm_list = []

Σ_ladderLoc, Σ_loc, imp_density, kG, gLoc_fft, gLoc_rfft, Γsp, Γch, χDMFTsp, χDMFTch, χ_sp_loc, γ_sp_loc, χ_ch_loc, γ_ch_loc, χ₀Loc, gImp = setup_LDGA(kGridsStr[1], mP, sP, env);

bubble = calc_bubble(gLoc_fft, gLoc_rfft, kG, mP, sP);

Fsp = F_from_χ(χDMFTsp, gImp[1,:], sP, mP.β);
λ₀ = calc_λ0(bubble, Fsp, χ_sp_loc, γ_sp_loc, mP, sP)

χ_sp, γ_sp = calc_χγ(:sp, Γsp, bubble, kG, mP, sP);
χ_ch, γ_ch = calc_χγ(:ch, Γch, bubble, kG, mP, sP);
Σ_ladder = calc_Σ(χ_sp, γ_sp, χ_ch, γ_ch, λ₀, gLoc_rfft, kG, mP, sP);
Σ_ladder_parts = calc_Σ_parts(χ_sp, γ_sp, χ_ch, γ_ch, λ₀, gLoc_rfft, kG, mP, sP);
c2_res = residuals(10, 10, Float64[], χ_sp, γ_sp, χ_ch, γ_ch, Σ_loc, gLoc_rfft, λ₀, kG, mP, sP; maxit=0)
λsp_old = c2_res[1,c2_res[2,:] .== 0][1]
λsp_new,λch_new,check_new = find_root(c2_res[:,c2_res[8,:] .== 1])
χ_λ!(χ_sp, χ_sp, λsp_old)
Σ_ladder_old = calc_Σ_parts(χ_sp, γ_sp, χ_ch, γ_ch, λ₀, gLoc_rfft, kG, mP, sP);
χ_λ!(χ_sp, χ_sp, -λsp_old)
χ_λ!(χ_sp, χ_sp, λsp_new)
χ_λ!(χ_ch, χ_ch, λch_new)
Σ_ladder_new = calc_Σ_parts(χ_sp, γ_sp, χ_ch, γ_ch, λ₀, gLoc_rfft, kG, mP, sP);
χ_λ!(χ_sp, χ_sp, -λsp_new)
χ_λ!(χ_ch, χ_ch, -λch_new)
# push!(Σ_ladder_list,Σ_ladder)
# push!(Σ_ladder_λm_list,Σ_ladder_old)
# push!(Σ_ladder_λdm_list,Σ_ladder_new)
