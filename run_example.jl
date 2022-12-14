using Pkg
Pkg.activate(@__DIR__)
using LadderDGA

wp, mP, sP, env, kGridsStr = readConfig(cfg_file);
Σ_ladderLoc, Σ_loc, imp_density, kG, gLoc_fft, gLoc_rfft, Γsp, Γch, χDMFTsp, χDMFTch, χ_sp_loc, γ_sp_loc, χ_ch_loc, γ_ch_loc, χ₀Loc, gImp = setup_LDGA(kGridsStr[1], mP, sP, env);

χ₀ = calc_bubble(gLoc_fft, gLoc_rfft, kG, mP, sP);
Fsp = F_from_χ(χDMFTsp, gImp[1,:], sP, mP.β);
λ₀ = calc_λ0(χ₀, Fsp, χ_sp_loc, γ_sp_loc, mP, sP)

χ_sp, γ_sp = calc_χγ(:sp, Γsp, χ₀, kG, mP, sP);
χ_ch, γ_ch = calc_χγ(:ch, Γch, χ₀, kG, mP, sP);

gLoc_i = gLoc_fft

λsp_old = λ_correction(:sp, imp_density, χ_sp, γ_sp, χ_ch, γ_ch, gLoc_i, λ₀, kG, mP, sP)
λsp_new = λ_correction(:sp_ch, imp_density, χ_sp, γ_sp, χ_ch, γ_ch, gLoc_i, λ₀, kG, mP, sP, parallel=false)

Σ_ladder = calc_Σ(χ_sp, γ_sp, χ_ch, γ_ch, λ₀, gLoc_rfft, kG, mP, sP);
gLoc_i = fft(G_from_Σ(Σ_ladder))
