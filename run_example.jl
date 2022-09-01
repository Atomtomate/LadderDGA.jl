using Pkg

Pkg.activate(@__DIR__)
using LadderDGA

cfg_file = "/home/julian/Hamburg/ED_data/asympt_tests/config_14_small.toml"
wp, mP, sP, env, kGridsStr = readConfig(cfg_file);
Σ_ladderLoc, Σ_loc, imp_density, kG, gLoc_fft, gLoc_rfft, Γsp, Γch, χDMFTsp, χDMFTch, locQ_sp, locQ_ch, χ₀Loc, gImp = setup_LDGA(kGridsStr[1], mP, sP, env);

χ₀ = calc_bubble(gLoc_fft, gLoc_rfft, kG, mP, sP);
Fsp = F_from_χ(χDMFTsp, gImp[1,:], sP, mP.β);
Fupdo = F_from_χ((χDMFTch .- χDMFTsp) .* 0.5 , gImp[1,:], sP, mP.β, diag_term=false);
λ₀ = calc_λ0(χ₀, Fsp, locQ_sp, mP, sP)

nlQ_sp = calc_χγ(:sp, Γsp, χ₀, kG, mP, sP);
nlQ_ch = calc_χγ(:ch, Γch, χ₀, kG, mP, sP);

gLoc_i = gLoc_fft
for i in 1:10
    λsp_old = λ_correction(:sp, imp_density, nlQ_sp, nlQ_ch, gLoc_i, λ₀, kG, mP, sP)
    λsp_new = λ_correction(:sp_ch, imp_density, nlQ_sp, nlQ_ch, gLoc_i, λ₀, kG, mP, sP, parallel=false)

    Σ_ladder = calc_Σ(nlQ_sp, nlQ_ch, λ₀, gLoc_rfft, kG, mP, sP);
    gLoc_i = fft(G_from_Σ(Σ_ladder))
end
