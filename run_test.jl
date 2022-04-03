using Base.GC
using TimerOutputs

using Pkg
Pkg.activate(@__DIR__)
using LadderDGA

using Distributed
#addprocs(2; topology=:master_worker)
@everywhere using Pkg
@everywhere println(@__DIR__)
@everywhere Pkg.activate(@__DIR__)
@everywhere using LadderDGA

cfg_file = "/home/julian/Hamburg/ED_data/asympt_tests/config_14_small.toml"
 mP, sP, env, kGridsStr = readConfig(cfg_file);
@timeit LadderDGA.to "setup" Σ_ladderLoc, Σ_loc, imp_density, kG, gLoc, gLoc_fft, Γsp, Γch, χDMFTsp, χDMFTch, locQ_sp, locQ_ch, χ₀Loc, gImp = setup_LDGA(kGridsStr[1], mP, sP, env);

# ladder quantities
@info "bubble"
@timeit LadderDGA.to "nl bblt" bubble = calc_bubble(gLoc_fft, kG, mP, sP);
@info "chi"
@timeit LadderDGA.to "nl xsp" nlQ_sp = calc_χγ(:sp, Γsp, bubble, kG, mP, sP);
@timeit LadderDGA.to "nl xsp par" qwp, qwr, rs, nlQ_sp_par = LadderDGA.calc_χγ_par(:sp, Γsp, bubble, kG, mP, sP);
#@timeit LadderDGA.to "nl xch" nlQ_ch = calc_χγ(:ch, Γch, bubble, kG, mP, sP);

#λsp_old = λ_correction(:sp, imp_density, FUpDo, Σ_loc, Σ_ladderLoc, nlQ_sp, nlQ_ch,bubble, gLoc_fft, kG, mP, sP)

@info "Σ"
#@timeit LadderDGA.to "nl Σ" Σ_ladder = calc_Σ(nlQ_sp, nlQ_ch, bubble, gLoc_fft, FUpDo, kG, mP, sP)
#Σ_ladder = Σ_loc_correction(Σ_ladder, Σ_ladderLoc, Σ_loc);
#G_λ = G_from_Σ(Σ_ladder)
#@timeit LadderDGA.to "nl Σ" Σ_ladder = calc_Σ(nlQ_sp, nlQ_ch, bubble, fft(G_λ), FUpDo, kG, mP, sP)
#Σ_ladder = Σ_loc_correction(Σ_ladder, Σ_ladderLoc, Σ_loc);
