#using Profile, ProfileSVG, FlameGraphs, Plots
#using JLD2, FileIO
#using Logging
using Distributed
using Base.GC
using TimerOutputs

#addprocs(2)

using Pkg
Pkg.activate(@__DIR__)
using LadderDGA

cfg_file =  "/home/julian/Hamburg/Julia_lDGA/lDGA_shift_tests/data/40_40_s0_b14_u1.0/config_j.toml";
@timeit LadderDGA.to "read" mP, sP, env, kGridsStr = readConfig(cfg_file);

println(typeof(kGridsStr[1]))
println("------------------------------------------- $(kGridsStr[1])")
@timeit LadderDGA.to "setup" impQ_sp, impQ_ch, gImp, kGridLoc, kG, gLoc, gLoc_fft, Σ_loc, FUpDo = setup_LDGA(kGridsStr[1], mP, sP, env);

@info "local"
@timeit LadderDGA.to "loc bbl" bubbleLoc = calc_bubble(gImp, kGridLoc, mP, sP);
@timeit LadderDGA.to "loc xsp" locQ_sp = calc_χ_trilex(impQ_sp.Γ, bubbleLoc, kGridLoc, mP.U, mP, sP);
@timeit LadderDGA.to "loc xch"  locQ_ch = calc_χ_trilex(impQ_ch.Γ, bubbleLoc, kGridLoc, -mP.U, mP, sP);


@timeit LadderDGA.to "loc Σ" Σ_ladderLoc = calc_Σ(locQ_sp, locQ_ch, bubbleLoc, gImp, FUpDo, kGridLoc, mP, sP)
any(isnan.(Σ_ladderLoc)) && error("Σ_ladderLoc contains NaN")

# ladder quantities
@info "bubble"
@timeit LadderDGA.to "nl bblt" bubble = calc_bubble(gLoc_fft, kG, mP, sP);
@info "chi"
@timeit LadderDGA.to "nl xsp" nlQ_sp = calc_χ_trilex(impQ_sp.Γ, bubble, kG, mP.U, mP, sP);
@timeit LadderDGA.to "nl xch" nlQ_ch = calc_χ_trilex(impQ_ch.Γ, bubble, kG, -mP.U, mP, sP);


imp_density = real(impQ_sp.χ_loc + impQ_ch.χ_loc)
λsp_old = λ_correction(:sp, imp_density, FUpDo, Σ_loc, Σ_ladderLoc, nlQ_sp, nlQ_ch,bubble, gLoc_fft, kG, mP, sP)

@info "Σ"
@timeit LadderDGA.to "nl Σ" Σ_ladder = calc_Σ(nlQ_sp, nlQ_ch, bubble, gLoc_fft, FUpDo, kG, mP, sP)
Σ_ladder = Σ_loc_correction(Σ_ladder, Σ_ladderLoc, Σ_loc);
@timeit LadderDGA.to "nl Σ" Σ_ladder = calc_Σ(nlQ_sp, nlQ_ch, bubble, fft(G_λ), FUpDo, kG, mP, sP)
Σ_ladder = Σ_loc_correction(Σ_ladder, Σ_ladderLoc, Σ_loc);
