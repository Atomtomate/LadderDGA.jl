using Distributed

nprocs() == 1 && addprocs(2, exeflags="--project=$(Base.active_project())")
@everywhere using LadderDGA

dir = dirname(@__FILE__)
dir = joinpath(dir, "test_data/config_b1u2.toml")
cfg_file = joinpath(dir)
wp, mP, sP, env, kGridsStr = readConfig(cfg_file);
Σ_ladderLoc, Σ_loc, imp_density, kG, gLoc_fft, gLoc_rfft, Γsp, Γch, χDMFTsp, χDMFTch, χ_sp_loc, γ_sp_loc, χ_ch_loc, γ_ch_loc, χ₀Loc, gImp = setup_LDGA(kGridsStr[1], mP, sP, env);
LadderDGA.@timeit LadderDGA.to "χ₀ ser" bubble = calc_bubble(gLoc_fft, gLoc_rfft, kG, mP, sP);
LadderDGA.@timeit LadderDGA.to "χ₀ par" bubble_par = calc_bubble_par(kG, mP, sP, collect_data=true);
LadderDGA.@timeit LadderDGA.to "χ₀ par 2" calc_bubble_par(kG, mP, sP, collect_data=false);
@test all(bubble.data .≈ bubble_par.data)
@test all(bubble.asym .≈ bubble_par.asym)
println(LadderDGA.to)
