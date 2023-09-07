using Distributed
using TimerOutputs
using Pkg

Pkg.activate(".")
nprocs() == 1 && addprocs(8, exeflags="--project=$(Base.active_project())")
@everywhere using LadderDGA

dir = dirname(@__FILE__)
dir = joinpath(dir, "/home/julian/Hamburg/ED_data/asympt_tests/b5.0_mu1.0_tp0.toml")
cfg_file = joinpath(dir)

println(" ======== Running: ========")
to = LadderDGA.to
println("   - Input")
@timeit to "input" wp, mP, sP, env, kGridsStr = readConfig(cfg_file);
@timeit to "setup" Σ_ladderLoc, Σ_loc, imp_density, kG, gLoc, gLoc_fft, gLoc_rfft, Γsp, Γch, χDMFTsp, χDMFTch, χ_sp_loc, γ_sp_loc, χ_ch_loc, γ_ch_loc, χ₀Loc, gImp = setup_LDGA(kGridsStr[1], mP, sP, env);
println("   - Bubble")
@timeit to "χ₀" bubble = calc_bubble(gLoc_fft, gLoc_rfft, kG, mP, sP);
@timeit to "χ₀ par" bubble_par = calc_bubble_par(kG, mP, sP, collect_data=true);
@timeit to "χ₀ par nc" calc_bubble_par(kG, mP, sP, collect_data=true);
#calc_bubble_par(kG, mP, sP, collect_data=false);

println("   - χγ")
@timeit to "χγ sp" χ_sp, γ_sp = calc_χγ(:sp, Γsp, bubble, kG, mP, sP);
@timeit to "χγ ch" χ_ch, γ_ch = calc_χγ(:ch, Γch, bubble, kG, mP, sP);
println("   - χγ par")
@timeit to "χγ sp par" χ_sp_par, γ_sp_par = calc_χγ_par(:sp, Γsp, kG, mP, sP);
@timeit to "χγ ch par" χ_ch_par, γ_ch_par = calc_χγ_par(:ch, Γsp, kG, mP, sP);
println("   - χγ par2")
@timeit to "χγ sp par nc" calc_χγ_par(:sp, Γch, kG, mP, sP, collect_data=false);
@timeit to "χγ ch par nc" calc_χγ_par(:ch, Γch, kG, mP, sP, collect_data=false);


println("   - λ₀")
@timeit to "Fsp" Fsp = F_from_χ(χDMFTsp, gImp[1,:], sP, mP.β);
@timeit to "λ₀" λ₀ = calc_λ0(bubble, Fsp, χ_sp_loc, γ_sp_loc, mP, sP)
println("   - Σ")
@timeit to "Σ" Σ_ladder = calc_Σ(χ_sp, γ_sp, χ_ch, γ_ch, λ₀, gLoc_rfft, kG, mP, sP, νmax=sP.n_iν);

println("   - Σ init2")
@timeit to "init Σ nc" initialize_EoM(gLoc_rfft, λ₀, 0:sP.n_iν-1, kG, mP, sP)
println("   - Σ init")
@timeit to "init Σ" initialize_EoM(gLoc_rfft, λ₀, 0:sP.n_iν-1, kG, mP, sP, 
                χsp = χ_sp, γsp = γ_sp,
                χch = χ_ch, γch = γ_ch)
@timeit to "Σ par" Σ_ladder_par = calc_Σ_par(kG, mP, sP, νrange=0:sP.n_iν-1);
@timeit to "Σ par2" calc_Σ_par(kG, mP, sP, νrange=0:sP.n_iν-1, collect_data=false);
println(to)
