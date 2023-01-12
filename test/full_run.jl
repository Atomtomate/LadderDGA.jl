using Distributed

rmprocs(workers(default_worker_pool()))
addprocs(2, exeflags="--project=$(Base.active_project())")
@everywhere using LadderDGA

dir = dirname(@__FILE__)
dir = joinpath(dir, "test_data/config_b1u2.toml")
cfg_file = joinpath(dir)
wp, mP, sP, env, kGridsStr = readConfig(cfg_file);
Σ_ladderLoc, Σ_loc, imp_density, kG, gLoc_fft, gLoc_rfft, Γsp, Γch, χDMFTsp, χDMFTch, χ_sp_loc, γ_sp_loc, χ_ch_loc, γ_ch_loc, χ₀Loc, gImp = setup_LDGA(kGridsStr[1], mP, sP, env);
bubble = calc_bubble(gLoc_fft, gLoc_rfft, kG, mP, sP);
bubble_par = calc_bubble_par(kG, mP, sP, collect_data=true);
calc_bubble_par(kG, mP, sP, collect_data=false);
@test all(bubble.data .≈ bubble_par.data)
@test all(bubble.asym .≈ bubble_par.asym)

χ_sp, γ_sp = calc_χγ(:sp, Γsp, bubble, kG, mP, sP);
χ_ch, γ_ch = calc_χγ(:ch, Γch, bubble, kG, mP, sP);
χ_sp_par, γ_sp_par = calc_χγ_par(:sp, Γsp, kG, mP, sP);
χ_ch_par, γ_ch_par = calc_χγ_par(:ch, Γch, kG, mP, sP);
@test all(χ_sp.data .≈ χ_sp_par.data)
@test all(χ_ch.data .≈ χ_ch_par.data)
@test all(χ_sp.tail_c .≈ χ_sp_par.tail_c)
@test all(χ_ch.tail_c .≈ χ_ch_par.tail_c)
@test all(χ_sp.usable_ω .≈ χ_sp_par.usable_ω)
@test all(χ_ch.usable_ω .≈ χ_ch_par.usable_ω)
@test all(γ_sp.data .≈ γ_sp_par.data)
@test all(γ_ch.data .≈ γ_ch_par.data)


Fsp = F_from_χ(χDMFTsp, gImp[1,:], sP, mP.β);
λ₀ = calc_λ0(bubble, Fsp, χ_sp_loc, γ_sp_loc, mP, sP)
Σ_ladder = calc_Σ(χ_sp, γ_sp, χ_ch, γ_ch, λ₀, gLoc_rfft, kG, mP, sP, νmax=sP.n_iν);

initialize_EoM(gLoc_rfft, λ₀, 0:sP.n_iν-1, kG, mP, sP, 
                χsp = χ_sp, γsp = γ_sp,
                χch = χ_ch, γch = γ_ch)
Σ_ladder_par = calc_Σ_par(kG, mP, sP, νrange=0:sP.n_iν-1);
@test all(Σ_ladder.parent .≈ Σ_ladder_par.parent)

initialize_EoM(gLoc_rfft, λ₀, 0:sP.n_iν-5, kG, mP, sP, 
                χsp = χ_sp, γsp = γ_sp,
                χch = χ_ch, γch = γ_ch)
Σ_ladder_par = calc_Σ_par(kG, mP, sP, νrange=0:sP.n_iν-5);
@test all(Σ_ladder.parent[:,1:end-4] .≈ Σ_ladder_par.parent)

χ_sp2 = collect_χ(:sp, kG, mP, sP)
χ_ch2 = collect_χ(:ch, kG, mP, sP)
@test all(χ_sp.data .≈ χ_sp2.data)
@test all(χ_ch.data .≈ χ_ch2.data)

c2_res_sc = residuals(2, 2, Float64[], χ_sp, γ_sp, χ_ch, γ_ch, Σ_loc, gLoc_rfft, λ₀, kG, mP, sP, conv_abs=1e-6, maxit=10, par=false)
c2_res_sc_par = residuals(2, 2, Float64[], χ_sp, γ_sp, χ_ch, γ_ch, Σ_loc, gLoc_rfft, λ₀, kG, mP, sP, conv_abs=1e-6, maxit=10, par=true)
ind = isfinite.(c2_res_sc) .&& isfinite.(c2_res_sc_par)
@test all(isapprox.(c2_res_sc[ind], c2_res_sc_par[ind], atol=1e-4))
c2_res = residuals(2, 2, Float64[], χ_sp, γ_sp, χ_ch, γ_ch, Σ_loc, gLoc_rfft, λ₀, kG, mP, sP; maxit=0, par=false)
r1 = find_root(c2_res_sc)
