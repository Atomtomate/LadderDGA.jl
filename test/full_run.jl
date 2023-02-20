using Distributed
using OffsetArrays

rmprocs(workers(default_worker_pool()))
addprocs(2, exeflags="--project=$(Base.active_project())")
@everywhere using LadderDGA

dir = dirname(@__FILE__)
dir = joinpath(dir, "test_data/config_b1u2.toml")
cfg_file = joinpath(dir)

LadderDGA.clear_wcache!()
wp, mP, sP, env, kGridsStr = readConfig(cfg_file);
Σ_ladderLoc, Σ_loc, imp_density, kG, gLoc, gLoc_fft, gLoc_rfft, Γsp, Γch, χDMFTsp, χDMFTch, χ_m_loc, γ_m_loc, χ_d_loc, γ_d_loc, χ₀Loc, gImp = setup_LDGA(kGridsStr[1], mP, sP, env);
bubble = calc_bubble(gLoc_fft, gLoc_rfft, kG, mP, sP);
bubble_par = calc_bubble_par(kG, mP, sP, collect_data=true);
calc_bubble_par(kG, mP, sP, collect_data=false);
@test all(bubble.data .≈ bubble_par.data)
@test all(bubble.asym .≈ bubble_par.asym)

χ_m, γ_m = calc_χγ(:sp, Γsp, bubble, kG, mP, sP);
χ_d, γ_d = calc_χγ(:ch, Γch, bubble, kG, mP, sP);
cs_χm = abs(sum(χ_m))
cs_χd = abs(sum(χ_d))
cs_γm = abs(sum(γ_m))
cs_γd = abs(sum(γ_d))

χ_m_par, γ_m_par = calc_χγ_par(:sp, Γsp, kG, mP, sP);
χ_d_par, γ_d_par = calc_χγ_par(:ch, Γch, kG, mP, sP);
@test all(χ_m.data .≈ χ_m_par.data)
@test all(χ_d.data .≈ χ_d_par.data)
@test all(χ_m.tail_c .≈ χ_m_par.tail_c)
@test all(χ_d.tail_c .≈ χ_d_par.tail_c)
@test all(χ_m.usable_ω .≈ χ_m_par.usable_ω)
@test all(χ_d.usable_ω .≈ χ_d_par.usable_ω)
@test all(γ_m.data .≈ γ_m_par.data)
@test all(γ_d.data .≈ γ_d_par.data)


Fsp = F_from_χ(χDMFTsp, gImp[1,:], sP, mP.β);
λ₀ = calc_λ0(bubble, Fsp, χ_m_loc, γ_m_loc, mP, sP)
Σ_ladder = calc_Σ(χ_m, γ_m, χ_d, γ_d, λ₀, gLoc_rfft, kG, mP, sP, νmax=sP.n_iν);
initialize_EoM(gLoc_rfft, λ₀, 0:sP.n_iν-5, kG, mP, sP, χ_m = χ_m, γ_m = γ_m,
                χ_d = χ_d, γ_d = γ_d, force_reinit=true)
Σ_ladder_par_2 = calc_Σ_par(kG, mP);
@test all(Σ_ladder.parent[:,1:end-4] .≈ Σ_ladder_par_2.parent)
Σ_ladder_inplace = similar(Σ_ladder_par_2)
LadderDGA.calc_Σ_par!(Σ_ladder_inplace, mP)
@test all(Σ_ladder.parent[:,1:end-4] .≈ Σ_ladder_inplace.parent)

initialize_EoM(gLoc_rfft, λ₀, 0:sP.n_iν-1, kG, mP, sP, 
                χ_m = χ_m, γ_m = γ_m,
                χ_d = χ_d, γ_d = γ_d, force_reinit=true)
Σ_ladder_par = calc_Σ_par(kG, mP);
@test all(Σ_ladder.parent .≈ Σ_ladder_par.parent)
@test abs(sum(χ_d)) ≈ cs_χd


run_res = run_sc(χ_m, γ_m, χ_d, γ_d, λ₀, gLoc_rfft, Σ_loc,0.1, kG, mP, sP;maxit=100, mixing=0.2, conv_abs=1e-8, update_χ_tail=false)
@test abs(sum(χ_d)) ≈ cs_χd
run_res_par = LadderDGA.LambdaCorrection.run_sc_par(χ_m, γ_m, χ_d, γ_d, λ₀, gLoc_rfft, Σ_loc,0.1, kG, mP, sP;maxit=100, mixing=0.2, conv_abs=1e-8, update_χ_tail=false)
@test abs(sum(χ_d)) ≈ cs_χd

@testset "run_sc" begin
for el in zip(run_res[2:end], run_res_par[2:end])
    @test all(isapprox.(el[1], el[2], rtol=1e-4))
end
end

χ_m2 = collect_χ(:sp, kG, mP, sP)
χ_d2 = collect_χ(:ch, kG, mP, sP)
@test all(χ_m.data .≈ χ_m2.data)
@test all(χ_d.data .≈ χ_d2.data)
@test abs(sum(χ_d)) ≈ cs_χd
res_λdm = λdm_correction(χ_m, γ_m, χ_d, γ_d, Σ_loc, gLoc_rfft, λ₀, kG, mP, sP; update_χ_tail=false, maxit=0, par=false, with_trace=true)
@test abs(sum(χ_d)) ≈ cs_χd
res_λdm_par = λdm_correction(χ_m, γ_m, χ_d, γ_d, Σ_loc, gLoc_rfft, λ₀, kG, mP, sP; update_χ_tail=false, maxit=0, par=true, with_trace=true)
@test abs(sum(χ_d)) ≈ cs_χd
@testset "λdm" begin
for el in zip(res_λdm[2:end-1], res_λdm_par[2:end-1])
    @test all(isapprox.(el[1], el[2], rtol=0.01))
end
end
Σ_ladder_dm_2 = calc_Σ(χ_m, γ_m, χ_d, γ_d, λ₀, gLoc_rfft, kG, mP, sP, νmax=last(axes(res_λdm[2],2))+1, λm=res_λdm[7], λd=res_λdm[end]);
@test all(Σ_ladder_dm_2 .≈ res_λdm[2])
_, G_ladder_dm_2 = G_from_Σladder(Σ_ladder_dm_2, Σ_loc, kG, mP, sP; fix_n=false);
@test all(G_ladder_dm_2 .≈ res_λdm[3])

res_λdm_sc = λdm_correction(χ_m, γ_m, χ_d, γ_d, Σ_loc, gLoc_rfft, λ₀, kG, mP, sP; update_χ_tail=false, maxit=10, par=false, with_trace=true)
@test abs(sum(χ_d)) ≈ cs_χd
res_λdm_sc_par = λdm_correction(χ_m, γ_m, χ_d, γ_d, Σ_loc, gLoc_rfft, λ₀, kG, mP, sP; update_χ_tail=false, maxit=10, par=true, with_trace=true)
@test abs(sum(χ_d)) ≈ cs_χd
@testset "λdm" begin
for el in zip(res_λdm_sc[2:end-1], res_λdm_sc_par[2:end-1])
    @test all(isapprox.(el[1], el[2], rtol=0.01))
end
end
@test abs(sum(χ_d)) ≈ cs_χd

tr = []
res_λdm_tsc = λdm_correction(χ_m, γ_m, χ_d, γ_d, Σ_loc, gLoc_rfft, λ₀, kG, mP, sP; update_χ_tail=true, maxit=10, par=false, with_trace=true)
@test abs(sum(χ_m)) ≈ cs_χm
@test abs(sum(χ_d)) ≈ cs_χd
#res_λdm_tsc = run_sc(χ_m, γ_m, χ_d, γ_d, λ₀, gLoc_rfft, Σ_loc, 0.1, kG, mP, sP; maxit=100, mixing=0.2, conv_abs=1e-8, update_χ_tail=true)
res_λdm_tsc_par = λdm_correction(χ_m, γ_m, χ_d, γ_d, Σ_loc, gLoc_rfft, λ₀, kG, mP, sP; update_χ_tail=true, maxit=10, par=true, with_trace=true)
@test abs(sum(χ_m)) ≈ cs_χm
@test abs(sum(χ_d)) ≈ cs_χd

@testset "λdm_tsc" begin
    for el in zip(res_λdm_tsc[2:end-1], res_λdm_tsc_par[2:end-1])
        @test all(isapprox.(el[1], el[2], atol=1e-3))
    end
end
@test abs(sum(χ_m)) ≈ cs_χm
@test abs(sum(χ_d)) ≈ cs_χd
@test abs(sum(γ_m)) ≈ cs_γm
@test abs(sum(γ_d)) ≈ cs_γd
