using Distributed
using TimerOutputs
using OffsetArrays

rmprocs(workers(default_worker_pool()))
addprocs(2, exeflags="--project=$(Base.active_project())")
@everywhere using LadderDGA

dir = dirname(@__FILE__)
dir = joinpath(dir, "test_data/config_b1u2.toml")
# dir = "/home/julian/Hamburg/eom_tails/b5.0_mu1.4_tp0.toml"
cfg_file = joinpath(dir)

LadderDGA.clear_wcache!()
wp, mP, sP, env, kGridsStr = readConfig(cfg_file);
lDGAhelper = setup_LDGA(kGridsStr[1], mP, sP, env);
bubble     = calc_bubble(lDGAhelper);
bubble_par = calc_bubble_par(lDGAhelper);
calc_bubble_par(lDGAhelper, collect_data=false);
@test all(bubble.data .≈ bubble_par.data)
@test all(bubble.asym .≈ bubble_par.asym)

χ_m, γ_m = calc_χγ(:m, lDGAhelper, bubble);
χ_d, γ_d = calc_χγ(:d, lDGAhelper, bubble);
cs_χm = abs(sum(χ_m))
cs_χd = abs(sum(χ_d))
cs_γm = abs(sum(γ_m))
cs_γd = abs(sum(γ_d))

χ_m_par, γ_m_par = calc_χγ_par(:m, lDGAhelper);
χ_d_par, γ_d_par = calc_χγ_par(:d, lDGAhelper);
@test all(χ_m.data .≈ χ_m_par.data)
@test all(χ_d.data .≈ χ_d_par.data)
@test all(χ_m.tail_c .≈ χ_m_par.tail_c)
@test all(χ_d.tail_c .≈ χ_d_par.tail_c)
@test all(χ_m.usable_ω .≈ χ_m_par.usable_ω)
@test all(χ_d.usable_ω .≈ χ_d_par.usable_ω)
@test all(γ_m.data .≈ γ_m_par.data)
@test all(γ_d.data .≈ γ_d_par.data)


λ₀ = calc_λ0(bubble, lDGAhelper)
Σ_ladder = calc_Σ(χ_m, γ_m, χ_d, γ_d, λ₀, lDGAhelper, νmax=sP.n_iν, λm=0.123, λd=1.234);
μnew, G_ladder = G_from_Σladder(Σ_ladder, lDGAhelper.Σ_loc, lDGAhelper.kG, lDGAhelper.mP, lDGAhelper.sP; fix_n=true)
initialize_EoM(lDGAhelper, λ₀, 0:sP.n_iν-1, χ_m = χ_m, γ_m = γ_m,
                χ_d = χ_d, γ_d = γ_d, force_reinit=true)
Σ_ladder_par_2 = calc_Σ_par(λm=0.123, λd=1.234);
@test all(Σ_ladder.parent .≈ Σ_ladder_par_2.parent)
Σ_ladder_inplace = similar(Σ_ladder_par_2)
LadderDGA.calc_Σ_par!(Σ_ladder_inplace, λm=0.123, λd=1.234)
@test all(Σ_ladder.parent .≈ Σ_ladder_inplace.parent)

initialize_EoM(lDGAhelper, λ₀, 0:sP.n_iν-1,
                χ_m = χ_m, γ_m = γ_m,
                χ_d = χ_d, γ_d = γ_d, force_reinit=true)
Σ_ladder_par = calc_Σ_par(λm=0.123, λd=1.234);
@test all(Σ_ladder.parent .≈ Σ_ladder_par.parent)
@test abs(sum(χ_d)) ≈ cs_χd


@timeit LadderDGA.to "run_res" run_res = run_sc(χ_m, γ_m, χ_d, γ_d, lDGAhelper.χloc_m_sum, λ₀, lDGAhelper.gLoc_rfft, lDGAhelper.Σ_loc,0.1, lDGAhelper.kG, lDGAhelper.mP, lDGAhelper.sP;maxit=100, mixing=0.2, conv_abs=1e-8, update_χ_tail=false)
@test abs(sum(χ_d)) ≈ cs_χd
@timeit LadderDGA.to "run_res_par" run_res_par = LadderDGA.LambdaCorrection.run_sc_par(χ_m, γ_m, χ_d, γ_d, lDGAhelper.χloc_m_sum, λ₀, lDGAhelper.gLoc_rfft, lDGAhelper.Σ_loc, 0.1, lDGAhelper.kG, lDGAhelper.mP, lDGAhelper.sP;maxit=100, mixing=0.2, conv_abs=1e-8, update_χ_tail=false)
@test abs(sum(χ_d)) ≈ cs_χd

@testset "run_sc" begin
for el in zip(run_res[2:end], run_res_par[2:end])
    @test all(isapprox.(el[1], el[2], rtol=1e-4))
end
end

χ_λ!(χ_d, 0.1)
λm_res_λd01 = LadderDGA.λ_correction(:m, χ_m, γ_m, χ_d, γ_d, λ₀, lDGAhelper, λ_val_only=true)
λm = λm_res_λd01.λm
χ_λ!(χ_m, λm)
@timeit LadderDGA.to "run_sc_new" run_res_new = LadderDGA.LambdaCorrection.run_sc_new(χ_m, γ_m, χ_d, γ_d, lDGAhelper.gLoc_rfft, λ₀, lDGAhelper; maxit=100, mixing=0.2, conv_abs=1e-8, trace=true)
reset!(χ_d)
reset!(χ_m)
@testset "run_sc" begin
    @test all(run_res[2] .≈ run_res_new.Σ_ladder)
    @test all(run_res[3] .≈ run_res_new.G_ladder)
    @test run_res[4] .≈ run_res_new.EKin
    @test run_res[5] .≈ run_res_new.EPot_p1
    @test run_res[6] .≈ run_res_new.μ   atol=1e-7
    @test run_res[7] .≈ run_res_new.λm
    @test run_res[8] .≈ run_res_new.PP_p2
    @test run_res[9] .≈ run_res_new.EPot_p2
    @test run_res[10] .≈ run_res_new.converged
end

χ_m2 = collect_χ(:m, lDGAhelper)
χ_d2 = collect_χ(:d, lDGAhelper)
@test all(χ_m.data .≈ χ_m2.data)
@test all(χ_d.data .≈ χ_d2.data)

# λm 
λm_res = LadderDGA.λ_correction(:m, χ_m, γ_m, χ_d, γ_d, λ₀, lDGAhelper, λ_val_only=true)
λm = λm_res.λm
t_sc = 0.5 * (sum_kω(lDGAhelper.kG, χ_λ(χ_m,λm)) + sum_kω(lDGAhelper.kG,χ_d))
t_sc2 = 0.5 * (sum_ωk(lDGAhelper.kG, χ_λ(χ_m,λm)) + sum_kω(lDGAhelper.kG,χ_d))
ωn_arr=ωn_grid(χ_m)
ωn2_tail = real(χ_m.tail_c[3] ./ ωn_arr .^ 2)
zero_ind = findfirst(x->!isfinite(x), ωn2_tail)
ωn2_tail[zero_ind] = 0.0
t_sc3 = 0.5 * sum_kω(lDGAhelper.kG, χ_λ(χ_m,λm) .+ χ_d, χ_m.β, χ_m.tail_c[3], ωn2_tail)
@test t_sc ≈ t_sc2
@test t_sc ≈ t_sc3 atol=0.001
@test t_sc ≈ lDGAhelper.mP.n/2 * (1 - lDGAhelper.mP.n/2)
@test λm_res.converged
Σ_ladder_λm = calc_Σ(χ_m, γ_m, χ_d, γ_d, λ₀, lDGAhelper, νmax=sP.n_iν, λm=λm, λd=0.0);
μnew, G_ladder_λm = G_from_Σladder(Σ_ladder_λm, lDGAhelper.Σ_loc, lDGAhelper.kG, lDGAhelper.mP, lDGAhelper.sP; fix_n=true)
E_kin_1, E_pot_1 = LadderDGA.calc_E(G_ladder, Σ_ladder, μnew, lDGAhelper.kG, lDGAhelper.mP)
Σ_ladder_λm = calc_Σ(χ_m, γ_m, χ_d, γ_d, λ₀, lDGAhelper, νmax=sP.n_iν);
μnew, G_ladder_λm = G_from_Σladder(Σ_ladder_λm, lDGAhelper.Σ_loc, lDGAhelper.kG, lDGAhelper.mP, lDGAhelper.sP; fix_n=true)
E_kin_1, E_pot_1 = LadderDGA.calc_E(G_ladder, Σ_ladder, μnew, lDGAhelper.kG, lDGAhelper.mP)


# λdm 
@test abs(sum(χ_d)) ≈ cs_χd
res_λdm_new = LadderDGA.LambdaCorrection.λdm_correction(χ_m, γ_m, χ_d, γ_d, λ₀, lDGAhelper, νmax=4, λ_val_only=true)
@test abs(sum(χ_d)) ≈ cs_χd
res_λdm_new_par = LadderDGA.LambdaCorrection.λdm_correction(χ_m, γ_m, χ_d, γ_d, λ₀, lDGAhelper, νmax=4, par=true, λ_val_only=true)
@test abs(sum(χ_d)) ≈ cs_χd
@testset "λdm" begin
    res_λdm_new[1] ≈ res_λdm_new_par[1]
    res_λdm_new[2] ≈ res_λdm_new_par[2] 
end


# Σ_ladder_dm_2 = calc_Σ(χ_m, γ_m, χ_d, γ_d, λ₀, lDGAhelper, νmax=last(axes(res_λdm[2],2))+1, λm=res_λdm[7], λd=res_λdm[end]);
# @test all(Σ_ladder_dm_2 .≈ res_λdm[2])
# _, G_ladder_dm_2 = G_from_Σladder(Σ_ladder_dm_2, Σ_loc, kG, mP, sP; fix_n=false);
# @test all(G_ladder_dm_2 .≈ res_λdm[3])

# SC

# @timeit LadderDGA.to "res_sc" res_λdm_sc = λdm_correction(χ_m, γ_m, χ_d, γ_d, Σ_loc, gLoc_rfft, χloc_m_sum, λ₀, kG, mP, sP; update_χ_tail=false, maxit=10, par=false, with_trace=true)
# @test abs(sum(χ_d)) ≈ cs_χd
# @timeit LadderDGA.to "res_sc_par" res_λdm_sc_par = λdm_correction(χ_m, γ_m, χ_d, γ_d, Σ_loc, gLoc_rfft, χloc_m_sum, λ₀, kG, mP, sP; update_χ_tail=false, maxit=10, par=true, with_trace=true)
# @test abs(sum(χ_d)) ≈ cs_χd
# @testset "λdm" begin
# for el in zip(res_λdm_sc[2:end-1], res_λdm_sc_par[2:end-1])
#     @test all(isapprox.(el[1], el[2], rtol=0.01))
# end
# end
# @test abs(sum(χ_d)) ≈ cs_χd

# # tr = []
# @timeit LadderDGA.to "res_tsc" res_λdm_tsc = λdm_correction(χ_m, γ_m, χ_d, γ_d, Σ_loc, gLoc_rfft, χloc_m_sum, λ₀, kG, mP, sP; update_χ_tail=true, maxit=10, par=false, with_trace=true)
# @test abs(sum(χ_m)) ≈ cs_χm
# @test abs(sum(χ_d)) ≈ cs_χd
# #res_λdm_tsc = run_sc(χ_m, γ_m, χ_d, γ_d, λ₀, gLoc_rfft, Σ_loc, 0.1, kG, mP, sP; maxit=100, mixing=0.2, conv_abs=1e-8, update_χ_tail=true)
# @timeit LadderDGA.to "res_tsc_par" res_λdm_tsc_par = λdm_correction(χ_m, γ_m, χ_d, γ_d, Σ_loc, gLoc_rfft, χloc_m_sum, λ₀, kG, mP, sP; update_χ_tail=true, maxit=10, par=true, with_trace=true)
# @test abs(sum(χ_m)) ≈ cs_χm
# @test abs(sum(χ_d)) ≈ cs_χd

# @testset "λdm_tsc" begin
#     for el in zip(res_λdm_tsc[2:end-1], res_λdm_tsc_par[2:end-1])
#         @test all(isapprox.(el[1], el[2], atol=1e-6))
#     end
# end
# @test abs(sum(χ_m)) ≈ cs_χm
# @test abs(sum(χ_d)) ≈ cs_χd
# @test abs(sum(γ_m)) ≈ cs_γm
# @test abs(sum(γ_d)) ≈ cs_γd
