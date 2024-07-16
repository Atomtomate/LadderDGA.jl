# ==================== Includes ====================
#nprocs() == 1 && addprocs(8)
using Pkg
path = joinpath(abspath(@__DIR__),"..")
println("activating: ", path)
Pkg.activate(path)
using LadderDGA
using JLD2

cfg_test_01 = joinpath(@__DIR__, "../test/test_data/config_b1u2.toml")
cfg_test_02 = joinpath(@__DIR__, "../test/test_data/config_AlDGA_example.toml")

wp, mP, sP, env, kGridsStr = readConfig(cfg_test_01);
wp, mP_01, sP_01, env_01, kGridsStr = readConfig(cfg_test_01);
wp, mP_02, sP_02, env_02, kGridsStr = readConfig(cfg_test_02);
lDGAhelper = setup_LDGA(kGridsStr[1], mP, sP, env);
AlDGAhelper_01 = setup_ALDGA(kGridsStr[1], mP_01, sP_01, env_01);
AlDGAhelper_02 = setup_ALDGA(kGridsStr[1], mP_02, sP_02, env_02);

# ====================== lDGA ======================
bubble     = calc_bubble(:DMFT, lDGAhelper);
χm, γm = calc_χγ(:m, lDGAhelper, bubble);
χd, γd = calc_χγ(:d, lDGAhelper, bubble);
λ₀ = calc_λ0(bubble, lDGAhelper)

# ===================== AlDGA ======================
bubble_01     = calc_bubble(:DMFT, AlDGAhelper_01);
χm_01, γm_01 = calc_χγ(:m, AlDGAhelper_01, bubble_01);
χd_01, γd_01 = calc_χγ(:d, AlDGAhelper_01, bubble_01);
# λ₀ = calc_λ0(bubble_01, AlDGAhelper_01)
λ₀_01 = -AlDGAhelper_01.mP.U .* deepcopy(core(bubble_01));

converged_01, μ_it_01, G_ladder_it_01, Σ_ladder_it_01 = LadderDGA.LambdaCorrection.run_sc(χm_01, γm_01, χd_01, γd_01, λ₀_01, 0.0, 0.0, AlDGAhelper_01;
                maxit=100, mixing=0.2, conv_abs=1e-8, tc=true)


AlDGAhelper_02_i = deepcopy(AlDGAhelper_02);
χm_02, γm_02  = nothing, nothing
χd_02, γd_02  = nothing, nothing
bubble_02     = nothing;
for i in 0:15
    global bubble_02     = calc_bubble(:DMFT, AlDGAhelper_02_i);
    global χm_02, γm_02 = calc_χγ(:m, AlDGAhelper_02_i, bubble_02);
    global χd_02, γd_02 = calc_χγ(:d, AlDGAhelper_02_i, bubble_02);
    # λ₀ = calc_λ0(bubble_01, AlDGAhelper_01)
    λ₀_02 = -AlDGAhelper_02_i.mP.U .* deepcopy(core(bubble_02));

    converged_02, μ_it_02, G_ladder_it_02, Σ_ladder_it_02 = LadderDGA.LambdaCorrection.run_sc(χm_02, γm_02, χd_02, γd_02, λ₀_02, 0.0, 0.0, AlDGAhelper_02;
                    maxit=100, mixing=0.2, conv_abs=1e-8, tc=true)
    update_ΓAsym!(χm_02, χd_02, χd_02, sP_02, mP_02, AlDGAhelper_02_i)
end

# ==================== Results =====================
res_m = LadderDGA.λ_correction(:m, χm, γm, χd, γd, λ₀, lDGAhelper)
res_dm = λ_correction(:dm, χm, γm, χd, γd, λ₀, lDGAhelper; fit_μ=true)
res_dm_sc = run_sc(χm, γm, χd, γd, λ₀, lDGAhelper.mP.μ, lDGAhelper; type=:pre_dm, fit_μ=true, maxit=100, mixing=0.2, conv_abs=1e-6, trace=true);
res_m_ntc = LadderDGA.λ_correction(:m, χm, γm, χd, γd, λ₀, lDGAhelper, tc=false)
res_dm_ntc = λ_correction(:dm, χm, γm, χd, γd, λ₀, lDGAhelper; fit_μ=true, tc=false)
res_dm_sc_ntc = run_sc(χm, γm, χd, γd, λ₀, lDGAhelper.mP.μ, lDGAhelper; type=:pre_dm, fit_μ=true, maxit=100, mixing=0.2, conv_abs=1e-6, tc=false, trace=true);
