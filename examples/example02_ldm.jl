dir = dirname(@__FILE__)

using Pkg; Pkg.activate(joinpath(dir, ".."));
using LadderDGA

cfg_file_3Dsc = joinpath(dir, "../test/test_data/config_b1u2.toml")
cfg_file_2Dsc = joinpath(dir, "../test/test_data/config_2D.toml")

wp, mP, sP, env, kGridsStr = readConfig(cfg_file_3Dsc);
lDGAhelper = setup_LDGA(kGridsStr[1], mP, sP, env, silent=true);
bubble     = calc_bubble(:DMFT, lDGAhelper);
χm, γm = calc_χγ(:m, lDGAhelper, bubble, ω_symmetric=true);
χd, γd = calc_χγ(:d, lDGAhelper, bubble, ω_symmetric=true);

check_χ_health(χm, :m, lDGAhelper; q0_check_eps = 0.1, λmmin_check_eps = 1000)
check_χ_health(χd, :d, lDGAhelper; q0_check_eps = 0.1, λmmin_check_eps = 1000)
χm = fix_χr(χm, negative_eps = 1e-2)
χd = fix_χr(χd, negative_eps = 1e-2)

λ₀ = calc_λ0(bubble, lDGAhelper);

res_m      = λm_correction(χm, γm, χd, γd, λ₀, lDGAhelper); print(res_m)
res_dm     = λdm_correction(χm, γm, χd, γd, λ₀, lDGAhelper); print(res_dm)
res_m_sc   = λm_sc_correction(χm, γm, χd, γd, λ₀, lDGAhelper); print(res_m_sc)
res_dm_sc  = λdm_sc_correction(χm, γm, χd, γd, λ₀, lDGAhelper; validation_threshold=1e-6); print(res_dm_sc)

#= performance omparison to clean version
using TimerOutputs
to = TimerOutput()

@timeit to "clean  sc" r1 = LadderDGA.LambdaCorrection.λdm_sc_correction_clean(χm, γm, χd, γd, λ₀, lDGAhelper; validation_threshold=1e-6);
@timeit to "fast   sc" r2 = LadderDGA.LambdaCorrection.λdm_sc_correction(χm, γm, χd, γd, λ₀, lDGAhelper; validation_threshold=1e-6);

@timeit to "clean tsc" r3 = LadderDGA.LambdaCorrection.λdm_tsc_correction_clean(χm, γm, χd, γd, λ₀, lDGAhelper; validation_threshold=1e-6);
@timeit to "fast  tsc" r4 = LadderDGA.LambdaCorrection.λdm_tsc_correction(χm, γm, χd, γd, λ₀, lDGAhelper; validation_threshold=1e-6);

=#