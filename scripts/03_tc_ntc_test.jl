dir = dirname(@__FILE__)

using Pkg; Pkg.activate(joinpath(dir, ".."));
using LadderDGA

cfg_file_3Dsc_ntc = joinpath(dir, "../test/test_data/config_b1u2_ntc.toml")
cfg_file_3Dsc_tc  = joinpath(dir, "../test/test_data/config_b1u2.toml")

wp, mP, sP, env, kGridsStr = readConfig(cfg_file_3Dsc_tc);
lDGAhelper = setup_LDGA(kGridsStr[1], mP, sP, env, silent=true);
bubble     = calc_bubble(:DMFT, lDGAhelper);
χm, γm = calc_χγ(:m, lDGAhelper, bubble);
χd, γd = calc_χγ(:d, lDGAhelper, bubble);
λ₀ = calc_λ0(bubble, lDGAhelper);

res_m      = λm_correction(χm, γm, χd, γd, λ₀, lDGAhelper, verbose=true); print(res_m)

wp_ntc, mP_ntc, sP_ntc, env_ntc, kGridsStr_ntc = readConfig(cfg_file_3Dsc_ntc);
lDGAhelper_ntc = setup_LDGA(kGridsStr_ntc[1], mP_ntc, sP_ntc, env_ntc, silent=true);
bubble_ntc     = calc_bubble(:DMFT, lDGAhelper_ntc);
χm_ntc, γm_ntc = calc_χγ(:m, lDGAhelper_ntc, bubble_ntc);
χd_ntc, γd_ntc = calc_χγ(:d, lDGAhelper_ntc, bubble_ntc);
λ₀_ntc = calc_λ0(bubble_ntc, lDGAhelper_ntc);

res_m_ntc      = λm_correction(χm_ntc, γm_ntc, χd_ntc, γd_ntc, λ₀_ntc, lDGAhelper_ntc, verbose=true); print(res_m_ntc)
# res_dm     = λdm_correction(χm, γm, χd, γd, λ₀, lDGAhelper); print(res_dm)
# res_m_sc   = λm_sc_correction(χm, γm, χd, γd, λ₀, lDGAhelper); print(res_m_sc)
# res_dm_sc  = λdm_sc_correction(χm, γm, χd, γd, λ₀, lDGAhelper; validation_threshold=1e-6); print(res_dm_sc)
# res_m_tsc  = λm_tsc_correction(χm, γm, χd, γd, λ₀, lDGAhelper; max_steps_sc = 200, validation_threshold=1e-6); print(res_m_tsc)
# res_dm_tsc = λdm_tsc_correction(χm, γm, χd, γd, λ₀, lDGAhelper; max_steps_sc = 200, validation_threshold=1e-6); print(res_dm_tsc)

#= performance omparison to clean version
using TimerOutputs
to = TimerOutput()

@timeit to "clean  sc" r1 = LadderDGA.LambdaCorrection.λdm_sc_correction_clean(χm, γm, χd, γd, λ₀, lDGAhelper; validation_threshold=1e-6);
@timeit to "fast   sc" r2 = LadderDGA.LambdaCorrection.λdm_sc_correction(χm, γm, χd, γd, λ₀, lDGAhelper; validation_threshold=1e-6);

@timeit to "clean tsc" r3 = LadderDGA.LambdaCorrection.λdm_tsc_correction_clean(χm, γm, χd, γd, λ₀, lDGAhelper; validation_threshold=1e-6);
@timeit to "fast  tsc" r4 = LadderDGA.LambdaCorrection.λdm_tsc_correction(χm, γm, χd, γd, λ₀, lDGAhelper; validation_threshold=1e-6);

=#

