using LadderDGA
using JLD2

cfg_file = ARGS[1]
fOutName = ARGS[2]

wp, mP, sP, env, kGridsStr = readConfig(cfg_file);
lDGAhelper = setup_LDGA(kGridsStr[1], mP, sP, env, silent=true);
bubble     = calc_bubble(:DMFT, lDGAhelper);
χm, γm = calc_χγ(:m, lDGAhelper, bubble);
χd, γd = calc_χγ(:d, lDGAhelper, bubble);
λ₀ = calc_λ0(bubble, lDGAhelper);

res_m_nat  = λm_correction(χm, γm, χd, γd, λ₀, lDGAhelper, λ_rhs=:native, verbose=true)
res_m_fix  = λm_correction(χm, γm, χd, γd, λ₀, lDGAhelper, λ_rhs=:fixed, verbose=true); print(res_m_fix)
res_dm     = try 
    λdm_correction(χm, γm, χd, γd, λ₀, lDGAhelper)
catch e
    println("Error in lambda dm: ", e)
    nothing
end
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

jldopen(fOutName, "w") do f
    f["chi_m"] = χm
    f["chi_d"] = χd
    f["res_m_nat"] = res_m_nat
    f["res_m_fix"] = res_m_fix
    f["res_dm"] = res_dm
end
