using LadderDGA
using JLD2

cfg_file = ARGS[1]
mode = parse(Int, ARGS[3])

wp, mP, sP, env, kGridsStr = readConfig(cfg_file);
lDGAhelper = setup_LDGA(kGridsStr[1], mP, sP, env, silent=true);
bubble     = calc_bubble(:DMFT, lDGAhelper);
χm, γm = calc_χγ(:m, lDGAhelper, bubble);
χd, γd = calc_χγ(:d, lDGAhelper, bubble);
λ₀ = calc_λ0(bubble, lDGAhelper);

res_m      = mode == 1 && try λm_correction(χm, γm, χd, γd, λ₀, lDGAhelper) catch e nothing end;
res_dm     = mode == 2 && try λdm_correction(χm, γm, χd, γd, λ₀, lDGAhelper) catch e nothing end;
res_m_sc   = mode == 3 && try λm_sc_correction(χm, γm, χd, γd, λ₀, lDGAhelper) catch e nothing end;
res_dm_sc  = mode == 4 && try λdm_sc_correction(χm, γm, χd, γd, λ₀, lDGAhelper; validation_threshold=1e-6) catch e nothing end;
res_m_tsc  = mode == 5 && try λm_tsc_correction(χm, γm, χd, γd, λ₀, lDGAhelper; max_steps_sc = 200, validation_threshold=1e-6) catch e nothing end;
res_dm_tsc = mode == 6 && try λdm_tsc_correction(χm, γm, χd, γd, λ₀, lDGAhelper; max_steps_sc = 200, validation_threshold=1e-6) catch e nothing end;

#= performance omparison to clean version
using TimerOutputs
to = TimerOutput()

@timeit to "clean  sc" r1 = LadderDGA.LambdaCorrection.λdm_sc_correction_clean(χm, γm, χd, γd, λ₀, lDGAhelper; validation_threshold=1e-6);
@timeit to "fast   sc" r2 = LadderDGA.LambdaCorrection.λdm_sc_correction(χm, γm, χd, γd, λ₀, lDGAhelper; validation_threshold=1e-6);

@timeit to "clean tsc" r3 = LadderDGA.LambdaCorrection.λdm_tsc_correction_clean(χm, γm, χd, γd, λ₀, lDGAhelper; validation_threshold=1e-6);
@timeit to "fast  tsc" r4 = LadderDGA.LambdaCorrection.λdm_tsc_correction(χm, γm, χd, γd, λ₀, lDGAhelper; validation_threshold=1e-6);

=#
if length(ARGS) > 1
    fname = ARGS[2]
    jldopen(fname, "w") do f
        f["chi_m"] = χm
        f["chi_d"] = χd
        f["res_m"] = res_m
        f["res_dm"] = res_dm
        f["res_m_sc"] = res_m_sc
        f["res_dm_sc"] = res_dm_sc
        f["res_m_tsc"] = res_m_tsc
        f["res_dm_tsc"] = res_dm_tsc
    end
end
