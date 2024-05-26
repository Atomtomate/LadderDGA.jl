dir = dirname(@__FILE__)

using Pkg; Pkg.activate(joinpath(dir, ".."));
using LadderDGA


cfg_dir = joinpath(dir, "../test/test_data/config_b1u2.toml")
cfg_file = joinpath(cfg_dir)

wp, mP, sP, env, kGridsStr = readConfig(cfg_file);
lDGAhelper = setup_LDGA(kGridsStr[1], mP, sP, env, silent=true);
bubble     = calc_bubble(:DMFT, lDGAhelper);
χm, γm = calc_χγ(:m, lDGAhelper, bubble);
χd, γd = calc_χγ(:d, lDGAhelper, bubble);
λ₀ = calc_λ0(bubble, lDGAhelper);

res_m      = LadderDGA.LambdaCorrection.λm_correction(χm, γm, χd, γd, λ₀, lDGAhelper); print(res_m)
res_dm     = LadderDGA.LambdaCorrection.λdm_correction(χm, γm, χd, γd, λ₀, lDGAhelper); print(res_dm)
res_m_sc   = LadderDGA.LambdaCorrection.λm_sc_correction(χm, γm, χd, γd, λ₀, lDGAhelper); print(res_m_sc)
res_dm_sc  = LadderDGA.LambdaCorrection.λdm_sc_correction(χm, γm, χd, γd, λ₀, lDGAhelper); print(res_dm_sc)
res_m_tsc  = LadderDGA.LambdaCorrection.λm_tsc_correction(χm, γm, χd, γd, λ₀, lDGAhelper; max_steps_sc = 200); print(res_m_tsc)
res_dm_tsc = LadderDGA.LambdaCorrection.λdm_tsc_correction(χm, γm, χd, γd, λ₀, lDGAhelper; max_steps_sc = 200); print(res_dm_tsc)