dir = dirname(@__FILE__)

using Pkg; Pkg.activate(joinpath(dir, ".."));
using LadderDGA

#cfg_file_3Dsc = joinpath(dir, "../test/test_data/config_b1u2.toml")
#cfg_file_2Dsc = joinpath(dir, "../test/test_data/config_2D.toml")
cfg_file = ARGS[1]

wp, mP, sP, env, kGridsStr = readConfig(cfg_file);
lDGAhelper = setup_LDGA(kGridsStr[1], mP, sP, env, silent=true);
bubble     = calc_bubble(:DMFT, lDGAhelper);
χm, γm = calc_χγ(:m, lDGAhelper, bubble);
χd, γd = calc_χγ(:d, lDGAhelper, bubble);
λ₀ = calc_λ0(bubble, lDGAhelper);
res_m      = λm_correction(χm, γm, χd, γd, λ₀, lDGAhelper); print(res_m);
res_dm     = λdm_correction(χm, γm, χd, γd, λ₀, lDGAhelper); print(res_dm);
se =  calc_Σ(χm, γm, χd, γd, λ₀, lDGAhelper; λm = 0.0, λd = 0.0, tc = true)
se_ntc = calc_Σ(χm, γm, χd, γd, λ₀, lDGAhelper; λm = 0.0, λd = 0.0, tc = false);