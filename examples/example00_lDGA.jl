dir = dirname(@__FILE__)

using Pkg; Pkg.activate(joinpath(dir, ".."));
using LadderDGA

#cfg_file_3Dsc = joinpath(dir, "../test/test_data/config_b1u2.toml")
#cfg_file_2Dsc = joinpath(dir, "../test/test_data/config_2D.toml")
cfg_file = ARGS[1]

wp, mP, sP, env, kGridsStr = readConfig(cfg_file);
lDGAhelper = setup_LDGA(kGridsStr[1], mP, sP, env, silent=false);
bubble     = calc_bubble(:DMFT, lDGAhelper);
χm, γm = calc_χγ(:m, lDGAhelper, bubble, ω_symmetric=true, use_threads=true);
χd, γd = calc_χγ(:d, lDGAhelper, bubble, ω_symmetric=true, use_threads=true);
check_χ_health(χm, :m, lDGAhelper; q0_check_eps = 0.1, λmin_check_eps = 1000)
check_χ_health(χd, :d, lDGAhelper; q0_check_eps = 0.1, λmin_check_eps = 1000)
χm2 = fix_χr(χm, negative_eps = 1e-2)
χd2 = fix_χr(χd, negative_eps = 1e-2);
λ₀ = calc_λ0(bubble, lDGAhelper);

#res_m      = λm_correction(χm, γm, χd, γd, λ₀, lDGAhelper); print(res_m);
#res_dm     = λdm_correction(χm, γm, χd, γd, λ₀, lDGAhelper); print(res_dm);
    
#se =  calc_Σ(χm, γm, χd, γd, λ₀, lDGAhelper; λm = 0.0, λd = 0.0, tc = ΣTail_Full)
#se_ntc = calc_Σ(χm, γm, χd, γd, λ₀, lDGAhelper; λm = 0.0, λd = 0.0, tc = ΣTail_Plain);