using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using LadderDGA
using JLD2

cfg_file = ARGS[1]

wp, mP, sP, env, kGridsStr = readConfig(cfg_file);
lDGAhelper = setup_LDGA(kGridsStr[1], mP, sP, env, silent=true);
bubble     = calc_bubble(:DMFT, lDGAhelper);
χm, γm = calc_χγ(:m, lDGAhelper, bubble);
χd, γd = calc_χγ(:d, lDGAhelper, bubble);
λ₀ = calc_λ0(bubble, lDGAhelper);

xr_m, fr_m = LadderDGA.LambdaCorrection.PPCond_curve(χm,γm,χd,γd,λ₀, lDGAhelper)

res_m      = λm_correction(χm, γm, χd, γd, λ₀, lDGAhelper)
