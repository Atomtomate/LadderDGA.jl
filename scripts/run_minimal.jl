using Pkg
Pkg.activate(joinpath(@__FILE__,"../.."))
using LadderDGA

cfg = ARGS[1]

wp, mP, sP, env, kGridsStr = readConfig(cfg);
lDGAhelper = setup_LDGA(kGridsStr[1], mP, sP, env);
bubble     = calc_bubble(lDGAhelper);

χm, γm = calc_χγ(:m, lDGAhelper, bubble);
χd, γd = calc_χγ(:d, lDGAhelper, bubble);
λ₀ = calc_λ0(bubble, lDGAhelper);
λdm_res = λdm_correction(χm, γm, χd, γd, λ₀, lDGAhelper; λ_val_only=false, sc_max_it=10, update_χ_tail=false, verbose=true);
