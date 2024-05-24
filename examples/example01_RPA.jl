using Pkg
Pkg.activate("$(@__DIR__)/..")
using LadderDGA

wp, mP, sP, env, kGridsStr = readConfig_RPA("examples/example01_RPA.toml")
RPAhelper = setup_RPA!(kGridsStr, mP, sP);

bubble     = calc_bubble(:RPA, RPAhelper);
#bubble_RPA = χ₀RPA_T()
χm, γm = calc_χγ(:m, RPAhelper, bubble);
χd, γd = calc_χγ(:d, RPAhelper, bubble);
λ₀ = calc_λ0(bubble, RPAhelper)

res_m = LadderDGA.λ_correction(:m, χm, χd, RPAhelper)
nh = ceil(Int,size(χm,2)/2)
χ_λ(χm, res_m[1])[end, nh] 
