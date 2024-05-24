using Pkg
Pkg.activate("$(@__DIR__)/..")
using LadderDGA

wp, mP, sP, env, kGridsStr = readConfig_RPA("examples/example01_RPA.toml")
RPAhelper = setup_RPA!(kGridsStr, mP, sP);
