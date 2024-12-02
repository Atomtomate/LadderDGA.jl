using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using LadderDGA
using JLD2

cfg_file = ARGS[1]
out_file = ARGS[2]
wp, mP, sP, env, kGridsStr = readConfig(cfg_file);
lDGAhelper = setup_LDGA((kGridsStr[1][1],4), mP, sP, env, silent=true);
Σ_nl = calc_local_EoM(lDGAhelper)
Σ_loc = lDGAhelper.Σ_loc[axes(Σ_nl,1)]

jldopen(out_file, "w") do f
    f["indices"] = axes(Σ_loc,2)
    f["Sigma_Imp"] = Σ_loc.parent
    f["Sigma_nl"] = Σ_nl.parent
end
