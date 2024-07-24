# ==================== Includes ====================
#nprocs() == 1 && addprocs(8)
using Pkg
path = joinpath(abspath(@__DIR__),"..")
println("activating: ", path)
Pkg.activate(path)
using LadderDGA
# using Plots
# using LaTeXStrings
using JLD2

cfg = ARGS[1]
out_dir = ARGS[2]


wp, mP, sP, env, kGridsStr = readConfig(cfg);
lDGAhelper = setup_LDGA(kGridsStr[1], mP, sP, env);
Nk = lDGAhelper.kG.Ns
Nω = 2*lDGAhelper.sP.n_iω

file_name = "chi_NK$(Nk)_Nw$(Nω).jld2"
output_file = joinpath(out_dir,file_name)
if isfile(output_file)
    println("Output file exists, aborting.")
    exit(1)
end
println("output file location: ", output_file)
flush(stdout)


# ====================== lDGA ======================
bubble     = calc_bubble(lDGAhelper);
χm, γm = calc_χγ(:m, lDGAhelper, bubble);
χd, γd = calc_χγ(:d, lDGAhelper, bubble);


# ==================== Results =====================
jldopen(joinpath(out_dir,file_name), "w") do f
    f["lDGAHelper"] = lDGAhelper
    f["χ0"] = bubble
    f["χm"] = χm
    f["χd"] = χd
end
