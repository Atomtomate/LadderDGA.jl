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

output_file = joinpath(out_dir,"ldm_sc.jld2")
println("output file location: ", output_file)
flush(stdout)


gen_EPot_diff(result::λ_result) = result.EPot_p1 - result.EPot_p2
gen_PP_diff(result::λ_result) = result.PP_p1 - result.EPot_p2


# ====================== lDGA ======================
wp, mP, sP, env, kGridsStr = readConfig(cfg);
lDGAhelper = setup_LDGA(kGridsStr[1], mP, sP, env);
bubble     = calc_bubble(lDGAhelper);
χm, γm = calc_χγ(:m, lDGAhelper, bubble);
χd, γd = calc_χγ(:d, lDGAhelper, bubble);
λ₀ = calc_λ0(bubble, lDGAhelper)


# ==================== Results =====================
res_dm = λdm_correction(χm, γm, χd, γd, λ₀, lDGAhelper; fit_μ=false)
res_dm_sc = run_sc(χm, γm, χd, γd, λ₀, lDGAhelper.mP.μ, lDGAhelper; type=:pre_dm, fit_μ=true, maxit=80, mixing=0.2, conv_abs=1e-8, trace=true);
Nk = lDGAhelper.kG.Ns
Nω = 2*lDGAhelper.sP.n_iω
jldopen(joinpath(out_dir,"new_res_ldga_NK$(Nk)_Nw$(Nω).jld2"), "w") do f
    f["lDGAHelper"] = lDGAhelper
    f["χ0"] = bubble
    f["χm"] = χm
    f["χd"] = χd
    f["res_dm"] = res_dm
    f["res_dm_sc"] = res_dm_sc
end