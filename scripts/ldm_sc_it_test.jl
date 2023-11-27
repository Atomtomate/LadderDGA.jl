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
out_dir = splitdir(cfg)[1]

wp, mP, sP, env, kGridsStr = readConfig(cfg);
lDGAhelper = setup_LDGA(kGridsStr[1], mP, sP, env);
Nk = lDGAhelper.kG.Ns
Nω = 2*lDGAhelper.sP.n_iω

file_name = "sc_it_ldga_NK$(Nk)_Nw$(Nω).jld2"
output_file = joinpath(out_dir,file_name)
if isfile(output_file)
    println("Output file exists, aborting.")
    exit(1)
end
println("output file location: ", output_file)
flush(stdout)



# ====================== lDGA ======================
bubble     = calc_bubble(:DMFT, lDGAhelper);
χm, γm = calc_χγ(:m, lDGAhelper, bubble);
χd, γd = calc_χγ(:d, lDGAhelper, bubble);
λ₀ = calc_λ0(bubble, lDGAhelper)


# ==================== Results =====================
res_m = LadderDGA.λ_correction(:m, χm, γm, χd, γd, λ₀, lDGAhelper)
res_dm = λdm_correction(χm, γm, χd, γd, λ₀, lDGAhelper; fit_μ=true)
res_dm_sc_list = []
for ii in 1:100
    res_dm_sc = run_sc(χm, γm, χd, γd, λ₀, lDGAhelper.mP.μ, lDGAhelper; type=:pre_dm, fit_μ=true, maxit=ii, mixing=0.2, conv_abs=1e-8, trace=true);
    push!(res_dm_sc_list, res_dm_sc)
    res_dm_sc.sc_converged && break
        
end
if isfinite(res_dm.λm) && isfinite(res_dm.λd)
    jldopen(joinpath(out_dir,file_name), "w") do f
        f["lDGAHelper"] = lDGAhelper
        #f["χ0"] = bubble
        f["χm"] = χm
        f["χd"] = χd
        f["res_m"] = res_m
        f["res_dm"] = res_dm
        f["res_dm_sc_list"] = res_dm_sc_list
    end
end

