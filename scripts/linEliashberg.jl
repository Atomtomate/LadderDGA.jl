# ==================== Includes ====================
#nprocs() == 1 && addprocs(8)
include("memInfo.jl")

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

file_name = "test_subSample_LinEliashberg_NK$(Nk)_Nw$(Nω).jld2"
output_file = joinpath(out_dir,file_name)

lDGA_fname = "lDGA_NK$(Nk)_Nw$(Nω)_res.jld"
χm, χd, γm, γd, bubble, λ₀, res_m, res_dm, res_dm_sc = nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing

if isfile(joinpath(out_dir,lDGA_fname))
    println("Found existing lDΓA results, loading from file!")
    jldopen(lDGA_fname, "r") do f
        global bubble = f["bubble"]
        global χm = f["chi_m"]
        global χd = f["chi_d"]
        global γm = f["gamma_m"]
        global γd = f["gamma_d"]
        global λ₀ = f["lambda0"]
        global res_m     = f["res_m"]
        global res_dm    = f["res_dm"]
        global res_dm_sc = f["res_dm_sc"]
    end
else
    println("No lDΓA results found, calculating from scratch!")
    if isfile(output_file)
        @warn "Output file exists!"
    end
    println("output file location: ", output_file)
    flush(stdout)


    # ====================== lDGA ======================
    bubble     = calc_bubble(:DMFT, lDGAhelper);
    χm, γm = calc_χγ(:m, lDGAhelper, bubble);
    χd, γd = calc_χγ(:d, lDGAhelper, bubble);
    λ₀ = calc_λ0(bubble, lDGAhelper)


    # =================== λ Results =====================
    res_m = LadderDGA.λ_correction(:m, χm, γm, χd, γd, λ₀, lDGAhelper)
    res_dm = λdm_correction(χm, γm, χd, γd, λ₀, lDGAhelper; fit_μ=true)
    res_dm_sc = run_sc(χm, γm, χd, γd, λ₀, lDGAhelper.mP.μ, lDGAhelper; type=:pre_dm, fit_μ=true, maxit=100, mixing=0.2, conv_abs=1e-8, trace=true);

    jldopen(lDGA_fname, "w") do f
        f["bubble"] = bubble
        f["chi_m"]  = χm
        f["chi_d"]  = χd
        f["gamma_m"] = γm
        f["gamma_d"] = γd
        f["lambda0"] = λ₀
        f["res_m"]     = res_m
        f["res_dm"]    = res_dm
        f["res_dm_sc"] = res_dm_sc
    end
end

# =========== Calculation of F_ladder pp ============
println(" ========== Step 01 ========== ")
meminfo_julia()
#meminfo_procfs()
λ_list = calc_λmax_linEliashberg(bubble, χm, χd, γm, γd, lDGAhelper, env; max_Nk=10)
println(" ========== Step 02 ========== ")
meminfo_julia()
#meminfo_procfs()
λ_m_gLoc_list = calc_λmax_linEliashberg(bubble, χ_λ(χm, res_m.λm), χ_λ(χd,res_m.λd), γm, γd, lDGAhelper, env; max_Nk=10)
println(" ========== Step 03 ========== ")
meminfo_julia()
#meminfo_procfs()
λ_m_list = calc_λmax_linEliashberg(bubble, χ_λ(χm, res_dm_sc.λm), χ_λ(χd,res_dm_sc.λd), γm, γd, lDGAhelper, env; GF=res_m.G_ladder, max_Nk=10)
println(" ========== Step 04 ========== ")
meminfo_julia()
#meminfo_procfs()
λ_dm_gLoc_list = calc_λmax_linEliashberg(bubble, χ_λ(χm, res_dm.λm), χ_λ(χd,res_dm.λd), γm, γd, lDGAhelper, env; max_Nk=10)
println(" ========== Step 05 ========== ")
meminfo_julia()
#meminfo_procfs()
λ_dm_list = calc_λmax_linEliashberg(bubble, χ_λ(χm, res_dm_sc.λm), χ_λ(χd,res_dm_sc.λd), γm, γd, lDGAhelper, env; GF=res_dm.G_ladder, max_Nk=10)
println(" ========== Step 06 ========== ")
meminfo_julia()
#meminfo_procfs()
λ_dm_sc_gLoc_list = calc_λmax_linEliashberg(bubble, χ_λ(χm, res_dm_sc.λm), χ_λ(χd,res_dm_sc.λd), γm, γd, lDGAhelper, env; max_Nk=10)
println(" ========== Step 07 ========== ")
meminfo_julia()
#meminfo_procfs()
λ_dm_sc_list = calc_λmax_linEliashberg(bubble, χ_λ(χm, res_dm_sc.λm), χ_λ(χd,res_dm_sc.λd), γm, γd, lDGAhelper, env; GF=res_dm_sc.G_ladder, max_Nk=10)
println(" ========== Step 08 ========== ")
meminfo_julia()
#meminfo_procfs()

jldopen(joinpath(out_dir,file_name), "w") do f
    f["lDGAHelper"] = lDGAhelper
    f["χ0"] = bubble
    f["χm"] = χm
    f["χd"] = χd
    f["res_m"] = res_m
    f["res_dm"] = res_dm
    f["res_dm_sc"] = res_dm_sc
    f["λ_list_DMFT"] = map(x->x[1], λ_list)
    f["λ_list_lDGA_m"] = map(x->x[1], λ_m_list)
    f["λ_list_lDGA_dm"] = map(x->x[1], λ_dm_list)
    f["λ_list_lDGA_dm_sc_gLoc"] = map(x->x[1], λ_dm_sc_gLoc_list)
    f["λ_list_lDGA_dm_sc"] = map(x->x[1], λ_dm_sc_list)
end

