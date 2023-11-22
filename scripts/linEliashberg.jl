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

force_new=true
max_Nk=4

cfg = ARGS[1]
out_dir = splitdir(cfg)[1]

wp, mP, sP, env, kGridsStr = readConfig(cfg);
lDGAhelper = setup_LDGA(kGridsStr[1], mP, sP, env);
Nk = lDGAhelper.kG.Ns
Nω = 2*lDGAhelper.sP.n_iω

file_name = "run1_subSample_LinEliashberg_NK$(Nk)_Nw$(Nω)_SubS$(max_Nk).jld2"
output_file = joinpath(out_dir,file_name)

lDGA_fname = "lDGA_NK$(Nk)_Nw$(Nω)_res.jld"
# χm, χd, γm, γd, bubble, λ₀, res_m, res_dm, res_dm_sc = nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing

# if false #!force_new && isfile(joinpath(out_dir,lDGA_fname))
#     println("Found existing lDΓA results, loading from file!")
#     jldopen(lDGA_fname, "r") do f
#         global bubble = f["bubble"]
#         global χm = f["chi_m"]
#         global χd = f["chi_d"]
#         global γm = f["gamma_m"]
#         global γd = f["gamma_d"]
#         global λ₀ = f["lambda0"]
#         global res_m     = f["res_m"]
#         global res_dm    = f["res_dm"]
#         global res_dm_sc = f["res_dm_sc"]
#     end
# else
#     println("No lDΓA results found, calculating from scratch!")
#     if false #isfile(output_file)
#         @warn "Output file exists!"
#     end
#     println("output file location: ", output_file)


#     jldopen(lDGA_fname, "w") do f
#         f["bubble"] = bubble
#         f["chi_m"]  = χm
#         f["chi_d"]  = χd
#         f["gamma_m"] = γm
#         f["gamma_d"] = γd
#         f["lambda0"] = λ₀
#         f["res_m"]     = res_m
#         f["res_dm"]    = res_dm
#         f["res_dm_sc"] = res_dm_sc
#     end
# end

    # ====================== lDGA ======================
    bubble     = calc_bubble(:DMFT, lDGAhelper);
    χm, γm = calc_χγ(:m, lDGAhelper, bubble);
    χd, γd = calc_χγ(:d, lDGAhelper, bubble);
    λ₀ = calc_λ0(bubble, lDGAhelper)


    # =================== λ Results =====================
    res_m = LadderDGA.λ_correction(:m, χm, γm, χd, γd, λ₀, lDGAhelper)
    res_dm = λdm_correction(χm, γm, χd, γd, λ₀, lDGAhelper; fit_μ=true)
    res_dm_sc = run_sc(χm, γm, χd, γd, λ₀, lDGAhelper.mP.μ, lDGAhelper; type=:pre_dm, fit_μ=true, maxit=100, mixing=0.2, conv_abs=1e-8, trace=true);


# =========== Calculation of F_ladder pp ============

lDGAhelper_Ur = deepcopy(lDGAhelper)
lDGAhelper_Ur.Γ_m[:,:,:] = lDGAhelper_Ur.Γ_m[:,:,:] .- (-lDGAhelper_Ur.mP.U / lDGAhelper_Ur.mP.β^2)
lDGAhelper_Ur.Γ_d[:,:,:] = lDGAhelper_Ur.Γ_d[:,:,:] .- ( lDGAhelper_Ur.mP.U / lDGAhelper_Ur.mP.β^2)

println("Generation generalized Susceptibility")
χm_star_gen = calc_gen_χ(lDGAhelper_Ur.Γ_m, bubble, lDGAhelper_Ur.kG);
χd_star_gen = calc_gen_χ(lDGAhelper_Ur.Γ_d, bubble, lDGAhelper_Ur.kG);

println(" ========== Step 01 ========== ")
meminfo_julia()
#meminfo_procfs()
ri = res_m
λ_m_list = (ri.converged && ri.sc_converged) ? calc_λmax_linEliashberg(bubble, χ_λ(χm, ri.λm), χ_λ(χd, ri.λd), γm, γd, lDGAhelper, env; GF=ri.G_ladder, max_Nk=max_Nk, χm_star_gen=χm_star_gen, χd_star_gen=χd_star_gen) : [NaN, NaN]
GC.gc()
println(" ========== Step 02 ========== ")
meminfo_julia()
λ_m_gLoc_list = (ri.converged && ri.sc_converged) ? calc_λmax_linEliashberg(bubble, χ_λ(χm, ri.λm), χ_λ(χd,ri.λd), γm, γd, lDGAhelper, env; max_Nk=max_Nk, χm_star_gen=χm_star_gen, χd_star_gen=χd_star_gen) : [NaN, NaN]
GC.gc()
println(" ========== Step 03 ========== ")
meminfo_julia()
ri = res_dm
λ_dm_list = (ri.converged && ri.sc_converged) ? calc_λmax_linEliashberg(bubble, χ_λ(χm, ri.λm), χ_λ(χd,ri.λd), γm, γd, lDGAhelper, env; GF=ri.G_ladder, max_Nk=max_Nk, χm_star_gen=χm_star_gen, χd_star_gen=χd_star_gen) : [NaN, NaN]
GC.gc()
println(" ========== Step 04 ========== ")
meminfo_julia()
λ_dm_gLoc_list = (ri.converged && ri.sc_converged) ? calc_λmax_linEliashberg(bubble, χ_λ(χm, ri.λm), χ_λ(χd,ri.λd), γm, γd, lDGAhelper, env; max_Nk=max_Nk, χm_star_gen=χm_star_gen, χd_star_gen=χd_star_gen) : [NaN, NaN]
GC.gc()
println(" ========== Step 05 ========== ")
meminfo_julia()
ri = res_dm_sc
λ_dm_sc_list = (ri.converged && ri.sc_converged) ? calc_λmax_linEliashberg(bubble, χ_λ(χm, ri.λm), χ_λ(χd,ri.λd), γm, γd, lDGAhelper, env; GF=ri.G_ladder, max_Nk=max_Nk, χm_star_gen=χm_star_gen, χd_star_gen=χd_star_gen) : [NaN, NaN]
GC.gc()
println(" ========== Step 06 ========== ")
meminfo_julia()
λ_dm_sc_gLoc_list = (ri.converged && ri.sc_converged) ? calc_λmax_linEliashberg(bubble, χ_λ(χm, ri.λm), χ_λ(χd,ri.λd), γm, γd, lDGAhelper, env; max_Nk=max_Nk, χm_star_gen=χm_star_gen, χd_star_gen=χd_star_gen) : [NaN, NaN]
GC.gc()
println(" ========== Step 07 ========== ")
meminfo_julia()
#meminfo_procfs()

jldopen(joinpath(out_dir,file_name), "w") do f
    f["lDGAHelper"] = lDGAhelper
    f["subSampling"] = max_Nk
    f["χ0"] = bubble
    f["χm"] = χm
    f["χd"] = χd
    f["res_m"] = res_m
    f["res_dm"] = res_dm
    f["res_dm_sc"] = res_dm_sc
    f["λ_list_lDGA_m_gLoc"] = map(x->x[1], λ_m_gLoc_list)
    f["λ_list_lDGA_m"] = map(x->x[1], λ_m_list)
    f["λ_list_lDGA_dm_gLoc"] = map(x->x[1], λ_dm_gLoc_list)
    f["λ_list_lDGA_dm"] = map(x->x[1], λ_dm_list)
    f["λ_list_lDGA_dm_sc_gLoc"] = map(x->x[1], λ_dm_sc_gLoc_list)
    f["λ_list_lDGA_dm_sc"] = map(x->x[1], λ_dm_sc_list)
end

