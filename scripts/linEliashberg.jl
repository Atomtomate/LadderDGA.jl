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

file_name = "test_res_ldga_LinEliashberg_NK$(Nk)_Nw$(Nω).jld2"
output_file = joinpath(out_dir,file_name)
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

# =========== Calculation of F_ladder pp ============
println("TODO: λ-correction for χ!!!!")

λ_list = calc_λmax_linEliashberg(bubble, χm, χd, γm, γd, lDGAhelper, env)
λ_m_list = calc_λmax_linEliashberg(bubble, χ_λ(χm, res_m.λm), χ_λ(χd,res_m.λd), γm, γd, lDGAhelper, env)
λ_dm_list = calc_λmax_linEliashberg(bubble, χ_λ(χm, res_dm.λm), χ_λ(χd,res_dm.λd), γm, γd, lDGAhelper, env)
λ_dm_sc_gLoc_list = calc_λmax_linEliashberg(bubble, χ_λ(χm, res_dm_sc.λm), χ_λ(χd,res_dm_sc.λd), γm, γd, lDGAhelper, env)
λ_dm_sc_list = calc_λmax_linEliashberg(bubble, χ_λ(χm, res_dm_sc.λm), χ_λ(χd,res_dm_sc.λd), γm, γd, lDGAhelper, env; GF=res_dm_sc.G_ladder)

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

