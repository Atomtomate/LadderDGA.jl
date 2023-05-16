# ==================== Includes ====================
using Distributed

#nprocs() == 1 && addprocs(8)
@everywhere using Pkg
@everywhere path = joinpath(abspath(@__DIR__),"..")
@everywhere println("activating: ", path)
@everywhere Pkg.activate(path)
@everywhere using LadderDGA
using Plots
using LaTeXStrings
using JLD2


# =================== Functions ====================
@everywhere function gen_sc(λ::Tuple{Float64,Float64}; maxit::Int=100, with_tsc::Bool=false)
    λm,λd = λ
    χ_λ!(χm, λm)
    χ_λ!(χd, λd)
    res = run_sc(χm, γm, χd, γd, λ₀, lDGAhelper; type=:O, maxit=maxit, mixing=0.2, conv_abs=1e-8, trace=true, update_χ_tail=with_tsc)
    reset!(χd)
    reset!(χm)
    return res
end

@everywhere function gen_sc_grid(λ_grid::AbstractMatrix{Tuple{Float64,Float64}}; maxit::Int=100, with_tsc::Bool=false)
    results = λ_result[]

    i = 1
    total = length(λm_grid)*length(λd_grid)
    for λ in λ_grid
        print("\r $(round(100.0*i/total,digits=2)) % done λ = $λ")
        res = gen_sc(λ, maxit=maxit, with_tsc=with_tsc)
        push!(results, res)
        i += 1
    end
    results = reshape(results, length(λm_grid), length(λd_grid))
    return results
end

gen_EPot_diff(result::λ_result) = result.EPot_p1 - result.EPot_p2
gen_PP_diff(result::λ_result) = result.PP_p1 - result.EPot_p2


# ====================== lDGA ======================
dir = dirname(@__FILE__)
dir = joinpath(dir, "../test/test_data/config_b1u2.toml")
dir = joinpath(dir, "config_b1u2.toml") #/home/julian/Hamburg/ED_data/U1b14m0.4_ldmsc_test/config.toml")
cfg_file = joinpath(dir)
LadderDGA.clear_wcache!()
wp, mP, sP, env, kGridsStr = readConfig(cfg_file);
lDGAhelper = setup_LDGA(kGridsStr[1], mP, sP, env);
bubble     = calc_bubble(lDGAhelper);
χm, γm = calc_χγ(:m, lDGAhelper, bubble);
χd, γd = calc_χγ(:d, lDGAhelper, bubble);
λ₀ = calc_λ0(bubble, lDGAhelper)


# ==================== Results =====================
λm_grid = -0.5:0.2:0.5
λd_grid = -0.5:0.2:1.5
λ_grid = collect(Base.product(λm_grid, λd_grid))
println("λdm grid")
results_0sc = gen_sc_grid(λ_grid, maxit=0);
println("sc grid")
results = gen_sc_grid(λ_grid);
println("tsc grid")
results_tsc = gen_sc_grid(λ_grid, maxit=100, with_tsc=true);

EPot_diff_grid     = map(gen_EPot_diff, results)
PP_diff_grid       = map(gen_PP_diff, results)
EPot_diff_grid_0sc = map(gen_EPot_diff, results_0sc)
PP_diff_grid_0sc   = map(gen_PP_diff, results_0sc)
EPot_diff_grid_tsc = map(gen_EPot_diff, results_tsc)
PP_diff_grid_tsc   = map(gen_PP_diff, results_tsc)


println("λdm")

λdm_res = try λdm_correction(χm, γm, χd, γd, λ₀, lDGAhelper; λ_val_only=false, sc_max_it=0, update_χ_tail=false) catch; nothing end
println("λdm sc")
λdm_sc_res = try λdm_correction(χm, γm, χd, γd, λ₀, lDGAhelper; λ_val_only=false, sc_max_it=100, update_χ_tail=false) catch; nothing end
println("λdm tsc")
λdm_tsc_res = try λdm_correction(χm, γm, χd, γd, λ₀, lDGAhelper; λ_val_only=false, sc_max_it=100, update_χ_tail=true) catch; nothing end

abs_diff = abs.(transpose(EPot_diff_grid)) .+ abs.(transpose(PP_diff_grid))
p1 = heatmap(λm_grid, λd_grid, transpose(PP_diff_grid),  clims=(-0.01,0.01), xlabel=L"\lambda_\mathrm{m}", ylabel=L"\lambda_\mathrm{d}", title=L"\sum_{q,\omega} \chi^\omega_{q,\!\!\uparrow\!\!\uparrow} - \frac{n}{2}\left(1-\frac{n}{2}\right)")
p2 = heatmap(λm_grid, λd_grid, transpose(EPot_diff_grid),clims=(-0.01,0.01), xlabel=L"\lambda_\mathrm{m}", ylabel=L"\lambda_\mathrm{d}", title=L"E^{(2)}_\mathrm{pot} - E^{(1)}_\mathrm{pot}")
p3 = heatmap(λm_grid, λd_grid, abs_diff, clims=(-0.01,0.01), xlabel=L"\lambda_\mathrm{m}", ylabel=L"\lambda_\mathrm{d}", title=L"\mathrm{Res} = |E^{(2)}_\mathrm{pot} - E^{(1)}_\mathrm{pot}| + |\sum_{q,\omega} \chi^\omega_{q,\!\!\uparrow\!\!\uparrow} - \frac{n}{2}\left(1-\frac{n}{2}\right)|")
p1_diff = heatmap(λm_grid, λd_grid, abs_diff .- abs_diff_0sc, clims=(-0.01,0.01), xlabel=L"\lambda_\mathrm{m}", ylabel=L"\lambda_\mathrm{d}", title=L"\Delta \mathrm{Res}")

abs_diff_0sc = abs.(transpose(EPot_diff_grid_0sc)) .+ abs.(transpose(PP_diff_grid_0sc))
p1_0sc = heatmap(λm_grid, λd_grid, transpose(PP_diff_grid_0sc),  clims=(-0.01,0.01), xlabel=L"\lambda_\mathrm{m}", ylabel=L"\lambda_\mathrm{d}", title=L"\sum_{q,\omega} \chi^\omega_{q,\!\!\uparrow\!\!\uparrow} - \frac{n}{2}\left(1-\frac{n}{2}\right)")
p2_0sc = heatmap(λm_grid, λd_grid, transpose(EPot_diff_grid_0sc),clims=(-0.01,0.01), xlabel=L"\lambda_\mathrm{m}", ylabel=L"\lambda_\mathrm{d}", title=L"E^{(2)}_\mathrm{pot} - E^{(1)}_\mathrm{pot}")
p3_0sc = heatmap(λm_grid, λd_grid, abs_diff_0sc, clims=(-0.01,0.01), xlabel=L"\lambda_\mathrm{m}", ylabel=L"\lambda_\mathrm{d}", title=L"\mathrm{Res} = |E^{(2)}_\mathrm{pot} - E^{(1)}_\mathrm{pot}| + |\sum_{q,\omega} \chi^\omega_{q,\!\!\uparrow\!\!\uparrow} - \frac{n}{2}\left(1-\frac{n}{2}\right)|")

abs_diff_tsc = abs.(transpose(EPot_diff_grid_tsc)) .+ abs.(transpose(PP_diff_grid_tsc))
p1_tsc = heatmap(λm_grid, λd_grid, transpose(PP_diff_grid_tsc),  clims=(-0.01,0.01), xlabel=L"\lambda_\mathrm{m}", ylabel=L"\lambda_\mathrm{d}", title=L"\sum_{q,\omega} \chi^\omega_{q,\!\!\uparrow\!\!\uparrow} - \frac{n}{2}\left(1-\frac{n}{2}\right)")
p2_tsc = heatmap(λm_grid, λd_grid, transpose(EPot_diff_grid_tsc),clims=(-0.01,0.01), xlabel=L"\lambda_\mathrm{m}", ylabel=L"\lambda_\mathrm{d}", title=L"E^{(2)}_\mathrm{pot} - E^{(1)}_\mathrm{pot}")
p3_tsc = heatmap(λm_grid, λd_grid, abs_diff_tsc, clims=(-0.01,0.01), xlabel=L"\lambda_\mathrm{m}", ylabel=L"\lambda_\mathrm{d}", title=L"\mathrm{Res} = |E^{(2)}_\mathrm{pot} - E^{(1)}_\mathrm{pot}| + |\sum_{q,\omega} \chi^\omega_{q,\!\!\uparrow\!\!\uparrow} - \frac{n}{2}\left(1-\frac{n}{2}\right)|")


jldopen("sc_grids.jld2", "w") do f
    f["lDGAHelper"] = lDGAhelper
    f["results_sc"]    = results
    f["results_0sc"]   = results_0sc
    f["results_tsc"]   = results_tsc
    f["λ_dm"] = λdm_res 
    f["λ_dm_sc"] = λdm_sc_res 
    f["λ_dm_tsc"] = λdm_tsc_res 
end
nothing
