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

cfg = ARGS[1]

# =================== Functions ====================
@everywhere function gen_sc(λ::Tuple{Float64,Float64,Float64}; maxit::Int=100, with_tsc::Bool=false)
    λm,λd,μ = λ
    χ_λ!(χm, λm)
    χ_λ!(χd, λd)
    res = run_sc(χm, γm, χd, γd, λ₀, μ, lDGAhelper; type=:O, maxit=maxit, mixing=0.2, conv_abs=1e-8, trace=true, update_χ_tail=with_tsc)
    reset!(χd)
    reset!(χm)
    res.G_ladder = nothing
    res.Σ_ladder = nothing
    return res
end

@everywhere function gen_sc_grid(λ_grid::Array{Tuple{Float64,Float64,Float64},3}; maxit::Int=100, with_tsc::Bool=false)
    results = λ_result[]

    i = 1
    total = length(λm_grid)*length(λd_grid)*length(μ_grid)
    for λ in λ_grid
        λp = rpad.(lpad.(round.(λ,digits=1),5),5)
        print("\r $(rpad(lpad(round(100.0*i/total,digits=2),5),6)) % done λ = $λp")
        res = gen_sc(λ, maxit=maxit, with_tsc=with_tsc)
        push!(results, res)
        i += 1
    end
    results = reshape(results, length(λm_grid), length(λd_grid), length(μ_grid))
    return results
end

gen_EPot_diff(result::λ_result) = result.EPot_p1 - result.EPot_p2
gen_PP_diff(result::λ_result) = result.PP_p1 - result.EPot_p2


# ====================== lDGA ======================
LadderDGA.clear_wcache!()
wp, mP, sP, env, kGridsStr = readConfig(cfg);
lDGAhelper = setup_LDGA(kGridsStr[1], mP, sP, env);
bubble     = calc_bubble(lDGAhelper);
χm, γm = calc_χγ(:m, lDGAhelper, bubble);
χd, γd = calc_χγ(:d, lDGAhelper, bubble);
λ₀ = calc_λ0(bubble, lDGAhelper)


# ==================== Results =====================
μDMFT = mP.μ
λm_grid = -10.0:2.5:10.0
λd_grid = -10.0:2.5:10.0
μ_grid  = (μDMFT-0.5):0.4:(μDMFT+0.5)
λ_grid = collect(Base.product(λm_grid, λd_grid, μ_grid))
println("\n\nλdm grid:")
results_0sc = gen_sc_grid(λ_grid, maxit=0);
println("\n\nsc grid:")
results = gen_sc_grid(λ_grid);
println("\n\ntsc grid:")
results_tsc = gen_sc_grid(λ_grid, maxit=100, with_tsc=true);

EPot_diff_grid     = map(gen_EPot_diff, results)
PP_diff_grid       = map(gen_PP_diff, results)
EPot_diff_grid_0sc = map(gen_EPot_diff, results_0sc)
PP_diff_grid_0sc   = map(gen_PP_diff, results_0sc)
EPot_diff_grid_tsc = map(gen_EPot_diff, results_tsc)
PP_diff_grid_tsc   = map(gen_PP_diff, results_tsc)


println("λdm")

# λdm_res = try λdm_correction(χm, γm, χd, γd, λ₀, lDGAhelper; λ_val_only=false, sc_max_it=0, update_χ_tail=false, verbose=true) catch; nothing end
# println("λdm sc")
# λdm_sc_res = try λdm_correction(χm, γm, χd, γd, λ₀, lDGAhelper; λ_val_only=false, sc_max_it=100, update_χ_tail=false, verbose=true) catch; nothing end
# println("λdm tsc")
# λdm_tsc_res = try λdm_correction(χm, γm, χd, γd, λ₀, lDGAhelper; λ_val_only=false, sc_max_it=100, update_χ_tail=true, verbose=true) catch; nothing end

# abs_diff = abs.(transpose(EPot_diff_grid)) .+ abs.(transpose(PP_diff_grid))
# p1 = heatmap(λm_grid, λd_grid, transpose(PP_diff_grid),  clims=(-0.01,0.01), xlabel=L"\lambda_\mathrm{m}", ylabel=L"\lambda_\mathrm{d}", title=L"\sum_{q,\omega} \chi^\omega_{q,\!\!\uparrow\!\!\uparrow} - \frac{n}{2}\left(1-\frac{n}{2}\right)")
# p2 = heatmap(λm_grid, λd_grid, transpose(EPot_diff_grid),clims=(-0.01,0.01), xlabel=L"\lambda_\mathrm{m}", ylabel=L"\lambda_\mathrm{d}", title=L"E^{(2)}_\mathrm{pot} - E^{(1)}_\mathrm{pot}")
# p3 = heatmap(λm_grid, λd_grid, abs_diff, clims=(-0.01,0.01), xlabel=L"\lambda_\mathrm{m}", ylabel=L"\lambda_\mathrm{d}", title=L"\mathrm{Res} = |E^{(2)}_\mathrm{pot} - E^{(1)}_\mathrm{pot}| + |\sum_{q,\omega} \chi^\omega_{q,\!\!\uparrow\!\!\uparrow} - \frac{n}{2}\left(1-\frac{n}{2}\right)|")

# abs_diff_0sc = abs.(transpose(EPot_diff_grid_0sc)) .+ abs.(transpose(PP_diff_grid_0sc))
# p1_0sc = heatmap(λm_grid, λd_grid, transpose(PP_diff_grid_0sc),  clims=(-0.01,0.01), xlabel=L"\lambda_\mathrm{m}", ylabel=L"\lambda_\mathrm{d}", title=L"\sum_{q,\omega} \chi^\omega_{q,\!\!\uparrow\!\!\uparrow} - \frac{n}{2}\left(1-\frac{n}{2}\right)")
# p2_0sc = heatmap(λm_grid, λd_grid, transpose(EPot_diff_grid_0sc),clims=(-0.01,0.01), xlabel=L"\lambda_\mathrm{m}", ylabel=L"\lambda_\mathrm{d}", title=L"E^{(2)}_\mathrm{pot} - E^{(1)}_\mathrm{pot}")
# p3_0sc = heatmap(λm_grid, λd_grid, abs_diff_0sc, clims=(-0.01,0.01), xlabel=L"\lambda_\mathrm{m}", ylabel=L"\lambda_\mathrm{d}", title=L"\mathrm{Res} = |E^{(2)}_\mathrm{pot} - E^{(1)}_\mathrm{pot}| + |\sum_{q,\omega} \chi^\omega_{q,\!\!\uparrow\!\!\uparrow} - \frac{n}{2}\left(1-\frac{n}{2}\right)|")
# p1_diff = heatmap(λm_grid, λd_grid, abs_diff .- abs_diff_0sc, clims=(-0.01,0.01), xlabel=L"\lambda_\mathrm{m}", ylabel=L"\lambda_\mathrm{d}", title=L"\Delta \mathrm{Res}")

# abs_diff_tsc = abs.(transpose(EPot_diff_grid_tsc)) .+ abs.(transpose(PP_diff_grid_tsc))
# p1_tsc = heatmap(λm_grid, λd_grid, transpose(PP_diff_grid_tsc),  clims=(-0.01,0.01), xlabel=L"\lambda_\mathrm{m}", ylabel=L"\lambda_\mathrm{d}", title=L"\sum_{q,\omega} \chi^\omega_{q,\!\!\uparrow\!\!\uparrow} - \frac{n}{2}\left(1-\frac{n}{2}\right)")
# p2_tsc = heatmap(λm_grid, λd_grid, transpose(EPot_diff_grid_tsc),clims=(-0.01,0.01), xlabel=L"\lambda_\mathrm{m}", ylabel=L"\lambda_\mathrm{d}", title=L"E^{(2)}_\mathrm{pot} - E^{(1)}_\mathrm{pot}")
# p3_tsc = heatmap(λm_grid, λd_grid, abs_diff_tsc, clims=(-0.01,0.01), xlabel=L"\lambda_\mathrm{m}", ylabel=L"\lambda_\mathrm{d}", title=L"\mathrm{Res} = |E^{(2)}_\mathrm{pot} - E^{(1)}_\mathrm{pot}| + |\sum_{q,\omega} \chi^\omega_{q,\!\!\uparrow\!\!\uparrow} - \frac{n}{2}\left(1-\frac{n}{2}\right)|")
# p1_diff = heatmap(λm_grid, λd_grid, abs_diff .- abs_diff_tsc, clims=(-0.01,0.01), xlabel=L"\lambda_\mathrm{m}", ylabel=L"\lambda_\mathrm{d}", title=L"\Delta \mathrm{Res}")


jldopen("sc_grids_U$(mP.U)_b$(mP.β)_n$(mP.n).jld2", "w") do f
    f["lDGAHelper"] = lDGAhelper
    f["results_sc"]    = results
    f["results_0sc"]   = results_0sc
    f["results_tsc"]   = results_tsc
    # f["λ_dm"] = λdm_res 
    # f["λ_dm_sc"] = λdm_sc_res 
    # f["λ_dm_tsc"] = λdm_tsc_res 
end
nothing
