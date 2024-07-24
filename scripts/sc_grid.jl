# ==================== Includes ====================
using Distributed

#nprocs() == 1 && addprocs(8)
@everywhere using Pkg
@everywhere path = joinpath(abspath(@__DIR__),"..")
@everywhere println("activating: ", path)
@everywhere Pkg.activate(path)
@everywhere using LadderDGA
# using Plots
# using LaTeXStrings
using JLD2

cfg = ARGS[1]
out_dir = ARGS[2]

output_file = joinpath(out_dir,"sc_grids.jld2")
println("output file location: ", output_file)
flush(stdout)



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
wp, mP, sP, env, kGridsStr = readConfig(cfg);
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
results_0sc = false # gen_sc_grid(λ_grid, maxit=0);
println("sc grid")
results = false #gen_sc_grid(λ_grid);
println("tsc grid")
results_tsc = false #gen_sc_grid(λ_grid, maxit=100, with_tsc=true);

# EPot_diff_grid     = map(gen_EPot_diff, results)
#nothing PP_diff_grid       = map(gen_PP_diff, results)
# EPot_diff_grid_0sc = map(gen_EPot_diff, results_0sc)
# PP_diff_grid_0sc   = map(gen_PP_diff, results_0sc)
# EPot_diff_grid_tsc = map(gen_EPot_diff, results_tsc)
# PP_diff_grid_tsc   = map(gen_PP_diff, results_tsc)



Nk = lDGAhelper.kG.Ns
Nω = 2*lDGAhelper.sP.n_iω

Σ_dmft    =  calc_Σ(χm, γm, χd, γd, λ₀, lDGAhelper);
jldopen(joinpath(out_dir,"res_ldga_NK$(Nk)_Nw$(Nω).jld2"), "w") do f
    f["lDGAHelper"] = lDGAhelper
    f["χ0"] = bubble
    f["χm"] = χm
    f["χd"] = χd
    f["Σ_dmft"] = Σ_dmft
end

λdm_res = try λdm_correction(χm, γm, χd, γd, λ₀, lDGAhelper; λ_val_only=false, sc_max_it=0, update_χ_tail=false, verbose=true) catch; nothing end
Σ_dm    = isnothing(λdm_res) ? nothing : calc_Σ(χm, γm, χd, γd, λ₀, lDGAhelper, λm=λdm_res.λm, λd=λdm_res.λd);

jldopen(joinpath(out_dir,"res_dm_NK$(Nk)_Nw$(Nω).jld2"), "w") do f
    f["λ_dm"] = λdm_res 
    f["Σ_dm"] = Σ_dm
end
println("λdm sc")
λdm_sc_res = try λdm_correction(χm, γm, χd, γd, λ₀, lDGAhelper; λ_val_only=false, sc_max_it=150, update_χ_tail=false, verbose=true) catch; nothing end
Σ_dm_sc    = isnothing(λdm_sc_res) ? nothing : calc_Σ(χm, γm, χd, γd, λ₀, lDGAhelper, λm=λdm_sc_res.λm, λd=λdm_sc_res.λd);
jldopen(joinpath(out_dir,"res_dm_sc_NK$(Nk)_Nw$(Nω).jld2"), "w") do f
    f["λ_dm_sc"] = λdm_sc_res 
    f["Σ_dm_sc"] = Σ_dm_sc
end
println("λdm tsc")
λdm_tsc_res = try λdm_correction(χm, γm, χd, γd, λ₀, lDGAhelper; λ_val_only=false, sc_max_it=150, update_χ_tail=true, verbose=true) catch; nothing end
Σ_dm_tsc    = isnothing(λdm_tsc_res) ? nothing : calc_Σ(χm, γm, χd, γd, λ₀, lDGAhelper, λm=λdm_tsc_res.λm, λd=λdm_tsc_res.λd);
jldopen(joinpath(out_dir,"res_dm_tsc_NK$(Nk)_Nw$(Nω).jld2"), "w") do f
    f["λ_dm_tsc"] = λdm_tsc_res 
    f["Σ_dm_tsc"] = Σ_dm_tsc
end

# abs_diff = abs.(transpose(EPot_diff_grid)) .+ abs.(transpose(PP_diff_grid))
# p1 = heatmap(λm_grid, λd_grid, transpose(PP_diff_grid),  clims=(-0.01,0.01), xlabel=L"\lambda_\mathrm{m}", ylabel=L"\lambda_\mathrm{d}", title=L"\sum_{q,\omega} \chi^\omega_{q,\!\!\uparrow\!\!\uparrow} - \frac{n}{2}\left(1-\frac{n}{2}\right)")
# p2 = heatmap(λm_grid, λd_grid, transpose(EPot_diff_grid),clims=(-0.01,0.01), xlabel=L"\lambda_\mathrm{m}", ylabel=L"\lambda_\mathrm{d}", title=L"E^{(2)}_\mathrm{pot} - E^{(1)}_\mathrm{pot}")
# p3 = heatmap(λm_grid, λd_grid, abs_diff, clims=(-0.01,0.01), xlabel=L"\lambda_\mathrm{m}", ylabel=L"\lambda_\mathrm{d}", title=L"\mathrm{Res} = |E^{(2)}_\mathrm{pot} - E^{(1)}_\mathrm{pot}| + |\sum_{q,\omega} \chi^\omega_{q,\!\!\uparrow\!\!\uparrow} - \frac{n}{2}\left(1-\frac{n}{2}\right)|")
# p1_diff = heatmap(λm_grid, λd_grid, abs_diff .- abs_diff_0sc, clims=(-0.01,0.01), xlabel=L"\lambda_\mathrm{m}", ylabel=L"\lambda_\mathrm{d}", title=L"\Delta \mathrm{Res}")

# abs_diff_0sc = abs.(transpose(EPot_diff_grid_0sc)) .+ abs.(transpose(PP_diff_grid_0sc))
# p1_0sc = heatmap(λm_grid, λd_grid, transpose(PP_diff_grid_0sc),  clims=(-0.01,0.01), xlabel=L"\lambda_\mathrm{m}", ylabel=L"\lambda_\mathrm{d}", title=L"\sum_{q,\omega} \chi^\omega_{q,\!\!\uparrow\!\!\uparrow} - \frac{n}{2}\left(1-\frac{n}{2}\right)")
# p2_0sc = heatmap(λm_grid, λd_grid, transpose(EPot_diff_grid_0sc),clims=(-0.01,0.01), xlabel=L"\lambda_\mathrm{m}", ylabel=L"\lambda_\mathrm{d}", title=L"E^{(2)}_\mathrm{pot} - E^{(1)}_\mathrm{pot}")
# p3_0sc = heatmap(λm_grid, λd_grid, abs_diff_0sc, clims=(-0.01,0.01), xlabel=L"\lambda_\mathrm{m}", ylabel=L"\lambda_\mathrm{d}", title=L"\mathrm{Res} = |E^{(2)}_\mathrm{pot} - E^{(1)}_\mathrm{pot}| + |\sum_{q,\omega} \chi^\omega_{q,\!\!\uparrow\!\!\uparrow} - \frac{n}{2}\left(1-\frac{n}{2}\right)|")

# abs_diff_tsc = abs.(transpose(EPot_diff_grid_tsc)) .+ abs.(transpose(PP_diff_grid_tsc))
# p1_tsc = heatmap(λm_grid, λd_grid, transpose(PP_diff_grid_tsc),  clims=(-0.01,0.01), xlabel=L"\lambda_\mathrm{m}", ylabel=L"\lambda_\mathrm{d}", title=L"\sum_{q,\omega} \chi^\omega_{q,\!\!\uparrow\!\!\uparrow} - \frac{n}{2}\left(1-\frac{n}{2}\right)")
# p2_tsc = heatmap(λm_grid, λd_grid, transpose(EPot_diff_grid_tsc),clims=(-0.01,0.01), xlabel=L"\lambda_\mathrm{m}", ylabel=L"\lambda_\mathrm{d}", title=L"E^{(2)}_\mathrm{pot} - E^{(1)}_\mathrm{pot}")
# p3_tsc = heatmap(λm_grid, λd_grid, abs_diff_tsc, clims=(-0.01,0.01), xlabel=L"\lambda_\mathrm{m}", ylabel=L"\lambda_\mathrm{d}", title=L"\mathrm{Res} = |E^{(2)}_\mathrm{pot} - E^{(1)}_\mathrm{pot}| + |\sum_{q,\omega} \chi^\omega_{q,\!\!\uparrow\!\!\uparrow} - \frac{n}{2}\left(1-\frac{n}{2}\right)|")

nothing
