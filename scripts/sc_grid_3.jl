using Pkg
using ProgressMeter
using JLD2
# using GLMakie, LinearAlgebra


cfg = ARGS[1]
const Nm = parse(Int,ARGS[2])
const Nd = parse(Int,ARGS[3])
const Nμ = parse(Int,ARGS[4])
codeDir = ARGS[5]
const version = parse(Int, ARGS[6])
output = ARGS[7]
Pkg.activate(codeDir)
using LadderDGA


# =================== Functions ====================
function gen_param_grid(h::lDΓAHelper, res_λdm::λ_result, Nm::Int, Nd::Int, Nμ::Int;
                        λm_range::Float64=0.2, λd_range::Float64=0.2, μ_range=0.2)
    λm_min = res_λdm.λm - λm_range * abs(res_λdm.λm)
    λd_min = res_λdm.λd - λd_range * abs(res_λdm.λd)
    λm_max = res_λdm.λm + λm_range * abs(res_λdm.λm)
    λd_max = res_λdm.λd + λd_range * abs(res_λdm.λd)
    μ_min  = h.mP.μ - μ_range*h.mP.μ
    μ_max  = h.mP.μ + μ_range*h.mP.μ
    λm_grid = LinRange(λm_min, λm_max,  Nm)
    λd_grid = LinRange(λd_min, λd_max, Nd)
    μ_grid  = LinRange(μ_min, μ_max, Nμ)
    grid = collect(Base.product(λm_grid, λd_grid, μ_grid))
    (any(isnan.(λm_grid)) || any(isnan.(λd_grid))) && error("could not determine λmin")
    return grid
end

function gen_param_grid_full(χm::χT, χd::χT, h::lDΓAHelper, Nm::Int, Nd::Int, Nμ::Int;
                        λm_range::Float64=0.2, λd_range::Float64=0.2, μ_range=0.2)
    λm_min = LadderDGA.LambdaCorrection.get_λ_min(χm)
    λd_min = LadderDGA.LambdaCorrection.get_λ_min(χd)
    λm_max = 10.0
    λd_max = 100.0
    λm_min = λm_min + 1e-4
    λd_min = λd_min + 1e-4
    μ_min  = h.mP.μ - h.mP.μ/4
    μ_max  = h.mP.μ + h.mP.μ/4
    λm_grid = LinRange(λm_min, λm_max, Nm)
    λd_grid = LinRange(λd_min, λd_max, Nd)
    μ_grid  = LinRange(μ_min, μ_max, Nμ)
    grid = collect(Base.product(λm_grid, λd_grid, μ_grid))
    (any(isnan.(λm_grid)) || any(isnan.(λd_grid))) && error("could not determine λmin")
    return grid
end

# types: :O, :m, :dm, :pre, :fix 
function gen_sc_grid(χm::χT, γm::γT, χd::χT, γd::γT, λ₀, lDGAhelper::lDΓAHelper, type, fit_μ, update_tail::Bool, param_grid::Array{Tuple{Float64,Float64,Float64},3})
    res = similar(param_grid)
    fit_params = similar(param_grid) 
    converged = falses(size(res))
    χm_bak = deepcopy(χm)
    χd_bak = deepcopy(χd)
    @showprogress for (i,p) in enumerate(param_grid)
        λm, λd, μ = p
        try
            run_res = run_sc(χm, γm, χd, γd, λ₀, μ, lDGAhelper; type=type, λm, λd, fit_μ=fit_μ, update_χ_tail=update_tail, maxit=20, mixing=0.2, conv_abs=1e-7, trace=false)
            Δpp   = run_res.PP_p1 - run_res.PP_p2
            ΔEPot = run_res.EPot_p1 - run_res.EPot_p2
            Δn    = lDGAhelper.mP.n - run_res.n
            converged[i] = run_res.sc_converged
            fit_params[i] = (run_res.λm, run_res.λd, run_res.μ)
            res[i] = (Δpp, ΔEPot, Δn)
        catch e
            println("error during sc: $e")
            converged[i] = false
            fit_params[i] = (NaN,NaN,NaN)
            res[i] = (NaN,NaN,NaN)
        end
        χm = deepcopy(χm_bak)
        χd = deepcopy(χd_bak)
    end
    res, fit_params, converged
end

function normalize_grid(grid)
    min_p = [minimum(map(x->x[i], grid)) for i in 1:3]
    max_p = [maximum(map(x->x[i] - min_p[i], grid)) for i in 1:3]
    grid_normed = map(x-> (x .- min_p) ./ max_p, grid)
end


# ===================== lDGA =======================
wp, mP, sP, env, kGridsStr = readConfig(cfg);
println("  ===================================  ")
println(mP)
println("  ===================================  ")
lDGAhelper = setup_LDGA(kGridsStr[1], mP, sP, env);
bubble     = calc_bubble(:DMFT, lDGAhelper);
χm, γm = calc_χγ(:m, lDGAhelper, bubble);
χd, γd = calc_χγ(:d, lDGAhelper, bubble);
λ₀ = calc_λ0(bubble, lDGAhelper);
Σ_ladder = calc_Σ(χm, γm, χd, γd, λ₀, lDGAhelper);


# ==================== script ======================

res_λdm = λdm_correction(χm, γm, χd, γd, λ₀, lDGAhelper; fit_μ=false)
# param_grid    = gen_param_grid(lDGAhelper, res_λdm, Nm, Nd, Nμ)
param_grid    = gen_param_grid_full(χm, χd, lDGAhelper, Nm, Nd, Nμ;)
sc_grid, param_sc_grid, converged = if version == 1
    gen_sc_grid(χm, γm, χd, γd, λ₀, lDGAhelper, :fix, false, false, param_grid);
elseif version == 2
    gen_sc_grid(χm, γm, χd, γd, λ₀, lDGAhelper, :fix, true, false, param_grid);
elseif version == 3
    gen_sc_grid(χm, γm, χd, γd, λ₀, lDGAhelper, :pre_dm, false, false, param_grid);
elseif version == 4
    gen_sc_grid(χm, γm, χd, γd, λ₀, lDGAhelper, :pre_dm, true, false, param_grid);
elseif version == 5
    gen_sc_grid(χm, γm, χd, γd, λ₀, lDGAhelper, :pre_m, false, false, param_grid);
elseif version == 6
    gen_sc_grid(χm, γm, χd, γd, λ₀, lDGAhelper, :pre_m, true, false, param_grid);
elseif version == 7
    gen_sc_grid(χm, γm, χd, γd, λ₀, lDGAhelper, :fix, false, true, param_grid);
elseif version == 8
    gen_sc_grid(χm, γm, χd, γd, λ₀, lDGAhelper, :fix, true, true, param_grid);
elseif version == 9
    gen_sc_grid(χm, γm, χd, γd, λ₀, lDGAhelper, :pre_dm, false, true, param_grid);
elseif version == 10
    gen_sc_grid(χm, γm, χd, γd, λ₀, lDGAhelper, :pre_dm, true, true, param_grid);
elseif version == 11
    gen_sc_grid(χm, γm, χd, γd, λ₀, lDGAhelper, :pre_m, false, true, param_grid);
elseif version == 12
    gen_sc_grid(χm, γm, χd, γd, λ₀, lDGAhelper, :pre_m, true, true, param_grid);
end;

param_grid_normalized = normalize_grid(param_grid)
param_sc_grid_normalized = normalize_grid(param_sc_grid)
sc_grid_normalized = normalize_grid(sc_grid);

jldopen(output, "w") do f
    f["res_dm"] = res_λdm
    f["param_grid"] = param_grid
    f["sc_grid"] = sc_grid
    f["param_sc_grid"] = param_sc_grid
    f["param_grid_norm"] = param_grid_normalized
    f["sc_grid_norm"] = sc_grid_normalized
    f["param_sc_grid_norm"] = param_sc_grid_normalized
    f["converged"] = converged
end
