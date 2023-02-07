# ==================================================================================================== #
#                                        residualCurves.jl                                             #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
#   Last Edit Date  : 03.10.22                                                                         #
# ----------------------------------------- Description ---------------------------------------------- #
#   Diagnostic tools for residuals of conditions, used to determine the λ-parameters.                  #
# -------------------------------------------- TODO -------------------------------------------------- #
# ==================================================================================================== #

#TODO: the transformation of the search range has been removed for now
# @info "λsp ∈ [$λsp_min, $λsp_max], λch ∈ [$λch_min, $λch_max]"
# trafo_bak(x) = [((λsp_max - λsp_min)/2)*(tanh(x[1])+1) + λsp_min, ((λch_max-λch_min)/2)*(tanh(x[2])+1) + λch_min]
# trafo(x) = x
# @info "After transformation: λsp ∈ [$(trafo(λsp_min)), $(trafo(λsp_max))], λch ∈ [$(trafo(λch_min)), $(trafo(λch_max))]"


# ===================================== Residuals of conditions ======================================

"""
    find_lin_interp_root(xdata::AbstractVector{Float64}, ydata::AbstractVector{Float64})

WARNING: this is a specialized function which assumes strictly monotonic data!
Given sampled `xdata` and `ydata`, find the root using linear interpolation.
Returns estimated `x₀`.

Returns: 
-------------
`x₀` : `Float64`, root of sampled function data.

Arguments:
-------------
- **`xdata`** : x-data of samples from strictly monotonic decreasing function.
- **`ydata`** : y-data of samples from strictly monotonic decreasing function
"""
function find_lin_interp_root(xdata::AbstractVector{Float64}, ydata::AbstractVector{Float64})::Float64
    ind = findlast(x -> x < 0, ydata)
    (ind === nothing) && return NaN
    (ind == 1) && return NaN
    (ind == length(xdata)) && return NaN
    Δx = xdata[ind+1] - xdata[ind]
    Δy = ydata[ind+1] - ydata[ind]
    m = Δy/Δx
    b = ydata[ind]
    xᵢ = xdata[ind]
    x₀ = -b/m + xᵢ
    return x₀
end

"""
    find_root(c2_data::Array{Float64,2})

Determines root (i.e. ``(\\lambda_\\text{sp}, \\lambda_\\text{ch})``) from residual curve computed via [`residuals`](@ref residuals).
The root is determined via linear interpolation from the given data.

Returns: 
-------------
``(\\lambda_\\text{sp}, \\lambda_\\text{ch}, \\text{check})``, roots for ``\\lambda`` values and a check (Boolean). Check is false, if the ``\\lambda`` values could not be determined (`NaN` )


Arguments:
-------------
- **`c2_data`** : `Matrix{Float64}`, data from [`residuals`](@ref residuals).
"""
function find_root(c2_data::Matrix{Float64})
    ydata = (c2_data[5,end] .- c2_data[6,end]) > 0 ? c2_data[5,:] .- c2_data[6,:] : c2_data[6,:] .- c2_data[5,:]
    xdata_sp = c2_data[1,:]
    xdata_ch = c2_data[2,:]
    λsp = find_lin_interp_root(xdata_sp, ydata)
    λch = find_lin_interp_root(xdata_ch, ydata)
    check = (isnan(λsp) || isnan(λch)) ? false : true
    λsp, λch, check
end

"""
    residuals(N_subdivisions::Int, 
            χ_m::χT, γ_m::γT, χ_d::χT, γ_d::γT, Σ_loc::Vector{ComplexF64},
            Gνω::GνqT, λ₀::Array{ComplexF64,3},
            kG::KGrid, mP::ModelParameters, sP::SimulationParameters; 
            method = :newton, 
            νmax::Int = -1, λd_min_δ = 0.1, λd_max = 500,
            maxit::Int = 50, update_χ_tail=false, mixing=0.2, conv_abs=1e-6, par=false)

Arguments:
-------------
- **`method`**: Options are `:newton`, `:bisection`, `:lingrid`. The first two try and esstimate the position of the `0`, while lingrid yields a linear grid of residuals.
"""
function residuals(N_Evals::Int, 
            χ_m::χT, γ_m::γT, χ_d::χT, γ_d::γT, Σ_loc::Vector{ComplexF64},
            Gνω::GνqT, λ₀::Array{ComplexF64,3},
            kG::KGrid, mP::ModelParameters, sP::SimulationParameters; 
            method = :newton,
            νmax::Int = -1, λd_min_δ = 0.1, λd_max = 500,
            maxit::Int = 50, update_χ_tail=false, mixing=0.2, conv_abs=1e-6, par=false)

    # general definitions
    rhs_c1 = mP.n/2 * (1 - mP.n/2)
    λd_min_tmp = get_λ_min(real(χ_d.data)) 
    λd_range = if method != :lingrid 
        [λd_min_tmp + λd_min_δ*abs(λd_min_tmp), λd_max]
    else
        sort(union([0.0], LinRange(λd_min_tmp + λd_min_δ*abs(λd_min_tmp), λd_max, N_Evals-1)))
    end
    χ_m_tmp::χT = deepcopy(χ_m)
    χ_d_tmp::χT = deepcopy(χ_d)

    residuals = zeros(8, length(λd_range))
    si = [1,2]  # index for search bracket
    last_λd = NaN

    for i in 1:N_Evals
        
        if i > length(λd_range)
            if method == :newton
            elseif method == :bisection
                λd_new = abs(residuals[2,si[1]] - residuals[2,si[2]])/2 + residuals[2,si[1]]
                push!(λd_range, λd_new)
            else
                error("Unkown root finding method!")
            end
        end
        println(λd_range, " // [", λd_range[si[1]] , " , ",λd_range[si[2]]  , "]")

        last_λd = λd_i = λd_range[i]
        λm_i, lhs_c1, rhs_c1, lhs_c2, rhs_c2, μ, conv = res_curve_int(
                        λd_i, χ_m, γ_m, χ_d, γ_d, Σ_loc, Gνω, λ₀, kG, mP, sP, 
                        mixing=mixing, maxit=maxit, conv_abs=conv_abs, update_χ_tail=update_χ_tail, par=par)
        if i > 2
            residuals = cat(residuals, [λm_i, λd_i, lhs_c1, rhs_c1, lhs_c2, rhs_c2, μ, conv], dims=2)
        else
            residuals[:,i] = [λm_i, λd_i, lhs_c1, rhs_c1, lhs_c2, rhs_c2, μ, conv]
        end
        χ_m = deepcopy(χ_m_tmp)
        χ_d = deepcopy(χ_d_tmp)

        if i > 2 && method != :lingrid
            res_l = (residuals[5,si[1]] - residuals[6,si[1]])
            res_m = lhs_c2 - rhs_c2
            res_r = (residuals[5,si[2]] - residuals[6,si[2]])
            println("RES: $res_l , $res_m , $res_r")
            if sign(res_l) != sign(res_m)
                si[2] = i
            elseif sign(res_r) != sign(res_m)
                si[1] = i
            else
                println("ERROR: Bisection failed to determine interval!")
                break
            end
        end
    end

    return residuals
end

function res_curve_int(λd_i::Float64, 
            χ_m::χT, γ_m::γT, χ_d::χT, γ_d::γT,
            Σ_loc::Vector{ComplexF64},gLoc_rfft_init::GνqT,
            λ₀::Array{ComplexF64,3}, kG::KGrid, mP::ModelParameters, sP::SimulationParameters;
            maxit=50, mixing=0.2, conv_abs=1e-6, update_χ_tail=false, par=false)

    conv_abs = maxit != 0 ? conv_abs : Inf

    Σ_ladder, gLoc_new, E_kin, E_pot, μnew, λm_i, lhs_c1, lhs_c2, converged = if par 
        run_sc_par(χ_m, γ_m, χ_d, γ_d, λ₀, gLoc_rfft_init, Σ_loc, λd_i, kG, mP, sP, 
               maxit=maxit, mixing=mixing, conv_abs=conv_abs, update_χ_tail=update_χ_tail)
    else
        run_sc(χ_m, γ_m, χ_d, γ_d, λ₀, gLoc_rfft_init, Σ_loc, λd_i, kG, mP, sP, 
               maxit=maxit, mixing=mixing, conv_abs=conv_abs, update_χ_tail=update_χ_tail)
    end
                
    rhs_c1 = mP.n/2 * (1 - mP.n/2)
    rhs_c2 = E_pot/mP.U - (mP.n/2) * (mP.n/2)
    return λm_i, lhs_c1, rhs_c1, lhs_c2, rhs_c2, μnew, converged
end
