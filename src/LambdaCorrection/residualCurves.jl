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

function residuals(NPoints_coarse::Int, NPoints_negative::Int, last_λ::Vector{Float64}, 
            χsp::χT, γsp::γT, χch::χT, γch::γT, Σ_loc::Vector{ComplexF64},
            Gνω::GνqT, λ₀::Array{ComplexF64,3},
            kG::KGrid, mP::ModelParameters, sP::SimulationParameters; 
            νmax::Int = -1, maxit::Int = 50, update_χ_tail=false, mixing=0.2, conv_abs=1e-6, par=false)

    # general definitions


    rhs_c1 = mP.n/2 * (1 - mP.n/2)
    λch_min, λch_max = [get_λ_min(real(χch.data)), 10]
    λch_max2 = 200

    λch_range_negative = 10.0.^(range(0,stop=log10(abs(λch_min)+0.9),length=NPoints_negative+2)) .+ λch_min .- 1
    λch_range_coarse = range(0,stop=λch_max,length=NPoints_coarse)
    λch_range_large = 10.0.^(range(0,stop=log10(λch_max2-2*λch_max+1),length=6)) .+ 2*λch_max .- 1
    #last_λch_range = isfinite(last_λ[2]) ? range(last_λ[2] - abs(last_λ[2]*0.1), stop = last_λ[2] + abs(last_λ[2]*0.1), length=8) : []
    λch_range = Float64.(sort(unique(union([0.0], λch_range_negative, λch_range_coarse, λch_range_large))))

    # EoM optimization related definitions

    # preallications
    χsp_tmp::χT = deepcopy(χsp)
    χch_tmp::χT = deepcopy(χch)

    residuals = zeros(8, length(λch_range))
    for (i,λch_i) in enumerate(λch_range)
            λsp_i, lhs_c1, rhs_c1, lhs_c2, rhs_c2, μ, conv = if maxit == 0
            res_curve_int(λch_i, χsp, γsp, χch, γch, Σ_loc, Gνω, λ₀, kG, mP, sP, maxit=0, conv_abs=Inf, par=par)
        else
            res_curve_int(λch_i, χsp, γsp, χch, γch, Σ_loc, Gνω, λ₀, kG, mP, sP, mixing=mixing, maxit=maxit, conv_abs=conv_abs, update_χ_tail=update_χ_tail, par=par)
        end

        residuals[:,i] = [λsp_i, λch_i, lhs_c1, rhs_c1, lhs_c2, rhs_c2, μ, conv]
        χsp = deepcopy(χsp_tmp)
        χch = deepcopy(χch_tmp)
    end

    return residuals
end

function res_curve_int(λch_i::Float64, 
            χsp::χT, γsp::γT, χch::χT, γch::γT,
            Σ_loc::Vector{ComplexF64},gLoc_rfft_init::GνqT,
            λ₀::Array{ComplexF64,3}, kG::KGrid, mP::ModelParameters, sP::SimulationParameters;
            maxit=50, mixing=0.2, conv_abs=1e-6, update_χ_tail=false, par=false)

    k_norm::Int = Nk(kG)
    _, νGrid, χ_tail = gen_νω_indices(χsp, χch, mP, sP)
    χ_λ!(χch, χch, λch_i)
    rhs_λsp = λsp_rhs(NaN, χsp, χch, kG, mP, sP)
    λsp_i = λsp_correction(χsp, real(rhs_λsp), kG, mP, sP)
    reset!(χch)
    if !isfinite(λsp_i)
        @warn "no finite λsp found!"
    end
    lhs_c1, lhs_c2 = lhs_int(χsp.data, χch.data, λsp_i, λch_i, 
                            χ_tail, kG.kMult, k_norm, χsp.tail_c[3], mP.β)

    Σ_ladder, gLoc_new, E_pot, μnew, converged = if par 
        initialize_EoM(gLoc_rfft_init, λ₀, νGrid, kG, mP, sP, 
                    χsp = χsp, γsp = γsp,
                    χch = χch, γch = γch)
        run_sc_par(gLoc_rfft_init, Σ_loc, νGrid, λsp_i, λch_i, kG, mP, sP, maxit=maxit, mixing=mixing, conv_abs=conv_abs)
    else
        run_sc(χsp, γsp, χch, γch, λ₀, gLoc_rfft_init, Σ_loc, λsp_i, λch_i, kG, mP, sP, maxit=maxit, mixing=mixing, conv_abs=conv_abs)
    end
                
    rhs_c1 = mP.n/2 * (1 - mP.n/2)
    rhs_c2 = E_pot/mP.U - (mP.n/2) * (mP.n/2)
    return λsp_i, lhs_c1, rhs_c1, lhs_c2, rhs_c2, μnew, converged
end
