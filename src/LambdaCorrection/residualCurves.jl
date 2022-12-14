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
    x₀ = Δx > 0 ? -(ydata[ind]-xdata[ind])/m : (ydata[ind]-xdata[ind+1])/m 
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
        χ_sp::χT, γ_sp::γT, χ_ch::χT, γ_ch::γT,
            Gνω::GνqT, λ₀::Array{ComplexF64,3},
            kG::KGrid, mP::ModelParameters, sP::SimulationParameters; νmax::Int = -1)

    # general definitions
    Nq, Nν, Nω = size(γ_sp)
    ωindices = usable_ωindices(sP, χ_sp, χ_ch)
    νmax::Int = νmax < 0 ? minimum([sP.n_iν,floor(Int,3*length(ωindices)/8)]) : νmax
    νGrid::UnitRange{Int} = 0:(νmax-1)
    iωn = 1im .* 2 .* (-sP.n_iω:sP.n_iω)[ωindices] .* π ./ mP.β
    iωn[findfirst(x->x ≈ 0, iωn)] = Inf
    χ_tail::Vector{ComplexF64} = mP.EKin_DMFT ./ (iωn.^2)

    # Therodynamics preallocations
    Σ_hartree::Float64 = mP.n * mP.U/2.0;
    E_pot_tail_c = [zeros(size(kG.ϵkGrid)),
            (mP.U^2 * 0.5 * mP.n * (1-0.5*mP.n) .+ Σ_hartree .* (kG.ϵkGrid .+ Σ_hartree .- mP.μ))]
    tail = [1 ./ (iν_array(mP.β, νGrid) .^ n) for n in 1:length(E_pot_tail_c)]
    E_pot_tail::Matrix{ComplexF64} = sum(E_pot_tail_c[i] .* transpose(tail[i]) for i in 1:length(tail))
    E_pot_tail_inv::Vector{Float64} = sum((mP.β/2)  .* [Σ_hartree .* ones(size(kG.ϵkGrid)), (-mP.β/2) .* E_pot_tail_c[2]])

    rhs_c1 = mP.n/2 * (1 - mP.n/2)
    λsp_min, λsp_max = λ_search_range(real.(χ_sp.data))
    λch_min, λch_max = λ_search_range(real.(χ_ch.data))

    λch_range_negative = 10.0.^(range(0,stop=log10(abs(λch_min)+1),length=NPoints_negative+2)) .+ λch_min .- 1
    λch_range_negative_2 = range(maximum([-200,λch_min]),stop=100,length=20)
    λch_range_coarse = range(0,stop=λch_max,length=NPoints_coarse)
    λch_range_large = 10.0.^(range(0,stop=log10(λch_max2-2*λch_max+1),length=6)) .+ 2*λch_max .- 1
    last_λch_range = isfinite(last_λ[2]) ? range(last_λ[2] - abs(last_λ[2]*0.1), stop = last_λ[2] + abs(last_λ[2]*0.1), length=8) : []
    λch_range = Float64.(sort(unique(union([0], last_λch_range, λch_range_negative, λch_range_negative_2, λch_range_coarse, λch_range_large))))


    # EoM optimization related definitions
    Kνωq_pre::Vector{ComplexF64} = Vector{ComplexF64}(undef, length(kG.kMult))
    Σ_ladder_ω::OffsetArray{ComplexF64,3,Array{ComplexF64,3}} = OffsetArray(Array{ComplexF64,3}(undef,Nq,νmax,length(ωindices)),
                              1:Nq, 0:νmax-1, ωindices)
    Σ_ladder::OffsetArray{ComplexF64,2,Array{ComplexF64,2}} = OffsetArray(Array{ComplexF64,2}(undef,Nq, νmax),
                              1:Nq, 0:νmax-1)

    # preallications
    χ_sp_tmp::χT = deepcopy(χ_sp)
    χ_ch_tmp::χT = deepcopy(χ_ch)
    G_corr::Matrix{ComplexF64} = Matrix{ComplexF64}(undef, Nq, νmax)

    residuals = zeros(6, length(λch_range))
    for (i,λch_i) in enumerate(λch_range)
            λsp_i, lhs_c1, rhs_c1, lhs_c2, rhs_c2 = cond_both_int(λch_i, χ_sp, γ_sp, χ_ch, γ_ch, χ_sp_tmp, χ_ch_tmp,
            ωindices, Σ_ladder_ω, Σ_ladder, Kνωq_pre, G_corr, νGrid, χ_tail, Σ_hartree,
            E_pot_tail, E_pot_tail_inv, Gνω,λ₀, kG, mP, sP)

        residuals[:,i] = [λsp_i, λch_i, lhs_c1, rhs_c1, lhs_c2, rhs_c2]
        χ_sp.data = deepcopy(χ_sp_tmp.data)
        χ_ch.data = deepcopy(χ_ch_tmp.data)
    end

    return residuals
end
