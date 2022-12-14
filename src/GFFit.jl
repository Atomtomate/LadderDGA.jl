# ==================================================================================================== #
#                                            GFFit.jl                                                  #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
#   Last Edit Date  : 14.11.22                                                                         #
# ----------------------------------------- Description ---------------------------------------------- #
#   Functionality for the estimation of valid ranges, sum extrapolations and fixing of tails.          #
#   This functionality hast mostly been replaces by analytic methods in BSE_SC.jl                      #
# -------------------------------------------- TODO -------------------------------------------------- #
#                                                                                                      #
# ==================================================================================================== #

# ======================================== Usable Inervals ===========================================

"""
    find_usable_χ_interval(χ_ω::Array{Float64,1/2}; sum_type::Union{Symbol,Tuple{Int,Int}}=:common, reduce_range_prct::Float64 = 0.1)

Determines usable range for physical susceptibilities ``\\chi^\\omega`` or ``\\chi^\\omega`` and additionally cut away `reduce_range_prct` % of the range.
The unusable region is given whenever the susceptibility becomes negative, or the first derivative changes sign.

Returns: 
-------------
`range::AbstractVector{Float64}` : Usable ``\\omega`` range for ``\\chi``

Arguments:
-------------
- **`χ_ω`**                : ``\\chi^\\omega`` 
- **`sum_type`**           : Optional, default `:common`. Can be set to `:full` to enforce full range, or a `::Tuple{Int,Int}` to enforce a specific interval size.
- **`reduce_range_prct`**  : Optional, default `0.1`. After finding the usable interval it is reduced by an additional percentage given by this value.
"""
function find_usable_χ_interval(χ_ω::AbstractVector{Float64}; sum_type::Union{Symbol,Tuple{Int,Int}}=:common, reduce_range_prct::Float64 = 0.1)::AbstractVector{Int}
    if length(χ_ω)%2 == 0
        throw(ArgumentError("Finding a usable interval is only implemented for uneven size of χ_ω!"))
    end
    mid_index = Int(ceil(length(χ_ω)/2))
    if sum_type == :full
        return 1:length(χ_ω)
    elseif typeof(sum_type) == Tuple{Int,Int}
        return sum_type[1]:sum_type[2]
    end

    res = 1:length(χ_ω)
    darr = diff(χ_ω; dims=1)
    if χ_ω[mid_index] < 0.0
        res = [mid_index]
        return res
    end
    # interval for condition 1 (positive values)
    cond1_intervall_range = 0
    # find range for positive values
    while (cond1_intervall_range < mid_index - 1) &&
        (χ_ω[(mid_index-cond1_intervall_range-1)] > 0) &&
        (χ_ω[(mid_index+cond1_intervall_range+1)] > 0)
        cond1_intervall_range = cond1_intervall_range + 1
    end

    # interval for condition 2 (monotonicity)
    cond2_intervall_range = 0
    # find range for first turning point
   while (cond2_intervall_range < mid_index-1) &&
        (darr[(mid_index-cond2_intervall_range-1)] > 0) &&
        (darr[(mid_index+cond2_intervall_range)] < 0)
        cond2_intervall_range = cond2_intervall_range + 1
    end
    intervall_range = minimum([cond1_intervall_range, cond2_intervall_range])
    range = ceil(Int64, intervall_range*(1-reduce_range_prct))
    res = ((mid_index-range):(mid_index+range))
    return res
end

function find_usable_χ_interval(χ_qω::Matrix; sum_type::Union{Symbol,Tuple{Int,Int}}=:common, reduce_range_prct::Float64 = 0.1)::AbstractVector{Int}
    intersect([find_usable_χ_interval(x, sum_type=:common, reduce_range_prct=0.1) for x in eachslice(χ_qω, dims=1)]...)
end

"""
    usable_ωindices(sP::SimulationParameters, χ_sp::χT, χ_ch::χT)

Helper function, returning the indices `n` for ``\\omega_n`` ranges of multiple channels. If `dbg_full_eom_omega` is set to `true` in the config,
the full range will be returned, otherwise an intersection of the usable ranges obtained from [`find_usable_χ_interval`](@ref find_usable_χ_interval).
"""
function usable_ωindices(sP::SimulationParameters, χ_list::Vararg{χT,N}) where N
    ωindices = (sP.dbg_full_eom_omega) ? (1:size(χ_list[1],2)) : intersect(getfield.(χ_list,:usable_ω)...)
    return ωindices
end
