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
- **`sum_type`**           : Optional, default `:common` computes sum range. For debug purposes: Can be set to `:full` to enforce full range, or a `::Tuple{Int,Int}` to enforce a specific interval size.
- **`reduce_range_prct`**  : Optional, default `0.1`. After finding the usable interval it is reduced by an additional percentage given by this value.
"""
function find_usable_χ_interval(
    χ_ω::AbstractVector{Float64};
    sum_type::Union{Symbol,Tuple{Int,Int}} = :common,
    reduce_range_prct::Float64 = 0.1,
)::AbstractVector{Int}
    if length(χ_ω) % 2 == 0
        throw(ArgumentError("Finding a usable interval is only implemented for uneven size of χ_ω!"))
    end
    mid_index = ceil(Int, (length(χ_ω) / 2))
    if sum_type == :full
        return 1:length(χ_ω)
    elseif typeof(sum_type) == Tuple{Int,Int}
        return sum_type[1]:sum_type[2]
    end
    χ_ω[mid_index] < 0.0 && return [mid_index]

    darr = diff(χ_ω; dims = 1)
    # find range for positive values
    cond_1 = χ_ω .> 0
    # find range for monotonic condition
    cond_2 = vcat(darr[1:mid_index-1] .> 0, [1,1], darr[mid_index+1:end] .< 0)
    usable_interval = cond_1 .&& cond_2

    red_range = ceil(Int,(count(usable_interval .== 1) / 2 ) * (reduce_range_prct))
    first_ind = findfirst(x->x==1, usable_interval) + red_range
    last_ind  = findlast(x->x==1, usable_interval)  - red_range 
    return first_ind:last_ind
end

function find_usable_χ_interval(
    χ_qω::Matrix;
    sum_type::Union{Symbol,Tuple{Int,Int}} = :common,
    reduce_range_prct::Float64 = 0.1,
)::AbstractVector{Int}
    intersect(
        [find_usable_χ_interval(x, sum_type = sum_type, reduce_range_prct = reduce_range_prct) for x in eachslice(χ_qω, dims = 1)]...,
    )
end

"""
    usable_ωindices(sP::SimulationParameters, χ_sp::χT, χ_ch::χT)

Helper function, returning the indices `n` for ``\\omega_n`` ranges of multiple channels. If `dbg_full_eom_omega` is set to `true` in the config,
the full range will be returned, otherwise an intersection of the usable ranges obtained from [`find_usable_χ_interval`](@ref find_usable_χ_interval).
"""
function usable_ωindices(sP::SimulationParameters, χ_list::Vararg{χT,N}) where {N}
    ωindices = (sP.dbg_full_eom_omega) ? (1:size(χ_list[1], 2)) : intersect(getfield.(χ_list, :usable_ω)...)
    return ωindices
end
