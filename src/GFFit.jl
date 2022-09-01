# ==================================================================================================== #
#                                            GFFit.jl                                                  #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
#   Last Edit Date  : 29.08.22                                                                         #
# ----------------------------------------- Description ---------------------------------------------- #
#   Functionality for the estimation of valid ranges, sum extrapolations and fixing of tails.          #
#   This functionality hast mostly been replaces by analytic methods in BSE_SC.jl                      #
# -------------------------------------------- TODO -------------------------------------------------- #
#                                                                                                      #
# ==================================================================================================== #

# ======================================== Usable Inervals ===========================================
function find_usable_interval(arr::Array{Float64,1}; sum_type::Union{Symbol,Tuple{Int,Int}}=:common, reduce_range_prct::Float64 = 0.1)
    mid_index = Int(ceil(length(arr)/2))
    if sum_type == :full
        return 1:length(arr)
    elseif typeof(sum_type) == Tuple{Int,Int}
        return default_sum_range(mid_index, sum_type)
    end

    darr = diff(arr; dims=1)
    if arr[mid_index] < 0.0
        res = [mid_index]
        return res
    end
    # interval for condition 1 (positive values)
    cond1_intervall_range = 1
    # find range for positive values
    @inbounds while (cond1_intervall_range < mid_index) &&
        (arr[(mid_index-cond1_intervall_range)] > 0) &&
        (arr[(mid_index+cond1_intervall_range)] > 0)
        cond1_intervall_range = cond1_intervall_range + 1
    end

    # interval for condition 2 (monotonicity)
    cond2_intervall_range = 1
    # find range for first turning point
   @inbounds while (cond2_intervall_range < mid_index-1) &&
        (darr[(mid_index-cond2_intervall_range)] > 0) &&
        (darr[(mid_index+cond2_intervall_range)] < 0)
        cond2_intervall_range = cond2_intervall_range + 1
    end
    intervall_range = minimum([cond1_intervall_range, cond2_intervall_range])
    range = ceil(Int64, intervall_range*(1-reduce_range_prct))
    if length(arr)%2 == 1
        res = ((mid_index-range):(mid_index+range-1) .+ 1)
    else
        res = ((mid_index-range):(mid_index+range-1) .+ 2)
    end

    if length(res) < 1
        @warn "   ---> WARNING: could not determine usable range. Defaulting to single frequency!"
        res = [mid_index]
    end
    return res
end

"""
    find_usable_γ(arr; threshold=50, prct_red=0.05)

Usable νₙ range for γ. γ(ωₙ) is usable, for a range in which `γ[n]/γ[n-1]` does not
exceed some threshold value.

Arguments
-------------
- **`arr`**    : 1D γ(νₙ) slice for fixed ω and q
- **`threshold`** : Optional, default `50`, unusable if `arr[i]/arr[i±1] > threshold`
- **`prct_red`**  : Optional, default `0.05`, reduce unusble range by this amount
"""
function find_usable_γ(arr; threshold=50, prct_red=0.05)
    indh = ceil(Int, length(arr)/2)
    i = indh + floor(Int,indh/5)
    red = ceil(Int,length(arr)*prct_red)
    lo = 1
    up = length(arr)
    while i <= length(arr)
        change = abs(real(arr[i]/arr[i-1]))
        (change > threshold) && (up = i-1; break)
        i += 1
    end
    i = indh - floor(Int,indh/5)
    while i > 0
        change = abs(real(arr[i]/arr[i+1]))
        (change > threshold) && (lo = i+1; break)
        i -= 1
    end
    (lo > 1) && (lo += red)
    (up < length(arr)) && (up -= red)
    return lo,up
end

function default_sum_range(mid_index::Int, lim_tuple::Tuple{Int,Int}) where T
    return union((mid_index - lim_tuple[2]):(mid_index - lim_tuple[1]), (mid_index + lim_tuple[1]):(mid_index + lim_tuple[2]))
end

function reduce_range(range::AbstractArray, red_prct::Float64)
    sub = floor(Int, length(range)/2 * red_prct)
    lst = maximum([last(range)-sub, ceil(Int,length(range)/2 + iseven(length(range)))])
    fst = minimum([first(range)+sub, ceil(Int,length(range)/2)])
    return fst:lst
end
