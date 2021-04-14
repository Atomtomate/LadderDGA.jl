"""
    build_fνmax_fast(f::AbstractArray{T,1}, nmin::Int)::Array{T, 1} where T <: Number
    build_fνmax_fast(f::AbstractArray{T,2}, nmin::Int)::Array{T, 1} where T <: Number

Description
-------------
Constructs array of partial sums of one or two-dimensional array `f` starting
at with `rmin` summands.
This assumes, that the array is symmetric around the mid index.

Examples
-------------
```
julia> LadderDGA.build_fνmax_fast([3,2,1,2,3],2)
[5, 11]
julia> arr = [3 3 3 3 3; 3 2 2 2 3; 3 2 1 2 3; 3 2 2 2 3; 3 3 3 3 3
julia> LadderDGA.build_fνmax_fast(arr,2)
[17, 65]
```
"""
function build_fνmax_fast(f::AbstractArray{T,2}, nmin::Int)::Array{T, 1} where T <: Number
    n_iν       = minimum(size(f))
    lo = ceil(Int,n_iν/2) - (nmin - 1)
    up = ceil(Int,n_iν/2) + iseven(n_iν) + (nmin - 1)
    f_νmax  = Array{eltype(f), 1}(undef, lo)

    f_νmax[1] = sum(f[lo:up, lo:up])
    for i in 2:length(f_νmax)
        lo = lo - 1
        up = up + 1
        f_νmax[i] = f_νmax[i-1] + sum(f[lo, lo:up]) + sum(f[up, lo:up]) + 
                    sum(f[(lo+1):(up-1),lo]) + sum(f[(lo+1):(up-1),up]) 
    end
    return f_νmax
end

function build_fνmax_fast(f::AbstractArray{T, 1}, nmin::Int)::Array{T, 1} where T <: Number
    n_iν       = minimum(size(f))
    lo = ceil(Int,n_iν/2) - (nmin - 1)
    up = ceil(Int,n_iν/2) + iseven(n_iν) + (nmin - 1)
    f_νmax  = Array{eltype(f), 1}(undef, lo)
    f_νmax[1] = sum(f[lo:up])
    for i in 2:length(f_νmax)
        lo = lo - 1
        up = up + 1
        f_νmax[i] = f_νmax[i-1] + f[lo] + f[up]
    end
    return f_νmax
end

"""
    get_sum_helper(range, sP::SimulationParameters)

Description
-------------
Construct helper for (improved) sums from setting in
`sP` over a given fit range `range`.
"""
function get_sum_helper(range, sP::SimulationParameters)
    fitRange = default_fit_range(range)
    sumHelper = if sP.tc_type == :nothing
        Naive()
    elseif sP.tc_type == :richardson
        Richardson(fitRange, sP.fermionic_tail_coeffs)
    else
        @error("Unrecognized tail correction, falling back to naive sums!")
        Naive()
    end
    return sumHelper
end

"""
    sum_freq(f, dims::AbstractArray{Int,1}, type::T, β::Float64; 
                  corr::Float64=0.0) where T <: SumHelper

Description
-------------
Wrapper for estimation of initite Matsubara sums from a finite set
of samples. Computes ``\\frac{1}{\\beta}^D \\sum_{n} f(i\\omega_n)``.
For convenience, a correction term (being the analytic sum over a
previously subtracted tail) can also be provided throught the `corr` argument.

Arguments
-------------
- **`f`**    : array containing function of Matsubara frequencies
- **`dims`** : dimensions over which to sum
- **`type`** : SumHelper object, used for estimation of limit
- **`β`**    : Inverse Temperature for normalization
- **`corr`** : When tail coefficients are know, the subtraction and 
               subsequent addition of the analytic infinite sum over
               the tail can be added back here.

Examples
-------------
```
julia> arr = ones(3,5,5)
julia> sum_freq(arr, [2,3], Naive(), 1.0)
3×1×1 Array{Float64, 3}:
    [:, :, 1] =
     25.0
     25.0
     25.0
```
"""
function sum_freq(f::AbstractArray{T1}, dims::Array{Int,1}, type::T2, β::Float64; 
        corr::Float64=0.0) where {T1 <: Number, T2 <: SumHelper}

    res = mapslices(x -> esum_c(build_fνmax_fast(x, 1) .+ corr, type), f, dims=dims)
    return res/(β^length(dims))
end


"""
    find_usable_interval(arr::Array{Float64,1}; sum_type::Union{Symbol,Tuple{Int,Int}}=:common, reduce_range_prct::Float64 = 0.0)

Finds a usable interval of indices for ``\\chi (\\omega_i)``, i.e. an interval where
known physical properties hold.

Description
-------------
Returns range of indices `i` where the following conditions hold:
 - arr[i] is positive
 - arr[i] is decreasing (for ``|i-i_0|``, ``i_0`` index of 0 frequency).

Arguments
-------------
 - **`arr`**      : array, containing ``\\chi (\\omega_i)``
 - **`sum_type`** : Optional. `:full` will always return `eachindex()`, any `Tuple{Int,Int}` 
                    will be itnerpreted as fixed limits and the range will be fixed to these.
 - **`reduce_range_prct`** : Optional. Will reduce the range by given percentage (usefull to
                    reduce effect of bad tails).
"""
function find_usable_interval(arr::Array{Float64,1}; sum_type::Union{Symbol,Tuple{Int,Int}}=:common, reduce_range_prct::Float64 = 0.0)
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
        res = ((mid_index-range+1):(mid_index+range-2) .+ 1)
    else
        res = ((mid_index-range+1):(mid_index+range-2) .+ 2)
    end

    if length(res) < 1
        println(stderr, "   ---> WARNING: could not determine usable range. Defaulting to single frequency!")
        res = [mid_index]
        println(res)
    end
    return res
end


"""
    find_usable_γ(arr)

Not tested yet!
"""
function find_usable_γ(arr)
    nh = ceil(Int64,length(arr)/2)
    darr = abs.(diff(real.(arr)))
    max_ok = darr[nh]
    i = 1
    @inbounds while (i < floor(Int64,length(arr)/2))
        if findmax(darr[nh-i:nh+i])[1] > max_ok
            max_ok = findmax(darr[nh-i:nh+i])[1]
        else
            break
        end
        i += 1
    end
    @inbounds max_range_i = findfirst(darr[nh:end] .> max_ok)

    range = max_range_i === nothing ? (1:length(arr)) : (nh-max_range_i+2):(nh+max_range_i-1)
    return range
end

"""
    extend_γ(arr, usable_ν)

Not tested yet!
"""
function extend_γ(arr, usable_ν)
    res = copy(arr)
    val = arr[first(usable_ν)]
    res[setdiff(1:length(arr), usable_ν)] .= 1.0
    return res
end

function extend_γ!(arr, usable_ν)
    val = arr[first(usable_ν)]
    arr[setdiff(1:length(arr), usable_ν)] .= 1.0
end
