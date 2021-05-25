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

default_fit_range(arr::AbstractArray) = default_fit_range(length(arr))
default_fit_range(s::Int) = ceil(Int,s/5):ceil(Int, s/2)

"""
    get_sum_helper(range, sP::SimulationParameters)

Description
-------------
Construct helper for (improved) sums from setting in
`sP` over a given fit range `range`.
"""
function get_sum_helper(range, sP::SimulationParameters, type)
    sumHelper = if sP.tc_type == :nothing
        Naive()
    elseif sP.tc_type == :richardson
        fitRange = default_fit_range(range)
        (type) == :f ? Richardson(fitRange, sP.fermionic_tail_coeffs, method=:rohringer) : Richardson(fitRange, sP.bosonic_tail_coeffs, method=:rohringer)
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
        corr::Float64=0.0) where {T1 <: Real, T2 <: SumHelper}
    res = mapslices(x -> esum_c(build_fνmax_fast(x, 1) .+ corr, type), f, dims=dims)
    return res/(β^length(dims))
end

function sum_freq(f::AbstractArray{T1}, dims::Array{Int,1}, type::T2, β::Float64; 
        corr::Float64=0.0) where {T1 <: Complex, T2 <: SumHelper}
    res_re = mapslices(x -> esum_c(build_fνmax_fast(x, 1) .+ corr, type), real.(f), dims=dims)
    res_im = mapslices(x -> esum_c(build_fνmax_fast(x, 1) .+ corr, type), imag.(f), dims=dims)
    res = res_re + res_im*im
    return res/(β^length(dims))
end

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
    find_usable_γ(arr; threshold=50, prct_red=0.05)

Usable νₙ range for γ.
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


"""
    extend_γ!(arr::AbstractArray{Complex{Float64},1})

Extends γ for fermionic frequencies, for which the sum extrapolation failed.
See also (`extend_γ(::AbstractArray, ::AbstractArray)`) for a version, that uses
the naive sum as reference.

Description
-------------
Assuming that ∀ ωₙ,q  γ[ωₙ,q,νₙ] → 1 + 0i for ν → ∞, this function cuts off
typical overfitting spikes for large νₙ. It is furthermore assumed, that
γ[ωₙ,q,νₙ] /  γ[ωₙ,q,νₙ ± ν_{n+1}] never exceeds 50.
If such a large change is detected all γ[ωₙ,q,νₙ] after that point are
set to df⋅γ[ωₙ,q,νₙ ∓ ν{n}], df being the difference between the last
accepted elements.
The tail behavior is approximated by the 0th to 2nd Taylor terms.
This will most likely lead to a discontinuiety at the first altered frequency
in the second derivative.
Since it is know, that the real part tends towards 1 and the imaginary part
towards 0, a frequency depndended dampening factor is multiplied with the 
first and second derivative. I.e. `arr[i] = (1 + df/h + ddf/h^2) arr[i-1]`
and `df = weight * df`, `ddf = weight^2 ddf` at each iteration.


Arguments
-------------
- **`arr`**    : 1D array containing sum-extrapolated γ[νₙ] (for fixed ωₙ and q)
- **`h`**      : 2 pi/ beta 
- **`weight`** : Optional, default `0.9`, see description above for effect of weight.

Examples
-------------
```
extend_γ!(view(γ,1,1,:), 0.5235987755982988)
 """
function extend_γ!(arr::AbstractArray{Complex{Float64},1}, h::Float64; weight=0.9)
    indh = ceil(Int, length(arr)/2)
    lo,up = find_usable_γ(arr)
    # left
    i = lo
    df = -conj(arr[i] - arr[i+1])/h
    ddf = -conj(arr[i+2] - 2*arr[i+1] + arr[i])/(2*h^2)
    wi = weight
    while i > 1
        i -= 1
        arr[i] = (ddf*wi + df*wi + 1) * arr[i+1]
        #df = df * wi
        #ddf = ddf * wi
        wi = 10*(arr[i] - 1.0 + 0im)
    end
    # right
    i = up
    df = -conj(arr[i] - arr[i-1])/h
    ddf = -conj(arr[i-2] - 2*arr[i-1] + arr[i])/(2*h^2)
    wi = weight
    while i < length(arr)
        i += 1
        arr[i] = (ddf*wi + df*wi + 1) * arr[i-1]
        #df = df * wi
        #ddf = ddf * wi
        wi = 10000*(arr[i] - 1.0 + 0im)
    end
end

"""
    extend_γ!(arr::AbstractArray{Complex{Float64},1}, ref::AbstractArray{Complex{Float64},1})

Extends γ for fermionic frequencies, for which the sum extrapolation failed.
See also (`extend_γ(::AbstractArray)`) for a version, that does not require
the naive sum as input.

Description
-------------
Assuming that ∀ ωₙ,q  γ[ωₙ,q,νₙ] → 0 for ν → ∞, this function cuts off
typical overfitting spikes for large νₙ. It is furthermore assumed, that
γ[ωₙ,q,νₙ] /  γ[ωₙ,q,νₙ ± ν_{n+1}] never exceeds 50.
If such a large change is detected all γ[ωₙ,q,νₙ] after that point are
set to df⋅γ[ωₙ,q,νₙ ∓ ν{n}], df being the difference between the last
accepted elements.
This will most likely lead to a discontinuiety at the first altered frequency.
See also (`extend_γ(::AbstractArray, ::AbstractArray)`) for a version, that uses
the naive sum as reference.


Arguments
-------------
- **`arr`**    : 1D array containing sum-extrapolated γ[νₙ] (for fixed ωₙ and q)
- **`ref`**    : 1D array containing naively summed γ[νₙ] (for fixed ωₙ and q)

Examples
-------------
```
extend_γ!(view(γ,1,1,:), view(γ_ref,1,1,:))
 """
function extend_γ!(arr::AbstractArray{Complex{Float64},1}, ref::AbstractArray{Complex{Float64},1})
    indh = ceil(Int, length(arr)/2)
    i = floor(Int,indh/5)
    override= false
    # right
    while i <= length(arr)
        change = abs(arr[i]/arr[i-1])
        (!override && change > 50) && (arr[i-1]=ref[i-1];override = true)
        override && (arr[i] = ref[i])
        i += 1
    end
    # left
    i = floor(Int,indh/5)
    override= false
    while i > 0
        change = abs(arr[i]/arr[i+1])
        (!override && change > 50) && (arr[i+1]=ref[i+1];override = true)
        override && (arr[i] = ref[i])
        i -= 1
    end
end
