# ==================================================================================================== #
#                                         backup/GFFit.jl                                              #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
#   Last Edit Date  : 04.08.22                                                                         #
# ----------------------------------------- Description ---------------------------------------------- #
#   Remaining functionality for sum extrapolation methods using Shanks and Richardson extrapolation    #
# ==================================================================================================== #


"""
    build_fνmax_fast(f::AbstractArray{T}, n::Int)::Array{T, 1} where T <: Number

Description
-------------
Constructs array of partial sums of one or two-dimensional array `f` with
`n` partial sum terms.
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
function build_fνmax_fast(f::AbstractArray{T}, n::Int)::Array{T, 1} where T <: Number
    n_iν = minimum(size(f))
    lo = n
    up = n_iν - n + 1 
    f_νmax  = Array{T, 1}(undef, lo)
    build_fνmax_fast!(f_νmax, f, lo, up)
    return f_νmax
end

function csum_helper!(arr1::AbstractArray{T,1}, arr2::AbstractArray{T,1}, i::Int, lo::Int, up::Int) where T <: Number
    @inbounds arr1[i] = arr1[i-1] + arr2[lo] + arr2[up]
end

function csum_helper!(arr1::AbstractArray{T,1}, arr2::AbstractArray{T,2}, i::Int, lo::Int, up::Int) where T <: Union{ComplexF64,Float64}
    @inbounds @views arr1[i] = arr1[i-1] + sum(arr2[lo, lo:up]) + sum(arr2[up, lo:up]) + 
                                    sum(arr2[(lo+1):(up-1),lo]) + sum(arr2[(lo+1):(up-1),up]) 
end
"""
    build_fνmax_fast!(f::AbstractArray{T,1}, lo::Int, up::Int)::Array{T, 1} where T <: Number
    build_fνmax_fast!(f::AbstractArray{T,2}, lo::Int, up::Int)::Array{T, 1} where T <: Number

Description
-------------
Constructs array of partial sums of one or two-dimensional array `f` starting
at with `lo` and ending wie `up` summands.
This assumes, that the array is symmetric around the mid index.
For a more convenient, but possible slower version, see also [`build_fνmax_fast`](@ref build_fνmax_fast!)
"""
function build_fνmax_fast!(f_νmax::AbstractArray{T,1}, f::AbstractArray{T,1}, lo::Int, up::Int) where T <: Union{ComplexF64,Float64}
    sum!(view(f_νmax,1),view(f,lo:up))
    @simd for i in 2:length(f_νmax)
        lo -= 1
        up += 1
        csum_helper!(f_νmax, f, i, lo, up)
    end
end

function build_fνmax_fast!(f_νmax::AbstractArray{T,1}, f::AbstractArray{T,2}, lo::Int, up::Int) where T <: Union{ComplexF64,Float64}
    sum!(view(f_νmax,1), view(f,lo:up, lo:up))
    @simd for i in 2:length(f_νmax)
        lo -= 1
        up += 1
        csum_helper!(f_νmax, f, i, lo, up)
    end
end


"""
    get_sum_helper(range, sP::SimulationParameters)

Description
-------------
Construct helper for (improved) sums from setting in
`sP` over a given fit range `range`.
"""
function get_sum_helper(range::AbstractArray{Int,1}, sP::SimulationParameters, type::Symbol)
    tc = if type == :f sP.tc_type_f else sP.tc_type_b end
    sEH = sP.sumExtrapolationHelper
    res = if sEH !== nothing
        fitRange = (type == :f) ? (default_fit_range(range)) : (1:length(sEH.bosonic_tail_coeffs)) 
        get_sum_helper(fitRange, (type == :f) ? sEH.fermionic_tail_coeffs : sEH.bosonic_tail_coeffs, tc) 
    else
        DirectSum()
    end
    return res
end

function get_sum_helper(range::AbstractArray{Int,1}, coeffs::AbstractArray{Int,1}, tc::Symbol)
    sumHelper = if tc == :nothing
        DirectSum()
    elseif tc == :richardson
        Richardson(range, coeffs, method=:bender)
    elseif tc == :coeffs
        DirectSum()
    else
        @error("Unrecognized sum helper type \"$tc\", falling back to direct sums!")
        DirectSum()
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
For a faster version of 1D arrays, see also [`sum_freq_1D`](@ref sum_freq_1D).

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
julia> sum_freq(arr, [2,3], DirectSum(), 1.0)
3×1×1 Array{Float64, 3}:
    [:, :, 1] =
     25.0
     25.0
     25.0
```
"""
function sum_freq(f::AbstractArray{T1}, dims::Array{Int,1}, type::T2, β::Float64, corr::Float64) where {T1 <: Real, T2 <: SumHelper}
    length(dims) == ndims(f) && return  sum_freq_full(f,type,β,corr)
    res = mapslices(x -> esum_c(build_fνmax_fast(x, npartial_sums(type)) .+ corr, type), f, dims=dims)
    return res/(β^length(dims))
end

#TODO: provide interface for tail subtraction in sum_freq

function sum_freq(f::AbstractArray{T1}, dims::Array{Int,1}, type::T2, β::Float64, corr::Float64) where {T1 <: Complex, T2 <: SumHelper}
    length(dims) == ndims(f) && return sum_freq_full(f,type,β,corr)
    @warn "using unoptimized sum_freq"
    res_re = mapslices(x -> esum_c(build_fνmax_fast(x,npartial_sums(type)) .+ corr, type), real.(f), dims=dims)
    res_im = mapslices(x -> esum_c(build_fνmax_fast(x,npartial_sums(type)) .+ corr, type), imag.(f), dims=dims)
    res = res_re + res_im*im
    return res/(β^length(dims))
end

function sum_freq(f::AbstractArray{T1}, dims::Array{Int,1}, type::T2, β::Float64) where {T1 <: Complex, T2 <: SumHelper}
    length(dims) == ndims(f) && return sum_freq_full(f,type,β)
    @warn "using unoptimized sum_freq"
    res_re = mapslices(x -> esum_c(build_fνmax_fast(x,npartial_sums(type)), type), real.(f), dims=dims)
    res_im = mapslices(x -> esum_c(build_fνmax_fast(x,npartial_sums(type)), type), imag.(f), dims=dims)
    res = res_re + res_im*im
    return res/(β^length(dims))
end

"""
    sum_freq_full(f, type::T, β::Float64; corr::Float64=0.0) where T <: SumHelper

Description
-------------
Fast wrapper of [`sum_freq`](@ref sum_freq), lacking the `dims` argument.
"""
function sum_freq_full(f::AbstractArray{Float64}, type::T2, β::Float64, corr::Float64) where {T2 <: SumHelper}
    esum_c(build_fνmax_fast(f, npartial_sums(type)) .+ corr, type)/(β^ndims(f))
end

function sum_freq_full(f::AbstractArray{Complex{Float64}}, type::T2, β::Float64, corr::Float64) where {T2 <: SumHelper}
    tmp = build_fνmax_fast(f, npartial_sums(type)) .+ corr
    @warn "using unoptimized sum_freq"
    return (esum_c(real.(tmp), type) + esum_c(imag.(tmp), type).*im)/(β^ndims(f))
end


function sum_freq_full(f::AbstractArray{Float64}, type::T2, β::Float64) where {T2 <: SumHelper}
    esum_c(build_fνmax_fast(f, npartial_sums(type)), type)/(β^ndims(f))
end

function sum_freq_full(f::AbstractArray{Complex{Float64}}, type::T2, β::Float64) where {T2 <: SumHelper}
    tmp = build_fνmax_fast(f, npartial_sums(type))
    @warn "using unoptimized sum_freq"
    return (esum_c(real.(tmp), type) + esum_c(imag.(tmp), type).*im)/(β^ndims(f))
end


"""
    sum_freq_full!(f::Array{T,1/2}, type::T, β::Float64, fνmax_cache::Array{T,1}, lo::Int, up::Int; corr::Float64=0.0) where T <: SumHelper

Description
-------------
Inplace version of [`sum_freq_full`](@ref sum_freq_full). A cache for the internal partial sums `fνmax_cache` and the boundaries
for summation of `f`, `lo/up` for the minimum/maximum number of summands in the array of partial sums.
"""
sum_freq_full_f!(f::AbstractArray, β::Float64, sEH::Nothing) = (sum(f))/(β^ndims(f))

#TODO: cleaner interface, but about 10% slower! fix this!
function sum_freq_full_f!(f::AbstractArray{Float64}, β::Float64, corr::Float64, sEH::SumExtrapolationHelper)::Float64
    build_fνmax_fast!(sEH.fνmax_cache_r, f, sEH.fνmax_lo, sEH.fνmax_up)
    @inbounds esum_c(sEH.fνmax_cache_r .+ corr, sEH.sh_f)/(β^ndims(f))
end

function sum_freq_full_f!(f::AbstractArray{ComplexF64}, β::Float64, corr::Float64, sEH::SumExtrapolationHelper)::ComplexF64
    build_fνmax_fast!(sEH.fνmax_cache_c, f, sEH.fνmax_lo, sEH.fνmax_up)
    @simd for i in 1:length(sEH.fνmax_cache)
        @inbounds sEH.fνmax_cache_c[i] = sEH.fνmax_cache_c[i] + corr
    end
    @inbounds v = reshape(reinterpret(Float64,sEH.fνmax_cache_c),2,1:length(sEH.fνmax_cache_c))
    @inbounds (esum_c(view(v,1,:), sEH.sh_f) + esum_c(view(v,2,:), sEH.sh_f).*im)/(β^ndims(f))
end

function sum_freq_full_f!(f::AbstractArray{Float64}, β::Float64, sEH::SumExtrapolationHelper)::Float64
    build_fνmax_fast!(sEH.fνmax_cache_r, f, sEH.fνmax_lo, sEH.fνmax_up)
    @inbounds esum_c(sEH.fνmax_cache_r, sEH.sh_f)/(β^ndims(f))
end

function sum_freq_full_f!(f::AbstractArray{ComplexF64}, β::Float64, sEH::SumExtrapolationHelper)::ComplexF64
    build_fνmax_fast!(sEH.fνmax_cache_c, f, sEH.fνmax_lo, sEH.fνmax_up)
    @inbounds v = reshape(reinterpret(Float64,sEH.fνmax_cache_c),2,1:length(sEH.fνmax_cache_c))
    #TODO: do we really need to split real and imaginary part?
    @inbounds (esum_c(view(v,1,:), sEH.sh_f) + esum_c(view(v,2,:), sEH.sh_f).*im)/(β^ndims(f))
end

function sum_freq_full!(f::AbstractArray{Float64}, type::SumHelper, β::Float64, fνmax_cache::Array{Float64,1}, lo::Int, up::Int, corr::Float64)
    build_fνmax_fast!(fνmax_cache, f, lo, up)
    @inbounds esum_c(fνmax_cache .+ corr, type)/(β^ndims(f))
end

function sum_freq_full!(f::AbstractArray{ComplexF64}, type::SumHelper, β::Float64, fνmax_cache::Array{ComplexF64,1}, lo::Int, up::Int, corr::Float64)
    build_fνmax_fast!(fνmax_cache, f, lo, up)
    fνmax_cache[:] = fνmax_cache[:] .+ corr
    @inbounds v = reshape(reinterpret(Float64,fνmax_cache),2,1:length(fνmax_cache))
    @inbounds (esum_c(view(v,1,:), type) + esum_c(view(v,2,:), type).*im)/(β^ndims(f))
end

function sum_freq_full!(f::AbstractArray{Float64}, type::SumHelper, β::Float64, fνmax_cache::Array{Float64,1}, lo::Int, up::Int)
    build_fνmax_fast!(fνmax_cache, f, lo, up)
    @inbounds esum_c(fνmax_cache, type)/(β^ndims(f))
end

function sum_freq_full!(f::AbstractArray{ComplexF64}, type::SumHelper, β::Float64, fνmax_cache::Array{ComplexF64,1}, lo::Int, up::Int)
    build_fνmax_fast!(fνmax_cache, f, lo, up)
    @inbounds v = reshape(reinterpret(Float64,fνmax_cache),2,1:length(fνmax_cache))
    @inbounds (esum_c(view(v,1,:), type) + esum_c(view(v,2,:), type).*im)/(β^ndims(f))
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
function extend_γ!(arr::AbstractArray{_eltype,1}, h::Float64; weight=0.01)
    indh = ceil(Int, length(arr)/2)
    lo,up = find_usable_γ(arr)
    # left
    i = lo
    df = -conj(arr[i] - arr[i+1])/h
    ddf = -conj(arr[i+2] - 2*arr[i+1] + arr[i])/(2*h^2)
    df = abs(df) > 0.1*abs(arr[i]) ? 0.0 : df
    ddf = abs(ddf) > 0.1*abs(arr[i]) ? 0.0 : ddf

    #println(" -----> $lo, $up / $df - $ddf")
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
    df = abs(df) > 0.1*abs(arr[i]) ? 0.0 : df
    ddf = abs(ddf) > 0.1*abs(arr[i]) ? 0.0 : ddf
    wi = weight
    while i < length(arr)
        i += 1
        arr[i] = (ddf*wi + df*wi + 1) * arr[i-1]
        #df = df * wi
        #ddf = ddf * wi
        wi = 10*(arr[i] - 1.0 + 0im)
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
function extend_γ!(arr::AbstractArray{_eltype,1}, ref::Array{_eltype,1})
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

"""
    extend_corr!(arr::AbstractArray{Complex{Float64},3})

Extends the `corr(ω,ν) = Σ_νp Fupdo(ω,ν,νp) ⋅ G(νp) ⋅ G(νp+ω)` term outside it's
stability range. This assumes, that `corr(ω,ν) → corr(ω)` for large ν and `corr(ω,ν) < 0`.
TODO: Especially the second assumption may be violated outside half filling!!
"""
function extend_corr!(arr::AbstractArray{Float64,3})
    indh = trunc(Int, size(arr,3)/2) + 1
    for qi in axes(arr,2)
        override = false
        for νi in (indh+1):size(arr,3)
            (!override) && any(arr[:,qi,νi] .> 0) && (override = true)  # if overide switch not set and any positive value found: set override switch
            override && (arr[:,qi,νi] .= arr[:,qi,νi-1])               # if override set: replace values with last known good values
        end
        override = false
        for νi in (indh-2):-1:1
            (!override) && any(arr[:,qi,νi] .> 0) && (override = true)
            override && (arr[:,qi,νi] .= arr[:,qi,νi+1])
        end
    end
end
