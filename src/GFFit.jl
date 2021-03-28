"""
    build_design_matrix(imin, imax, ncoeffs)

Helper function that build the matrix ``M`` used to fit data obtained in a
`imin` to `imax` interval to a ``\\sum_{n=0}^{`ncoeffs`} c_n/i^n`` tail.
The coefficients ``c_n`` are obtained by solving ``M c = b``. 
``b`` can be constructed from the data using [`build_rhs`](@ref).
"""
function build_design_matrix(imin, imax, coeff_exp_list::Array)
    ncoeffs = length(coeff_exp_list)
    M = zeros(BigFloat, (ncoeffs, ncoeffs))
    for i = imin:imax, (ki,k) = enumerate(coeff_exp_list), (li,l) in enumerate(coeff_exp_list)
        M[li,ki] += 1.0 / ((BigFloat(i)^(k+l)))
    end
    return M
end

"""
    build_rhs(imin, imax, ncoeffs, data)

Contructs rhs for ``M c = b``. Use [`build_design_matrix`](@ref) to obtain
``M`` matrix. ``ncoeffs`` is the dimension of ``c`` and must match in both calls.
"""
function build_rhs(imin, imax, ncoeffs, data)
    b = zeros(BigFloat, ncoeffs)
    for l in 0:(ncoeffs-1)       
        for (ji,j) in enumerate(imin:imax)
            b[l+1] += real(data[ji]) / (BigFloat(j)^(l))
        end
    end
    return b
end

function build_rhs!(b, imin, imax, ncoeffs, data)
    for (ji,j) in enumerate(imin:imax)
        for l in 0:(ncoeffs-1)       
            b[l+1] += real(data[ji]) / (BigFloat(j)^(l))
        end
    end
    return b
end

"""
   build_weights(imin, imax, ncoeffs)

Build weight matrix i.e. ``W = M^{-1} R`` with M from [`build_design_matrix`](@ref)
and ``R_{ij} = \\frac{1}{j^i}``.
Fit coefficients can be obtained by multiplying `w` with data: ``a_k = W_{kj} g_j``
"""
function build_weights(imin::Int64, imax::Int64, coeff_exp_list::Array)::Array{Float64, 2}
    if (imax - imin) < length(coeff_exp_list) 
        @error "Trying to construct Richardson matrix for insufficient number of array elements."
        return Matrix{Float64}(undef, 0,0)
    end
    M = build_design_matrix(imin, imax, coeff_exp_list)
    Minv = inv(M)
    ncoeffs = length(coeff_exp_list)
    w = zeros(Float64, (ncoeffs, length(imin:imax)))
    for k=1:ncoeffs, (li,l) = enumerate(coeff_exp_list), (ii,i) = enumerate(imin:imax)
        w[k,ii] += Float64(Minv[k, li] / (BigFloat(i)^l), RoundDown)
    end
    return w
end

function fit_νsum_shanks(data; order=0)
    result = ndims(data) > 1 ? Array{eltype(data)}(undef, (size(data,1))) : zero(eltype(data))
    if ndims(data) == 1
        order = order == 0 ? floor(Int64,(size(data,1)-1)/2) : order
        if order > 0
            result = Shanks.shanks(data, order=order, csum_inp=true)[1]
        end
    else
        order = order == 0 ? floor(Int64,(size(data,2)-1)/2) : order
        for ωi in 1:size(data,1)
            result[ωi] = Shanks.shanks(data[ωi,:], csum_inp=true)[1]
        end
    end
    return result
end

function fit_νsum_richardson(W, data; precision = 100000)
    result = ndims(data) > 1 ? Array{eltype(data)}(undef, (size(data,1))) : zero(eltype(data))
    setprecision(precision) do
        if ndims(data) == 1
            result = dot(data, W[1,:])
        else
            for ωi in 1:size(data,1)
                dot(data[ωi,:] , W[1,:])
            end
        end
    end
    return result
end


"""
    Sums first νmax entries of any array along given dim for νmax = 1:size(arr,dim).
"""
function build_fνmax(f, W, dims; ω_shift = 0)
    n_iν   = minimum(size(f)[dims])
    νmax_end = floor(Int64,n_iν/2)
    νmax_start =  νmax_end - size(W, 2) + 1
    if νmax_start < 1
        throw(BoundsError("ERROR: negative range for summ approximation!"))
    end
    dims   = Tuple(dims)
    ν_cut  = νmax_end - νmax_start

    #TODO: keep dins as singleton dims?
    f_νmax = dropdims(sum_νmax(f, ν_cut+1, dims=dims); dims=dims)
    νdim   = ndims(f_νmax) + 1
    for ν_cut_i in (ν_cut):-1:1
        f_νmax = cat(f_νmax, dropdims(sum_νmax(f, ν_cut_i; dims=dims); dims=dims); dims=νdim)
    end

    return f_νmax
end

#TODO: test for square arrays
"""
    Faster version for build_νmax. WARNING: only tested for square arrays
"""
function build_fνmax_fast(f::Union{Array{<:Number, 1}, Array{<:Number,2}}, νmin::Int)
    #
    n_iν       = minimum(size(f))
    νmax_end   = floor(Int64,n_iν/2)
    νmax_start =  νmax_end - νmin  + 1
    ν_cut  = νmax_end - νmax_start + 1
    f_νmax  = Array{eltype(f)}(undef, ν_cut)
    if νmax_start < 1
        throw("ERROR: negative range for sum approximation!")
    end
    lo = ν_cut-0
    up = n_iν-ν_cut+0+1

    if ndims(f) == 1
        f_νmax[1] = sum(f[lo:up])
        for i in 2:length(f_νmax)
            lo = lo - 1
            up = up + 1
            f_νmax[i] = f_νmax[i-1] + f[lo] + f[up]
        end
    elseif ndims(f) == 2
        f_νmax[1] = sum(f[lo:up, lo:up])
        for i in 2:length(f_νmax)
            lo = lo - 1
            up = up + 1
            f_νmax[i] = f_νmax[i-1] + sum(f[lo, lo:up]) + sum(f[up, lo:up]) + 
                        sum(f[(lo+1):(up-1),lo]) + sum(f[(lo+1):(up-1),up]) 
        end
    else
        error("Frequency fit only implemented for 1 and 2 dimensions.")
    end
    return f_νmax
end


"""
    Computes an approximation for the infinite sum over f, by fitting to
    a function g = c_0 + c_1/x + c_2/x^2 ... 
    arguments are the function, the weights constructed from  [`build_design_weights`](@ref)
    and the dimensions over which to fit.
"""
function approx_full_sum_shanks(f; correction::Float64=0.0)
    f_νmax = build_fνmax_fast(f, 4) .+ correction
    return fit_νsum_shanks(f_νmax)[1]
end


function approx_full_sum_richardson(f; correction::Float64=0.0, W::Array{Float64,2})
    dims = collect(1:ndims(f))
    N = floor(Int64, size(f, dims[1])/2)
    if (N < size(W,1)) || (N < 5) 
        @error "WARNING: could not extrapolate sum, there were only $(size(f,dims[1])) terms. Falling back to naive sum."
        sum_approx = sum(f, dims=dims) .+ correction
    else
        f_νmax = build_fνmax_fast(f, size(W, 2)) .+ correction
        sum_approx = fit_νsum_richardson(W, f_νmax)
    end
    return sum_approx
end


#TODO: this is about 2 times slower than sum, why?
function sum_freq(arr, dims::Array{Int,1}, type::Symbol, β::Float64; 
                  correction::Float64=0.0, weights::Union{Nothing, Array{Float64,2}}=nothing)
    if type == :richardson
        if weights === nothing
            @warn "constructing fit matrix in place"
            N = floor(Int64, size(arr, dims[1])/2)
            ωmin = Int(floor(N*1/4))
            ωmax = N 
            weights = build_weights(ωmin, ωmax, [0,1,2,3])
        end
        res = mapslices(x -> approx_full_sum_richardson(x, correction=correction, W=weights), arr, dims=dims)
    elseif type == :shanks
        res = mapslices(x -> approx_full_sum_shanks(x, correction=correction), arr, dims=dims)
    else
        res = sum(arr, dims=dims) .+ correction
    end
    return res/(β^length(dims))
end


sum_inner(a::Array{T, 1}, cut::Int64) where T <: Number = sum(a[cut:(end-cut+1)])
sum_inner(a::Array{T, N}, cut::Int64) where {T <: Number, N} = sum(mapslices(x -> sum_inner(x,cut), a; dims=2:ndims(a))[cut:(end-cut+1)])

"""
    Sums first νmax entries of any array along given dimension.
    Warning: This has NOT been tested for multiple dimensions.
"""
sum_νmax(a, cut; dims) = mapslices(x -> sum_inner(x, (cut)), a; dims=dims)

"""
    Returns rang of indeces that are usable under 2 conditions.
    TODO: This is temporary and should be replace with a function accepting general predicates.
"""
function find_usable_interval(arr::Array{Float64,1};sum_type::Union{Symbol,Tuple{Int,Int}}=:common, reduce_range_prct::Float64 = 0.0)
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
    Returns rang of indeces that are usable for the ν range of γ.
    This assumes γ → 1 for ν → ∞.
    TODO: This is temporary and should be replace with a function accepting general predicates.
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


