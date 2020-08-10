function tail_func_full(iωn, c)
    res = [c[1] for i = 1:length(iωn)]
    for  i = 2:length(c)
        res = res .+ c[i]./(iωn .^ (i-1))
    end
    return res
end

function tail_func_cmplx(iωn, c)
    res = [c[1] for i = 1:length(iωn)]
    for  i =  2:Int(floor(length(c)/2))
        res = res .+ (c[2*i-1]+c[2*i]*im)./(iωn .^ (2*i-1))
    end
    return res
end


function fit_tail(G, iν_array, tail_func = tail_func_full, n_tail = 8)
    #println(length(G))
    #println(length(iν_array))
    @assert length(G) == length(iν_array)
    cost(c) = sum(abs2.(G .- tail_func(iν_array, c)))
    #print(Optim.optimize(cost, zeros(n_tail), Optim.Newton(); autodiff = :forward))
    res = Optim.minimizer(Optim.optimize(cost, zeros(n_tail), Optim.Newton(); autodiff = :forward))
    return res
end

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
function build_weights(imin, imax, coeff_exp_list::Array)
    M = build_design_matrix(imin, imax, coeff_exp_list)
    Minv = inv(M)
    ncoeffs = length(coeff_exp_list)
    w = zeros(Float64, (ncoeffs, length(imin:imax)))
    for k=1:ncoeffs, (li,l) = enumerate(coeff_exp_list), (ii,i) = enumerate(imin:imax)
        w[k,ii] += Float64(Minv[k, li] / (BigFloat(i)^l), RoundDown)
    end
    return w
end

function fit_νsum(W, data; precision = 100000)
    result = Array{eltype(data)}(undef, (size(data,1)))
    setprecision(precision) do
        if ndims(data) == 1
            result = dot(data, W[1,:])
        else
            for ωi in 1:size(data,1)
                result[ωi] = dot(data[ωi,:] , W[1,:])
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
function build_fνmax_fast(f, W)
    n_iν       = minimum(size(f))
    νmax_end   = floor(Int64,n_iν/2)
    νmax_start =  νmax_end - size(W, 2) + 1
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
            #println("$(i): $(lo), $(lo:up), $(up), $(lo:up), $(lo), $((lo+1):(up-1)), $(up), $((lo+1):(up-1))")
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
function approx_full_sum(f, dims; W::Union{Nothing, Array{Float64,2}}=nothing, fast::Bool=true)
    N = floor(Int64, size(f, dims[1])/2)

    if !(W === nothing) && N < size(W,1)
        println(stderr, "WARNING: could not extrapolate sum, there were only $(size(f,dims[1])) terms.",
                " Falling back to naive sum.")
        sum_approx = sum(f, dims=dims)
    else
        if W === nothing
            ωmin = Int(floor(N*3/4))
            ωmax = N 
            W = build_weights(ωmin, ωmax, [0,1,2,3])
        end
        if fast
            if !all(dims .== 1:ndims(f))
                throw(BoundsError("incorrect dimension. Fast approximate sum not implemented for aritrary dimensions! Use the setting fast=false instead"))
            end
            f_νmax = build_fνmax_fast(f, W)
        else
            f_νmax = build_fνmax(f, W, dims)
        end
        tail_c = fit_νsum(W, f_νmax)
        sum_approx = tail_c#tail_c[ax...]
    end
    return sum_approx
end


function find_inner_maximum(arr)
    darr = diff(arr; dims=1)
    mid_index = Int(floor(size(arr,1)/2))
    intervall_range = 1

    # find interval
    while (intervall_range < mid_index) &&
        (darr[(mid_index-intervall_range)] * darr[(mid_index+intervall_range-1)] > 0)
            intervall_range = intervall_range+1
    end

    index_maximum = mid_index-intervall_range+1
    if (mid_index-intervall_range) < 1 || index_maximum >= length(arr)
        return 1
    end
    # find index
    while darr[(mid_index-intervall_range)]*darr[index_maximum] > 0
        index_maximum = index_maximum + 1
    end
    return index_maximum
end

#TODO: this is about 2 times slower than sum, why?
function sum_freq(arr, dims, tail_corrected::Bool, β::Float64; weights::Union{Nothing, Array{Float64,2}}=nothing)
    if tail_corrected
        res = mapslices(x -> approx_full_sum(x, 1:length(dims), W=weights, fast=true), arr, dims=dims)
    else
        res = sum(arr, dims=dims)
    end
    return res/(β^length(dims))
end

#TODO: sum_q
