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
function build_design_matrix(imin, imax, ncoeffs)
    M = zeros(BigFloat, (ncoeffs, ncoeffs))
    for i = imin:imax, k = 0:(ncoeffs-1), l in 0:(ncoeffs-1)
        M[l+1,k+1] += 1.0 / ((BigFloat(i)^(k+l)))
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
function build_weights(imin, imax, ncoeffs)
    M = build_design_matrix(imin, imax, ncoeffs)
    Minv = inv(M)
    w = zeros(Float64, (ncoeffs, length(imin:imax)))
    for k=1:ncoeffs, l = 0:(ncoeffs-1), (ii,i) = enumerate(imin:imax)
        w[k,ii] += Float64(Minv[k, l+1] / (BigFloat(i)^l), RoundDown)
    end
    return w
end

function fit_νsum(νmax_start, νmax_end, ncoeffs, data; W = nothing, precision = 100000)
    result = zeros((size(data,1)))
    setprecision(precision) do
        W = if W == nothing build_weights(νmax_start, νmax_end, ncoeffs) else W end
        if ndims(data) == 1
            result = dot(real(data) , W[1,:])
        else
            for ωi in 1:size(data,1)
                result[ωi] = dot(real(data)[ωi,:] , W[1,:])
            end
        end
    end
    return result
end


"""
    Sums first νmax entries of any array along given dim for νmax = 1:size(arr,dim).
"""
function fit_ν_sum(f, νmax_start, modelParams, simParams, dims; W = nothing, ω_shift = 0,  n_tail = 6)
    n_iν   = minimum(size(f)[dims])
    dims   = Tuple(dims)
    νmax_end = Int(n_iν/2)
    ν_cut  = νmax_end - νmax_start

    #TODO: keep dins as singleton dims?
    f_νmax = dropdims(sum_νmax(f, ν_cut+1, dims=dims); dims=dims)/(modelParams.β^length(dims))
    νdim   = ndims(f_νmax) + 1
    for ν_cut_i in (ν_cut):-1:1
        f_νmax = cat(f_νmax, dropdims(sum_νmax(f, ν_cut_i; dims=dims); dims=dims)/(modelParams.β^length(dims)); dims=νdim)
    end

    #TODO: allow numeric fit and use method = :analytic_lsq,
    #fit_internal(arr) = fit_tail(arr, iν_arr, tail_func, n_tail)
    #tail = mapslices(fit_internal, f_νmax; dims=νdim)
    @time tail = fit_νsum(νmax_start, νmax_end, n_tail, f_νmax, W = W)
    return f_νmax, tail
end

function approx_full_sum(f, start, modelParams, simParams, dims; W = nothing, ω_shift = 0,  n_tail = 6)
    _, tail_c = fit_ν_sum(f, start, modelParams, simParams, dims, W = W, ω_shift = ω_shift, n_tail = n_tail)
    ax = collect(collect.(axes(tail_c)))
    ax[end] = [1]
    sum_approx = tail_c[ax...]
    return sum_approx
end

function fit_bubble_test(χ, modelParams, simParams)
    iν_arr = iν_array(modelParams.β, simParams.n_iν)
    c0 = zeros(length(1:(size(χ,2)-3)), size(χ,1), size(χ,3))
    start = 25
    qi = 1
    ωi = 3
    for start in 15:(size(χ,2)-3)
        for ωi in 1:size(χ,1)
            for qi in 1:size(χ,3)
                c0[start, ωi, qi] = fit_tail(-χ[ωi,start:end,qi], iν_arr[start:end], tail_func_full, 6)[1]
            end
        end
    end
    return c0
end
