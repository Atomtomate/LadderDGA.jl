using Plots

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


function fit_tail(G, iν_array, tail_func = tail_func_cmplx, n_tail = 8)
    @assert length(G) == length(iν_array)
    cost(c) = sum(abs2.(G .- tail_func(iν_array, c)))
    #print(Optim.optimize(cost, zeros(n_tail), Optim.Newton(); autodiff = :forward))
    res = Optim.minimizer(Optim.optimize(cost, zeros(n_tail), Optim.Newton(); autodiff = :forward))
    return res
end


"""
    Sums first νmax entries of any array along given dim for νmax = 1:size(arr,dim).
"""
function fit_F_sum(f, start, modelParams, simParams, dims; ω_shift = 0, tail_func = tail_func_cmplx, n_tail = 6)
    n_iν   = minimum(size(f)[dims])
    iν_arr = iν_array(modelParams.β, n_iν)[start:end]
    dims   = Tuple(dims)
    #TODO: keep dins as singleton dims?
    f_νmax = dropdims(sum_νmax(f, start, dims=dims); dims=dims)/(modelParams.β^length(dims))
    νdim   = ndims(f_νmax) + 1
    for νmax in (start+1):n_iν
        f_νmax = cat(f_νmax, dropdims(sum_νmax(f, νmax; dims=dims); dims=dims)/(modelParams.β^length(dims)); dims=νdim)
    end

    fit_internal(arr) = fit_tail(arr, iν_arr, tail_func, n_tail)
    tail = mapslices(fit_internal, f_νmax; dims=νdim)
    return f_νmax, tail
end

function approx_full_sum(f, start, modelParams, simParams, dims; ω_shift = 0, tail_func = tail_func_cmplx, n_tail = 6)
    _, tail_c = fit_F_sum(f, start, modelParams, simParams, dims, ω_shift = ω_shift, tail_func = tail_func, n_tail = n_tail)
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
