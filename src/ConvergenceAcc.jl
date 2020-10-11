module Shanks

struct SumApproximation{T}
    res::Array{T,1}
    method
    #Order? 
    absTol::Float64
    relTol::Float64
end

function _shanks(arr::AbstractArray{T, 1}; atol_denom = 10*eps(Float64)) where T
    v0 = view(arr, 2:(length(arr)-1))
    vm1 = view(arr, 1:(length(arr)-2))
    vp1 = view(arr, 3:(length(arr)-0))
    denom = (vp1 .- v0) .- (v0 .- vm1)
    any(abs.(denom) .< atol_denom) && println("hit atol_denom")
    return any(abs.(denom) .< atol_denom) ? (vp1,true) : (vp1 .- (((vp1 .- v0) .^ 2) ./ denom),false)
end

function shanks(arr::AbstractArray{T,1}; order::Int=floor(Int64,(length(arr)-1)/2), csum_inp=false) where T
    partial = !csum_inp ? cumsum(arr) : arr
    i = 0
    while i < order
        i += 1
        partial, conv = _shanks(partial)
        conv && break
    end
    return partial[end], i
end

function build_weights(N::Int)
    M = zeros(BigFloat, (N, N))
    for i in 0:(N-1)
        for j in 0:(i)
            #println("i=$i, j=$j, res = ", j^i * (-1)^(i+j))
            M[i+1,j+1] = (big(j)^i * (-1)^(j+i)) / (factorial(big(j)) * factorial(big(i-j)))
        end
    end
    return M
end

function _richardson_naive(arr::AbstractArray{T,1}) where T
    N = length(arr) - 1
    return [(arr[i+1] * (0 + i)^N * (-1)^(i+N)) / (factorial(i) * factorial(N-i)) for i in 0:N]
end

function richardson(arr::AbstractArray{T,1}, csum_inp=false) where T
    partial = !csum_inp ? cumsum(arr) : arr
end


end
