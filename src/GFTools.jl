"""
Most functions in this files are not used in this project.
"""
# TODO: test everything for single orbital case
iν_array(β::Real, grid::AbstractArray{Int64,1}) = ComplexF64[1.0im*((2.0 *el + 1)* π/β) for el in grid]
iν_array(β::Real, size::Integer)    = ComplexF64[1.0im*((2.0 *i + 1)* π/β) for i in 0:size-1]
iω_array(β::Real, grid::AbstractArray{Int64,1}) = ComplexF64[1.0im*((2.0 *el)* π/β) for el in grid]
iω_array(β::Real, size::Integer)    = ComplexF64[1.0im*((2.0 *i)* π/β) for i in 0:size-1]


function FUpDo_from_χDMFT(χupdo::AbstractArray{T,3}, GImp, env, mP, sP) where T <: Number
    FUpDo = similar(χupdo)
    #TODO: indices should not be computed by hand here, get them from input
    jldopen(env.freqFile) do f
        freqList = f["freqList"]
        for f in freqList
            i = f[1] + sP.n_iω+1
            j = f[2] + sP.n_iν+1 + trunc(Int,sP.shift*f[1]/2)
            k = f[3] + sP.n_iν+1 + trunc(Int,sP.shift*f[1]/2)
    #TODO: fix different permdims in file and julia code (inconsistency!, names axes?)
            FUpDo[i,j,k] = χupdo[i,j,k]/(mP.β^2 * get_symm_f(GImp,f[2]) * get_symm_f(GImp,f[1]+f[2])
                               * get_symm_f(GImp,f[3]) * get_symm_f(GImp,f[1]+f[3]))
        end
    end
    return FUpDo
end


"""
    G(ind::Int64, Σ::Array{Complex{Float64},1}, ϵkGrid, β::Float64, μ)

Constructs GF from k-independent self energy, using the Dyson equation
and the dispersion relation of the lattice.
"""
#TODO: preallocate and compute
@inline function G(ind::Int64, Σ::Array{Complex{Float64},1},
                   ϵkGrid::Union{Array{Float64,1},Base.Generator}, β::Float64, μ::Float64)
    Σν = get_symm_f(Σ,ind)
    return map(ϵk -> G_from_Σ(ind, β, μ, ϵk, Σν), ϵkGrid)
end

#TODO: preallocate and compute
@inline function G(ind::Int64, Σ::Array{Complex{Float64},2},
                   ϵkGrid, β::Float64, μ::Float64)
    Σνk = get_symm_f(Σ,ind)
    return reshape(map(((ϵk, Σνk_i),) -> G_from_Σ(ind, β, μ, ϵk, Σνk_i), zip(ϵkGrid, Σνk)), size(ϵkGrid)...)
end


function Σ_Dyson(GBath::Array{Complex{Float64},1}, GImp::Array{Complex{Float64},1}, eps = 1e-3)
    @inbounds Σ::Array{Complex{Float64},1} =  1 ./ GBath .- 1 ./ GImp
    return Σ
end

@inline @fastmath G_from_Σ(n::Int64, β::Float64, μ::Float64, ϵₖ::T, Σ::Complex{Float64}) where T <: Real =
                    1/((π/β)*(2*n + 1)*1im + μ - ϵₖ - Σ)
@inline @fastmath G_from_Σ(mf::Complex{Float64}, β::Float64, μ::Float64, ϵₖ::Float64, Σ::Complex{Float64}) =
                    1/(mf + μ - ϵₖ - Σ)

#TODO optimize these helpers
G_from_Σ(Σ, ϵkGrid,
                 range::UnitRange{Int64}, mP::ModelParameters) = [G(ind, Σ, ϵkGrid, mP.β, mP.μ) for ind in range]


function subtract_tail!(outp::AbstractArray{T,1}, inp::AbstractArray{T,1}, c::Float64, iω::Array{Complex{Float64},1}) where T <: Number
    for n in 1:length(inp)
        if iω[n] != 0
            outp[n] = inp[n] - (c/(iω[n]^2))
        else
            outp[n] = inp[n]
        end
    end
end

function subtract_tail(inp::AbstractArray{T,1}, c::Float64, iω::Array{Complex{Float64},1}) where T <: Number
    res = Array{eltype(inp),1}(undef, length(inp))
    for n in 1:length(inp)
        if iω[n] != 0
            res[n] = inp[n] - (c/(iω[n]^2))
        else
            res[n] = inp[n]
        end
    end
    return res
end
