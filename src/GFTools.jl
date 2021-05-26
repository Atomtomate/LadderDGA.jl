"""
Most functions in this files are not used in this project.
"""
# TODO: test everything for single orbital case
iν_array(β::Real, grid::AbstractArray{Int64,1}) = [1.0im*((2.0 *el + 1)* π/β) for el in grid]
iν_array(β::Real, size::Integer)    = [1.0im*((2.0 *i + 1)* π/β) for i in 0:size-1]
iω_array(β::Real, grid::AbstractArray{Int64,1}) = [1.0im*((2.0 *el)* π/β) for el in grid]
iω_array(β::Real, size::Integer)    = [1.0im*((2.0 *i)* π/β) for i in 0:size-1]


function FUpDo_from_χDMFT(χupdo, GImp, freqList, mP, sP)
    FUpDo = Array{eltype(χupdo)}(undef, 2*sP.n_iω+1, 2*sP.n_iν, 2*sP.n_iν)
    #TODO: indices should not be computed by hand here, get them from input
    for f in freqList
        i = f[1] + sP.n_iω+1
        j = f[2] + sP.n_iν+1 + trunc(Int64,sP.shift*f[1]/2)
        k = f[3] + sP.n_iν+1 + trunc(Int64,sP.shift*f[1]/2)
        FUpDo[i,j,k] = χupdo[i,j,k]/(mP.β^2 * get_symm_f(GImp,f[2]) * get_symm_f(GImp,f[1]+f[2])
                           * get_symm_f(GImp,f[3]) * get_symm_f(GImp,f[1]+f[3]))
    end
    return FUpDo
end

function Σ_Dyson(GBath::Array{Complex{Float64},1}, GImp::Array{Complex{Float64},1}, eps = 1e-3) 
    @inbounds Σ::Array{Complex{Float64},1} =  1 ./ GBath .- 1 ./ GImp
    return Σ
end

@inline @fastmath G_from_Σ(n::Int64, β::Float64, μ::Float64, ϵₖ::T, Σ::Complex{Float64}) where T <: Real =
                    1/((π/β)*(2*n + 1)*1im + μ - ϵₖ - Σ)

@inline G_from_Σ(Σ::Array{Complex{Float64}}, ϵkGrid, 
                 range::UnitRange{Int64}, mP::ModelParameters) = [G(ind, Σ, ϵkGrid, mP.β, mP.μ) for ind in range]

@inline @fastmath G_from_Σ(n::Int64, β::Float64, μ::Float64, ϵₖ::T, Σ::Complex{Interval{Float64}}) where T <: Real =
                    1/((π/β)*(2*n + 1)*1im + μ - ϵₖ - Σ)
@inline G_from_Σ(Σ::Array{Complex{Interval{Float64}}}, ϵkGrid, 
                 range::UnitRange{Int64}, mP::ModelParameters) = [G(ind, Σ, ϵkGrid, mP.β, mP.μ) for ind in range]
