"""
Most functions in this files are not used in this project.
"""
# TODO: test everything for single orbital case
iν_array(β::Real, grid::AbstractArray{Int64,1}) = ComplexF64[1.0im*((2.0 *el + 1)* π/β) for el in grid]
iν_array(β::Real, size::Integer)    = ComplexF64[1.0im*((2.0 *i + 1)* π/β) for i in 0:size-1]
iω_array(β::Real, grid::AbstractArray{Int64,1}) = ComplexF64[1.0im*((2.0 *el)* π/β) for el in grid]
iω_array(β::Real, size::Integer)    = ComplexF64[1.0im*((2.0 *i)* π/β) for i in 0:size-1]
"""
    G(ind::Int64, Σ::Array{ComplexF64,1}, ϵkGrid, β::Float64, μ)

Constructs GF from k-independent self energy, using the Dyson equation
and the dispersion relation of the lattice.
"""
#TODO: preallocate and compute
@inline function G(ind::Int64, Σ::Array{ComplexF64,1},
                   ϵkGrid::Union{Array{Float64,1},Base.Generator}, β::Float64, μ::Float64)
    Σν = get_symm_f(Σ,ind)
    return map(ϵk -> G_from_Σ(ind, β, μ, ϵk, Σν), ϵkGrid)
end

#TODO: preallocate and compute
@inline function G(ind::Int64, Σ::Array{ComplexF64,2},
                   ϵkGrid, β::Float64, μ::Float64)
    Σνk = get_symm_f_2(Σ,ind)
    return reshape(map(((ϵk, Σνk_i),) -> G_from_Σ(ind, β, μ, ϵk, Σνk_i), zip(ϵkGrid, Σνk)), size(ϵkGrid)...)
end


Σ_Dyson(GBath::Array{ComplexF64,1}, GImp::Array{ComplexF64,1}, eps = 1e-3) =
    Σ::Array{ComplexF64,1} =  1 ./ GBath .- 1 ./ GImp

@inline @fastmath G_from_Σ(n::Int64, β::Float64, μ::Float64, ϵₖ::Float64, Σ::ComplexF64) =
                    1/((π/β)*(2*n + 1)*1im + μ - ϵₖ - Σ)
@inline @fastmath G_from_Σ(mf::ComplexF64, β::Float64, μ::Float64, ϵₖ::Float64, Σ::ComplexF64) =
                    1/(mf + μ - ϵₖ - Σ)

#TODO optimize these helpers
G_from_Σ(Σ::AbstractArray, ϵkGrid::AbstractArray, range::UnitRange{Int64}, mP::ModelParameters) = [G(ind, Σ, ϵkGrid, mP.β, mP.μ) for ind in range]


function subtract_tail!(outp::AbstractArray{T,1}, inp::AbstractArray{T,1}, c::Float64, iω::Array{ComplexF64,1}) where T <: Number
    for n in 1:length(inp)
        if iω[n] != 0
            outp[n] = inp[n] - (c/(iω[n]^2))
        else
            outp[n] = inp[n]
        end
    end
end

function subtract_tail(inp::AbstractArray{T,1}, c::Float64, iω::Array{ComplexF64,1}) where T <: Number
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
