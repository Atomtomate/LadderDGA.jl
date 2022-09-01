# ==================================================================================================== #
#                                           GFTools.jl                                                 #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
#   Last Edit Date  : 01.09.22                                                                         #
# ----------------------------------------- Description ---------------------------------------------- #
#   Green's function and Matsubara frequency related functions                                         #
# -------------------------------------------- TODO -------------------------------------------------- #
#   This file could be a separate module                                                               #
#   Most functions in this files are not used in this project.                                         #
#   Test and optimize functions                                                                        #
#   Rename subtrac_tail and make it more general for arbitrary tails                                   #
# ==================================================================================================== #



# =================================== Matsubara Frequency Helpers ====================================

iν_array(β::Real, grid::AbstractArray{Int64,1}) = ComplexF64[1.0im*((2.0 *el + 1)* π/β) for el in grid]
iν_array(β::Real, size::Integer)    = ComplexF64[1.0im*((2.0 *i + 1)* π/β) for i in 0:size-1]
iω_array(β::Real, grid::AbstractArray{Int64,1}) = ComplexF64[1.0im*((2.0 *el)* π/β) for el in grid]
iω_array(β::Real, size::Integer)    = ComplexF64[1.0im*((2.0 *i)* π/β) for i in 0:size-1]


# ===================================== Dyson Equations Helpers ======================================

"""
    G(ind::Int64, Σ::Array{ComplexF64,1}, ϵkGrid, β::Float64, μ)

Constructs GF from k-independent self energy, using the Dyson equation
and the dispersion relation of the lattice.
"""
@inline function G(ind::Int64, Σ::Array{ComplexF64,1},
                   ϵkGrid::Union{Array{Float64,1},Base.Generator}, β::Float64, μ::Float64)
    Σν = get_symm_f(Σ,ind)
    return map(ϵk -> G_from_Σ(ind, β, μ, ϵk, Σν), ϵkGrid)
end

@inline function G(ind::Int64, Σ::Array{ComplexF64,2},
                   ϵkGrid, β::Float64, μ::Float64)
    Σνk = get_symm_f_2(Σ,ind)
    return reshape(map(((ϵk, Σνk_i),) -> G_from_Σ(ind, β, μ, ϵk, Σνk_i), zip(ϵkGrid, Σνk)), size(ϵkGrid)...)
end


@inline @fastmath G_from_Σ(n::Int64, β::Float64, μ::Float64, ϵₖ::Float64, Σ::ComplexF64) =
                    1/((π/β)*(2*n + 1)*1im + μ - ϵₖ - Σ)
@inline @fastmath G_from_Σ(mf::ComplexF64, β::Float64, μ::Float64, ϵₖ::Float64, Σ::ComplexF64) =
                    1/(mf + μ - ϵₖ - Σ)

G_from_Σ(Σ::AbstractArray, ϵkGrid::AbstractArray, range::UnitRange{Int64}, mP::ModelParameters) = [G(ind, Σ, ϵkGrid, mP.β, mP.μ) for ind in range]


Σ_Dyson(GBath::Array{ComplexF64,1}, GImp::Array{ComplexF64,1}, eps = 1e-3) =
    Σ::Array{ComplexF64,1} =  1 ./ GBath .- 1 ./ GImp


# =============================== Frequency Tail Modification Helpers ================================

"""
    subtract_tail(inp::AbstractArray{T,1}, c::Float64, iω::Array{ComplexF64,1}) where T <: Number

subtract the c/(iω)^2 high frequency tail from `inp`.
"""
function subtract_tail(inp::AbstractArray{T,1}, c::Float64, iω::Array{ComplexF64,1}) where T <: Number
    res = Array{eltype(inp),1}(undef, length(inp))
    subtract_tail!(res, inp, c, iω)
    return res
end

"""
    subtract_tail!(outp::AbstractArray{T,1}, inp::AbstractArray{T,1}, c::Float64, iω::Array{ComplexF64,1}) where T <: Number

subtract the c/(iω)^2 high frequency tail from `inp` and store in `outp`. See also [`subtract_tail`](@ref subtract_tail)
"""
function subtract_tail!(outp::AbstractArray{T,1}, inp::AbstractArray{T,1}, c::Float64, iω::Array{ComplexF64,1}) where T <: Number
    for n in 1:length(inp)
        if iω[n] != 0
            outp[n] = inp[n] - (c/(iω[n]^2))
        else
            outp[n] = inp[n]
        end
    end
end

