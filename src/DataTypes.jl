# ==================================================================================================== #
#                                           DataTypes.jl                                               #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#   Data types for Matsubara functions.                                                                #
# -------------------------------------------- TODO -------------------------------------------------- #
#   define getproperty to mask internals of MatsubaraFunction                                          #
#      Base.propertynames(F::TYPE, private::Bool=false) = ... see LU type (lu.jl 332)                  #
#   Σ data type                                                                                        #
#   Struct with asymptotics for Green's fnctions                                                       #
#   χ₀: calculate t1,t2 of bubble from GFtails (first: define GF struct)                               #
# ==================================================================================================== #


import Base.copy
const ω_axis = 3;
const ν_axis = 2;
const q_axis = 1;

# =========================================== Static Types ===========================================
const _eltype = ComplexF64
const ΓT = Array{_eltype,3}
const FT = Array{_eltype,3}
const GνqT = OffsetMatrix{ComplexF64, Matrix{ComplexF64}}

abstract type MatsubaraFunction{T,N} <: AbstractArray{T,N} end


# =========================================== Struct Types ===========================================
"""
    χ₀T <: MatsubaraFunction

Struct for the bubble term. The `q`, `ω` dependent asymptotic behavior is computed from the 
`t1` and `t2` input.

Fields
-------------
- **`data`**         : `Array{ComplexF64,3}`, data
- **`asym`**         : `Array{ComplexF64,2}`, `[q, ω]` dependent asymptotic behavior.
- **`axes`**         : `Dict{Symbol,Int}`, Dictionary mapping `:q, :ν, :ω` to the axis indices.
"""
struct χ₀T <: MatsubaraFunction{_eltype,3}
    data::Array{_eltype,3}
    asym::Array{_eltype,2}
    axes::Dict{Symbol, Int}
    function χ₀T(data::Array{_eltype,3}, kG::KGrid, t1::Vector{ComplexF64}, t2::Float64,
                 β::Float64, ω_grid::AbstractVector{Int}, n_iν::Int, shift::Int)
        χ₀_rest = χ₀_shell_sum_core(β, ω_grid, n_iν, shift)
        c1 = real.(kintegrate(kG, t1))
        c2 = real.(conv(kG, t1, t1))
        c3 = real.(kintegrate(kG, t1 .^ 2) .+ t2)
        asym = Array{_eltype, 2}(undef, length(c2), length(ω_grid))
        for (ωi,ωn) in enumerate(ω_grid)
            for (qi,c2i) in enumerate(c2)
                asym[qi,ωi] = χ₀_shell_sum(χ₀_rest, ωn, β, c1, c2[qi], c3)
            end
        end
        new(data,asym,Dict(:q => 1, :ν => 2, :ω => 3))
    end
end


"""
    χT <: MatsubaraFunction

Struct for the non-local susceptibilities. 

Fields
-------------
- **`data`**         : `Array{ComplexF64,3}`, data
- **`axes`**         : `Dict{Symbol,Int}`, Dictionary mapping `:q, :ω` to the axis indices.
- **`λ`**            : `Float64`, λ correction parameter.
- **`usable_ω`**     : `AbstractArray{Int}`, usable indices for which data is assumed to be correct. See also [`find_usable_interval`](@ref find_usable_interval)
"""
struct χT <: MatsubaraFunction{_eltype, 2}
    data::Array{_eltype,2}
    axes::Dict{Symbol, Int}
    λ::Float64
    usable_ω::AbstractArray{Int}
    function χT(data::Array{_eltype, 2})
        @warn "DBG: currently forcing omega FULL range!!"
        new(data, Dict(:q => 1, :ω => 2), 0.0, 1:size(data,2))
    end
end


"""
    γT <: MatsubaraFunction

Struct for the non-local triangular vertex. 

Fields
-------------
- **`data`**         : `Array{ComplexF64,3}`, data
- **`axes`**         : `Dict{Symbol,Int}`, Dictionary mapping `:q, :ν, :ω` to the axis indices.
"""
struct γT <: MatsubaraFunction{ComplexF64,3}
    data::Array{ComplexF64,3}
    axes::Dict{Symbol, Int}
    function γT(data::Array{ComplexF64, 3})
        new(data, Dict(:q => 1, :ν => 2, :ω => 3))
    end
end

# ============================================= Interface ============================================

# ---------------------------------------------- Indexing --------------------------------------------
Base.size(arr::T) where T <: MatsubaraFunction = size(arr.data)
Base.getindex(arr::T, i::Int) where T <: MatsubaraFunction = Base.getindex(arr.data, i)
Base.getindex(arr::χ₀T, I::Vararg{Int,3}) = Base.getindex(arr.data, I...)
Base.getindex(arr::χT, I::Vararg{Int,2}) = Base.getindex(arr.data, I...)
Base.getindex(arr::γT, I::Vararg{Int,3}) = Base.getindex(arr.data, I...)
Base.setindex!(arr::T, v, i::Int) where T <: MatsubaraFunction = Base.setindex!(arr.data, v, i)
Base.setindex!(arr::χ₀T, v, I::Vararg{Int,3}) = Base.setindex!(arr.data, v, I...)
Base.setindex!(arr::χT, v, I::Vararg{Int,2}) = Base.setindex!(arr.data, v, I...)
Base.setindex!(arr::γT, v, I::Vararg{Int,3}) = Base.setindex!(arr.data, v, I...)

# --------------------------------------------- Iteration --------------------------------------------
# --------------------------------------------- Broadcast --------------------------------------------
