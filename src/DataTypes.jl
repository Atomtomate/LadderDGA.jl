import Base.copy

const ω_axis = 3;
const ν_axis = 2;
const q_axis = 1;

const _eltype = ComplexF64
const ΓT = Array{_eltype,3}
const BubbleT = Array{_eltype,3}
const FUpDoT = Array{_eltype,3}
const γT = Array{_eltype,3}
const χT = Array{_eltype,2}
#TODO: overload for 2D and 3D grids
const GνqT = Array
const qGridT = Array{Tuple{Int64,Int64,Int64},1}


# TODO: define getproperty to mask internals Base.propertynames(F::TYPE, private::Bool=false) =
# see LU type (lu.jl 332)
"""
    ImpurityQuantities

Contains all quantities of a given channel, computed by DMFT
"""
struct ImpurityQuantities
    Γ::Array{_eltype,3}
    χ::Array{_eltype,3}
    χ_ω::Array{_eltype,1}
    χ_loc::Complex{Float64}
    usable_ω::AbstractArray{Int,1}
    tailCoeffs::AbstractArray{Float64,1}
end

"""
    NonLocalQuantities

Contains all non local quantities computed by the lDGA code
"""
mutable struct NonLocalQuantities
    χ::Array{_eltype,2}
    γ::Array{_eltype,3}
    usable_ω::AbstractArray
    λ::Float64
end


Base.copy(x::T) where T <: Union{NonLocalQuantities, ImpurityQuantities} = T([deepcopy(getfield(x, k)) for k ∈ fieldnames(T)]...)

