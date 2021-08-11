import Base.copy

const ω_axis = 3;
const ν_axis = 2;
const q_axis = 1;

# TODO: define getproperty to mask internals Base.propertynames(F::TYPE, private::Bool=false) =
# see LU type (lu.jl 332)
"""
    ImpurityQuantities

Contains all quantities of a given channel, computed by DMFT
"""
struct ImpurityQuantities
    Γ::AbstractArray{Complex{Float64},3}
    χ::AbstractArray{Complex{Float64},3}
    χ_ω::Array{Complex{Float64},1}
    χ_loc::Complex{Float64}
    usable_ω::AbstractArray{Int,1}
    tailCoeffs::AbstractArray{Float64,1}
end

"""
    NonLocalQuantities

Contains all non local quantities computed by the lDGA code
"""
mutable struct NonLocalQuantities{T1 <: Union{Complex{Float64}, Float64}, T2 <: Union{Complex{Float64}, Float64}}
    χ::AbstractArray{T1,2}
    γ::AbstractArray{T2,3}
    usable_ω::AbstractArray
    λ::Float64
end


Base.copy(x::T) where T <: Union{NonLocalQuantities, ImpurityQuantities} = T([deepcopy(getfield(x, k)) for k ∈ fieldnames(T)]...)

const ΓT = AbstractArray{Complex{Float64},3}
const BubbleT = AbstractArray{Complex{Float64},3}
const GνqT = AbstractArray
const qGridT = Array{Tuple{Int64,Int64,Int64},1}
