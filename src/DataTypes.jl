"""
Contains all quantities of a given channel, computed by DMFT
"""
struct ImpurityQuantities
    Γ::SharedArray{Complex{Float64},3}
    χ::SharedArray{Complex{Float64},3}
    χ_ω::SharedArray{Complex{Float64},1}
    χ_loc::Complex{Float64}
    usable_ω::AbstractArray
end

struct NonLocalQuantities{T1 <: Union{Complex{Float64}, Float64}, T2 <: Union{Complex{Float64}, Float64}}
    χ::SharedArray{T1,2}
    γ::SharedArray{T2,3}
    usable_ω::AbstractArray
    λ::Float64
end

struct Simulation
end

const ΓT = SharedArray{Complex{Float64},3}
const BubbleT = SharedArray{Complex{Float64},3}
const GνqT = SharedArray{Complex{Float64},2}
const qGridT = Array{Tuple{Int64,Int64,Int64},1}
