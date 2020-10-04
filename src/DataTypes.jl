"""
Contains all quantities of a given channel, computed by DMFT
"""
struct ImpurityQuantities
    Γ::SharedArray{Complex{Float64},3}
    χ::SharedArray{Complex{Float64},3}
    χ_ω::SharedArray{Complex{Float64},1}
    χ_loc::Complex{Float64}
    usable_ω::UnitRange{Int64}
end

struct NonLocalQuantities{T <: Union{Complex{Float64}, Float64}}
    χ::SharedArray{T,2}
    γ::SharedArray{T,3}
    usable_ω::UnitRange{Int64}
    λ::Float64
end

const ΓT = SharedArray{Complex{Float64},3}
const BubbleT = SharedArray{Complex{Float64},3}
const GνqT = SharedArray{Complex{Float64},2}
const qGridT = Array{Tuple{Int64,Int64,Int64},1}
