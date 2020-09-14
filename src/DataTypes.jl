"""
Contains all quantities of a given channel, computed by DMFT
"""
struct ImpurityQuantities
    Γ::SharedArray{Complex{Float64},3}
    χ_general::SharedArray{Complex{Float64},3}
    χ_physical::SharedArray{Complex{Float64},1}
    χ_loc::Complex{Float64}
    usable_ω::UnitRange{Int64}
end

struct NonLocalQuantities
    χ_general::SharedArray{Complex{Float64},2}
    χ_physical::SharedArray{Complex{Float64},1}
    γ::SharedArray{Complex{Float64},3}
    usable_ω::UnitRange{Int64}
end
