

"""
Contains all quantities of a given channel, computed by DMFT
"""
struct LocalQuantities
    Γ::Array{Complex{Float64},3}
    χ_general::Array{Complex{Float64},3}
    χ_physical::Array{Complex{Float64},1}
    usable_ω::UnitRange{Int64}
end

struct NonLocalQuantities
    χ_general::Array{Complex{Float64},3}
    χ_physical::Array{Complex{Float64},1}
    γ::Array{Complex{Float64},1}
    usable_ω::UnitRange{Int64}
end
