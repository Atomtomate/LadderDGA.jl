import Base.copy
#TODO: all data types with tails should have their own data structure and know about their summation type
#TODO: overload data types to behave like AbstractArrays for math and I/O

#TODO: each struct should know about its axis labels
const ω_axis = 3;
const ν_axis = 2;
const q_axis = 1;

const _eltype = ComplexF64
const ΓT = Array{_eltype,3}
const FT = Array{_eltype,3}
const γT = Array{_eltype,3}
const χT = Array{_eltype,2}
const GνqT = OffsetMatrix
const qGridT = Array{Tuple{Int64,Int64,Int64},1}

struct χ₀T
    data::Array{_eltype,3}
    asym::Array{_eltype,2}
    axes::Dict{Symbol, Int}
    #TODO: grid::FreqGridType
    #TODO: calculate t1,t2 of bubble from GFtails (first: define GF struct)
    function χ₀T(data::Array{_eltype,3}, kG::ReducedKGrid, t1::Vector{ComplexF64}, t2::Float64,
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
    χ::Array{ComplexF64,2}
    γ::Array{ComplexF64,3}
    usable_ω::AbstractArray
    λ::Float64
end


Base.copy(x::T) where T <: Union{NonLocalQuantities, ImpurityQuantities} = T([deepcopy(getfield(x, k)) for k ∈ fieldnames(T)]...)

