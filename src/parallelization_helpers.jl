@inline _parallel_decision(Niω::Int, Nk::Int)::Bool = false# (Niω < 10 || Nk < 100) ? false : true

"""
    par_partition(set::AbstractVector, N::Int)

Returns list of indices for partition of `set` into `N` (almost) equally large segements.
"""
function par_partition(set::AbstractVector, N::Int)
    N = N <= 0 ? 1 : N
    s,r = divrem(length(set),N)
    [(i*s+1+(i<r)*(i)+(i>=r)*r):(i+1)*s+(i<r)*(i+1)+(i>=r)*r for i in 0:(N-1)]
end

function qω_partition(qList::AbstractVector, ωList::AbstractVector, N::Int)
    qωi_range = collect(Base.product(qList, ωList))[:]
    p = par_partition(set, N)
end

mutable struct WorkerCache
    initialized::Bool
    νω_range::Array{NTuple{4,Int}}
    ωind_map::Dict{Int,Int}
    χsp::Array{ComplexF64,2}
    χch::Array{ComplexF64,2}
    γsp::Array{ComplexF64,3}
    γch::Array{ComplexF64,3}
    λ₀::Array{ComplexF64,3}
    G::GνqT
    kG::Union{KGrid, Nothing}
    function WorkerCache()
        new(false, NTuple{4,Int}[], Dict{Int,Int}(), Array{ComplexF64,2}(undef,0,0), Array{ComplexF64,2}(undef,0,0),  
            Array{ComplexF64,3}(undef,0,0,0), Array{ComplexF64,3}(undef,0,0,0), Array{ComplexF64,3}(undef,0,0,0),
            OffsetMatrix(Array{ComplexF64,2}(undef,0,0)), nothing)
    end
end
