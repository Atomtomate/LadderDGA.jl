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

"""
    gen_νω_part(sP::SimulationParameters, workerpool::AbstractWorkerPool)

Returns 
"""
function gen_νω_part(sP::SimulationParameters, workerpool::AbstractWorkerPool)
    ωνi_range::Vector{NTuple{4, Int64}} = [(ωn,νn,ωi,νi) for (ωi,ωn) in enumerate(-sP.n_iω:sP.n_iω) 
                 for (νi,νn) in enumerate(((-(sP.n_iν+sP.n_iν_shell)):(sP.n_iν+sP.n_iν_shell-1)) .- trunc(Int,sP.shift*ωn/2))]
    ωνi_part = par_partition(ωνi_range, length(workerpool))
    return ωνi_range, ωνi_part
end

mutable struct WorkerCache
    initialized::Bool
    νω_map::Array{NTuple{4,Int}}
    ωind_map::Dict{Int,Int}
    χsp::Array{ComplexF64,2}
    χch::Array{ComplexF64,2}
    γsp::Array{ComplexF64,3}
    γch::Array{ComplexF64,3}
    λ₀::Array{ComplexF64,3}
    G::GνqT
    kG::Union{KGrid, Nothing}
    function WorkerCache()
        new(false, NTuple{4,Int}[], Dict{Int,Int}(), Array{ComplexF64,2}(undef,0,0),
            Array{ComplexF64,2}(undef,0,0), Array{ComplexF64,3}(undef,0,0,0), Array{ComplexF64,3}(undef,0,0,0),
            Array{ComplexF64,3}(undef,0,0,0), OffsetMatrix(Array{ComplexF64,2}(undef,0,0)), nothing)
    end
end

function initialize_cache(νω_map::Array{NTuple{4,Int}}, ωind_map::Dict{Int,Int}, χsp::χT, χch::χT, 
                          γsp::γT, γch::γT, λ₀::γT, G::GνqT, kG::KGrid; override=false)
    if !wcache.initialized || override
        wcache.νω_map = νω_map
        wcache.ωind_map = ωind_map
        wcache.χsp = χsp
        wcache.χch = χch
        wcache.γsp = γsp
        wcache.γch = γch
        wcache.λ₀ = λ₀
        wcache.G =G
        wcache.kG = kG
        wcache.initialized = true
    end
end

function update_χ(type::Symbol, χ::χT)
    if type == :sp
        wcache.χsp = χ
    elseif type == :ch
        wcache.χch = χ
    else
        throw("Unkown channel type for worker update of chi")
    end
end
function update_χ(type::Symbol, χ_uncorrected::χT, λ::Float64)
    if type == :sp
        χ_λ!(wcache.χsp, χ_uncorrected, λ)
    elseif type == :ch
        χ_λ!(wcache.χch, χ_uncorrected, λ)
    else
        throw("Unkown channel type for worker update of chi")
    end
end
