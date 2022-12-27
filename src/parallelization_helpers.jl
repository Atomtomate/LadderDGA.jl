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

Returns partition of frequency grid, according to the number of workers in `wp`.
"""
function gen_νω_part(sP::SimulationParameters, workerpool::AbstractWorkerPool)
    ωνi_range::Vector{NTuple{4, Int64}} = [(ωn,νn,ωi,νi) for (ωi,ωn) in enumerate(-sP.n_iω:sP.n_iω) 
                 for (νi,νn) in enumerate(((-(sP.n_iν+sP.n_iν_shell)):(sP.n_iν+sP.n_iν_shell-1)) .- trunc(Int,sP.shift*ωn/2))]
    ωνi_part = par_partition(ωνi_range, length(workerpool))
    return ωνi_range, ωνi_part
end

"""
    gen_ω_part(sP::SimulationParameters, workerpool::AbstractWorkerPool)

Returns partition of bosonic frequencies grid, according to the number of workers in `wp`.
"""
function gen_ω_part(sP::SimulationParameters, workerpool::AbstractWorkerPool)
    ωi_range::Vector{NTuple{2, Int64}} = [(ωn,ωi) for (ωi,ωn) in enumerate(-sP.n_iω:sP.n_iω)]
    ωi_part = par_partition(ωi_range, length(workerpool))
    return ωi_range, ωi_part
end

mutable struct WorkerCache
    initialized::Dict{Symbol,Bool}
    GLoc_fft::GνqT
    GLoc_fft_reverse::GνqT
    kG::Union{KGrid, Nothing}
    mP::Union{ModelParameters, Nothing}
    sP::Union{SimulationParameters, Nothing}
    χ₀::Array{_eltype, 3}
    χ₀Asym::Array{_eltype, 2}
    χ₀Indices::Vector{NTuple{2,Int}}
    # χ_ind::Dict{Int,Int}
    # γ_ind::Dict{Int,Int}
    # χsp::Array{ComplexF64,2}
    # χch::Array{ComplexF64,2}
    # γsp::Array{ComplexF64,2}
    # γch::Array{ComplexF64,2}
    # λ₀::Array{ComplexF64,3}
    function WorkerCache()
            # false, NTuple{4,Int}[], Dict{Int,Int}(), 
            # Array{ComplexF64,2}(undef,0,0), Array{ComplexF64,2}(undef,0,0), 
            # Array{ComplexF64,3}(undef,0,0,0), Array{ComplexF64,3}(undef,0,0,0), Array{ComplexF64,3}(undef,0,0,0), 
            # OffsetMatrix(Array{ComplexF64,2}(undef,0,0)), OffsetMatrix(Array{ComplexF64,2}(undef,0,0)), 
            # nothing)
        new(Dict(:GLoc_fft => false, :GLoc_fft_reverse => false, :kG => false), 
            OffsetMatrix(Array{ComplexF64,2}(undef,0,0)),
            OffsetMatrix(Array{ComplexF64,2}(undef,0,0)),
            nothing, nothing, nothing,
            Array{_eltype,3}(undef,0,0,0), Array{_eltype,2}(undef,0,0), 
            Vector{NTuple{3,Int}}(undef, 0)
        )
    end
end

function update_wcache!(name::Symbol, val; override=true)
    if !haskey(wcache[].initialized, name) || !wcache[].initialized[name] || override
        if name == :kG
            val = gen_kGrid(val[1], val[2])
        end
        setfield!(wcache[], name, val)
        wcache[].initialized[name] = true
    else
        @warn "Value of $name already set."
    end
end

function get_workerpool()
    default_worker_pool()
end
