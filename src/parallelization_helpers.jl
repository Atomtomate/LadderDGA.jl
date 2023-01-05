# ==================================================================================================== #
#                                    parallelization_helpers.jl                                        #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#   Functionality for parallel computation.                                                            #
# -------------------------------------------- TODO -------------------------------------------------- #
#   collect* functions need to be refactored                                                           #
# ==================================================================================================== #


# ============================================== Helpers =============================================

function get_workerpool()
    default_worker_pool()
end

@inline _parallel_decision(Niω::Int, Nk::Int)::Bool = false# (Niω < 10 || Nk < 100) ? false : true

"""
    par_partition(set::AbstractVector, N::Int)

Returns list of indices for partition of `set` into `N` (almost) equally large segements.
"""
function par_partition(set::AbstractVector, N::Int)
    N > length(set) && error("Cannot split set of length $(length(set)) into $N partitions.")
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
    ωi_range::Vector{NTuple{2, Int64}} = [el for el in enumerate(-sP.n_iω:sP.n_iω)]
    ωi_part = par_partition(ωi_range, length(workerpool))
    return ωi_range, ωi_part
end

"""
gen_ν_part(νGrid::AbstractVector{Int}, sP::SimulationParameters, N::Int)

Returns partition of fermionic frequencies grid, according to the number of workers `N`.
The partition is given as a list (of length `N`) of lists with 4-Tuples `(ωi, ωn, νi, νn)`. 
`νi` and `ωi` are the indices for the Matsubara frequencies `νn` and `ωn`.
"""
function gen_ν_part(νGrid::AbstractArray{Int}, sP::SimulationParameters, N::Int)
    νi_part = par_partition(νGrid, N)
    # (ωi, ωn, νi, νn)
    ωνgrids = [(ωel[1],ωel[2],νi,j - trunc(Int,sP.shift*ωel[2]/2)) for ωel in enumerate(-sP.n_iω:sP.n_iω), (νi,j) in enumerate(-sP.n_iν:sP.n_iν-1)]
    index_lists = [sort(filter(x->x[4] in νGrid[segment], ωνgrids), by=x->x[4]) for segment in νi_part]
    return index_lists
end

"""
    gen_ν_part_slices(data::Array{ComplexF64,3}, index_list::Vector{NTuple{4,Int}})

Rearragnes data over `q`, `ν` and `ω` axes, for EoM (see [`calc_Σ_par`](@ref calc_Σ_par)) given `index_list`, which is one element of the list of lists obtained from [`gen_ν_part`](@ref gen_ν_part).

Returns three arrays: 
    - data_res: Rearranged data, only containes values for ν, given in `index_list`, ω values not contained in `data` are set to 0.
    - νn_list: Has length `size(data_res,3)`. Contained fermionic Matsubara frequency for each index.
    - ωn_ranges: Has length `size(data_res,3)`. Containes bosonic Matsubara frequencies for each ν value.
"""
function gen_ν_part_slices(data::Array{ComplexF64,3}, index_list::Vector{NTuple{4,Int}})
    νn = map(x->x[4], index_list)
    res::Array{eltype(data),3} = zeros(eltype(data), size(data,q_axis), size(data,ω_axis), length(unique(νn)))
    νn_list = unique(νn)
    ωn_ranges = Vector{UnitRange{Int}}(undef, length(νn_list))
    for (i,νn) in enumerate(νn_list)
        slice = map(x->x[2], filter(x->x[4]==νn, index_list))
        ωn_ranges[i] = first(slice):last(slice)
    end
    νi = 1
    νn = index_list[1][4]
    for i in 1:length(index_list)
        ωi_i, _, νi_i, νn_i = index_list[i]
        if νn_i != νn
            νn = νn_i
            νi += 1
        end
        res[:, ωi_i, νi] = data[:,νi_i,ωi_i]
    end
    return res, νn_list, ωn_ranges
end


# ============================================ Worker Cache ==========================================
# ----------------------------------------------- Main -----------------------------------------------
mutable struct WorkerCache
    initialized::Dict{Symbol,Bool}
    G_fft::GνqT
    G_fft_reverse::GνqT
    kG::Union{KGrid, Nothing}
    mP::Union{ModelParameters, Nothing}
    sP::Union{SimulationParameters, Nothing}
    χ₀::Array{_eltype, 3}
    χ₀Asym::Array{_eltype, 2}
    χ₀Indices::Vector{NTuple{2,Int}}
    χsp::Array{ComplexF64,2}
    χch::Array{ComplexF64,2}
    γsp::Array{ComplexF64,3}
    γch::Array{ComplexF64,3}
    # EoM related:
    ωn_ranges::Vector{UnitRange{Int}}
    νn_indices::Vector{Int}
    λ₀::Array{ComplexF64,3}
    Kνωq_pre::Vector{ComplexF64}
    Kνωq_post::Vector{ComplexF64}
    Σ_ladder::Array{ComplexF64,2}
    Σ_initialized::Bool
    function WorkerCache()
        new(Dict(:G_fft => false, :G_fft_reverse => false, :kG => false), 
            OffsetMatrix(Array{ComplexF64,2}(undef,0,0)),
            OffsetMatrix(Array{ComplexF64,2}(undef,0,0)),
            nothing, nothing, nothing,
            Array{_eltype,3}(undef,0,0,0), Array{_eltype,2}(undef,0,0), 
            Vector{NTuple{3,Int}}(undef, 0),
            Array{ComplexF64,2}(undef, 0,0),Array{ComplexF64,2}(undef, 0,0),
            Array{ComplexF64,3}(undef, 0,0,0),Array{ComplexF64,3}(undef, 0,0,0),
            # EoM
            Vector{UnitRange{Int}}(undef, 0),
            Vector{Int}(undef, 0),
            Array{ComplexF64,3}(undef, 0,0,0),
            Vector{ComplexF64}(undef, 0),
            Vector{ComplexF64}(undef, 0),
            Array{ComplexF64,2}(undef, 0,0),
            false
        )
    end
end

# ----------------------------------------- Updates and Inits ----------------------------------------
"""
    update_wcache!(name::Symbol, val; override=true)

Updates worker cache with given name and value. Typically used through `remotecall()` on specific worker.
"""
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

function update_wcaches_G_rfft!(G::GνqT)
    wp = workers(get_workerpool())
    @sync begin
    for wi in wp
        @async remotecall_fetch(update_wcache!, wi, :G_fft_reverse, G)
    end
    end
end

"""
    initialize_EoM_cache!()

Initializes cache to correct size. `kG` and `sP` must be set first.
"""
function initialize_EoM_cache!(Nν::Int)
    !wcache[].initialized[:kG] || !wcache[].initialized[:sP] && error("Worker cache must be initialized before calling initializiation of EoM cache.")
    Nq = length(wcache[].kG.kMult)
    update_wcache!(:Kνωq_pre, Vector{ComplexF64}(undef, Nq))
    update_wcache!(:Kνωq_post, Vector{ComplexF64}(undef, Nq))
    update_wcache!(:Σ_ladder, Matrix{ComplexF64}(undef, Nq, Nν))
    update_wcache!(:Σ_initialized, true)
end


"""
function initialize_EoM(G_fft_reverse, λ₀::Array{ComplexF64,3}, νGrid::AbstractVector{Int}, 
                        kG::KGrid, mP::ModelParameters, sP::SimulationParameters; 
                        force_reinit = false,
                        χsp::χT = collect_χ(:sp, kG, mP, sP),
                        γsp::γT = collect_γ(:sp, kG, mP, sP),
                        χch::χT = collect_χ(:ch, kG, mP, sP),
                        γch::γT = collect_γ(:ch, kG, mP, sP))

Worker cache initialization. Must be called before [`calc_Σ_par`](@ref calc_Σ_par).
"""
function initialize_EoM(G_fft_reverse, λ₀::Array{ComplexF64,3}, νGrid::AbstractVector{Int}, kG::KGrid, mP::ModelParameters, sP::SimulationParameters; 
                        force_reinit = false,
                        χsp::χT = collect_χ(:sp, kG, mP, sP),
                        γsp::γT = collect_γ(:sp, kG, mP, sP),
                        χch::χT = collect_χ(:ch, kG, mP, sP),
                        γch::γT = collect_γ(:ch, kG, mP, sP))
    #TODO: calculate and distribute lambda0 directly here
    workerpool = get_workerpool()
    !(length(workerpool) > 0) && throw(ErrorException("Add workers and rund lDGA_setup before calling parallel functions!"))

    indices = gen_ν_part(νGrid, sP, length(workerpool))

    @sync begin
        for (i,wi) in enumerate(workers(workerpool))
            @async begin
            remotecall_fetch(update_wcache!, wi, :χsp, χsp.data)
            remotecall_fetch(update_wcache!, wi, :χch, χch.data)
            data_sl, νn_indices, ωn_ranges = gen_ν_part_slices(γsp.data, indices[i])
            remotecall_fetch(update_wcache!, wi, :γsp, data_sl)
            remotecall_fetch(update_wcache!, wi, :ωn_ranges,  ωn_ranges)
            remotecall_fetch(update_wcache!, wi, :νn_indices, νn_indices)
            data_sl, _, _ = gen_ν_part_slices(γch.data, indices[i])
            remotecall_fetch(update_wcache!, wi, :γch, data_sl)
            data_sl, _, _ = gen_ν_part_slices(λ₀, indices[i])
            remotecall_fetch(update_wcache!, wi, :λ₀, data_sl)
            remotecall_fetch(initialize_EoM_cache!, wi, length(νn_indices))
            remotecall_fetch(update_wcache!, wi, :G_fft_reverse, G_fft_reverse)
            end
        end
    end
end


# --------------------------------------- Collect from workers ---------------------------------------
"""
    collect_χ(type::Symbol, kG::KGrid, mP::ModelParameters, sP::SimulationParameters)
    collect_γ(type::Symbol, kG::KGrid, mP::ModelParameters, sP::SimulationParameters)

Collects susceptibility and triangular vertex from workers, after parallel computation (see [`calc_χγ_par`](@ref calc_χγ_par)).
"""
function collect_χ(type::Symbol, kG::KGrid, mP::ModelParameters, sP::SimulationParameters)
    workerpool = get_workerpool()
    !(length(workerpool) > 0) && throw(ErrorException("Add workers and rund lDGA_setup before calling parallel functions!"))
    χ_data::Array{ComplexF64,2} = Array{ComplexF64,2}(undef, length(kG.kMult), 2*sP.n_iω+1)
    χfield = Symbol(string("χ",type))

    #TODO: check if Σ_initialization is set. if true, use different indices (also for γ, remove error after implementation)
    #
    w_list = workers(workerpool)
    if @fetchfrom w_list[1] LadderDGA.wcache[].Σ_initialized
        χ_data[:,:] = @fetchfrom w_list[1] getfield(LadderDGA.wcache[], χfield)
    else
        @sync begin
        for w in w_list
            @async begin
                indices = @fetchfrom w LadderDGA.wcache[].χ₀Indices
                χdata   = @fetchfrom w getfield(LadderDGA.wcache[], χfield)
                for i in 1:length(indices)
                    ωi,_ = indices[i]
                    χ_data[:,ωi] = χdata[:,i]
                end
            end
        end
        end
    end
    log_q0_χ_check(kG, sP, χ_data, type)
    χT(χ_data, tail_c=[0,0,mP.Ekin_DMFT])
end

function collect_γ(type::Symbol, kG::KGrid, mP::ModelParameters, sP::SimulationParameters)
    workerpool = get_workerpool()
    !(length(workerpool) > 0) && throw(ErrorException("Add workers and rund lDGA_setup before calling parallel functions!"))
    γ_data::Array{ComplexF64,3} = Array{ComplexF64,3}(undef, length(kG.kMult), 2*sP.n_iν, 2*sP.n_iω+1)
    γfield = Symbol(string("γ",type))

    w_list = workers(workerpool)
    if @fetchfrom w_list[1] LadderDGA.wcache[].Σ_initialized
        error("Cannot collect γ after initialize_EoM has been called!")
    else
        @sync begin
            for w in w_list
                @async begin
                    indices = @fetchfrom w LadderDGA.wcache[].χ₀Indices
                    γdata   = @fetchfrom w getfield(LadderDGA.wcache[], γfield)
                    for i in 1:length(indices)
                        ωi,_ = indices[i]
                        γ_data[:,:,ωi] = γdata[:,:,i]
                    end
                end
            end
        end
    end
    γT(γ_data)
end
