include("lambdaCorrection_aux.jl")
include("lambdaCorrection_singleCore.jl")

#TODO: refactor code repetition
function f_c1(λ::Float64, kMult::Vector{Float64}, χ::Matrix{Float64}, 
        tail::Vector{Float64})::Float64
    res = 0.0
    resi = 0.0
    norm = sum(kMult)
    for (i,ωi) in enumerate(tail)
        resi = 0.0
        for (ki,km) in enumerate(kMult)
            resi += χ_λ(χ[ki,i],λ) * km
        end
        res += resi/norm - ωi
    end
    return res
end

function df_c1(λ::Float64, kMult::Vector{Float64}, χ::Matrix{Float64}, 
        tail::Vector{Float64})::Float64
    res = 0.0
    resi = 0.0
    norm = sum(kMult)
    for (i,ωi) in enumerate(tail)
        resi = 0.0
        for (ki,km) in enumerate(kMult)
            resi += dχ_λ(χ[ki,i],λ) * km
        end
        res += resi/norm - ωi
    end
    return res
end

function cond_both_int_par!(F::Vector{Float64}, λ::Vector{Float64}, νωi_part, νω_range::Array{NTuple{4,Int}},
        χsp::χT, χch::χT, γsp::γT, γch::γT, χsp_bak::χT, χch_bak::χT,
        remote_results::Vector{Future},Σ_ladder::Array{ComplexF64,2}, 
        G_corr::Matrix{ComplexF64},νGrid::UnitRange{Int},χ_tail::Vector{ComplexF64},Σ_hartree::Float64,
        E_pot_tail::Matrix{ComplexF64},E_pot_tail_inv::Vector{Float64},Gνω::GνqT,
        λ₀::Array{ComplexF64,3}, kG::KGrid, mP::ModelParameters, workerpool::AbstractWorkerPool, trafo::Function)::Nothing
    λi::Vector{Float64} = trafo(λ)
    χ_λ!(χsp, χsp_bak, λi[1])
    χ_λ!(χch, χch_bak, λi[2])
    k_norm::Int = Nk(kG)

    ### Untroll start
    
    n_iν = size(Σ_ladder, 2)
    # distribute
    # workers = collect(workerpool.workers)
    # for (i,ind) in enumerate(νωi_part)
    #     ωi = sort(unique(map(x->x[1],νω_range[ind])))
    #     remote_results[i] = remotecall(update_χ, workers[i], :sp, χsp[:,ωi])
    #     remote_results[i] = remotecall(update_χ, workers[i], :ch, χch[:,ωi])
    #     remote_results[i] = remotecall(calc_Σ_eom_par, workers[i], n_iν, mP.U)
    # end
    for (i,ind) in enumerate(νωi_part)
        ωi = sort(unique(map(x->x[1],νω_range[ind])))
        ωind_map::Dict{Int,Int} = Dict(zip(ωi, 1:length(ωi)))
        remote_results[i] = remotecall(calc_Σ_eom, workerpool, νω_range[ind], ωind_map, n_iν, χsp[:,ωi],
                                       χch[:,ωi], γsp[:,:,ωi], γch[:,:,ωi], Gνω, λ₀[:,:,ωi], mP.U, kG)
    end

    # collect results
    fill!(Σ_ladder, Σ_hartree .* mP.β)
    for i in 1:length(remote_results)
        data_i = fetch(remote_results[i])
        Σ_ladder[:,:] += data_i
    end
    Σ_ladder = Σ_ladder ./ mP.β
    ### Untroll end

    lhs_c1 = 0.0
    lhs_c2 = 0.0
    #TODO: sum can be done on each worker
    for (ωi,t) in enumerate(χ_tail)
        tmp1 = 0.0
        tmp2 = 0.0
        for (qi,km) in enumerate(kG.kMult)
            tmp1 += 0.5 * real(χch[qi,ωi] .+ χsp[qi,ωi]) * km
            tmp2 += 0.5 * real(χch[qi,ωi] .- χsp[qi,ωi]) * km
        end
        lhs_c1 += tmp1/k_norm - t
        lhs_c2 += tmp2/k_norm
    end

    lhs_c1 = lhs_c1/mP.β - mP.Ekin_DMFT*mP.β/12
    lhs_c2 = lhs_c2/mP.β

    #TODO: the next two lines are expensive
    G_corr[:] = G_from_Σ(Σ_ladder, kG.ϵkGrid, νGrid, mP);
    E_pot = EPot1(kG, G_corr, Σ_ladder, E_pot_tail, E_pot_tail_inv, mP.β)


    rhs_c1 = mP.n/2 * (1 - mP.n/2)
    rhs_c2 = E_pot/mP.U - (mP.n/2) * (mP.n/2)
    F[1] = lhs_c1 - rhs_c1
    F[2] = lhs_c2 - rhs_c2
    return nothing
end

#TODO: this is manually unrolled...
# after optimization, revert to:
# calc_Σ, correct Σ, calc G(Σ), calc E
function extended_λ_par(χsp::χT, γsp::γT, χch::χT, γch::γT,
            Gνω::GνqT, λ₀::Array{ComplexF64,3}, x₀::Vector{Float64},
            kG::KGrid, mP::ModelParameters, sP::SimulationParameters, workerpool::AbstractWorkerPool;
            νmax::Int = -1, iterations::Int=20, ftol::Float64=1e-6)
        # --- prepare auxiliary vars ---
    @info "Using DMFT GF for second condition in new lambda correction"

        # general definitions
        #
    Nq, Nν, Nω = size(γsp)
    EKin::Float64 = mP.Ekin_DMFT
    ωindices::UnitRange{Int} = (sP.dbg_full_eom_omega) ? (1:size(χch,2)) : intersect(χsp.usable_ω, χch.usable_ω)
    νmax::Int = νmax < 0 ? min(sP.n_iν,floor(Int,3*length(ωindices)/8)) : νmax
    νGrid::UnitRange{Int} = 0:(νmax-1)
    iωn = 1im .* 2 .* (-sP.n_iω:sP.n_iω)[ωindices] .* π ./ mP.β
    iωn[findfirst(x->x ≈ 0, iωn)] = Inf
    χ_tail::Vector{ComplexF64} = EKin ./ (iωn.^2)
    k_norm::Int = Nk(kG)

    # EoM optimization related definitions
    Σ_ladder::Array{ComplexF64,2} = Array{ComplexF64,2}(undef, Nq, νmax)
    νω_range::Array{NTuple{4,Int}} = Array{NTuple{4,Int}}[]
    for (ωi,ωn) in enumerate(-sP.n_iω:sP.n_iω)
        νZero = ν0Index_of_ωIndex(ωi, sP)
        maxn = min(size(γch,ν_axis), νZero + νmax - 1)
        for (νii,νi) in enumerate(νZero:maxn)
                push!(νω_range, (ωi, ωn, νi, νii))
        end
    end
    νωi_part = par_partition(νω_range, length(workerpool))
    remote_results = Vector{Future}(undef, length(νωi_part))

    # preallications
    χsp_tmp::χT = deepcopy(χsp)
    χch_tmp::χT = deepcopy(χch)
    G_corr::Matrix{ComplexF64} = Matrix{ComplexF64}(undef, Nq, νmax)

    # Therodynamics preallocations
    Σ_hartree::Float64 = mP.n * mP.U/2.0;
    E_pot_tail_c = [zeros(size(kG.ϵkGrid)),
            (mP.U^2 * 0.5 * mP.n * (1-0.5*mP.n) .+ Σ_hartree .* (kG.ϵkGrid .+ Σ_hartree .- mP.μ))]
    tail = [1 ./ (iν_array(mP.β, νGrid) .^ n) for n in 1:length(E_pot_tail_c)]
    E_pot_tail::Matrix{ComplexF64} = sum(E_pot_tail_c[i] .* transpose(tail[i]) for i in 1:length(tail))
    E_pot_tail_inv::Vector{Float64} = sum((mP.β/2)  .* [Σ_hartree .* ones(size(kG.ϵkGrid)), (-mP.β/2) .* E_pot_tail_c[2]])

    rhs_c1 = mP.n/2 * (1 - mP.n/2)
    λsp_min = get_λ_min(real.(χsp_tmp.data))
    λch_min = get_λ_min(real.(χch_tmp.data))
    λsp_max = 50.0#sum(kintegrate(kG,χ_λ(real.(χch_tmp.data), λch_min + 1e-8), 1)) / mP.β - rhs_c1
    λch_max = 1000.0#sum(kintegrate(kG,χ_λ(real.(χsp_tmp.data), λsp_min + 1e-8), 1)) / mP.β - rhs_c1
    @info "λsp ∈ [$λsp_min, $λsp_max], λch ∈ [$λch_min, $λch_max]"

    #trafo(x) = [((λsp_max - λsp_min)/2)*(tanh(x[1])+1) + λsp_min, ((λch_max-λch_min)/2)*(tanh(x[2])+1) + λch_min]
    trafo(x) = x
    
    cond_both!(F::Vector{Float64}, λ::Vector{Float64})::Nothing = 
        cond_both_int_par!(F, λ, νωi_part, νω_range,
        χsp, χch, γsp, γch, χsp_tmp, χch_tmp,  remote_results,Σ_ladder,
        G_corr, νGrid, χ_tail, Σ_hartree, E_pot_tail, E_pot_tail_inv, Gνω, λ₀, kG, mP, workerpool, trafo)
    
    println("λ search interval: $(trafo([-Inf, -Inf])) to $(trafo([Inf, Inf]))")

    
    # TODO: test this for a lot of data before refactor of code
    
    δ   = 1.0 # safety from first pole. decrese this if no roots are found
    λssp = λsp_min + abs.(λsp_min/10.0)
    λsch = λch_min + abs.(λch_min/10.0)
    λmin = [λsp_min, λch_min]
    λs = x₀
    all(x₀ .< λmin) && @warn "starting point $x₀ is not compatible with λmin $λmin !"
    λnew = nlsolve(cond_both!, λs, ftol=ftol, iterations=iterations)
    λnew.zero = trafo(λnew.zero)
    println(λnew)
    χsp.data = deepcopy(χsp_tmp.data)
    χch.data = deepcopy(χch_tmp.data)
    return λnew, ""
end
    

function λ_correction(type::Symbol, imp_density::Float64,
            χsp::χT, γsp::γT, χch::χT, γch::γT,
            Gνω::GνqT, λ₀::Array{ComplexF64,3}, kG::KGrid,
            mP::ModelParameters, sP::SimulationParameters;
            workerpool::AbstractWorkerPool=default_worker_pool(),init_sp=nothing, init_spch=nothing, parallel=false, x₀::Vector{Float64}=[0.0,0.0])
    res = if type == :sp
        rhs,usable_ω_λc = LambdaCorrection.λsp_rhs(imp_density, χsp, χch, kG, mP, sP)
        @timeit to "λsp" λsp = λsp_correction(χsp, mP.Ekin_DMFT, rhs, kG, mP, sP)
        λsp
    elseif type == :sp_ch
        @timeit to "λspch 2" λspch, dbg_string = if parallel
                extended_λ_par(χsp, γsp, χch, γch, Gνω, λ₀, x₀, kG, mP, sP, workerpool)
            else
                extended_λ(χsp, γsp, χch, γch, Gνω, λ₀, x₀, kG, mP, sP)
        end
        @warn "extended λ correction dbg string: " dbg_string
        λspch
    else
        error("unrecognized λ correction type: $type")
    end
    return res
end

function λ_correction!(type::Symbol, imp_density, F, Σ_loc_pos, Σ_ladderLoc,
                       χsp::χT, γsp::γT, χch::χT, γch::γT,
                      χ₀::χ₀T, Gνω::GνqT, kG::KGrid,
                      mP::ModelParameters, sP::SimulationParameters; init_sp=nothing, init_spch=nothing)

    λ = λ_correction(type, imp_density, F, Σ_loc_pos, Σ_ladderLoc, χsp, γsp, χch, γch,
                  χ₀, Gνω, kG, mP, sP; init_sp=init_sp, init_spch=init_spch)
    res = if type == :sp
        χ_λ!(χsp, λ)
    elseif type == :sp_ch
        χ_λ!(χsp, λ[1])
        χ_λ!(χch, λ[2])
    end
end
