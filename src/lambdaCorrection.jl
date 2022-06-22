include("lambdaCorrection_aux.jl")
include("lambdaCorrection_clean.jl")
include("lambdaCorrection_singleCore.jl")

function calc_λsp_rhs_usable(imp_density::Float64, nlQ_sp::NonLocalQuantities, nlQ_ch::NonLocalQuantities, kG::KGrid, mP::ModelParameters, sP::SimulationParameters)
    usable_ω = intersect(nlQ_sp.usable_ω, nlQ_ch.usable_ω)
    # min(usable_sp, usable_ch) = min($(nlQ_sp.usable_ω),$(nlQ_ch.usable_ω)) = $(usable_ω) for all calculations. relax this?"

    iωn = 1im .* 2 .* (-sP.n_iω:sP.n_iω)[usable_ω] .* π ./ mP.β
    χch_ω = kintegrate(kG, nlQ_ch.χ[:,usable_ω], 1)[1,:]
    #TODO: this should use sum_freq instead of naiive sum()
    χch_sum = real(sum(subtract_tail(χch_ω, mP.Ekin_DMFT, iωn)))/mP.β - mP.Ekin_DMFT*mP.β/12

    @info "λsp correction infos:"
    rhs = if (( (typeof(sP.χ_helper) != Nothing || sP.tc_type_f != :nothing) && sP.λ_rhs == :native) || sP.λ_rhs == :fixed)
        @info "  ↳ using n/2 * (1 - n/2) - Σ χch as rhs"
        mP.n * (1 - mP.n/2) - χch_sum
    else
        @info "  ↳ using χupup_DMFT - Σ χch as rhs"
        2*imp_density - χch_sum
    end

    @info """  ↳ Found usable intervals for non-local susceptibility of length 
                 ↳ sp: $(nlQ_sp.usable_ω), length: $(length(nlQ_sp.usable_ω))
                 ↳ ch: $(nlQ_ch.usable_ω), length: $(length(nlQ_ch.usable_ω))
                 ↳ total: $(usable_ω), length: $(length(usable_ω))
               ↳ χch sum = $(χch_sum), rhs = $(rhs)"""
    return rhs, usable_ω
end

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

function calc_λsp_correction(χ_in::AbstractArray, usable_ω::AbstractArray{Int64},
                            EKin::Float64, rhs::Float64, kG::KGrid, 
                            mP::ModelParameters, sP::SimulationParameters)
    χr::Matrix{Float64}    = real.(χ_in[:,usable_ω])
    iωn = (1im .* 2 .* (-sP.n_iω:sP.n_iω)[usable_ω] .* π ./ mP.β)
    iωn[findfirst(x->x ≈ 0, iωn)] = Inf
    χ_tail::Vector{Float64} = real.(EKin ./ (iωn.^2))

    f_c1_int(λint::Float64)::Float64 = f_c1(λint, kG.kMult, χr, χ_tail)/mP.β - EKin*mP.β/12 - rhs
    df_c1_int(λint::Float64)::Float64 = df_c1(λint, kG.kMult, χr, χ_tail)/mP.β - EKin*mP.β/12 - rhs

    λsp = newton_right(f_c1_int, df_c1_int, get_χ_min(χr))
    return λsp
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
    G_corr[:] = transpose(flatten_2D(G_from_Σ(Σ_ladder, kG.ϵkGrid, νGrid, mP)));
    E_pot = calc_E_pot(kG, G_corr, Σ_ladder, E_pot_tail, E_pot_tail_inv, mP.β)


    rhs_c1 = mP.n/2 * (1 - mP.n/2)
    rhs_c2 = E_pot/mP.U - (mP.n/2) * (mP.n/2)
    F[1] = lhs_c1 - rhs_c1
    F[2] = lhs_c2 - rhs_c2
    return nothing
end

#TODO: this is manually unrolled...
# after optimization, revert to:
# calc_Σ, correct Σ, calc G(Σ), calc E
function extended_λ_par(nlQ_sp::NonLocalQuantities, nlQ_ch::NonLocalQuantities,
            Gνω::GνqT, λ₀::Array{ComplexF64,3}, x₀::Vector{Float64},
            kG::KGrid, mP::ModelParameters, sP::SimulationParameters, workerpool::AbstractWorkerPool;
            νmax::Int = -1, iterations::Int=20, ftol::Float64=1e-6)
        # --- prepare auxiliary vars ---
    @info "Using DMFT GF for second condition in new lambda correction"

        # general definitions
    Nq::Int = size(nlQ_sp.γ,1)
    Nν::Int = size(nlQ_sp.γ,2)
    Nω::Int = size(nlQ_sp.γ,3)
    EKin::Float64 = mP.Ekin_DMFT
    ωindices::UnitRange{Int} = (sP.dbg_full_eom_omega) ? (1:size(nlQ_ch.χ,2)) : intersect(nlQ_sp.usable_ω, nlQ_ch.usable_ω)
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
        maxn = min(size(nlQ_ch.γ,ν_axis), νZero + νmax - 1)
        for (νii,νi) in enumerate(νZero:maxn)
                push!(νω_range, (ωi, ωn, νi, νii))
        end
    end
    νωi_part = par_partition(νω_range, length(workerpool))
    remote_results = Vector{Future}(undef, length(νωi_part))
    # distribute
    # workers = collect(workerpool.workers)
    # for (i,ind) in enumerate(νωi_part)
    #     ωi = sort(unique(map(x->x[1],νω_range[ind])))
    #     ωind_map::Dict{Int,Int} = Dict(zip(ωi, 1:length(ωi)))
    #     remote_results[i] = remotecall(initialize_cache, workers[i], νω_range[ind], ωind_map, 
    #                nlQ_sp.χ[:,ωi], nlQ_ch.χ[:,ωi], nlQ_sp.γ[:,:,ωi], nlQ_ch.γ[:,:,ωi], λ₀[:,:,ωi], Gνω, kG)
    # end
    # wait.(remote_results)
    ###


    # preallications
    χsp_tmp::Matrix{ComplexF64}  = deepcopy(nlQ_sp.χ)
    χch_tmp::Matrix{ComplexF64}  = deepcopy(nlQ_ch.χ)
    G_corr::Matrix{ComplexF64} = Matrix{ComplexF64}(undef, Nq, νmax)

    # Therodynamics preallocations
    Σ_hartree::Float64 = mP.n * mP.U/2.0;
    E_pot_tail_c = [zeros(size(kG.ϵkGrid)),
            (mP.U^2 * 0.5 * mP.n * (1-0.5*mP.n) .+ Σ_hartree .* (kG.ϵkGrid .+ Σ_hartree .- mP.μ))]
    tail = [1 ./ (iν_array(mP.β, νGrid) .^ n) for n in 1:length(E_pot_tail_c)]
    E_pot_tail::Matrix{ComplexF64} = sum(E_pot_tail_c[i] .* transpose(tail[i]) for i in 1:length(tail))
    E_pot_tail_inv::Vector{Float64} = sum((mP.β/2)  .* [Σ_hartree .* ones(size(kG.ϵkGrid)), (-mP.β/2) .* E_pot_tail_c[2]])

    rhs_c1 = mP.n/2 * (1 - mP.n/2)
    λsp_min = get_χ_min(real.(χsp_tmp))
    λch_min = get_χ_min(real.(χch_tmp))
    λch_min = if λch_min > 10000
        @warn "found λch_min=$λch_min, resetting to -400"
        -400.0
    else
        λch_min
    end
    λsp_max = 50.0#sum(kintegrate(kG,χ_λ(real.(χch_tmp), λch_min + 1e-8), 1)) / mP.β - rhs_c1
    λch_max = 1000.0#sum(kintegrate(kG,χ_λ(real.(χsp_tmp), λsp_min + 1e-8), 1)) / mP.β - rhs_c1
    @info "λsp ∈ [$λsp_min, $λsp_max], λch ∈ [$λch_min, $λch_max]"

    #trafo(x) = [((λsp_max - λsp_min)/2)*(tanh(x[1])+1) + λsp_min, ((λch_max-λch_min)/2)*(tanh(x[2])+1) + λch_min]
    trafo(x) = x
    
    cond_both!(F::Vector{Float64}, λ::Vector{Float64})::Nothing = 
        cond_both_int_par!(F, λ, νωi_part, νω_range,
        nlQ_sp.χ, nlQ_ch.χ, nlQ_sp.γ, nlQ_ch.γ, χsp_tmp, χch_tmp,  remote_results,Σ_ladder,
        G_corr, νGrid, χ_tail, Σ_hartree, E_pot_tail, E_pot_tail_inv, Gνω, λ₀, kG, mP, workerpool, trafo)
    
    println("λ search interval: $(trafo([-Inf, -Inf])) to $(trafo([Inf, Inf]))")

    
    # TODO: test this for a lot of data before refactor of code
    
    δ   = 1.0 # safety from first pole. decrese this if no roots are found
    λs_sp = λsp_min + abs.(λsp_min/10.0)
    λs_ch = λch_min + abs.(λch_min/10.0)
    λs = x₀#[λs_sp, λs_ch]
    λnew = nlsolve(cond_both!, λs, ftol=ftol, iterations=20)
    λnew.zero = trafo(λnew.zero)
    println(λnew)
    
    return λnew, ""
end
    

function λ_correction(type::Symbol, imp_density::Float64,
            nlQ_sp::NonLocalQuantities, nlQ_ch::NonLocalQuantities, 
            Gνω::GνqT, λ₀::Array{ComplexF64,3}, kG::KGrid,
            mP::ModelParameters, sP::SimulationParameters;
            workerpool::AbstractWorkerPool=default_worker_pool(),init_sp=nothing, init_spch=nothing, parallel=false, x₀::Vector{Float64}=[0.0,0.0])
    res = if type == :sp
        rhs,usable_ω_λc = calc_λsp_rhs_usable(imp_density, nlQ_sp, nlQ_ch, kG, mP, sP)
        @timeit to "λsp" λsp = calc_λsp_correction(real.(nlQ_sp.χ), usable_ω_λc, mP.Ekin_DMFT, rhs, kG, mP, sP)
        #@timeit to "λsp clean" λsp_clean = calc_λsp_correction_clean(real.(nlQ_sp.χ), usable_ω_λc, mP.Ekin_DMFT, rhs, kG, mP, sP)
        #@info "old: " λsp_clean " vs. " λsp
        λsp
    elseif type == :sp_ch
        #@warn "using unoptimized λ correction algorithm"
        #@time λ_spch_clean = extended_λ_clean(nlQ_sp, nlQ_ch, Gνω, λ₀, kG, mP, sP)
        #@time λ_spch = extended_λ(nlQ_sp, nlQ_ch, Gνω, λ₀, kG, mP, sP)
        @timeit to "λspch 2" λ_spch, dbg_string = if parallel
                extended_λ_par(nlQ_sp, nlQ_ch, Gνω, λ₀, x₀, kG, mP, sP, workerpool)
            else
                extended_λ(nlQ_sp, nlQ_ch, Gνω, λ₀, x₀, kG, mP, sP)
        end
        @warn "extended λ correction dbg string: " dbg_string
        #@timeit to "λspch clean 2" λ_spch_clean = extended_λ_clean(nlQ_sp, nlQ_ch, Gνω, λ₀, kG, mP, sP)
        #@info "new: " λ_spch_clean " vs. " λ_spch
        λ_spch
    else
        error("unrecognized λ correction type: $type")
    end
    return res
end

function λ_correction!(type::Symbol, imp_density, F, Σ_loc_pos, Σ_ladderLoc,
                       nlQ_sp::NonLocalQuantities, nlQ_ch::NonLocalQuantities, 
                       locQ::NonLocalQuantities,
                      χ₀::χ₀T, Gνω::GνqT, kG::KGrid,
                      mP::ModelParameters, sP::SimulationParameters; init_sp=nothing, init_spch=nothing)

    λ = λ_correction(type, imp_density, F, Σ_loc_pos, Σ_ladderLoc, nlQ_sp, nlQ_ch, locQ,
                  χ₀, Gνω, kG, mP, sP; init_sp=init_sp, init_spch=init_spch)
    res = if type == :sp
        nlQ_sp.χ = χ_λ(nlQ_sp.χ, λ)
        nlQ_sp.λ = λ
    elseif type == :sp_ch
        nlQ_sp.χ = χ_λ(nlQ_sp.χ, λ[1])
        nlQ_sp.λ = λ[1]
        nlQ_ch.χ = χ_λ(nlQ_ch.χ, λ[2])
        nlQ_ch.λ = λ[2]
    end
end

function newton_right(f::Function, df::Function,
                            start::Float64; nsteps=5000, atol=1e-11)
    done = false
    δ = 0.1
    x0 = start + δ
    xi = x0
    i = 1
    while !done
        fi = f(xi)
        dfii = 1 / df(xi)
        xlast = xi
        xi = x0 - dfii * fi
        (norm(xi-x0) < atol) && break
        if xi < x0               # only ever search to the right!
            δ  = δ/2.0
            x0  = start + δ      # reset with smaller delta
            xi = x0
        else
            x0 = xi
        end
        (i >= nsteps ) && (done = true)
        i += 1
    end
    return xi
end

#TODO: not working for now. fallback to nlsolve
function newton_2d_right(f::Function, 
                start::Vector{Float64}; nsteps=500, atol=1e-6, max_x2=3.0)
    done = false
    δ = [0.1, 0.1]
    x0 = start .+ δ
    xi = x0
    i = 1
    cache = FiniteDiff.JacobianCache(xi)
    J2 = Matrix{Float64}(undef, 2,2)
    while !done
        fi = f(xi)
        dfii = inv(FiniteDiff.finite_difference_jacobian(f, xi, cache))
        xlast = xi
        xi = x0 - dfii * fi
        (norm(xi-x0) < atol) && break
        if xi[1] .< x0[1]        # only ever search to the right!
            println("$(xi[1]) .< $(x0[1]) reset to $δ to $(δ .+ 0.1 .* δ)")
            flush(stdout)
            δ  = δ .+ 0.1 .* δ
            x0 = start .+ δ      # reset with larger delta
            xi = x0
        elseif xi[2] > max_x2    # λch has some cutoff value, here just 0, later to be determined
            println("$(xi[2]) > $(max_x2) reset to $δ to $(δ ./ 2)")
            δ  = δ ./ 2 
            x0 = start .+ δ      # reset with smaller delta
            xi = x0
        else
            x0 = xi
        end
        (i >= nsteps ) && (done = true)
        i += 1
    end
    return xi
end
