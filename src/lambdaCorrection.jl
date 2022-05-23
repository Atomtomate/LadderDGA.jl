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
    @warn "calc λsp assumes real χ_sp/ch"
    χr::Matrix{Float64}    = real.(χ_in[:,usable_ω])
    iωn = (1im .* 2 .* (-sP.n_iω:sP.n_iω)[usable_ω] .* π ./ mP.β)
    iωn[findfirst(x->x ≈ 0, iωn)] = Inf
    χ_tail::Vector{Float64} = real.(EKin ./ (iωn.^2))

    f_c1_int(λint::Float64)::Float64 = f_c1(λint, kG.kMult, χr, χ_tail)/mP.β - EKin*mP.β/12 - rhs
    df_c1_int(λint::Float64)::Float64 = df_c1(λint, kG.kMult, χr, χ_tail)/mP.β - EKin*mP.β/12 - rhs

    λsp = newton_right(f_c1_int, df_c1_int, get_χ_min(χr))
    return λsp
end

function cond_both_int!(F::Vector{Float64}, λ::Vector{Float64}, 
        χsp::χT, χch::χT, γsp::γT, γch::γT,
        χsp_bak::χT, χch_bak::χT,
        νω_range::Array{NTuple{4,Int}}, νωi_part::Vector{UnitRange{Int64}}, remote_results::Vector{Future},
        Σ_ladder::Array{ComplexF64,2}, 
        G_corr::Matrix{ComplexF64},νGrid::UnitRange{Int},χ_tail::Vector{ComplexF64},Σ_hartree::Float64,
        E_pot_tail::Matrix{ComplexF64},E_pot_tail_inv::Vector{Float64},Gνω::GνqT,
        λ₀::Array{ComplexF64,3}, kG::KGrid, mP::ModelParameters, workerpool::AbstractWorkerPool)::Nothing
    χ_λ!(χsp, χsp_bak, λ[1])
    χ_λ!(χch, χch_bak, λ[2])
    k_norm::Int = Nk(kG)

    #Σ_ladder[:,:] = calc_Σ_par(nlQ_sp, nlQ_ch, λ₀, Gνω, kG, mP, sP, workerpool=wp)[:,νGrid]
    ### Untroll start
    
    n_iν = size(Σ_ladder, 2)
    # distribute
    for (i,ind) in enumerate(νωi_part)
        ωi = sort(unique(map(x->x[1],νω_range[ind])))
        ωind_map::Dict{Int,Int} = Dict(zip(ωi, 1:length(ωi)))
        remote_results[i] = remotecall(calc_Σ_eom, workerpool, νω_range[ind], ωind_map, n_iν, χsp[:,ωi],
                                       χch[:,ωi], γsp[:,:,ωi], γch[:,:,ωi], Gνω, λ₀[:,:,ωi], mP.U, kG)
    end

    # collect results
    fill!(Σ_ladder, Σ_hartree .* mP.β)
    for (i,ind) in enumerate(νωi_part)
        data_i = fetch(remote_results[i])
        Σ_ladder[:,:] += data_i
    end
    Σ_ladder = Σ_ladder ./ mP.β
    ### Untroll end

    lhs_c1 = 0.0
    lhs_c2 = 0.0
    for (ωi,t) in enumerate(χ_tail)
        tmp1 = 0.0
        tmp2 = 0.0
        for (qi,km) in enumerate(kG.kMult)
            χsp_i_λ = real(χsp[qi,ωi])
            χch_i_λ = real(χch[qi,ωi])
            tmp1 += 0.5 * (χch_i_λ + χsp_i_λ) * km
            tmp2 += 0.5 * (χch_i_λ - χsp_i_λ) * km
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
            Gνω::GνqT, λ₀::Array{ComplexF64,3},
            kG::KGrid, mP::ModelParameters, sP::SimulationParameters, workerpool::AbstractWorkerPool;
            νmax::Int = -1, iterations::Int=400, ftol::Float64=1e-8, x₀ = [0.1, 0.1])
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
    ###

    # preallications
    χsp_bak::Matrix{ComplexF64}  = deepcopy(nlQ_sp.χ)
    χch_bak::Matrix{ComplexF64}  = deepcopy(nlQ_ch.χ)
    G_corr::Matrix{ComplexF64} = Matrix{ComplexF64}(undef, Nq, νmax)

    # Therodynamics preallocations
    Σ_hartree::Float64 = mP.n * mP.U/2.0;
    E_pot_tail_c = [zeros(size(kG.ϵkGrid)),
            (mP.U^2 * 0.5 * mP.n * (1-0.5*mP.n) .+ Σ_hartree .* (kG.ϵkGrid .+ Σ_hartree .- mP.μ))]
    tail = [1 ./ (iν_array(mP.β, νGrid) .^ n) for n in 1:length(E_pot_tail_c)]
    E_pot_tail::Matrix{ComplexF64} = sum(E_pot_tail_c[i] .* transpose(tail[i]) for i in 1:length(tail))
    E_pot_tail_inv::Vector{Float64} = sum((mP.β/2)  .* [Σ_hartree .* ones(size(kG.ϵkGrid)), (-mP.β/2) .* E_pot_tail_c[2]])

    
    cond_both!(F::Vector{Float64}, λ::Vector{Float64})::Nothing = 
        cond_both_int!(F, λ, nlQ_sp.χ, nlQ_ch.χ, nlQ_sp.γ, nlQ_ch.γ, 
        χsp_bak, χch_bak,  νω_range, νωi_part, remote_results,Σ_ladder,
        G_corr, νGrid, χ_tail, Σ_hartree, E_pot_tail, E_pot_tail_inv, Gνω, λ₀, kG, mP, workerpool)
    
    # TODO: test this for a lot of data before refactor of code
    
    δ   = 0.0 # safety from first pole. decrese this if no roots are found
    λl = [get_χ_min(real.(χsp_bak)), get_χ_min(real.(χch_bak))] .+ δ
    λr = [0.0, 0.0]
    Fr = [0.0, 0.0]
    Fm = [0.0, 0.0]
    Fl = [0.0, 0.0]
    
    dbg_log = IOBuffer()
    cond_both!(Fr, λr)
    #find λr
    println(dbg_log, "correct_margins: λl=$(round.(λl,digits=3)), λr=$(round.(λr,digits=3)) F=$(round.(Fr,digits=3))")
    while any(Fr .> 0)
        λl, λr  = correct_margins(λl, λr, Fl, Fr)
        println(dbg_log, λl, " ...... ", λr)
        cond_both!(Fr, λr)
        println(dbg_log, "correct_margins: λl=$(round.(λl,digits=3)), λr=$(round.(λr,digits=3))Fr=$(round.(Fr,digits=3))")
    end
    
    #bisect
    for i in 1:5
        Δh = (λr .- λl)./2
        λm = λl .+ Δh
        cond_both!(Fm, λm)
        cond_both!(Fl, λl)
        i > 1 && cond_both!(Fr, λr)
        println(dbg_log, "$i: λl=$(round.(λl,digits=4)), λm=$(round.(λm,digits=4)), λr=$(round.(λr,digits=4))")
        println(dbg_log, "    Fl=$(round.(Fl,digits=4)), Fm=$(round.(Fm,digits=4)), Fr=$(round.(Fr,digits=4))")
        println(dbg_log, "<- λl=$(round.(λl,digits=4)),                     λr=$(round.(λr,digits=4))")
        λl, λr = correct_margins(λl, λr, Fl, Fr)
        println(dbg_log, "-> λl=$(round.(λl,digits=4)),                     λr=$(round.(λr,digits=4))")
        λl, λr = bisect(λl, λm, λr, Fm)
        
    end
    @info "start: " λl .+ (λr .- λl)./2
    
    λnew = nlsolve(cond_both!, λl .+ (λr .- λl)./2, ftol=1e-8)
    println(λnew)
    
    return λnew, String(take!(dbg_log))
end

function λ_correction(type::Symbol, imp_density::Float64,
            nlQ_sp::NonLocalQuantities, nlQ_ch::NonLocalQuantities, 
            Gνω::GνqT, λ₀::Array{ComplexF64,3}, kG::KGrid,
            mP::ModelParameters, sP::SimulationParameters;
            workerpool::AbstractWorkerPool=default_worker_pool(),init_sp=nothing, init_spch=nothing, parallel=false)
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
                extended_λ_par(nlQ_sp, nlQ_ch, Gνω, λ₀, kG, mP, sP, workerpool, νmax=sP.n_iν)
            else
                extended_λ(nlQ_sp, nlQ_ch, Gνω, λ₀, kG, mP, sP)
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
