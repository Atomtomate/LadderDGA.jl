include("lambdaCorrection_aux.jl")
include("lambdaCorrection_clean.jl")

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
        tail::Vector{ComplexF64})::Float64
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
    return real(res)
end

function df_c1(λ::Float64, kMult::Vector{Float64}, χ::Matrix{Float64}, 
        tail::Vector{ComplexF64})::Float64
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
    return real(res)
end

function calc_λsp_correction(χ_in::AbstractArray, usable_ω::AbstractArray{Int64},
                            EKin::Float64, rhs::Float64, kG::KGrid, 
                            mP::ModelParameters, sP::SimulationParameters)
    @warn "calc λsp assumes real χ_sp/ch"
    χr    = real.(χ_in[:,usable_ω])
    iωn = (1im .* 2 .* (-sP.n_iω:sP.n_iω)[usable_ω] .* π ./ mP.β)
    iωn[findfirst(x->x ≈ 0, iωn)] = Inf
    χ_tail = EKin ./ (iωn.^2)

    f_c1_int(λint::Float64)::Float64 = f_c1(λint, kG.kMult, χr, χ_tail)/mP.β - EKin*mP.β/12 - rhs
    df_c1_int(λint::Float64)::Float64 = df_c1(λint, kG.kMult, χr, χ_tail)/mP.β - EKin*mP.β/12 - rhs

    λsp = newton_right(f_c1_int, df_c1_int, get_χ_min(χr))
    return λsp
end

#TODO: this is manually unrolled...
# after optimization, revert to:
# calc_Σ, correct Σ, calc G(Σ), calc E
function extended_λ(nlQ_sp::NonLocalQuantities, nlQ_ch::NonLocalQuantities,
            Gνω::GνqT, λ₀::AbstractArray{ComplexF64,3},
            kG::KGrid, mP::ModelParameters, sP::SimulationParameters;
            νmax = -1, iterations=400, ftol=1e-8, x₀ = [0.1, 0.1])
        # --- prepare auxiliary vars ---
    @info "Using DMFT GF for second condition in new lambda correction"
    
    # general definitions
    Nq = size(nlQ_sp.γ,1)
    Nν = size(nlQ_sp.γ,2)
    Nω = size(nlQ_sp.γ,3)
    EKin = mP.Ekin_DMFT
    ωindices = (sP.dbg_full_eom_omega) ? (1:size(nlQ_ch.χ,2)) : intersect(nlQ_sp.usable_ω, nlQ_ch.usable_ω)
    ωrange = (-sP.n_iω:sP.n_iω)[ωindices]
    ωrange = first(ωrange):last(ωrange)
    νmax = νmax < 0 ? minimum([sP.n_iν,floor(Int,3*length(ωindices)/8)]) : νmax
    νGrid = 0:(νmax-1)
    iωn = 1im .* 2 .* (-sP.n_iω:sP.n_iω)[ωindices] .* π ./ mP.β
    iωn[findfirst(x->x ≈ 0, iωn)] = Inf
    χ_tail = EKin ./ (iωn.^2)
    k_norm = Nk(kG)

    # EoM optimization related definitions
    Kνωq_pre = Array{ComplexF64, 1}(undef, length(kG.kMult))
    Σ_ladder_ω = OffsetArray( Array{ComplexF64,3}(undef,Nq, νmax, length(ωrange)),
                              1:Nq, 0:νmax-1, ωrange)
    Σ_ladder = OffsetArray( Array{ComplexF64,2}(undef,Nq, νmax),
                              1:Nq, 0:νmax-1)

    # preallications
    nlQ_sp_λ = deepcopy(nlQ_sp)
    nlQ_ch_λ = deepcopy(nlQ_ch)
    χ_sp_λ = real.(nlQ_sp.χ[:,ωindices])
    χ_ch_λ = real.(nlQ_ch.χ[:,ωindices])

    # Therodynamics preallocations
    Σ_hartree = mP.n * mP.U/2.0;
    E_pot_tail_c = [zeros(size(kG.ϵkGrid)),
            (mP.U^2 * 0.5 * mP.n * (1-0.5*mP.n) .+ Σ_hartree .* (kG.ϵkGrid .+ Σ_hartree .- mP.μ))]
    tail = [1 ./ (iν_array(mP.β, νGrid) .^ n) for n in 1:length(E_pot_tail_c)]
    E_pot_tail = sum(E_pot_tail_c[i] .* transpose(tail[i]) for i in 1:length(tail))
    E_pot_tail_inv = sum((mP.β/2)  .* [Σ_hartree .* ones(size(kG.ϵkGrid)), (-mP.β/2) .* E_pot_tail_c[2]])
        
    function cond_both!(F::Vector{Float64}, λ::Vector{Float64})
        #TODO: captured variables that are no reassigned, should be assigned in a let block
        @timeit to "λc" χ_λ!(nlQ_sp_λ.χ, nlQ_sp.χ, λ[1])
        @timeit to "λc" χ_λ!(nlQ_ch_λ.χ, nlQ_ch.χ, λ[2])

        #TODO: unroll 
        calc_Σ_ω!(Σ_ladder_ω, Kνωq_pre, ωindices, nlQ_sp_λ, nlQ_ch_λ, Gνω, λ₀, mP.U, kG, sP)
        Σ_ladder[:] = dropdims(sum(Σ_ladder_ω, dims=[3]),dims=3) ./ mP.β .+ Σ_hartree

        lhs_c1 = 0.0
        lhs_c2 = 0.0
        for (ωi,t) in enumerate(χ_tail)
            tmp1 = 0.0
            tmp2 = 0.0
            for (qi,km) in enumerate(kG.kMult)
                χsp_i_λ = real(nlQ_sp.χ[qi,ωi])
                χch_i_λ = real(nlQ_ch.χ[qi,ωi])
                tmp1 += 0.5 * (χch_i_λ + χsp_i_λ) * km
                tmp2 += 0.5 * (χch_i_λ - χsp_i_λ) * km
            end
            lhs_c1 += tmp1/k_norm - t
            lhs_c2 += tmp2/k_norm
        end

        lhs_c1 = lhs_c1/mP.β - mP.Ekin_DMFT*mP.β/12
        lhs_c2 = lhs_c2/mP.β

        #TODO: the next line is expensive: Optimize G_from_Σ
        @timeit to "GCorr" G_corr = transpose(flatten_2D(G_from_Σ(Σ_ladder.parent, kG.ϵkGrid, νGrid, mP)));
        @timeit to "Ep" E_pot = calc_E_pot(kG, G_corr, Σ_ladder.parent, E_pot_tail, E_pot_tail_inv, mP.β)
        rhs_c1 = mP.n/2 * (1 - mP.n/2)
        rhs_c2 = E_pot/mP.U - (mP.n/2) * (mP.n/2)
        F[1] = lhs_c1 - rhs_c1
        F[2] = lhs_c2 - rhs_c2
        return lhs_c1, rhs_c1, lhs_c2, rhs_c2
    end
    Fint = x₀

    λnew = nlsolve(cond_both!, Fint, iterations=iterations, ftol=ftol)
    #λnew.zero[:] = [tanh(λnew.zero[1]+a_sp+χsp_min)*a_sp, tanh(λnew.zero[2]+a_ch+χch_min)*a_ch]
    return λnew
end

function λ_correction(type::Symbol, imp_density::Float64,
            nlQ_sp::NonLocalQuantities, nlQ_ch::NonLocalQuantities, 
            Gνω::GνqT, λ₀::AbstractArray{ComplexF64,3}, kG::KGrid,
            mP::ModelParameters, sP::SimulationParameters; init_sp=nothing, init_spch=nothing)
    res = if type == :sp
        rhs,usable_ω_λc = calc_λsp_rhs_usable(imp_density, nlQ_sp, nlQ_ch, kG, mP, sP)
        @timeit to "λsp opt" λsp = calc_λsp_correction(real.(nlQ_sp.χ), usable_ω_λc, mP.Ekin_DMFT, rhs, kG, mP, sP)
        @timeit to "λsp clean" λsp_clean = calc_λsp_correction_clean(real.(nlQ_sp.χ), usable_ω_λc, mP.Ekin_DMFT, rhs, kG, mP, sP)
        @info "old: " λsp_clean " vs. " λsp
        λsp
    elseif type == :sp_ch
        @warn "using unoptimized λ correction algorithm"
        @timeit to "λspch clean" λ_clean = extended_λ_clean(nlQ_sp, nlQ_ch, Gνω, λ₀, kG, mP, sP)
        @timeit to "λspch opt" λ_opt = extended_λ(nlQ_sp, nlQ_ch, Gνω, λ₀, kG, mP, sP)
        @info "new: " λ_clean " vs. " λ_opt
        λ_clean
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
