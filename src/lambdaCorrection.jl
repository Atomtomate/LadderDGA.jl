dχ_λ(χ, λ::Float64) = map(χi -> - ((1.0 / χi) + λ)^(-2), χ)
χ_λ2(χ, λ) = map(χi -> 1.0 / ((1.0 / χi) + λ), χ)

function χ_λ(nlQ::NonLocalQuantities, λ::Float64) where T <: Union{ComplexF64, Float64}
    nlQ_new = deepcopy(nlQ)
    χ_λ!(nlQ_new.χ, nlQ.χ, λ)
    return nlQ_new
end

function χ_λ(χ::AbstractArray{T}, λ::Float64) where T <: Union{ComplexF64, Float64}
    res = similar(χ)
    χ_λ!(res, χ, λ)
    return res
end

function χ_λ!(χ_λ::AbstractArray{ComplexF64}, χ::AbstractArray{ComplexF64}, λ::Float64)
    @simd for i in eachindex(χ_λ)
        @inbounds χ_λ[i] = 1.0 / ((1.0 / χ[i]) + λ)
    end
end

function χ_λ!(χ_λ::AbstractArray{Float64}, χ::AbstractArray{Float64}, λ::Float64)
    @simd for i in eachindex(χ_λ)
        @inbounds χ_λ[i] = 1.0 / ((1.0 / χ[i]) + λ)
    end
end

function χ_λ!(χ_λ::AbstractArray{T,2}, χ::AbstractArray{T,2}, λ::Float64, ωindices::AbstractArray{Int,1}) where T <: Number
    for i in ωindices
        χ_λ!(view(χ_λ, :, i),view(χ,:,i), λ)
    end
end

function get_χ_min(χr::AbstractArray{Float64,2})
    nh  = ceil(Int64, size(χr,2)/2)
    -minimum(1 ./ view(χr,:,nh))
end


function calc_λsp_rhs_usable(imp_density::Float64, nlQ_sp::NonLocalQuantities, nlQ_ch::NonLocalQuantities, kG::ReducedKGrid, mP::ModelParameters, sP::SimulationParameters)
    usable_ω = intersect(nlQ_sp.usable_ω, nlQ_ch.usable_ω)
    @warn "currently using min(usable_sp, usable_ch) = min($(nlQ_sp.usable_ω),$(nlQ_ch.usable_ω)) = $(usable_ω) for all calculations. relax this?"

    iωn = 1im .* 2 .* (-sP.n_iω:sP.n_iω)[usable_ω] .* π ./ mP.β
    χch_ω = kintegrate(kG, nlQ_ch.χ[:,usable_ω], 1)[1,:]
    #TODO: this should use sum_freq instead of naiive sum()
    χch_sum = real(sum(subtract_tail(χch_ω, mP.Ekin_DMFT, iωn)))/mP.β - mP.Ekin_DMFT*mP.β/12

    rhs = if (( (typeof(sP.χ_helper) != Nothing || sP.tc_type_f != :nothing) && sP.λ_rhs == :native) || sP.λ_rhs == :fixed)
        @info " using n/2 * (1 - n/2) - Σ χch as rhs"
        mP.n * (1 - mP.n/2) - χch_sum
    else
        @info " using χupup_DMFT - Σ χch as rhs"
        2*imp_density - χch_sum
    end

    @info """Found usable intervals for non-local susceptibility of length 
          sp: $(nlQ_sp.usable_ω), length: $(length(nlQ_sp.usable_ω))
          ch: $(nlQ_ch.usable_ω), length: $(length(nlQ_ch.usable_ω))
          usable: $(usable_ω), length: $(length(usable_ω))
          χch sum = $(χch_sum), rhs = $(rhs)"""
    return rhs, usable_ω
end

function λsp(χr::Array{Float64,2}, iωn::Array{ComplexF64,1}, EKin::Float64,
                            rhs::Float64, kG::ReducedKGrid, mP::ModelParameters)
    #TODO: this should use sum_freq instead of naiive sum()
    f(λint) = real(sum(subtract_tail(kintegrate(kG, χ_λ(χr, λint), 1)[1,:],EKin, iωn)))/mP.β  -EKin*mP.β/12 - rhs
    df(λint) = real(sum(kintegrate(kG, -χ_λ(χr, λint) .^ 2, 1)[1,:]))/mP.β

    λsp = newton_right(f, df, get_χ_min(χr))
    return λsp
end

function calc_λsp_correction(χ_in::AbstractArray{Float64,2}, usable_ω::AbstractArray{Int64},
                            EKin::Float64, rhs::Float64, kG::ReducedKGrid, 
                            mP::ModelParameters, sP::SimulationParameters)
    χr    = χ_in[:,usable_ω]
    iωn = 1im .* 2 .* (-sP.n_iω:sP.n_iω)[usable_ω] .* π ./ mP.β
    #TODO: this should use sum_freq instead of naiive sum()
    f(λint) = sum(subtract_tail(kintegrate(kG, χ_λ(χr, λint), 1)[1,:],EKin, iωn))/mP.β  -EKin*mP.β/12 - rhs
    df(λint) = sum(kintegrate(kG, -χ_λ(χr, λint) .^ 2, 1)[1,:])/mP.β

    λsp = newton_right(f, df, get_χ_min(χr))
    @info "Found λsp " λsp
    return λsp
end

function extended_λ_clean(nlQ_sp::NonLocalQuantities, nlQ_ch::NonLocalQuantities,
        Gνω::GνqT, λ₀::AbstractArray{ComplexF64,3},
        kG::ReducedKGrid, mP::ModelParameters, sP::SimulationParameters; 
        νmax::Int = -1, iterations=1000, ftol=1e-8)

    ωindices = (sP.dbg_full_eom_omega) ? (1:size(nlQ_ch.χ,2)) : intersect(nlQ_sp.usable_ω, nlQ_ch.usable_ω)
    νmax = νmax < 0 ? minimum([sP.n_iν,floor(Int,3*length(ωindices)/8)]) : νmax
    iωn = 1im .* 2 .* (-sP.n_iω:sP.n_iω)[ωindices] .* π ./ mP.β
    nlQ_sp_λ = deepcopy(nlQ_sp)
    nlQ_ch_λ = deepcopy(nlQ_ch)

    Σ_hartree = mP.n * mP.U/2.0;
    νGrid = 0:(νmax-1) 
    iν_n = iν_array(mP.β, νGrid)
     E_pot_tail_c = [zeros(size(kG.ϵkGrid)),
            (mP.U^2 * 0.5 * mP.n * (1-0.5*mP.n) .+ Σ_hartree .* (kG.ϵkGrid .+ Σ_hartree .- mP.μ))]

    tail = [1 ./ (iν_n .^ n) for n in 1:length(E_pot_tail_c)]
    E_pot_tail = sum(E_pot_tail_c[i] .* transpose(tail[i]) for i in 1:length(tail))
    E_pot_tail_inv = sum((mP.β/2)  .* [Σ_hartree .* ones(size(kG.ϵkGrid)), (-mP.β/2) .* E_pot_tail_c[2]])



    function cond_both!(F, λ)
        χ_λ!(nlQ_sp_λ.χ, nlQ_sp.χ, λ[1])
        χ_λ!(nlQ_ch_λ.χ, nlQ_ch.χ, λ[2])
        Σ_ladder = calc_Σ(nlQ_sp_λ, nlQ_ch_λ, λ₀, Gνω, kG, mP, sP).parent[:,1:νmax]
        
        χupup_ω = subtract_tail(0.5 * kintegrate(kG,nlQ_ch_λ.χ .+ nlQ_sp_λ.χ,1)[1,ωindices], mP.Ekin_DMFT, iωn)
        χupdo_ω = 0.5 * kintegrate(kG,nlQ_ch_λ.χ .- nlQ_sp_λ.χ,1)[1,ωindices]
        E_kin, E_pot = calc_E(Σ_ladder, kG, mP, sP)
        G_corr = transpose(flatten_2D(G_from_Σ(Σ_ladder, kG.ϵkGrid, νGrid, mP)));
        E_pot2 = calc_E_pot(kG, G_corr, Σ_ladder, E_pot_tail, E_pot_tail_inv, mP.β)

        lhs_c1 = real(sum(χupup_ω))/mP.β - mP.Ekin_DMFT*mP.β/12
        lhs_c2 = real(sum(χupdo_ω))/mP.β
        rhs_c1 = mP.n/2 * (1 - mP.n/2)
        rhs_c2 = E_pot/mP.U - (mP.n/2) * (mP.n/2)
        F[1] = lhs_c1 - rhs_c1
        F[2] = lhs_c2 - rhs_c2
        return lhs_c1, rhs_c1, lhs_c2, rhs_c2
    end
    Fint = [0.1, 0.1]

    res_nls = nlsolve(cond_both!, Fint, iterations=iterations, ftol=ftol)
end


#TODO: this is manually unrolled...
# after optimization, revert to:
# calc_Σ, correct Σ, calc G(Σ), calc E
function extended_λ(nlQ_sp::NonLocalQuantities, nlQ_ch::NonLocalQuantities,
            Gνω::GνqT, λ₀::AbstractArray{ComplexF64,3},
            kG::ReducedKGrid, mP::ModelParameters, sP::SimulationParameters;
            νmax = -1, iterations=1000, ftol=1e-8)
        # --- prepare auxiliary vars ---
    Nq = size(nlQ_sp.χ,1)
    Nω = size(nlQ_sp.χ,2)
    
    ωindices = (sP.dbg_full_eom_omega) ? (1:size(nlQ_ch.χ,2)) : intersect(nlQ_sp.usable_ω, nlQ_ch.usable_ω)
    ωrange = (-sP.n_iω:sP.n_iω)[ωindices]
    
    νmax = νmax < 0 ? minimum([sP.n_iν,floor(Int,3*length(ωindices)/8)]) : νmax
    νGrid = 0:(νmax-1) 
    iν_n = iν_array(mP.β, νGrid)
    iωn = 1im .* 2 .* (-sP.n_iω:sP.n_iω)[ωindices] .* π ./ mP.β
    nlQ_sp_λ = deepcopy(nlQ_sp)
    nlQ_ch_λ = deepcopy(nlQ_ch)

    Σ_hartree = mP.n * mP.U/2.0;
    Kνωq = Array{ComplexF64, length(gridshape(kG))}(undef, gridshape(kG)...)
    Kνωq_pre = Array{ComplexF64, 1}(undef, length(kG.kMult))
    Σ_ladder_ω = OffsetArray( Array{ComplexF64,3}(undef,Nq, νmax, length(ωrange)),
                              1:Nq, 0:νmax-1, ωrange)
    Σ_ladder = OffsetArray( Array{ComplexF64,2}(undef,Nq, νmax),
                              1:Nq, 0:νmax-1)
    E_pot_tail_c = [zeros(size(kG.ϵkGrid)),
            (mP.U^2 * 0.5 * mP.n * (1-0.5*mP.n) .+ Σ_hartree .* (kG.ϵkGrid .+ Σ_hartree .- mP.μ))]
    tail = [1 ./ (iν_n .^ n) for n in 1:length(E_pot_tail_c)]
    E_pot_tail = sum(E_pot_tail_c[i] .* transpose(tail[i]) for i in 1:length(tail))
    E_pot_tail_inv = sum((mP.β/2)  .* [Σ_hartree .* ones(size(kG.ϵkGrid)), (-mP.β/2) .* E_pot_tail_c[2]])
    
    function cond_both!(F, λ)
        χ_λ!(nlQ_sp_λ.χ, nlQ_sp.χ, λ[1])
        χ_λ!(nlQ_ch_λ.χ, nlQ_ch.χ, λ[2])
        calc_Σ_ω!(Σ_ladder_ω, Kνωq, Kνωq_pre, ωindices, nlQ_sp_λ, nlQ_ch_λ, Gνω, λ₀, mP.U, kG, sP)
        Σ_ladder[:] = dropdims(sum(Σ_ladder_ω, dims=[3]),dims=3) ./ mP.β .+ Σ_hartree

    
        χupup_ω = subtract_tail(0.5 * kintegrate(kG,nlQ_ch_λ.χ .+ nlQ_sp_λ.χ,1)[1,ωindices], mP.Ekin_DMFT, iωn)
        χupdo_ω = 0.5 * kintegrate(kG,nlQ_ch_λ.χ .- nlQ_sp_λ.χ,1)[1,ωindices]
        #TODO: the next line is expensive: Optimize G_from_Σ
        G_corr = transpose(flatten_2D(G_from_Σ(Σ_ladder.parent, kG.ϵkGrid, νGrid, mP)));
        E_pot = calc_E_pot(kG, G_corr, Σ_ladder.parent, E_pot_tail, E_pot_tail_inv, mP.β)
        lhs_c1 = real(sum(χupup_ω))/mP.β - mP.Ekin_DMFT*mP.β/12
        lhs_c2 = real(sum(χupdo_ω))/mP.β
        rhs_c1 = mP.n/2 * (1 - mP.n/2)
        rhs_c2 = E_pot/mP.U - (mP.n/2) * (mP.n/2)
        F[1] = lhs_c1 - rhs_c1
        F[2] = lhs_c2 - rhs_c2
        return lhs_c1, rhs_c1, lhs_c2, rhs_c2
    end
    Fint = [0.1, 0.1]

    λnew = nlsolve(cond_both!, Fint, iterations=iterations, ftol=ftol)
    return λnew
end

function λ_correction(type::Symbol, imp_density::Float64,
            nlQ_sp::NonLocalQuantities, nlQ_ch::NonLocalQuantities, 
            Gνω::GνqT, λ₀::AbstractArray{ComplexF64,3}, kG::ReducedKGrid,
            mP::ModelParameters, sP::SimulationParameters; init_sp=nothing, init_spch=nothing)
    res = if type == :sp
        rhs,usable_ω_λc = calc_λsp_rhs_usable(imp_density, nlQ_sp, nlQ_ch, kG, mP, sP)
        calc_λsp_correction(real.(nlQ_sp.χ), usable_ω_λc, mP.Ekin_DMFT, rhs, kG, mP, sP)
    elseif type == :sp_ch
        @warn "using unoptimized λ correction algorithm"
        extended_λ_clean(nlQ_sp, nlQ_ch, Gνω, λ₀, kG, mP, sP)
    else
        error("unrecognized λ correction type: $type")
    end
    return res
end

function λ_correction!(type::Symbol, imp_density, F, Σ_loc_pos, Σ_ladderLoc,
                       nlQ_sp::NonLocalQuantities, nlQ_ch::NonLocalQuantities, 
                       locQ::NonLocalQuantities,
                      χ₀::χ₀T, Gνω::GνqT, 
                      kG::ReducedKGrid,
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
