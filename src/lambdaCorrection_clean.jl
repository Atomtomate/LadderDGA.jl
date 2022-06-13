function χ_λSum(f::Function, λ::Float64, kMult::Vector{Float64}, χ::Matrix{Float64}, 
        tail::Vector{ComplexF64})::Float64
    res = 0.0
    resi = 0.0
    norm = sum(kMult)
    for (i,ωi) in enumerate(tail)
        resi = 0.0
        for (ki,km) in enumerate(kMult)
            resi += f(χ[ki,i], λ) * km
        end
        res += resi/norm - ωi
    end
    return real(res)
end

function calc_λsp_correction_clean(χ_in::AbstractArray, usable_ω::AbstractArray{Int64},
                            EKin::Float64, rhs::Float64, kG::KGrid, 
                            mP::ModelParameters, sP::SimulationParameters)
    χr    = real.(χ_in[:,usable_ω])
    iωn = (1im .* 2 .* (-sP.n_iω:sP.n_iω)[usable_ω] .* π ./ mP.β)
    f_c1_clean(λint::Float64) = sum(subtract_tail(kintegrate(kG, χ_λ(χr, λint), 1)[1,:],mP.Ekin_DMFT,iωn))/mP.β  -mP.Ekin_DMFT*mP.β/12 - rhs
    df_c1_clean(λint::Float64) = sum(kintegrate(kG, -χ_λ(χr, λint) .^ 2, 1)[1,:])/mP.β

    λsp = newton_right(f_c1_clean, df_c1_clean, get_χ_min(χr))
    return λsp
end

function extended_λ_clean(nlQ_sp::NonLocalQuantities, nlQ_ch::NonLocalQuantities,
        Gνω::GνqT, λ₀::Array{ComplexF64,3},
        kG::KGrid, mP::ModelParameters, sP::SimulationParameters; 
        νmax::Int = -1, iterations=1000, ftol=1e-6)

    # general definitions
    ωindices::UnitRange{Int} = (sP.dbg_full_eom_omega) ? (1:size(nlQ_ch.χ,2)) : intersect(nlQ_sp.usable_ω, nlQ_ch.usable_ω)
    νmax::Int = νmax < 0 ? minimum([sP.n_iν,floor(Int,3*length(ωindices)/8)]) : νmax
    νGrid::UnitRange{Int} = 0:(νmax-1)

    iωn = 1im .* 2 .* (-sP.n_iω:sP.n_iω)[ωindices] .* π ./ mP.β
    χsp_bak::Matrix{ComplexF64}  = deepcopy(nlQ_sp.χ)
    χch_bak::Matrix{ComplexF64}  = deepcopy(nlQ_ch.χ)


    # Therodynamics preallocations
    Σ_hartree::Float64 = mP.n * mP.U/2.0;
    E_pot_tail_c = [zeros(size(kG.ϵkGrid)),
            (mP.U^2 * 0.5 * mP.n * (1-0.5*mP.n) .+ Σ_hartree .* (kG.ϵkGrid .+ Σ_hartree .- mP.μ))]
    tail = [1 ./ (iν_array(mP.β, νGrid) .^ n) for n in 1:length(E_pot_tail_c)]
    E_pot_tail::Matrix{ComplexF64} = sum(E_pot_tail_c[i] .* transpose(tail[i]) for i in 1:length(tail))
    E_pot_tail_inv::Vector{Float64} = sum((mP.β/2)  .* [Σ_hartree .* ones(size(kG.ϵkGrid)), (-mP.β/2) .* E_pot_tail_c[2]])

    function cond_both!(F, λ)
        χ_λ!(nlQ_sp.χ, χsp_bak, λ[1])
        χ_λ!(nlQ_ch.χ, χch_bak, λ[2])
        Σ_ladder = calc_Σ(nlQ_sp, nlQ_ch, λ₀, Gνω, kG, mP, sP).parent[:,1:νmax]
        χupup_ω = subtract_tail(0.5 * kintegrate(kG,nlQ_ch.χ .+ nlQ_sp.χ,1)[1,ωindices], mP.Ekin_DMFT, iωn)
        χupdo_ω = 0.5 * kintegrate(kG,nlQ_ch.χ .- nlQ_sp.χ,1)[1,ωindices]
        E_kin, E_pot = calc_E(Σ_ladder, kG, mP)
        G_corr = transpose(flatten_2D(G_from_Σ(Σ_ladder, kG.ϵkGrid, νGrid, mP)));
        E_pot2 = calc_E_pot(kG, G_corr, Σ_ladder, E_pot_tail, E_pot_tail_inv, mP.β)
        lhs_c1 = real(sum(χupup_ω))/mP.β - mP.Ekin_DMFT*mP.β/12
        lhs_c2 = real(sum(χupdo_ω))/mP.β
        rhs_c1 = mP.n/2 * (1 - mP.n/2)
        rhs_c2 = E_pot/mP.U - (mP.n/2) * (mP.n/2)
        F[1] = lhs_c1 - rhs_c1
        F[2] = lhs_c2 - rhs_c2
        return nothing
    end
    Fint = [0.1, 0.1]

    res_nls = nlsolve(cond_both!, Fint, iterations=iterations, ftol=ftol)
    nlQ_sp.χ = χsp_bak
    nlQ_ch.χ = χch_bak
    return res_nls
end

