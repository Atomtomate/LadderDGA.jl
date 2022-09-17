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


function extended_λ_clean(χ_sp::χT, γ_sp::γT, χ_ch::χT, γ_ch::γT,
        Gνω::GνqT, λ₀::Array{ComplexF64,3},
        kG::KGrid, mP::ModelParameters, sP::SimulationParameters; 
        νmax::Int = -1, iterations=1000, ftol=1e-6)

    # general definitions
    ωindices::UnitRange{Int} = (sP.dbg_full_eom_omega) ? (1:size(χ_ch,2)) : intersect(χ_sp.usable_ω, χ_ch.usable_ω)
    νmax::Int = νmax < 0 ? minimum([sP.n_iν,floor(Int,3*length(ωindices)/8)]) : νmax
    νGrid::UnitRange{Int} = 0:(νmax-1)

    iωn = 1im .* 2 .* (-sP.n_iω:sP.n_iω)[ωindices] .* π ./ mP.β
    χsp_bak::Matrix{ComplexF64}  = deepcopy(χ_sp.data)
    χch_bak::Matrix{ComplexF64}  = deepcopy(χ_ch.data)


    # Therodynamics preallocations
    Σ_hartree::Float64 = mP.n * mP.U/2.0;
    E_pot_tail_c = [zeros(size(kG.ϵkGrid)),
            (mP.U^2 * 0.5 * mP.n * (1-0.5*mP.n) .+ Σ_hartree .* (kG.ϵkGrid .+ Σ_hartree .- mP.μ))]
    tail = [1 ./ (iν_array(mP.β, νGrid) .^ n) for n in 1:length(E_pot_tail_c)]
    E_pot_tail::Matrix{ComplexF64} = sum(E_pot_tail_c[i] .* transpose(tail[i]) for i in 1:length(tail))
    E_pot_tail_inv::Vector{Float64} = sum((mP.β/2)  .* [Σ_hartree .* ones(size(kG.ϵkGrid)), (-mP.β/2) .* E_pot_tail_c[2]])

    function cond_both!(F, λ)
        χ_λ!(χ_sp, χsp_bak, λ[1])
        χ_λ!(χ_ch, χch_bak, λ[2])
        Σ_ladder = calc_Σ(χ_sp, γ_sp, χ_ch, γ_ch, λ₀, Gνω, kG, mP, sP).parent[:,1:νmax]
        χupup_ω = subtract_tail(0.5 * kintegrate(kG,χ_ch.data .+ χ_sp.data,1)[1,ωindices], mP.Ekin_DMFT, iωn)
        χupdo_ω = 0.5 * kintegrate(kG,χ_ch.data .- χ_sp.data,1)[1,ωindices]
        #E_kin, E_pot = calc_E(Σ_ladder, kG, mP)
        G_corr = G_from_Σ(Σ_ladder, kG.ϵkGrid, νGrid, mP);
        E_pot2 = calc_E_pot(kG, G_corr, Σ_ladder, E_pot_tail, E_pot_tail_inv, mP.β)
        lhs_c1 = real(sum(χupup_ω))/mP.β - mP.Ekin_DMFT*mP.β/12
        lhs_c2 = real(sum(χupdo_ω))/mP.β
        rhs_c1 = mP.n/2 * (1 - mP.n/2)
        rhs_c2 = E_pot2/mP.U - (mP.n/2) * (mP.n/2)
        F[1] = lhs_c1 - rhs_c1
        F[2] = lhs_c2 - rhs_c2
        return nothing
    end
    Fint = [0.1, 0.1]

    res_nls = nlsolve(cond_both!, Fint, iterations=iterations, ftol=ftol)
    χ_sp.data = χsp_bak
    χ_ch.data = χch_bak
    return res_nls
end
function cond_both_int_clean(
        χ_sp::χT, γ_sp::γT, χ_ch::χT, γ_ch::γT,
        ωindices::UnitRange{Int}, Σ_ladder_ω::OffsetArray{ComplexF64,3,Array{ComplexF64,3}}, 
        Σ_ladder::OffsetArray{ComplexF64,2,Array{ComplexF64,2}}, Kνωq_pre::Vector{ComplexF64},
        G_corr::Matrix{ComplexF64},νGrid::UnitRange{Int},χ_tail::Vector{ComplexF64},Σ_hartree::Float64,
        E_pot_tail::Matrix{ComplexF64},E_pot_tail_inv::Vector{Float64},Gνω::GνqT,
        λ₀::Array{ComplexF64,3}, kG::KGrid, mP::ModelParameters, sP::SimulationParameters)

    k_norm::Int = Nk(kG)

    #TODO: unroll 
    calc_Σ_ω!(eom, Σ_ladder_ω, Kνωq_pre, ωindices, χ_sp, γ_sp, χ_ch, γ_ch, Gνω, λ₀, mP.U, kG, sP)
    Σ_ladder[:,:] = dropdims(sum(Σ_ladder_ω, dims=[3]),dims=3) ./ mP.β .+ Σ_hartree

    lhs_c1 = 0.0
    lhs_c2 = 0.0
    for (ωi,t) in enumerate(χ_tail)
        tmp1 = 0.0
        tmp2 = 0.0
        for (qi,km) in enumerate(kG.kMult)
            χsp_i_λ = real(χ_sp[qi,ωi])
            χch_i_λ = real(χ_ch[qi,ωi])
            tmp1 += (χch_i_λ + χsp_i_λ) * km
            tmp2 += (χch_i_λ - χsp_i_λ) * km
        end
        lhs_c1 += 0.5*tmp1/k_norm - t
        lhs_c2 += 0.5*tmp2/k_norm
    end
    iωn = 1im .* 2 .* (-sP.n_iω:sP.n_iω) .* π ./ mP.β

    lhs_c1 = lhs_c1/mP.β - mP.Ekin_DMFT*mP.β/12
    lhs_c2 = lhs_c2/mP.β

    #TODO: the next line is expensive: Optimize G_from_Σ
    G_corr[:] = G_from_Σ(Σ_ladder.parent, kG.ϵkGrid, νGrid, mP);
    E_pot = calc_E_pot(kG, G_corr, Σ_ladder.parent, E_pot_tail, E_pot_tail_inv, mP.β)
    rhs_c1 = mP.n/2 * (1 - mP.n/2)
    rhs_c2 = E_pot/mP.U - (mP.n/2) * (mP.n/2)
    return lhs_c1, rhs_c1, lhs_c2, rhs_c2
end
