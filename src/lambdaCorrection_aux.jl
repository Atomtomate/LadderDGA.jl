dχ_λ(χ, λ::Float64) = map(χi -> - ((1.0 / χi) + λ)^(-2), χ)
χ_λ2(χ, λ) = map(χi -> 1.0 / ((1.0 / χi) + λ), χ)

@inline χ_λ(χ::Float64, λ::Float64)::Float64 = χ/(λ*χ + 1)
@inline dχ_λ(χ::Float64, λ::Float64)::Float64 = -χ_λ(χ, λ)^2

function χ_λ(χ::χT, λ::Float64) where T <: Union{ComplexF64, Float64}
    χ_new = χT(χ.data)
    χ_λ!(χ_new, χ, λ)
    return χ_new 
end

function χ_λ!(χ_new::χT, χ::χT, λ::Float64) where T <: Union{ComplexF64, Float64}
    χ_λ!(χ_new.data, χ.data, λ)
    χ_new.λ = χ.λ + λ
    return χ_new 
end

χ_λ!(χ::χT, λ::Float64) = χ_λ!(χ, χ, λ)

function χ_λ(χ::AbstractArray{T}, λ::Float64) where T <: Union{ComplexF64, Float64}
    res = similar(χ)
    χ_λ!(res, χ, λ)
    return res
end

function χ_λ!(χ_λ::AbstractArray{ComplexF64}, χ::AbstractArray{ComplexF64}, λ::Float64)
    @simd for i in eachindex(χ_λ)
        @inbounds χ_λ[i] = χ[i] / ((λ * χ[i]) + 1)
    end
end

function χ_λ!(χ_λ::AbstractArray{Float64}, χ::AbstractArray{Float64}, λ::Float64)
    @simd for i in eachindex(χ_λ)
        @inbounds χ_λ[i] = χ[i] / ((λ * χ[i]) + 1)
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

function correct_margins(λl::Float64, λr::Float64, Fl::Float64, Fr::Float64)::Tuple{Float64,Float64}
    Δ = 2 .* (λr .- λl)
    Fr > 0 && (λr = λr + Δ)
    Fl < 0 && (λl = λl - Δ)
    λl, λr
end

function correct_margins(λl::Vector{Float64}, λr::Vector{Float64},
                         Fl::Vector{Float64},Fr::Vector{Float64})::Tuple{Vector{Float64},Vector{Float64}}
    λl_1, λr_1 = correct_margins(λl[1], λr[1], Fl[1], Fr[1])
    λl_2, λr_2 = correct_margins(λl[2], λr[2], Fl[2], Fr[2])
    ([λl_1, λl_2], [λr_1, λr_2])
end

function bisect(λl::Float64, λm::Float64, λr::Float64, Fm::Float64)::Tuple{Float64,Float64}
    Fm > 0 ? (λm,λr) : (λl,λm)
end


function bisect(λl::Vector{Float64}, λm::Vector{Float64}, λr::Vector{Float64},
        Fm::Vector{Float64})::Tuple{Vector{Float64},Vector{Float64}}
    λl_1, λr_1 = bisect(λl[1], λm[1], λr[1], Fm[1])
    λl_2, λr_2 = bisect(λl[2], λm[2], λr[2], Fm[2])
    ([λl_1, λl_2], [λr_1, λr_2])
end

function lhs_c2_fast(χ_sp::χT, χ_ch::χT, χ_tail::Vector{Float64},
                  kMult::Vector{Float64}, k_norm::Int, β::Float64)
    lhs_c2 = 0.0
    for (ωi,t) in enumerate(χ_tail)
        tmp2 = 0.0
        for (qi,km) in enumerate(kMult)
            χsp_i_λ = real(χ_sp[qi,ωi])
            χch_i_λ = real(χ_ch[qi,ωi])
            tmp2 += (χch_i_λ - χsp_i_λ) * km
        end
        lhs_c2 += 0.5*tmp2/k_norm
    end
    lhs_c2 = lhs_c2/β
    return lhs_c2 
end

function lhs_fast(χ_sp::χT, χ_ch::χT, χ_tail::Vector{ComplexF64},
                  kMult::Vector{Float64}, k_norm::Int, Ekin_DMFT::Float64, β::Float64)
    lhs_c1 = 0.0
    lhs_c2 = 0.0
    for (ωi,t) in enumerate(χ_tail)
        tmp1 = 0.0
        tmp2 = 0.0
        for (qi,km) in enumerate(kMult)
            χsp_i_λ = real(χ_sp[qi,ωi])
            χch_i_λ = real(χ_ch[qi,ωi])
            tmp1 += (χch_i_λ + χsp_i_λ) * km
            tmp2 += (χch_i_λ - χsp_i_λ) * km
        end
        lhs_c1 += 0.5*tmp1/k_norm - t
        lhs_c2 += 0.5*tmp2/k_norm
    end
    lhs_c1 = lhs_c1/β - Ekin_DMFT*β/12
    lhs_c2 = lhs_c2/β
    return lhs_c1, lhs_c2 
end


function cond_both_int(
        λch_i::Float64, 
        χ_sp::χT, γ_sp::γT, χ_ch::χT, γ_ch::γT,
        χsp_tmp::χT, χch_tmp::χT,
        ωindices::UnitRange{Int}, Σ_ladder_ω::OffsetArray{ComplexF64,3,Array{ComplexF64,3}}, 
        Σ_ladder::OffsetArray{ComplexF64,2,Array{ComplexF64,2}}, Kνωq_pre::Vector{ComplexF64},
        G_corr::Matrix{ComplexF64},νGrid::UnitRange{Int},χ_tail::Vector{ComplexF64},Σ_hartree::Float64,
        E_pot_tail::Matrix{ComplexF64},E_pot_tail_inv::Vector{Float64},Gνω::GνqT,
        λ₀::Array{ComplexF64,3}, kG::KGrid, mP::ModelParameters, sP::SimulationParameters)

    k_norm::Int = Nk(kG)

    χ_λ!(χ_ch, χch_tmp, λch_i)
    rhs_c1 = 0.0
    for (ωi,t) in enumerate(χ_tail)
        tmp1 = 0.0
        for (qi,km) in enumerate(kG.kMult)
            χch_i_λ = χ_ch[qi,ωi]
            tmp1 += χch_i_λ * km
        end
        rhs_c1 -= real(tmp1/k_norm - t)
    end
    rhs_c1 = rhs_c1/mP.β + mP.Ekin_DMFT*mP.β/12 + mP.n * (1 - mP.n/2)
    λsp_i = calc_λsp_correction(χ_sp, ωindices, mP.Ekin_DMFT, real(rhs_c1), kG, mP, sP)
    χ_λ!(χ_sp, χsp_tmp, λsp_i)
    χsp_sum = sum(kintegrate(kG,real(χ_sp),1)[1,ωindices])/mP.β
    χch_sum = sum(kintegrate(kG,real(χ_ch),1)[1,ωindices])/mP.β
    @info "c1 check: $χsp_sum + $χch_sum  = $(χsp_sum + χch_sum) ?=? 1/2" 

    #TODO: unroll 
    calc_Σ_ω!(eom, Σ_ladder_ω, Kνωq_pre, ωindices, χ_sp, γ_sp, χ_ch, γ_ch, Gνω, λ₀, mP.U, kG, sP)
    Σ_ladder[:,:] = dropdims(sum(Σ_ladder_ω, dims=[3]),dims=3) ./ mP.β .+ Σ_hartree

    lhs_c1, lhs_c2 = lhs_fast(χ_sp, χ_ch, χ_tail, kG.kMult, k_norm, mP.Ekin_DMFT, mP.β)

    #TODO: the next line is expensive: Optimize G_from_Σ
    G_corr[:] = G_from_Σ(Σ_ladder.parent, kG.ϵkGrid, νGrid, mP);
    E_pot = calc_E_pot(kG, G_corr, Σ_ladder.parent, E_pot_tail, E_pot_tail_inv, mP.β)
    rhs_c1 = mP.n/2 * (1 - mP.n/2)
    rhs_c2 = E_pot/mP.U - (mP.n/2) * (mP.n/2)
    χ_sp.data = deepcopy(χsp_tmp.data)
    χ_ch.data = deepcopy(χch_tmp.data)
    return λsp_i, lhs_c1, rhs_c1, lhs_c2, rhs_c2
    return 
end

function cond_both_int!(F::Vector{Float64}, λ::Vector{Float64}, 
        χ_sp::χT, γ_sp::γT, χ_ch::χT, γ_ch::γT,
        χsp_tmp::χT, χch_tmp::χT,
        ωindices::UnitRange{Int}, Σ_ladder_ω::OffsetArray{ComplexF64,3,Array{ComplexF64,3}}, 
        Σ_ladder::OffsetArray{ComplexF64,2,Array{ComplexF64,2}}, Kνωq_pre::Vector{ComplexF64},
        G_corr::Matrix{ComplexF64},νGrid::UnitRange{Int},χ_tail::Vector{ComplexF64},Σ_hartree::Float64,
        E_pot_tail::Matrix{ComplexF64},E_pot_tail_inv::Vector{Float64},Gνω::GνqT,
        λ₀::Array{ComplexF64,3}, kG::KGrid, mP::ModelParameters, sP::SimulationParameters, trafo::Function)::Nothing

    λi = trafo(λ)
    χ_λ!(χ_sp, χsp_tmp, λi[1])
    χ_λ!(χ_ch, χch_tmp, λi[2])
    k_norm::Int = Nk(kG)

    #TODO: unroll 
    calc_Σ_ω!(eom, Σ_ladder_ω, Kνωq_pre, ωindices, χ_sp, γ_sp, χ_ch, γ_ch, Gνω, λ₀, mP.U, kG, sP)
    Σ_ladder[:,:] = dropdims(sum(Σ_ladder_ω, dims=[3]),dims=3) ./ mP.β .+ Σ_hartree

    lhs_c1, lhs_c2 = lhs_fast(χ_sp, χ_ch, χ_tail, kG.kMult, k_norm, mP.Ekin_DMFT, mP.β)

    #TODO: the next line is expensive: Optimize G_from_Σ
    G_corr[:] = G_from_Σ(Σ_ladder.parent, kG.ϵkGrid, νGrid, mP)
    E_pot = calc_E_pot(kG, G_corr, Σ_ladder.parent, E_pot_tail, E_pot_tail_inv, mP.β)
    rhs_c1 = mP.n/2 * (1 - mP.n/2)
    rhs_c2 = E_pot/mP.U - (mP.n/2) * (mP.n/2)
    F[1] = lhs_c1 - rhs_c1
    F[2] = lhs_c2 - rhs_c2
    return nothing
end

