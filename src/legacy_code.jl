"""
    to_m_index(arr::AbstractArray{T,2/3}, sP::SimulationParameters)

Converts array with simpel `1:N` index to larger array, where the index matches the Matsubara
Frequency number. This function is not optimized!
"""
function to_m_index(arr::AbstractArray{T,3}, sP::SimulationParameters) where T
    ωrange = -sP.n_iω:sP.n_iω
    νrange = -2*sP.n_iν:2*sP.n_iν
    length(ωrange) != size(arr,3) && @error "Assumption -n_iω:n_iω for ω grid not fulfilled."
    ωl = length(ωrange)
    νl = length(νrange)
    res = OffsetArray(zeros(ComplexF64, size(arr,1), νl, ωl), 1:size(arr,1) ,νrange, ωrange)
    for qi in 1:size(arr,1)
        to_m_index!(view(res,qi,:,:),view(arr,qi,:,:), sP)
    end
    return res
end

function to_m_index(arr::AbstractArray{T,2}, sP::SimulationParameters) where T
    ωrange = -sP.n_iω:sP.n_iω
    νrange = -2*sP.n_iν:2*sP.n_iν
    length(ωrange) != size(arr,2) && @error "Assumption -n_iω:n_iω for ω grid not fulfilled."
    ωl = length(ωrange)
    νl = length(νrange)
    res = OffsetArray(zeros(ComplexF64, νl,ωl), νrange, ωrange)
    to_m_index!(res, arr, sP)
    return res
end

function to_m_index!(res::AbstractArray{T,2}, arr::AbstractArray{T,2}, sP::SimulationParameters) where T
    for ωi in 1:size(arr,2)
        for νi in 1:size(arr,1)
            ωn,νn = OneToIndex_to_Freq(ωi, νi, sP)
            @inbounds res[νn, ωn] = arr[νi,ωi]
        end
    end
    return res
end

function ωindex_range(sP::SimulationParameters)
    return 1:(2*sP.n_iω+1)
    # TODO: placeholder for reduced omega-range computations
end




function cond_both_int(λch_i::Float64, 
            χ_sp::χT, γ_sp::γT, χ_ch::χT, γ_ch::γT,
            Σ_loc,gLoc_rfft::GνqT,
            λ₀::Array{ComplexF64,3}, kG::KGrid, mP::ModelParameters, sP::SimulationParameters)

    k_norm::Int = Nk(kG)
    ωindices = usable_ωindices(sP, χ_sp, χ_ch)
    νmax::Int = minimum([sP.n_iν,floor(Int,3*length(ωindices)/8)])
    iωn = 1im .* 2 .* (-sP.n_iω:sP.n_iω)[ωindices] .* π ./ mP.β
    iωn[findfirst(x->x ≈ 0, iωn)] = Inf
    χ_tail::Vector{ComplexF64} = χ_ch.tail_c[3] ./ (iωn.^2)


    χ_λ!(χ_ch, χ_ch, λch_i)
    rhs_c1 = 0.0
    for (ωi,t) in enumerate(χ_tail)
        tmp1 = 0.0
        for (qi,km) in enumerate(kG.kMult)
            χch_i_λ = χ_ch[qi,ωi]
            tmp1 += χch_i_λ * km
        end
        rhs_c1 -= real(tmp1/k_norm - t)
    end
    rhs_c1 = rhs_c1/mP.β + χ_ch.tail_c[3]*mP.β/12 + mP.n * (1 - mP.n/2)
    λsp_i = λsp_correction(χ_sp, real(rhs_c1), kG, mP, sP)
    χ_λ!(χ_sp, χ_sp, λsp_i)

    #TODO: use parallel version! 
    Σ_ladder = calc_Σ(χ_sp, γ_sp, χ_ch, γ_ch, λ₀, gLoc_rfft, kG, mP, sP);

    lhs_c1, lhs_c2 = lhs_int(χ_sp.data, χ_ch.data, 0.0, 0.0, 
                            χ_tail, kG.kMult, k_norm, χ_sp.tail_c[3], mP.β)

    #TODO: the next line is expensive: Optimize G_from_Σ
    μnew, GLoc_new, _, _ = G_from_Σladder(Σ_ladder[:,0:νmax-1], Σ_loc, kG, mP, sP)
    E_kin, E_pot = calc_E(GLoc_new[:,0:νmax-1].parent, Σ_ladder.parent, kG, mP, νmax = νmax)
    rhs_c1 = mP.n/2 * (1 - mP.n/2)
    rhs_c2 = E_pot/mP.U - (mP.n/2) * (mP.n/2)
    reset!(χ_sp)
    reset!(χ_ch)
    return λsp_i, lhs_c1, rhs_c1, lhs_c2, rhs_c2, μnew, true
end
