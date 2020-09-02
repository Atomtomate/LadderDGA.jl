#TODO: define GF type that knows about which dimension stores which variable
using Base.Iterators

function calc_bubble_fft(Gνω, Nk::Int64, mP::ModelParameters, sP::SimulationParameters)
    res = Array{Complex{Float64}}(undef, 2*sP.n_iω+1, Nk, 2*sP.n_iν)
    norm = (Nk == 1) ? -mP.β : -mP.β /(sP.Nk^(mP.D));
    transform = Nk == 1 ? identity : reduce_kGrid ∘ ifft_cut_mirror ∘ ifft
    for ωi in 1:(2*sP.n_iω+1), νi in 1:sP.n_iν
        @inbounds res[ωi,:,νi+sP.n_iν] .= norm .* transform(Gνω[νi+sP.n_iω] .* Gνω[νi+ωi-1])
    end
    res[:,:,1:sP.n_iν] = conj.(res[end:-1:1,:,end:-1:(sP.n_iν+1)])
    return res
end

function calc_tmp(Γ::Array{T,2}, bubble::Array{T,2})::Array{T,2} where T <: Number
    ((Diagonal(bubble[qi, :]) * Γ + Matrix{eltype(Γ)}(I, size(Γ[:,:])...))\Diagonal(bubble[qi, :]) for qi in 1:size(bubble,1))
end


#TODO: this can probabily be optimized a lot
"""
Solve χ = χ₀ - 1/β² χ₀ Γ χ
    ⇔ (1 + 1/β² χ₀ Γ) χ = χ₀
    ⇔      (χ⁻¹ - χ₀⁻¹) = 1/β² Γ
    with indices: χ[ω, q] = χ₀[]
"""
function calc_χ_trilex(Γsp::Array{T,3}, Γch::Array{T,3}, bubble::Array{T,3}, qMultiplicity,
                              mP::ModelParameters, sP::SimulationParameters) where T <: Number
    νmin = Int(floor(size(bubble, 3)))                                 # min freq for freq fit
    νmax = Int(floor(size(bubble, 3)/2))                               # max freq for freq fit
    W    = sP.tail_corrected ? build_weights(νmin, νmax, [0,1,2,3]) : nothing

    χsp   = zeros(eltype(bubble), size(bubble)[1:2]...)    # ωₙ x q (summed over νₙ)
    χsp_ω = zeros(eltype(bubble), size(bubble)[1])    # ωₙ (summed over νₙ and ωₙ)
    trilexsp = zeros(eltype(bubble), size(bubble)...)

    χch   = zeros(eltype(bubble), size(bubble)[1:2]...)    # ωₙ x q (summed over νₙ)
    χch_ω = zeros(eltype(bubble), size(bubble)[1])    # ωₙ (summed over νₙ and ωₙ)
    trilexch = zeros(eltype(bubble), size(bubble)...)

    UnitM = Matrix{eltype(Γsp)}(I, size(Γsp[1,:,:])...)
    usable_sp = nothing
    usable_ch = nothing


    indh = ceil(Int64, size(bubble,1)/2)
    ωindices = sP.fullChi ? 
            ((i == 0) ? indh : ((i % 2 == 0) ? indh+floor(Int64,i/2) : indh-floor(Int64,i/2)) for i in 1:size(bubble,1)) :
                1:size(bubble,2)

    for ωi in ωindices
        ΓspView = view(Γsp,ωi,:,:)
        ΓchView = view(Γch,ωi,:,:)
        for qi in 1:size(bubble, 2)
            bubble_i = bubble[ωi, qi, :]
            bubbleD = Diagonal(bubble_i)
            # input: vars: bubble, gamma, U, tc, beta ; functions: sum_freq ; out: χ, γ 

            @inbounds A = bubbleD * ΓspView + UnitM 
            χ_full_sp = A\bubbleD
            @inbounds χsp[ωi, qi] = sum_freq(χ_full_sp, [1,2], sP.tail_corrected, mP.β, weights=W)[1,1]
            @inbounds A = bubbleD * ΓchView + UnitM
            χ_full_ch = A\bubbleD
            @inbounds χch[ωi, qi] = sum_freq(χ_full_ch, [1,2], sP.tail_corrected, mP.β, weights=W)[1,1]

            @inbounds trilexsp[ωi, qi, :] .= sum_freq(χ_full_sp, [2], sP.tail_corrected, 1.0, weights=W)[:,1] ./ (bubble_i* (1.0 + mP.U * χsp[ωi, qi]))
            @inbounds trilexch[ωi, qi, :] .= sum_freq(χ_full_ch, [2], sP.tail_corrected, 1.0, weights=W)[:,1] ./  (bubble_i* (1.0 - mP.U * χch[ωi, qi]))
        end

        χsp_ω[ωi] = sum(χsp[ωi,:] .* qMultiplicity) / sum(qMultiplicity)
        χch_ω[ωi] = sum(χch[ωi,:] .* qMultiplicity) / sum(qMultiplicity)
        if (!sP.fullChi)
            usable_sp = find_usable_interval(real(χsp_ω))
            usable_ch = find_usable_interval(real(χch_ω))
            first(usable_sp) > ωi && first(usable_ch) > ωi && break
        end
    end
    usable_sp = find_usable_interval(real(χsp_ω), reduce_range_prct=0.1)
    usable_ch = find_usable_interval(real(χch_ω), reduce_range_prct=0.1)

    return χsp, χch, χsp_ω, χch_ω, trilexsp, trilexch, usable_sp, usable_ch
end

"""

"""
function calc_DΓA_Σ_fft(χsp, χch, γsp, γch, bubble, Gνω, FUpDo, ϵkGrid, qIndices, usable_ω, usable_ν, Nk,mP::ModelParameters, sP::SimulationParameters, tc::Bool)
    #usable_ν = 1:(last(usable_ν) - sP.n_iν)
    ωindices = sP.fullSums ? (1:(2*sP.n_iω+1)) : usable_ω
    Σ_ladder_ω = zeros(Complex{Float64}, length(ωindices), length(usable_ν), length(qIndices))
    νpindices = sP.n_iν #(sP.n_iν - last(usable_ν) + 1):(sP.n_iν  + last(usable_ν) - 1)
    Nq = size(bubble,2)

    @info "using ω range: " ωindices
    @info "using νp range: " νpindices
    Wν    = tc ? build_weights(Int(floor(last(νpindices)*3/5)), Int(last(νpindices)), [0,1,2,3]) : nothing
    Wω    = tc ? build_weights(Int(floor(last(ωindices .- sP.n_iω)*3/5)), Int(last(ωindices) - sP.n_iω), [0,1,2,3]) : nothing
    transform = length(qIndices) == 1 ? identity : reduce_kGrid ∘ ifft_cut_mirror ∘ ifft
    transformK(x) = length(qIndices) == 1 ? identity(x) : fft(expand_kGrid(qIndices, x))
    norm = mP.U / (mP.β * (Nk^mP.D))
    
    for (ω_ind, ωi) in enumerate(ωindices)
        f1 = 1.5 .* (1 .+ mP.U*χsp[ωi, :])
        f2 = 0.5 .* (1 .- mP.U*χch[ωi, :])
        for νi in usable_ν
            tmp = [sum_freq(bubble[ωi,qi,:] .* FUpDo[ωi,νi+sP.n_iν,:], [1], tc, 1.0, weights=Wν)[1] for qi in 1:Nq]
            Kνωq = γsp[ωi, :, νi+sP.n_iν] .* f1 .-
                   γch[ωi, :, νi+sP.n_iν] .* f2 .- 1.5 .+ 0.5 .+ tmp
            Kνωq = transformK(Kνωq)
            Σ_ladder_ω[ω_ind, νi, :] += transform(Kνωq .* Gνω[νi + ωi - 1])
        end
    end
    return  norm .* sum_freq(Σ_ladder_ω, [1], tc, 1.0, weights=Wω)[1,:,:]
end

function Σ_DMFT_correction
end
