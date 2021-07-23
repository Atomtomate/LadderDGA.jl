#TODO: define GF type that knows about which dimension stores which variable
using Base.Iterators

function calc_bubble(νGrid::Vector{AbstractArray}, Gνω::GνqT, kGrid::T, 
        mP::ModelParameters, sP::SimulationParameters) where T <: Union{ReducedKGrid,Nothing}
    res = SharedArray{Complex{Float64},3}(2*sP.n_iω+1, length(kGrid.kMult), 2*sP.n_iν)
    @sync @distributed for ωi in axes(res,1)
        for νi in axes(res,3)
            ωn, νn = OneToIndex_to_Freq(ωi, νi, sP)
            v1 = view(Gνω, νn, :)
            v2 = view(Gνω, νn+ωn, :)
            res[ωi,:,νi] .= -mP.β .* conv_fft(kGrid, v1, v2)[:]
        end
    end
    return res
end

"""
Solve χ = χ₀ - 1/β² χ₀ Γ χ
    ⇔ (1 + 1/β² χ₀ Γ) χ = χ₀
    ⇔      (χ⁻¹ - χ₀⁻¹) = 1/β² Γ
    with indices: χ[ω, q] = χ₀[]
"""
function calc_χ_trilex(Γr::SharedArray{Complex{Float64},3}, bubble::SharedArray{Complex{Float64},3}, 
                       kGrid::T2, νGrid, sumHelper::T1, U::Float64,
                       mP::ModelParameters, sP::SimulationParameters) where {T1 <: SumHelper, T2 <: ReducedKGrid}
    χ = SharedArray{eltype(bubble), 2}((size(bubble)[1:2]...))
    γ = SharedArray{eltype(bubble), 3}((size(bubble)...))
    χ_ω = Array{Float64, 1}(undef, size(bubble,1))  # ωₙ (summed over νₙ and ωₙ)
    ωZero = sP.n_iω

    indh = ceil(Int64, size(bubble,1)/2)
    fixed_ω = typeof(sP.ωsum_type) == Tuple{Int,Int}
    ωindices = if sP.fullChi
            (1:size(bubble,1)) 
        elseif fixed_ω
            mid_index = Int(ceil(size(bubble,1)/2))
            default_sum_range(mid_index, sP.ωsum_type)
        else
            [(i == 0) ? indh : ((i % 2 == 0) ? indh+floor(Int64,i/2) : indh-floor(Int64,i/2)) for i in 1:size(bubble,1)]
        end

    UnitM = Matrix{eltype(Γr)}(I, size(bubble,3), size(bubble,3))
    lower_flag = false
    upper_cut = false
    @sync @distributed for ωi in ωindices
        Γview = view(Γr,ωi,:,:)
        for qi in axes(bubble,2)
            bubble_i = view(bubble,ωi, qi, :)
            bubbleD  = Diagonal(bubble_i)
            χ_full   = (bubbleD * Γview + UnitM)\bubbleD
            @inbounds χ[ωi, qi] = sum_freq_full(χ_full, sumHelper, mP.β)
            #TODO: absor this loop into sum_freq, partial sum is carried out twice
            for νp in axes(γ, 3)
                @inbounds γ[ωi, qi, νp] = sum_freq_full((@view χ_full[νp,:]), sumHelper, 1.0) / (bubble[ωi, qi, νp] * (1.0 + U * χ[ωi, qi]))
            end
            (sP.tc_type_f != :nothing) && extend_γ!(view(γ,ωi, qi, :), 2*π/mP.β)
        end
        if (!sP.fullChi && !fixed_ω)
            @warn "Deactivating the fullChi option can lead to issues and is not recomended for this version."
            #usable = find_usable_interval(real(χ_ω), sum_type=sP.ωsum_type, reduce_range_prct=sP.usable_prct_reduction)
            #(first(usable) > ωi) && (lower_flag = true)
            #(last(usable) < ωi) && (upper_flag = true)
            #(lower_flag && upper_flag) && break
        end
    end

    if sP.ω_smoothing != :full
        for ωi in ωindices
            χ_ω[ωi] = real.(kintegrate(kGrid, χ[ωi,:])[1])
        end
    end
    if sP.ω_smoothing == :full
        for qi in 1:size(bubble, 2)
            filter_MA!(χ[1:ωZero,qi],3,χ[1:ωZero,qi])
            filter_MA!(χ[ωZero:end,qi],3,χ[ωZero:end,qi])
        end
        for ωi in ωindices
            χ_ω[ωi] = real.(kintegrate(kGrid, χ[ωi,:])[1])
        end
    elseif sP.ω_smoothing == :range
        filter_MA!(χ_ω[1:ωZero],3,χ_ω[1:ωZero])
        filter_MA!(χ_ω[ωZero:end],3,χ_ω[ωZero:end])
    end

    usable = !fixed_ω ? find_usable_interval(χ_ω, sum_type=sP.ωsum_type, reduce_range_prct=sP.usable_prct_reduction) : ωindices
    return NonLocalQuantities(χ, γ, usable, 0.0)
end


function Σ_internal!(tmp::SharedArray{Float64,3}, ωindices::AbstractArray{Int,1}, bubble::BubbleT, 
                     FUpDo, sumHelper::T) where T <: SumHelper
    @sync @distributed for ωii in 1:length(ωindices)
        ωi = ωindices[ωii]
        for qi in axes(bubble,2)
            for νi in axes(tmp,3)
                @inbounds tmp[ωii, qi, νi] = real(sum_freq_full(FUpDo[ωi,νi,:] .* bubble[ωi,qi,:], sumHelper, 1.0))
            end
        end
    end
end

"""
    calc_Σ_ω!

TODO: docstring
- provide `Σ` with `ν` range (third rank) half size in order to only compute positive fermionic indices.
"""
function calc_Σ_ω!(Σ::SharedArray{Complex{Float64},3}, ωindices::AbstractArray{Int,1}, Q_sp::TQ, Q_ch::TQ,
                    Gνω::GνqT, tmp::SharedArray{Float64,3}, U::Float64, kGrid::ReducedKGrid, 
                    sP::SimulationParameters; onlyPositive::Bool=true, lopWarn=false) where TQ <: Union{NonLocalQuantities, ImpurityQuantities}
    (!onlyPositive) && @error "Full nu range not tested!!!"

    @sync @distributed for ωii in 1:length(ωindices)
        ωi = ωindices[ωii]
        ωn = (ωi - sP.n_iω) - 1
        fsp = 1.5 .* (1 .+ U*Q_sp.χ[ωi, :])
        fch = 0.5 .* (1 .- U*Q_ch.χ[ωi, :])
        νZero = ν0Index_of_ωIndex(ωi, sP)
        maxn = minimum([νZero + sP.n_iν-1, size(tmp,3)])
        if lopWarn && (νZero + sP.n_iν-1 > size(tmp,3)) 
            ωn1, νn1 = OneToIndex_to_Freq(ωi, size(tmp,3) + 1, sP)
            ωn2, νn2 = OneToIndex_to_Freq(ωi, νZero + sP.n_iν - 1, sP)
            @warn "running out of data for νn = $(νn1) to $(νn2) at ωn = $ωn1, $ωn"
        end
        for (νn,νi) in enumerate(νZero:maxn)
            #ωn, νn = OneToIndex_to_Freq(ωi, νi, sP)
            #@warn "$ωii -> $ωi -> $ωn : $((νn,νi))"
            Kνωq = Q_sp.γ[ωi, :, νi] .* fsp .- Q_ch.γ[ωi, :, νi] .* fch .- 1.5 .+ 0.5 .+ tmp[ωii,:,νi]
            Σ[νn,:, ωii] = conv_fft1(kGrid, Kνωq, view(Gνω, (νn-1) + ωn,:))
        end
    end
end

function calc_Σ_dbg(Q_sp::NonLocalQuantities, Q_ch::NonLocalQuantities, bubble::BubbleT,
                Gνω::GνqT, FUpDo::SharedArray{Complex{Float64},3}, kGrid::T1,
                sumHelper_f::T2, mP::ModelParameters, sP::SimulationParameters) where {T1 <:  ReducedKGrid, T2 <: SumHelper}
    ωindices = (sP.dbg_full_eom_omega) ? (1:size(bubble,1)) : intersect(Q_sp.usable_ω, Q_ch.usable_ω)
    sh_b = Naive()#get_sum_helper(ωindices, sP, :b)

    tmp = SharedArray{Float64,3}(length(ωindices), size(bubble,2), size(bubble,3))
    Σ_ladder_ω = SharedArray{Complex{Float64},3}( sP.n_iν, size(bubble,2), size(bubble,1))

    Σ_internal!(tmp, ωindices, bubble, FUpDo, sumHelper_f)
    (sP.tc_type_f != :nothing) && extend_tmp!(tmp)

    calc_Σ_ω!(Σ_ladder_ω, ωindices, Q_sp, Q_ch, Gνω, tmp, mP.U, kGrid, sP, lopWarn=true)
    res = mP.U .* sum_freq(Σ_ladder_ω, [3], sh_b, mP.β)[:,:,1]
    Σ_ladder_ω = permutedims(Σ_ladder_ω, [3,2,1])
    return  res, Σ_ladder_ω, tmp
end

function calc_Σ(Q_sp::NonLocalQuantities, Q_ch::NonLocalQuantities, bubble::BubbleT,
                Gνω::GνqT, FUpDo::SharedArray{Complex{Float64},3}, kGrid::T1,
                sumHelper_f::T2, mP::ModelParameters, sP::SimulationParameters) where {T1 <:  ReducedKGrid, T2 <: SumHelper}
    ωindices = (sP.dbg_full_eom_omega) ? (1:size(bubble,1)) : intersect(Q_sp.usable_ω, Q_ch.usable_ω)
    sh_b = Naive()#get_sum_helper(ωindices, sP, :b)

    tmp = SharedArray{Float64,3}(length(ωindices), size(bubble,2), size(bubble,3))
    Σ_ladder_ω = SharedArray{Complex{Float64},3}( sP.n_iν, size(bubble,2), length(ωindices))

    Σ_internal!(tmp, ωindices, bubble, FUpDo, sumHelper_f)
    (sP.tc_type_f != :nothing) && extend_tmp!(tmp)
    calc_Σ_ω!(Σ_ladder_ω, ωindices, Q_sp, Q_ch, Gνω, tmp, mP.U, kGrid, sP)
    res = mP.U .* sum_freq(Σ_ladder_ω, [3], sh_b, mP.β)[:,:,1]
    return  res
end
