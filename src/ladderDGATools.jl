#TODO: define GF type that knows about which dimension stores which variable
using Base.Iterators

function calc_bubble(νGrid::Vector{AbstractArray}, Gνω::SharedArray{Complex{Float64},2}, kGrid::T, 
        mP::ModelParameters, sP::SimulationParameters) where T <: Union{ReducedKGrid,Nothing}
    res = SharedArray{Complex{Float64},3}(2*sP.n_iω+1, length(kGrid.kMult), 2*sP.n_iν)
    @sync @distributed for ωi in 1:2*sP.n_iω+1
        for (j,νₙ) in enumerate(νGrid[ωi])
            v1 = view(Gνω, νₙ+sP.n_iω, :)
            v2 = view(Gνω, νₙ+ωi-1, :)
            res[ωi,:,j] .= -mP.β .* conv_fft(kGrid, v1, v2)[:]
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

    νIndices = 1:size(bubble,3)
    lower_flag = false
    upper_cut = false
    @sync @distributed for ωi in ωindices

        Γview = view(Γr,ωi,νIndices,νIndices)
        UnitM = Matrix{eltype(Γr)}(I, length(νIndices),length(νIndices))
        for qi in 1:size(bubble, 2)
            bubble_i = view(bubble,ωi, qi, νIndices)
            bubbleD = Diagonal(bubble_i)
            χ_full = (bubbleD * Γview + UnitM)\bubbleD
            @inbounds χ[ωi, qi] = sum_freq_full(χ_full, sumHelper, mP.β)
            #TODO: absor this loop into sum_freq
            tmp = (bubble_i * (1.0 + U * χ[ωi, qi]))
            for νp in νIndices
                @inbounds γ[ωi, qi, νp] = sum_freq_full((@view χ_full[νp,:]), sumHelper, 1.0) / tmp[νp]
            end
            if sP.tc_type_f != :nothing
                extend_γ!(view(γ,ωi, qi, :), 2*π/mP.β)
            end
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


function Σ_internal2!(tmp, ωindices, bubble::BubbleT, FUpDo, sumHelper::T) where T <: SumHelper
    @sync @distributed for ωi in 1:length(ωindices)
        ωₙ = ωindices[ωi]
        for qi in 1:size(bubble,2)
            for νi in 1:size(tmp,3)
                val = bubble[ωₙ,qi,:] .* FUpDo[ωₙ,νi,:]
                @inbounds tmp[ωi, qi, νi] = sum_freq_full(val, sumHelper, 1.0)
            end
        end
    end
end

#TODO: use views here
function Σ_internal!(Σ, ωindices, ωZero, νZero, shift, χsp, χch, γsp, γch, Gνω,
                     tmp, U::Float64, kGrid)
    @sync @distributed for ωi in 1:length(ωindices)
        ωₙ = ωindices[ωi]
        @inbounds f1 = 1.5 .* (1 .+ U*χsp[ωₙ, :])
        @inbounds f2 = 0.5 .* (1 .- U*χch[ωₙ, :])
        νZerop = νZero + shift*(trunc(Int64,(ωₙ - ωZero - 1)/2) + 1)
        for νi in 1:size(Σ,3)
            @inbounds Kνωq = γsp[ωₙ, :, νZerop+νi] .* f1 .- γch[ωₙ, :, νZerop+νi] .* f2 .- 1.5 .+ 0.5 .+ tmp[ωi,:,νZerop+νi]
            @inbounds Σ[ωi,:, νi] = conv_fft1(kGrid, Kνωq[:], view(Gνω,νZero + νi + ωₙ - 1,:))[:]
        end
    end
end

function calc_Σ(Q_sp::NonLocalQuantities, Q_ch::NonLocalQuantities, bubble::BubbleT,
                Gνω::GνqT, FUpDo::SharedArray{Complex{Float64},3}, kGrid::T1,
                sumHelper_f::T2, mP::ModelParameters, sP::SimulationParameters) where {T1 <:  ReducedKGrid, T2 <: SumHelper}
    #TODO: move transform stuff to Dispersions.jl
    ωindices = sP.dbg_full_eom_omega ? (1:size(bubble,1)) : intersect(Q_sp.usable_ω, Q_ch.usable_ω)
    sh_b = Naive() #get_sum_helper(ωindices, sP, :b)

    tmp = SharedArray{Complex{Float64},3}(length(ωindices), size(bubble,2), size(bubble,3))
    Σ_ladder_ω = SharedArray{Complex{Float64},3}(length(ωindices), size(bubble,2), trunc(Int,sP.n_iν-sP.shift*sP.n_iω/2))

    Σ_internal2!(tmp, ωindices, bubble, FUpDo, sumHelper_f)
    Σ_internal!(Σ_ladder_ω, ωindices, sP.n_iω, sP.n_iν, sP.shift, Q_sp.χ, Q_ch.χ,
        Q_sp.γ, Q_ch.γ,Gνω, tmp, mP.U, kGrid)
    res = permutedims( mP.U .* sum_freq(Σ_ladder_ω, [1], Naive(), mP.β)[1,:,:], [2,1])
    return  res
end
