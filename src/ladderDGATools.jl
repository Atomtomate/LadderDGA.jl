#TODO: define GF type that knows about which dimension stores which variable
using Base.Iterators

function calc_bubble(νGrid::Vector{AbstractArray}, Gνω::SharedArray{Complex{Float64},2}, kGrid::T, 
        mP::ModelParameters, sP::SimulationParameters) where T <: Union{ReducedKGrid,Nothing}
    res = SharedArray{Complex{Float64},3}(2*sP.n_iω+1, length(kGrid.kMult), 2*sP.n_iν)
    @sync @distributed for ωi in 1:2*sP.n_iω+1
        for (j,νₙ) in enumerate(νGrid[ωi])
            v1 = view(Gνω, νₙ+sP.n_iω, :)
            v2 = view(Gνω, νₙ+ωi-1, :)
            res[ωi,:,j] .= -mP.β .* conv_transform(kGrid, v1 .* v2)
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
                       mP::ModelParameters, sP::SimulationParameters) where {T1 <: SumHelper, T2 <: Union{ReducedKGrid,Nothing}}
    χ = SharedArray{eltype(bubble), 2}((size(bubble)[1:2]...))
    γ = SharedArray{eltype(bubble), 3}((size(bubble)...))
    #γ_2 = Array{eltype(bubble), 1}(size(bubble,3))
    χ_ω = SharedArray{eltype(bubble), 1}(size(bubble)[1])  # ωₙ (summed over νₙ and ωₙ)

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
            @inbounds χ[ωi, qi] = sum_freq(χ_full, [1,2], sumHelper, mP.β)[1,1]
            @inbounds γ[ωi, qi, νIndices] .= sum_freq(χ_full, [2], sumHelper, 1.0)[:,1] ./ (bubble_i * (1.0 + U * χ[ωi, qi]))
            if sP.tc_type != :nothing
                extend_γ!(view(γ,ωi, qi, :), 2*π/mP.β)
            end
        end
        χ_ω[ωi] = kintegrate(kGrid, χ[ωi,:])[1]
        if (!sP.fullChi && !fixed_ω)
            usable = find_usable_interval(real(χ_ω), sum_type=sP.ωsum_type, reduce_range_prct=sP.usable_prct_reduction)
            (first(usable) > ωi) && (lower_flag = true)
            (last(usable) < ωi) && (upper_flag = true)
            (lower_flag && upper_flag) && break
        end
    end
    usable = !fixed_ω ? find_usable_interval(real(χ_ω), sum_type=sP.ωsum_type, reduce_range_prct=sP.usable_prct_reduction) : ωindices
    # if sP.tc_type != :nothing
    #    γ = convert(SharedArray, mapslices(x -> extend_γ(x, find_usable_γ(x)), γ, dims=[3]))
    # end
    return NonLocalQuantities(χ, γ, usable, 0.0)
end


function Σ_internal2!(tmp, ωindices, bubble::BubbleT, FUpDo, sumHelper::T) where T <: SumHelper
    @sync @distributed for ωi in 1:length(ωindices)
        ωₙ = ωindices[ωi]
        for qi in 1:size(bubble,2)
            for νi in 1:size(tmp,3)
                val = bubble[ωₙ,qi,:] .* FUpDo[ωₙ,νi,:]
                @inbounds tmp[ωi, qi, νi] = sum_freq(val, [1], sumHelper, 1.0)[1]
            end
        end
    end
end

function Σ_internal!(Σ, ωindices, ωZero, νZero, shift, χsp, χch, γsp, γch, Gνω,
                     tmp, U::Float64, kGrid)
    transformK(x) = (kGrid.Nk == 1) ? identity(x) : fft(expandKArr(kGrid, x))
    @sync @distributed for ωi in 1:length(ωindices)
        ωₙ = ωindices[ωi]
        @inbounds f1 = 1.5 .* (1 .+ U*χsp[ωₙ, :])
        @inbounds f2 = 0.5 .* (1 .- U*χch[ωₙ, :])
        νZerop = νZero + shift*(trunc(Int64,(ωₙ - ωZero - 1)/2) + 1)
        for νi in 1:size(Σ,3)
            @inbounds val = γsp[ωₙ, :, νZerop+νi] .* f1 .- γch[ωₙ, :, νZerop+νi] .* f2 .- 1.5 .+ 0.5 .+ tmp[ωi,:,νZerop+νi]
            Kνωq = transformK(val)
            @inbounds Σ[ωi,:, νi] = conv_transform(kGrid, Kνωq[:] .* view(Gνω,νZero + νi + ωₙ - 1,:))
        end
    end
end

function calc_Σ(Q_sp::NonLocalQuantities, Q_ch::NonLocalQuantities, bubble::BubbleT,
                Gνω::GνqT, FUpDo::SharedArray{Complex{Float64},3}, kGrid::T1,
                sumHelper_f::T2, mP::ModelParameters, sP::SimulationParameters) where {T1 <:  ReducedKGrid, T2 <: SumHelper}
    #TODO: move transform stuff to Dispersions.jl
    νZero = sP.n_iν
    ωZero = sP.n_iω
    ωindices = intersect(Q_sp.usable_ω, Q_ch.usable_ω)

    tmp = SharedArray{Complex{Float64},3}(length(ωindices), size(bubble,2), size(bubble,3))
    Σ_ladder_ω = SharedArray{Complex{Float64},3}(length(ωindices), size(bubble,2), trunc(Int,sP.n_iν-sP.shift*sP.n_iω/2))

    Σ_internal2!(tmp, ωindices, bubble, FUpDo, Naive())
    Σ_internal!(Σ_ladder_ω, ωindices, ωZero, νZero, sP.shift, Q_sp.χ, Q_ch.χ,
        Q_sp.γ, Q_ch.γ,Gνω, tmp, mP.U, kGrid)
    res = permutedims( mP.U .* sum_freq(Σ_ladder_ω, [1], Naive(), mP.β)[1,:,:] ./ kGrid.Nk, [2,1])
    return  res
end
