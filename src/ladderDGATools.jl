#TODO: define GF type that knows about which dimension stores which variable
using Base.Iterators


function calc_bubble_int(Gνω::SharedArray{Complex{Float64},2}, Nq::Int64, 
                         mP::ModelParameters, sP::SimulationParameters, channel::RemoteChannel)
    res = SharedArray{Complex{Float64},3}((2*sP.n_iω+1, Nq, 2*sP.n_iν))
    gridShape = (Nq == 1) ? [1] : repeat([sP.Nk], mP.D)
    norm = (Nq == 1) ? -mP.β : -mP.β /(sP.Nk^(mP.D));
    transform = (Nq == 1) ? identity :  reduce_kGrid ∘ ifft_cut_mirror ∘ ifft ∘ (x->reshape(x, gridShape...))
    @sync @distributed for ωi in 1:(2*sP.n_iω+1)
        for νi in 1:sP.n_iν
            v1 = view(Gνω, νi+sP.n_iω, :)
            v2 = view(Gνω, νi+ωi-1, :)
            @inbounds res[ωi,:,νi+sP.n_iν] .= norm .* transform(v1 .* v2)
        end
    end
    res[:,:,1:sP.n_iν] = conj.(res[end:-1:1,:,end:-1:(sP.n_iν+1)])
    return res
end

"""
Solve χ = χ₀ - 1/β² χ₀ Γ χ
    ⇔ (1 + 1/β² χ₀ Γ) χ = χ₀
    ⇔      (χ⁻¹ - χ₀⁻¹) = 1/β² Γ
    with indices: χ[ω, q] = χ₀[]
"""

function calc_χ_trilex_int(Γr::SharedArray{Complex{Float64},3}, bubble::SharedArray{Complex{Float64},3}, 
                       qMultiplicity::Array{Float64,1}, U::Float64, β::Float64, channel::RemoteChannel; 
                       tc::Bool=false, fullRange::Bool=false)
    W = tc ? build_weights(floor(Int64,size(bubble, 3)/4), floor(Int64,size(bubble, 3)/2), [0,1,2,3]) : nothing
    sum_χ(arr::Array{Complex{Float64},2}) = sum_freq(arr, [1,2], tc, β, weights=W)[1,1]
    sum_γ(arr::Array{Complex{Float64},2}) = sum_freq(arr, [2], tc, 1.0, weights=W)[:,1]

    χ = SharedArray{eltype(bubble), 2}((size(bubble)[1:2]...))
    γ = SharedArray{eltype(bubble), 3}((size(bubble)...))
    χ_ω = SharedArray{eltype(bubble), 1}(size(bubble)[1])  # ωₙ (summed over νₙ and ωₙ)

    UnitM = Matrix{eltype(Γr)}(I, size(Γr[1,:,:])...)

    indh = ceil(Int64, size(bubble,1)/2)
    ωindices = fullRange ? 
        (1:size(bubble,1)) : 
        [(i == 0) ? indh : ((i % 2 == 0) ? indh+floor(Int64,i/2) : indh-floor(Int64,i/2)) for i in 1:size(bubble,1)]

    @sync @distributed for ωi in ωindices
        Γview = view(Γr,ωi,:,:)
        for qi in 1:size(bubble, 2)
            bubble_i = view(bubble,ωi, qi, :)
            bubbleD = Diagonal(bubble_i)
            χ_full = (bubbleD * Γview + UnitM)\bubbleD
            @inbounds χ[ωi, qi] = sum_χ(χ_full)
            @inbounds γ[ωi, qi, :] .= sum_γ(χ_full) ./ (bubble_i * (1.0 + U * χ[ωi, qi]))
        end
        χ_ω[ωi] = sum_q(χ[ωi,:], qMultiplicity)[1]
        if (!fullRange)
            usable = find_usable_interval(real(χ_ω))
            first(usable) > ωi && break
        end
    end
    usable = find_usable_interval(real(χ_ω), reduce_range_prct=0.05)
    if tc
        γ = convert(SharedArray, mapslices(x -> extend_γ(x, find_usable_γ(x)), γ, dims=[3]))
    end
    return NonLocalQuantities(χ, γ, usable, 0.0)
end

function Σ_internal2!(tmp::Union{Array,SharedArray{Complex{Float64},3}}, ωindices,
                     bubble::BubbleT, FUpDo::SubArray, tc::Bool, Wν)
    @sync @distributed for ωi in 1:length(ωindices)
        ωₙ = ωindices[ωi]
        for qi in 1:size(bubble,2)
            for νi in 1:size(FUpDo,2)
                @inbounds tmp[ωi, qi, νi] = sum_freq(bubble[ωₙ,qi,:] .* FUpDo[ωₙ,νi,:], [1], tc, 1.0, weights=Wν)[1]
            end
        end
    end
end

#TODO: specify chi type (SharedArray{Complex{T}}, T = Union{Interval, Float64, AudoDiff:w
function Σ_internal!(Σ, ωindices::Union{Array{Int64,1},UnitRange{Int64}},
                     χsp, χch, γsp::SubArray, γch::SubArray, Gνω::GνqT,
                     tmp::Union{Array,SharedArray{Complex{Float64},3}}, U::Float64,
                     transformG::Function, transformK::Function, transform::Function) where T Float64
    @sync @distributed for ωi  in 1:length(ωindices)
        ωₙ = ωindices[ωi]
        @inbounds f1 = 1.5 .* (1 .+ U*χsp[ωₙ, :])
        @inbounds f2 = 0.5 .* (1 .- U*χch[ωₙ, :])
        for νi in 1:size(γsp,3)
            @inbounds Kνωq = transformK(γsp[ωₙ, :, νi] .* f1 .-
                              γch[ωₙ, :, νi] .* f2 .- 1.5 .+ 0.5 .+ tmp[ωi,:,νi])
            @inbounds Σ[ωi, νi, :] = transform(Kνωq .* transformG(view(Gνω,νi + ωₙ - 1,:)))
        end
    end
end;

function calc_DΓA_Σ_int(nlQ_sp::NonLocalQuantities, nlQ_ch::NonLocalQuantities, bubble::BubbleT, 
                        Gνω::GνqT, FUpDo::Array{Complex{Float64},3}, qIndices::qGridT, 
                        ωindices::UnitRange{Int64}, Nk::Int64,
                        mP::ModelParameters, sP::SimulationParameters, tc::Bool)
    gridShape = (Nk == 1) ? [1] : repeat([sP.Nk], mP.D)
    transform = (Nk == 1) ? identity : reduce_kGrid ∘ ifft_cut_mirror ∘ ifft 
    transformG(x) = reshape(x, gridShape...)
    transformK(x) = (Nk == 1) ? identity(x) : fft(expand_kGrid(qIndices, x))

    Σ_ladder_ω = SharedArray{Complex{Float64},3}(length(ωindices), length(1:sP.n_iν), length(qIndices))
    tmp = SharedArray{Complex{Float64},3}(length(ωindices), size(bubble,2), sP.n_iν)

    Wν    = tc ? build_weights(1, sP.n_iν, [0,1,2,3]) : nothing
    Wω    = tc ? build_weights(1, floor(Int64, length(ωindices)/2), [0,1,2,3,4]) : nothing
    norm = mP.U / (mP.β * (Nk^mP.D))
    Σ_internal2!(tmp, ωindices, bubble, view(FUpDo,:,(sP.n_iν+1):size(FUpDo,2),:), tc, Wν)
    Σ_internal!(Σ_ladder_ω, ωindices, nlQ_sp.χ, nlQ_ch.χ,
                view(nlQ_sp.γ,:,:,(sP.n_iν+1):size(nlQ_sp.γ,3)), view(nlQ_ch.γ,:,:,(sP.n_iν+1):size(nlQ_ch.γ,3)),
                Gνω, tmp, mP.U, transformG, transformK, transform)
    return  norm .* sum_freq(Σ_ladder_ω, [1], tc, 1.0, weights=Wω)[1,:,:]
end
