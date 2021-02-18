#TODO: define GF type that knows about which dimension stores which variable
using Base.Iterators


function calc_bubble(νGrid::Vector{AbstractArray}, Gνω::SharedArray{Complex{Float64},2}, Nq::Int64, 
                         mP::ModelParameters, sP::SimulationParameters)
    res = SharedArray{Complex{Float64},3}(2*sP.n_iω+1,Nq,2*sP.n_iν)
    gridShape = (Nq == 1) ? [1] : repeat([sP.Nk], mP.D)
    norm = (Nq == 1) ? -mP.β : -mP.β /(sP.Nk^(mP.D));
    transform = (Nq == 1) ? identity : LadderDGA.reduce_kGrid ∘ LadderDGA.ifft_cut_mirror ∘ LadderDGA.ifft ∘ (x->reshape(x, gridShape...))
    @sync @distributed for ωi in 1:length(νGrid)
        for (j,νi) in enumerate(νGrid[ωi])
            v1 = view(Gνω, νi+sP.n_iω, :)
            v2 = view(Gνω, νi+ωi-1, :)
            res[ωi,:,j] .= norm .* transform(v1 .* v2)
        end
    end
    return res#SharedArray(permutedims(reshape(res,(2*sP.n_iν,2*sP.n_iω+1,Nq)),(2,3,1)))
end

"""
Solve χ = χ₀ - 1/β² χ₀ Γ χ
    ⇔ (1 + 1/β² χ₀ Γ) χ = χ₀
    ⇔      (χ⁻¹ - χ₀⁻¹) = 1/β² Γ
    with indices: χ[ω, q] = χ₀[]
"""
function calc_χ_trilex(Γr::SharedArray{Complex{Float64},3}, bubble::SharedArray{Complex{Float64},3}, 
                       qMultiplicity::Array{Float64,1}, U::Float64,
                       mP::ModelParameters, sP::SimulationParameters)
    W = sP.tail_corrected ? build_weights(floor(Int64,size(bubble, 3)/4), floor(Int64,size(bubble, 3)/2), [0,1,2,3]) : nothing
    sum_χ(arr::Array{Complex{Float64},2}) = sum_freq(arr, [1,2], sP.tail_corrected, mP.β, weights=W)[1,1]
    sum_γ(arr::Array{Complex{Float64},2}) = sum_freq(arr, [2], sP.tail_corrected, 1.0, weights=W)[:,1]

    χ = SharedArray{eltype(bubble), 2}((size(bubble)[1:2]...))
    γ = SharedArray{eltype(bubble), 3}((size(bubble)...))
    χ_ω = SharedArray{eltype(bubble), 1}(size(bubble)[1])  # ωₙ (summed over νₙ and ωₙ)

    UnitM = Matrix{eltype(Γr)}(I, size(Γr[1,:,:])...)

    indh = ceil(Int64, size(bubble,1)/2)
    ωindices = sP.fullChi ? 
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
        if (!sP.fullChi)
            usable = find_usable_interval(real(χ_ω))
            first(usable) > ωi && break
        end
    end
    usable = find_usable_interval(real(χ_ω), reduce_range_prct=0.05)
    if sP.tail_corrected
        γ = convert(SharedArray, mapslices(x -> extend_γ(x, find_usable_γ(x)), γ, dims=[3]))
    end
    return NonLocalQuantities(χ, γ, usable, 0.0)
end

#TODO: specify chi type (SharedArray{Complex{T}}, T = Union{Interval, Float64, AudoDiff:w
function Σ_internal_new!(Σ, ωindices::Union{Array{Int64,1},AbstractArray{Int64}},νGrid,
        χsp::SharedArray{ComplexF64,2}, χch::SharedArray{ComplexF64,2}, 
        γsp::SharedArray{ComplexF64,3}, γch::SharedArray{ComplexF64,3}, Gνω::GνqT,
        tmp::Union{Array,SharedArray{Complex{Float64},3}}, U::Float64,
        transformG::Function, transformK::Function, transform::Function) where T Float64
    #TODO: no need to loop over all nu
    @sync @distributed for ωi  in 1:length(ωindices)
        ωₙ = ωindices[ωi]
        @inbounds f1 = 1.5 .* (1 .+ U*χsp[ωₙ, :])
        @inbounds f2 = 0.5 .* (1 .- U*χch[ωₙ, :])
        println("$(ωₙ) : $(νGrid[ωₙ])")
        for (j,νi) in enumerate(νGrid[ωₙ])
            Kνωq = transformK(γsp[ωₙ, :, j] .* f1 .-
                              γch[ωₙ, :, j] .* f2 .- 1.5 .+ 0.5 .+ tmp[ωi,:,j])
            Σ[ωi, j, :] = transform(Kνωq .* transformG(view(Gνω,νi + ωₙ - 1,:)))
        end
    end
end;

function calc_Σ_new(nlQ_sp::NonLocalQuantities, nlQ_ch::NonLocalQuantities, bubble::BubbleT, 
                        Gνω::GνqT, FUpDo::Array{Complex{Float64},3}, qIndices::AbstractArray, 
                        ωindices::AbstractArray{Int64},νGrid::Vector{AbstractArray}, Nk::Int64,
                        mP::ModelParameters, sP::SimulationParameters)
    gridShape = (Nk == 1) ? [1] : repeat([sP.Nk], mP.D)
    transform = (Nk == 1) ? identity : reduce_kGrid ∘ ifft_cut_mirror ∘ ifft 
    transformG(x) = reshape(x, gridShape...)
    transformK(x) = (Nk == 1) ? identity(x) : fft(expand_kGrid(qIndices, x))

    Σ_ladder_ω = SharedArray{Complex{Float64},3}(length(ωindices), size(bubble,3), length(qIndices))
    tmp = SharedArray{Complex{Float64},3}(length(ωindices), size(bubble,2), size(bubble,3))

    Wν    = sP.tail_corrected ? build_weights(1, sP.n_iν, [0,1,2,3,4]) : nothing
    Wω    = sP.tail_corrected ? build_weights(1, floor(Int64, length(ωindices)/2), [0,1,2,3]) : nothing
    #Σ_internal2!(tmp, ωindices, bubble, view(FUpDo,:,(sP.n_iν+1):size(FUpDo,2),:), sP.tail_corrected, Wν)
    Σ_internal2!(tmp, ωindices, bubble, FUpDo, sP.tail_corrected, Wν)
    # Σ_internal!(Σ_ladder_ω, ωindices,νGrid, nlQ_sp.χ, nlQ_ch.χ,
    #             view(nlQ_sp.γ,:,:,(sP.n_iν+1):size(nlQ_sp.γ,3)), view(nlQ_ch.γ,:,:,(sP.n_iν+1):size(nlQ_ch.γ,3)),
    #             Gνω, tmp, mP.U, transformG, transformK, transform)
    Σ_internal!(Σ_ladder_ω, ωindices,νGrid, nlQ_sp.χ, nlQ_ch.χ,
                nlQ_sp.γ, nlQ_ch.γ, Gνω, tmp, mP.U, transformG, transformK, transform)
    return  mP.U .* sum_freq(Σ_ladder_ω, [1], sP.tail_corrected, mP.β, weights=Wω)[1,:,:] ./ (Nk^mP.D)
end

function Σ_internal2!(tmp::Union{Array,SharedArray{Complex{Float64},3}}, ωindices, bubble::BubbleT,
                          FUpDo::SubArray, tc::Bool, Wν)
    @sync @distributed for ωi in 1:length(ωindices)
        ωₙ = ωindices[ωi]
        for qi in 1:size(bubble,2)
            for νi in 1:size(tmp,3)
                tmp[ωi, qi, νi] = LadderDGA.sum_freq(bubble[ωₙ,qi,:] .* FUpDo[ωₙ,νi,:], [1], tc, 1.0, weights=Wν)[1]
            end
        end
    end
end

function Σ_internal!(Σ, ωindices, νGrid, νZero, χsp, χch, γsp, γch, Gνω,
                     tmp, U,transformG, transformK, transform)
    @sync @distributed for ωi  in 1:length(ωindices)
        ωₙ = ωindices[ωi]
        @inbounds f1 = 1.5 .* (1 .+ U*χsp[ωₙ, :])
        @inbounds f2 = 0.5 .* (1 .- U*χch[ωₙ, :])
        for (νi,νₙ) in enumerate(νGrid[ωₙ][1:size(tmp,3)])
            Kνωq = transformK(γsp[ωₙ, :, νZero+νi-1] .* f1 .-
                              γch[ωₙ, :, νZero+νi-1] .* f2 .- 1.5 .+ 0.5 .+ tmp[ωi,:,νi])
            Σ[ωi, νi, :] = transform(Kνωq .* transformG(view(Gνω,νi + ωₙ - 1,:)))
        end
    end
end

function calc_Σ(nlQ_sp::NonLocalQuantities, nlQ_ch::NonLocalQuantities, bubble::BubbleT,
                        Gνω::GνqT, FUpDo::Array{Complex{Float64},3}, qIndices::Vector,
                        ωindices::UnitRange{Int64}, νGrid, νZero, Nk::Int64,
                        mP::ModelParameters, sP::SimulationParameters)

    gridShape = (Nk == 1) ? [1] : repeat([sP.Nk], mP.D)
    transform = (Nk == 1) ? LadderDGA.identity : LadderDGA.reduce_kGrid ∘ LadderDGA.ifft_cut_mirror ∘ LadderDGA.ifft
    transformG(x) = reshape(x, gridShape...)
    transformK(x) = (Nk == 1) ? identity(x) : LadderDGA.fft(LadderDGA.expand_kGrid(qIndices, x))
    νSize = length(νZero:size(nlQ_ch.γ,3))
    Σ_ladder_ω = SharedArray{Complex{Float64},3}(length(ωindices), νSize, length(qIndices))
    tmp = SharedArray{Complex{Float64},3}(length(ωindices), size(bubble,2), νSize)

    Wν    = sP.tail_corrected ? build_weights(1, νSize, [0,1,2,3,4]) : nothing
    Wω    = sP.tail_corrected ? build_weights(1, floor(Int64, length(ωindices)/2), [0,1,2,3]) : nothing

    Σ_internal2!(tmp, ωindices, bubble, view(FUpDo,:,νZero:size(FUpDo,2),:), sP.tail_corrected, Wν)
    Σ_internal!(Σ_ladder_ω, ωindices, νGrid, νZero, nlQ_sp.χ, nlQ_ch.χ,
                    nlQ_sp.γ, nlQ_ch.γ,Gνω, tmp, mP.U, transformG, transformK, transform)
    return  mP.U .* sum_freq(Σ_ladder_ω, [1], sP.tail_corrected, mP.β, weights=Wω)[1,:,:] ./ (Nk^mP.D)
end
