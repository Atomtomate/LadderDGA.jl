#TODO: nw and niv grid as parameters? 
#TODO: define GF type that knows about which dimension stores which variable
using Base.Iterators

@everywhere @inline @fastmath GF_from_Σ(n::Int64, β::Float64, μ::Float64, ϵₖ::T, Σ::Complex{T}) where T =
                    1/((π/β)*(2*n + 1)*1im + μ - ϵₖ - Σ)
@everywhere @inline @fastmath w_from_Σ(n::Int64, β::Float64, μ::Float64, Σ::Complex{Float64}) =
                    ((π/β)*(2*n + 1)*1im + μ - Σ)

@inline function G_fft(ind::Int64, Σ::Array{Complex{Float64},1}, 
                                 ϵkGrid::Base.Generator, β::Float64, μ::Float64)
    Σν = get_symm_f(Σ,ind)
    @inbounds Gν  = map(ϵk -> GF_from_Σ(ind, β, μ, ϵk, Σν), ϵkGrid)
    return fft!(Gν)
end

function calc_bubble_fft(Σ::Array{Complex{Float64},1}, ϵkGrid::Base.Generator, redGridSize::Int64,
                              mP::ModelParameters, sP::SimulationParameters)

    res = Array{Complex{Float64}}(undef, 2*sP.n_iω+1, redGridSize, sP.n_iν)
    println("in bubble: ", size(Σ))
    println("gnuomega indices: ", length(collect((-sP.n_iω):(sP.n_iν+sP.n_iω))))
    println("in bubble: ", mP)
    println("in bubble: ", sP)
    @inbounds Gνω = [G_fft(ind, Σ, ϵkGrid, mP.β, mP.μ) for ind in (-sP.n_iω):(sP.n_iν+sP.n_iω-1)]
    #println(Gνω)
    norm = -mP.β ./ (sP.Nk^mP.D) 

    for ωi in 0:2*sP.n_iω
        for νi in 1:sP.n_iν
            @inbounds res[ωi+1,:,νi] = norm .* reduce_kGrid(ifft_cut_mirror(ifft(Gνω[νi+sP.n_iω] .* Gνω[νi+ωi])))
        end
    end
    @inbounds res = cat(conj.(res[end:-1:1,:,end:-1:1]),res, dims=3)
    return res
end


"""
Solve χ = χ₀ - 1/β² χ₀ Γ χ
    ⇔ (1 + 1/β² χ₀ Γ) χ = χ₀
    ⇔      (χ⁻¹ - χ₀⁻¹) = 1/β² Γ
    with indices: χ[ω, q] = χ₀[]
    TODO: use 4.123 with B.6+B.7 instead of inversion
"""
function calc_χ_trilex(Γsp::Array{T,3}, Γch::Array{T,3}, bubble::Array{T,3}, qMultiplicity,
                              modelParams::ModelParameters, simParams::SimulationParameters) where T <: Number
    Nω = floor(Int64, size(bubble,1)/2)
    Nν = floor(Int64,size(bubble, 3)/2)
    νmin = Int(floor((Nν)*2/4))                                 # min freq for freq fit
    νmax = Int(floor(Nν))                                       # max freq for freq fit
    W    = simParams.tail_corrected ? build_weights(νmin, νmax, [0,1,2,3]) : nothing

    χsp   = zeros(eltype(bubble), size(bubble)[1:2]...)    # ωₙ x q (summed over νₙ)
    χsp_ω = zeros(eltype(bubble), size(bubble)[1])    # ωₙ (summed over νₙ and ωₙ)
    trilexsp = zeros(eltype(bubble), size(bubble)...)

    χch   = zeros(eltype(bubble), size(bubble)[1:2]...)    # ωₙ x q (summed over νₙ)
    χch_ω = zeros(eltype(bubble), size(bubble)[1])    # ωₙ (summed over νₙ and ωₙ)
    trilexch = zeros(eltype(bubble), size(bubble)...)

    UnitM = Matrix{eltype(Γsp)}(I, size(Γsp[1,:,:])...)
    qNorm = 8*(Int(simParams.Nk/2))^(modelParams.D)
    usable_sp = nothing
    usable_ch = nothing


    indh = ceil(Int64, size(bubble,1)/2)
    ωi_list = ((i == 0) ? indh : 
               ((i % 2 == 0) ? indh+floor(Int64,i/2) : indh-floor(Int64,i/2)) for i in 1:2*Nω+1)

    for ωi in ωi_list
        ΓspView = view(Γsp,ωi,:,:)
        ΓchView = view(Γch,ωi,:,:)
        for qi in 1:size(bubble, 2)
            bubble_i = bubble[ωi, qi, :]
            bubbleD = Diagonal(bubble_i)
            # input: vars: bubble, gamma, U, tc, beta ; functions: sum_freq ; out: χ, γ 

            @inbounds A = bubbleD * ΓspView + UnitM 
            χ_full_sp = A\bubbleD
            @inbounds χsp[ωi, qi] = sum_freq(χ_full_sp, [1,2], simParams.tail_corrected, modelParams.β, weights=W)[1,1]
            @inbounds trilexsp[ωi, qi, :] .= sum_freq(χ_full_sp, [2], simParams.tail_corrected, 1.0, weights=W)[:,1] ./ (bubble_i * (1.0 + modelParams.U * χsp[ωi, qi]))

            @inbounds A = bubbleD * ΓchView + UnitM
            χ_full_ch = A\bubbleD
            @inbounds χch[ωi, qi] = sum_freq(χ_full_ch, [1,2], simParams.tail_corrected, modelParams.β, weights=W)[1,1]
            @inbounds trilexch[ωi, qi, :] .= sum_freq(χ_full_ch, [2], simParams.tail_corrected, 1.0, weights=W)[:,1] ./  (bubble_i * (1.0 - modelParams.U * χch[ωi, qi]))
        end

        χsp_ω[ωi] = sum(χsp[ωi,:] .* qMultiplicity) / (qNorm)
        χch_ω[ωi] = sum(χch[ωi,:] .* qMultiplicity) / (qNorm)
        if (!simParams.fullChi)
            usable_sp = find_usable_interval(real(χsp_ω))
            usable_ch = find_usable_interval(real(χch_ω))
            first(usable_sp) > ωi && first(usable_ch) > ωi && break
        end
    end
    if simParams.fullChi
        usable_sp = find_usable_interval(real(χsp_ω))
        usable_ch = find_usable_interval(real(χch_ω))
    end

    return χsp, χch, χsp_ω, χch_ω, trilexsp, trilexch, usable_sp, usable_ch
end

function calc_DΓA_Σ_fft(χsp, χch, γsp, γch, bubble, Σ_loc, FUpDo, ϵkGrid, qIndices, usable_ω, mP::ModelParameters, sP::SimulationParameters)
    norm = mP.U / (mP.β * (sP.Nk^mP.D))
    @inbounds Gνω = [G_fft(ind, Σ_loc, ϵkGrid, mP.β, mP.μ) for ind in (-sP.n_iω):(sP.n_iν+sP.n_iω-1)]

    Σ_ladder = zeros(eltype(χch), sP.n_iν, length(qIndices))

    #for ωi in 1:2*sP.n_iω+1
    #    for νi in 1:sP.n_iν
    #        @inbounds res[ωi,:,νi] = norm .* reduce_kGrid(ifft_cut_mirror(ifft(Gνω[νi+sP.n_iω] .* Gνω[νi-1+ωi])))
    #for ωi in 0:2*sP.n_iω
    #    for νi in 1:sP.n_iν
    #        @inbounds res[ωi+1,:,νi] = norm .* reduce_kGrid(ifft_cut_mirror(ifft(Gνω[νi+sP.n_iω] .* Gνω[νi+ωi])))
    #    end
    println(usable_ω)
    println("TODO: qMult instead of expansion")
    for ωi in usable_ω
        for νi in 1:sP.n_iν
            Kνωq = (1.5 .* γsp[ωi, :, νi+sP.n_iν] .* (1 .+ mP.U*χsp[ωi, :]) .-
                   0.5 .* γch[ωi, :, νi+sP.n_iν].* (1 .- mP.U*χch[ωi, :]) .- 1.5 .+ 0.5) .+
                   sum(bubble[ωi,:,vpi] .* FUpDo[ωi,νi+sP.n_iν,vpi] for vpi = 1:size(bubble,3))
            Kνωq = expand_kGrid(qIndices, Kνωq)
            Kνωq = fft(Kνωq)
            Σ_ladder[νi, :] += norm .* reduce_kGrid(ifft_cut_mirror(ifft(Kνωq .* Gνω[νi + ωi])))
        end
    end
    return Σ_ladder
end


