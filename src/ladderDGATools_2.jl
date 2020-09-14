#TODO: kList must be template parameter for dimensions
#TODO: nw and niv grid as parameters? 
#TODO: define GF type that knows about which dimension stores which variable

@everywhere @inline @fastmath GF_from_Σ(n::Int64, β::Float64, μ::Float64, ϵₖ::T, Σ::Complex{T}) where T =
                    1/((π/β)*(2*n + 1)*1im + μ - ϵₖ - Σ)
@everywhere @inline @fastmath w_from_Σ(n::Int64, β::Float64, μ::Float64, Σ::Complex{Float64}) =
                    ((π/β)*(2*n + 1)*1im + μ - Σ)

@inline @inbounds function G_fft(ind::Int64, Σ::Array{Complex{Float64},1}, 
                                  ϵkIntGrid::Array{Float64}, β::Float64, μ::Float64, 
                                  phasematrix::Array{Complex{Float64}})
    Σν = get_symm_f(Σ,ind)
    Gν  = map(ϵk -> GF_from_Σ(ind, β, μ, ϵk, Σν), ϵkIntGrid)
    return fft!(Gν) .* phasematrix
end


function calc_bubble_fft(Gνω::Array{Complex{Float64},1}, Nk::Int64, mP::ModelParameters, sP::SimulationParameters)
    res = SharedArray{Complex{Float64},3}((2*sP.n_iω+1, Nk, 2*sP.n_iν))
    gridShape = (Nk == 1) ? [1] : repeat([sP.Nk], mP.D)
    norm = (Nk == 1) ? -mP.β : -mP.β /(sP.Nk^(mP.D));
    transform = Nk == 1 ? identity : reduce_kGrid ∘ ifft_cut_mirror ∘ ifft
    @sync @distributed for ωi in 1:(2*sP.n_iω+1)
        for νi in 1:sP.n_iν
            v1 = view(Gνω, νi+sP.n_iω, :)
            v2 = view(Gνω, νi+ωi-1, :)
            @inbounds res[ωi,:,νi+sP.n_iν] .= norm .* transform(reshape(v1 .* v2, gridShape...))
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
    TODO: use 4.123 with B.6+B.7 instead of inversion
"""
function calc_χ_trilex_old(Γsp::Array{T,3}, Γch::Array{T,3}, bubble::Array{T,3},
                              modelParams::ModelParameters, simParams::SimulationParameters) where T <: Number
    Nω = floor(Int64,size(bubble, 1)/2)
    Nq = size(bubble, 2)
    Nν = floor(Int64,size(bubble, 3)/2)
    χsp = SharedArray{eltype(bubble)}(2*Nω+1, Nq)    # ωₙ x q (summed over νₙ)
    γsp = SharedArray{eltype(bubble)}(2*Nω+1, Nq, 2*Nν)
    χch = SharedArray{eltype(bubble)}(2*Nω+1, Nq)    # ωₙ x q (summed over νₙ)
    γch = SharedArray{eltype(bubble)}(2*Nω+1, Nq, 2*Nν)

    W = nothing
    if simParams.tail_corrected
        νmin = Int(floor((Nν)*2/4))
        νmax = Int(floor(Nν))
        W = build_weights(νmin, νmax, [0,1,2,3])
    end
    UnitM = Matrix{eltype(Γsp)}(I, size(Γsp[1,:,:])...)
    for (ωi,ωₙ) in enumerate((-Nω):Nω)
        ΓspView = view(Γsp,ωi,:,:)
        ΓchView = view(Γch,ωi,:,:)
        for qi in 1:Nq
            bubbleD = Diagonal(bubble[ωi, qi, :])
            # input: vars: bubble, gamma, U, tc, beta ; functions: sum_freq ; out: χ, γ 

            @inbounds A_sp = bubbleD * ΓspView + UnitM 
            χ_full_sp = A_sp\bubbleD
            @inbounds χsp[ωi, qi] = sum_freq(χ_full_sp, [1,2], simParams.tail_corrected, modelParams.β, weights=W)[1,1]
            @inbounds γsp[ωi, qi, :] .= modelParams.β .* sum_freq(χ_full_sp, [2], simParams.tail_corrected, modelParams.β, weights=W)[:,1] ./ (1.0 + modelParams.U * χsp[ωi, qi])

            @inbounds A_ch = bubbleD * ΓchView + UnitM
            χ_full_ch = A_ch\bubbleD
            @inbounds χch[ωi, qi] = sum_freq(χ_full_ch, [1,2], simParams.tail_corrected, modelParams.β, weights=W)[1,1]
            @inbounds γch[ωi, qi, :] .= modelParams.β .* sum_freq(χ_full_ch, [2], simParams.tail_corrected, modelParams.β, weights=W)[:,1] ./  (1.0 - modelParams.U * χch[ωi, qi])
        end
    end
    return χsp, χch, γsp, γch
end

function calc_DΓA_Σ_fft_old(χsp, χch, γsp, γch, bubble, Σ_loc, FUpDo, qIndices, modelParams::ModelParameters, simParams::SimulationParameters; full_input = false)
    Nω = floor(Int64,size(bubble,1)/2)
    Nν = floor(Int64,size(bubble,3)/2)
    tsc =  modelParams.D == 3 ? 0.40824829046386301636 : 0.5
    kIndices, kGrid         = gen_kGrid(simParams.Nk, modelParams.D; min = 0, max = 2π)
    ϵkGrid   = squareLattice_ekGrid(kGrid, tsc)
    Σ_ladder = zeros(eltype(χch), Nν, length(collect(kGrid)))

    for (νi,νₙ) in enumerate(0:Nν-1)
        for (ωi,ωₙ) in enumerate(-Nω:Nω)
            Σνω  = get_symm_f(Σ_loc,ωₙ + νₙ)
            Kνωq = (1.5 .* γsp[ωi, :, νi+Nν] .* (1 .+ modelParams.U*χsp[ωi, :]) .-
                   0.5 .* γch[ωi, :, νi+Nν].* (1 .- modelParams.U*χch[ωi, :]) .- 1.5 .+ 0.5) .+
                   sum([bubble[ωi,:,vpi] .* FUpDo[ωi,νi+Nν,vpi] for vpi = 1:size(bubble,3)])

            Gνω = G_fft(νₙ + ωₙ, Σ_loc, ϵkGrid, modelParams.β, modelParams.μ, phasematrix)
            if !full_input
                Kνωq = expand_mirror(expand_kGrid(collect(qIndices), Kνωq))#) 
            else
                Kνωq = reshape(Kνωq, (4,4))
            end
            println(size(Kνωq))
            Kνωq = fft(Kνωq).* phasematrix2
            Σ_ladder[νi, :] -= (bw_plan * (Kνωq  .* Gνω))[1:end]
        end
    end
    Σ_ladder = modelParams.U .* Σ_ladder ./ (modelParams.β * (simParams.Nk^modelParams.D))
    return Σ_ladder
end


function calc_DΓA_Σ_old(χsp, χch, γsp, γch, 
                             bubble, Σ_loc, FUpDo, qMult, qGrid, 
                             modelParams::ModelParameters, simParams::SimulationParameters)
    _, kGrid         = reduce_kGrid.(gen_kGrid(simParams.Nk, modelParams.D; min = 0, max = π, include_min = true))
    kList = collect(kGrid)
    Nω = floor(Int64,size(bubble,1)/2)
    Nq = size(bubble,2)
    Nν = floor(Int64,size(bubble,3)/2)
    ϵkqList = gen_squareLattice_full_ekq_grid(kList, collect(qGrid))
    multFac =  if (modelParams.D == 2) 8.0 else 48.0 end # 6 for permutation 8 for mirror

    Σ_ladder = zeros(eltype(χch), Nν, length(kList))
    for (νi,νₙ) in enumerate(0:Nν-1)
        for qi in 1:Nq
            qiNorm = qMult[qi]/((2*(Int(simParams.Nk/2)))^(modelParams.D)*multFac)#8*(Nq-1)^(modelParams.D)
            #qiNorm = qMult[qi]/((2*(Nq-1))^2*8)#/(4.0*8.0)
            for (ωi,ωₙ) in enumerate((-simParams.n_iω):simParams.n_iω)
                Σν = get_symm_f(Σ_loc,ωₙ + νₙ)
                tmp = (1.5 * γsp[ωi, qi, νi+Nν]*(1 + modelParams.U*χsp[ωi, qi]) -
                       0.5 * γch[ωi, qi, νi+Nν]*(1 - modelParams.U*χch[ωi, qi])-1.5+0.5) +
                       sum(bubble[ωi, qi, :] .* FUpDo[ωi, νi+Nν, :])
                for ki in 1:length(kList)
                    for perm in 1:size(ϵkqList,3)
                        Gνω = GF_from_Σ(ωₙ + νₙ, modelParams.β, modelParams.μ, ϵkqList[ki,qi,perm], Σν) 
                        Σ_ladder[νi, ki] -= tmp*Gνω*qiNorm*modelParams.U/modelParams.β
                    end
                end
            end
        end
    end
    return sdata(Σ_ladder)
end
