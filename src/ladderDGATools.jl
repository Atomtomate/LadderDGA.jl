#TODO: kList must be template parameter for dimensions
#TODO: nw and niv grid as parameters? 
#TODO: define GF type that knows about which dimension stores which variable

@everywhere @inline @fastmath GF_from_Σ(n::Int64, β::Float64, μ::Float64, ϵₖ::T, Σ::Complex{T}) where T =
                    1/((π/β)*(2*n + 1)*1im + μ - ϵₖ - Σ)
@everywhere @inline @fastmath w_from_Σ(n::Int64, β::Float64, μ::Float64, Σ::Complex{Float64}) =
                    ((π/β)*(2*n + 1)*1im + μ - Σ)

function calc_bubble_fft_internal(ind::Int64, Σ::Array{Complex{Float64},1}, 
                                  ϵkIntGrid::Array{Float64}, β::Float64, μ::Float64, 
                                  phasematrix::Array{Complex{Float64}})
    Σν = get_symm_f(Σ,ind)
    Gν  = map(ϵk -> GF_from_Σ(ind, β, μ, ϵk, Σν), ϵkIntGrid)
    return fft!(Gν) .* phasematrix
end

#TODO: get rid of kInt, remember symmetry G(k')G(k'+q) so 2*LQ-2 = kInt
function calc_bubble_fft(Σ::Array{Complex{Float64},1}, 
                              modelParams::ModelParameters, simParams::SimulationParameters)
    tsc =  modelParams.D == 3 ? 0.40824829046386301636 : 0.5
    kIndices, kIntGrid = gen_kGrid(simParams.Nint, modelParams.D; min = 0, max = 2π, include_min = false) 
    ϵkIntGrid   = squareLattice_ekGrid(kIntGrid, tsc)
    _, qGrid = gen_kGrid(simParams.Nq, modelParams.D; min = 0, max = 2π, include_min = false) 
    qGrid = reduce_kGrid(qGrid)
    phasematrix = map(x -> exp(-2π*1im*(sum(x)-modelParams.D)/simParams.Nint)*(-1)^(sum(x)-modelParams.D), kIndices)

    res = Array{Complex{Float64}}(undef, 2*simParams.n_iω+1, simParams.n_iν, length(qGrid))
    bw_plan = plan_ifft(ϵkIntGrid,[1,2,3])
    for νₙ in 0:simParams.n_iν-1
        Gν = calc_bubble_fft_internal(νₙ, Σ, ϵkIntGrid, modelParams.β, modelParams.μ, phasematrix)
        for (ωi, ωₙ) in enumerate((-simParams.n_iω):simParams.n_iω)
            Gνω = calc_bubble_fft_internal(νₙ + ωₙ, Σ, ϵkIntGrid, modelParams.β, modelParams.μ, phasematrix)
            tmp = reshape(bw_plan * (Gν  .* Gνω), (repeat([simParams.Nint], modelParams.D)...))
            res[ωi, νₙ+1, :] = reduce_kGrid(tmp)[1:length(qGrid)]
        end
    end
    res = -modelParams.β .* res ./ (simParams.Nint^modelParams.D)
    res = cat(conj.(res[end:-1:1,end:-1:1,:]),res, dims=2)
    return res
end

function calc_χ_2(Γ::Array{T,3}, bubble::Array{T,3},
                              modelParams::ModelParameters, simParams::SimulationParameters; Usign= 1) where T
    Nω = floor(Int64,size(bubble, 1)/2)
    Nν = floor(Int64,size(bubble, 2)/2)
    Nq = size(bubble, 3)
    χ = SharedArray{eltype(bubble)}(2*Nω+1, Nq)    # ωₙ x q (summed over νₙ)
    β = (modelParams.β)
    for (ωi,ωₙ) in enumerate((-Nω):Nω)
        Γ[ωi,:,:]
        for qi in 1:Nq
            #χ = χ₀ - 1/β² χ₀ Γ χ
        end
    end
end


"""
Solve χ = χ₀ - 1/β² χ₀ Γ χ
    ⇔ (1 + 1/β² χ₀ Γ) χ = χ₀
    ⇔      (χ⁻¹ - χ₀⁻¹) = 1/β² Γ
    with indices: χ[ω, q] = χ₀[]
    TODO: use 4.123 with B.6+B.7 instead of inversion
"""
function calc_χ_trilex(Γ::Array{T,3}, bubble::Array{T,3},
                              modelParams::ModelParameters, simParams::SimulationParameters; Usign= 1) where T
    Nω = floor(Int64,size(bubble, 1)/2)
    Nν = floor(Int64,size(bubble, 2)/2)
    Nq = size(bubble, 3)
    γres = Array{eltype(bubble)}(undef, 2*Nω+1, 2*Nν, Nq)
    χ = Array{eltype(bubble)}(undef, 2*Nω+1, Nq)    # ωₙ x q (summed over νₙ)

    W = nothing
    if simParams.tail_corrected
        νmin = Int(floor((Nν)*2/4))
        νmax = Int(floor(Nν))
        if νmax - νmin < 4
            throw("not enough frequencies to use fit. Trid to fit from "* string(νmin)* " to "* string(νmax))
        end
        W = build_weights(νmin, νmax, [0,1,2,3])
    end
    for qi in 1:Nq
        for (ωi,ωₙ) in enumerate((-Nω):Nω)
            χ_tmp = copy(Γ[ωi, :, :])
            tmpSum = Array{eltype(Γ)}(undef, size(Γ,2))
            for νi in 1:size(Γ,2)
                @inbounds χ_tmp[νi, νi] += 1.0 / bubble[ωi, νi, qi] 
            end
            χ_tmp = inv(χ_tmp)

            #TODO: HUGE bottleneck here. this needs to be optimized
            @inbounds χ[ωi, qi] = sum_freq(χ_tmp, [1,2], simParams, modelParams, weights=W)[1,1]
            @inbounds tmpSum = sum_freq(χ_tmp, [2], simParams, modelParams, weights=W)[:,1]
            for νi in 1:size(tmpSum,1) #enumerate((-Nν):(Nν-1))
                @inbounds γres[ωi, νi, qi] = tmpSum[νi] ./ 
                       (bubble[ωi, νi, qi]  * (1.0 - Usign*modelParams.U * χ[ωi, qi]))
            end
        end
    end
    #sdata
    χ = sdata(χ) 
    γres = sdata(γres)
    return χ, γres
end

function calc_DΓA_Σ(χch, χsp,
                             γch, γsp,
                             bubble, Σ_loc, FUpDo, qMult, qGrid, 
                             modelParams::ModelParameters, simParams::SimulationParameters)
    _, kGrid         = reduce_kGrid.(gen_kGrid(simParams.Nk, modelParams.D; min = 0, max = π, include_min = true))
    qList = collect(qGrid)
    Nν = simParams.n_iν
    kList = collect(kGrid)
    if modelParams.D == 3
        tsc = 0.40824829046386301636
    elseif modelParams.D == 2
        tsc = 0.5
    end
    ϵkqList = gen_squareLattice_full_ekq_grid(kList, qList, tsc)

    Σ_ladder = SharedArray{eltype(χch)}(length(kList), Nν)
    #TODO: qifactor??
    #Σ_ladder = zeros(Complex{Float64}, Nν, length(kList))
    for νi in 1:Nν
        νₙ = νi - 1
        for qi in 1:length(qList)
            @inbounds qiNorm = qMult[qi]/((2*(simParams.Nq-1))^2*8)#/(4.0*8.0)
            for (ωi,ωₙ) in enumerate((-simParams.n_iω):simParams.n_iω)
                Σν = get_symm_f(Σ_loc,ωₙ + νₙ)
                #TODO: access trilx nd chi by freq, instead of index
                @inbounds tmp = (1.5 * γsp[ωi, νi+Nν, qi]*(1 + modelParams.U*χsp[ωi, qi])-
                                                     0.5 * γch[ωi, νi+Nν, qi]*(1 - modelParams.U*χch[ωi, qi])-1.5+0.5)
                for (νpi, νpₙ) in enumerate((-Nν):(Nν-1))
                    @inbounds tmp += bubble[ωi, νpi, qi]*FUpDo[ωi, νi+Nν, νpi]
                end
                for ki in 1:length(kList)
                    for perm in 1:size(ϵkqList,3)
                        @inbounds Gν = GF_from_Σ(ωₙ + νₙ, modelParams.β, modelParams.μ, ϵkqList[ki,qi,perm], Σν) 
                        @inbounds Σ_ladder[ki, νi] += tmp*Gν*qiNorm*modelParams.U/modelParams.β
                    end
                end
            end
        end
    end
    return sdata(Σ_ladder)
end
