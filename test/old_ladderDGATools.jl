using Printf 

function calc_bubble_parallel2(Σ::Array{Complex{Float64},1}, kList, qList, ϵkList, ϵkqList::Array{Float64,2}, modelParams::ModelParameters, simParams::SimulationParameters)
    kGrid  = gen_kGrid(simParams.Nint, modelParams.D; min = 0, max = 2π, include_min = false) 
    qGrid  = reduce_kGrid(gen_kGrid(simParams.Nint, modelParams.D; min = 0, max = π, include_min = true))
    ϵkList = squareLattice_ekGrid(kGrid)
    ϵkqList = gen_squareLattice_ekq_grid(kGrid, qGrid)
    kList  = collect(kGrid)
    qList  = collect(qGrid)

    res= SharedArray{Complex{Float64}}(  2*simParams.n_iω+1, simParams.n_iν, size(qList, 1))

    @sync @distributed for νₙ in 0:simParams.n_iν-1
        Σν = get_symm_f(Σ,νₙ)
        @simd for ki in 1:length(ϵkList)
            Gν = GF_from_Σ(νₙ, modelParams.β, modelParams.μ, ϵkList[ki], Σν) 
            @simd for qi in 1:size(qList, 1)
                @inbounds ϵₖ₂ = ϵkqList[ki, qi]
                #@inbounds ϵₖ₂ = 0.5*sum(cos.(kList[ki] .+ qList[qi]))
                @simd for ωₙ in (-simParams.n_iω):simParams.n_iω
                    Σων = get_symm_f(Σ,νₙ + ωₙ)
                    @inbounds res[ωₙ+simParams.n_iω+1, νₙ+1, qi]-= Gν*
                                       GF_from_Σ(νₙ + ωₙ, modelParams.β, modelParams.μ, ϵₖ₂, Σων)
                end
            end
        end
    end
    return 2*modelParams.β*sdata(res)/(simParams.n_iν^2)
end
function calc_bubble_naive_macro(Σ::Array{Complex{Float64},1}, kList, modelParams, simParams)
    kGrid  = gen_kGrid(simParams.Nint, modelParams.D; min = 0, max = 2π, include_min = false) 
    qGrid  = reduce_kGrid(gen_kGrid(simParams.Nint, modelParams.D; min = 0, max = π, include_min = true))
    ϵkList = squareLattice_ekGrid(kGrid)
    kList = collect(kGrid)
    qList = collect(qGrid)

    res= zeros(Complex{Float64},  2*simParams.n_iω+1, simParams.n_iν,size(qList, 1))
    @simd for νₙ in 0:simParams.n_iν-1
        Σν = get_symm_f(Σ,νₙ)
        @simd for ki in 1:length(ϵkList)
            Gν = GF_from_Σ(νₙ, modelParams.β, modelParams.μ, ϵkList[ki], Σν) 
            @simd for qi in 1:size(qList, 1)
                #@inbounds ϵₖ₂ = ϵkqList[ki, qi]
                @inbounds ϵₖ₂ = 0.5*sum(cos.(kList[ki] .+ qList[qi]))
                @simd for ωₙ in (-simParams.n_iω):simParams.n_iω
                    Σων = get_symm_f(Σ,νₙ + ωₙ)
                    @inbounds res[ωₙ+simParams.n_iω+1, νₙ+1, qi]-= Gν*
                                       GF_from_Σ(νₙ + ωₙ, modelParams.β, modelParams.μ, ϵₖ₂, Σων)
                end
            end
        end
    end
    return 2*modelParams.β*res/(simParams.n_iν^2)
end

function calc_bubble_naive(Σ::Array{Complex{Float64},1}, modelParams, simParams)
    Nint = simParams.Nint
    D = modelParams.D
    kGrid  = gen_kGrid(simParams.Nint, modelParams.D; min = 0, max = 2π, include_min = false) 
    qGrid  = reduce_kGrid(gen_kGrid(simParams.Nint, modelParams.D; min = 0, max = π, include_min = true))
    ϵkList = squareLattice_ekGrid(kGrid)
    ϵkqList = gen_squareLattice_ekq_grid(kGrid, qGrid)

    res= zeros(Complex{Float64},  2*simParams.n_iω+1, simParams.n_iν,length(collect(qGrid)))
    for ωₙ in (-simParams.n_iω):simParams.n_iω
    for νₙ in 0:simParams.n_iν-1
    for ki in 1:length(ϵkList)
    for qi in 1:size(qList, 1)
            ϵₖ₂ = ϵkqList[ki, qi]
                Σν = get_symm_f(Σ,νₙ)
                Gν = GF_from_Σ(νₙ, modelParams.β, modelParams.μ, ϵkList[ki], Σν) 
                    Σων = get_symm_f(Σ,νₙ + ωₙ)
                    tmp = Gν*GF_from_Σ(νₙ + ωₙ, modelParams.β, modelParams.μ, ϵₖ₂, Σων)
                    res[ωₙ+simParams.n_iω+1, νₙ+1, qi]-= 4*Gν*
                                       GF_from_Σ(νₙ + ωₙ, modelParams.β, modelParams.μ, ϵₖ₂, Σων)
                end
            end
        end
    end
    return modelParams.β*res/(2*simParams.n_iν^2)
end

#TODO: nu macro
function calc_χ_γ_v1(Γ::Array{Complex{Float64},3}, bubble::Array{Complex{Float64}, 3}, β::Float64, U::Float64, simParams, qList)
    #TODO: generalize, don't assume fixed dimensions
    Nωₙ = size(bubble, 1)
    Nνₙ = simParams.n_iν
    Nq = size(bubble, 3)
    Nνₙ_bubble = size(bubble, 2) 

    γres = zeros(Complex{Float64}, (Nωₙ, 2*Nνₙ, Nq))
    χ = zeros(Complex{Float64}, (Nωₙ, Nq))    # ωₙ x q (summed over νₙ)

    for (qi,q) in enumerate(qList)
        for ωₙ in 1:Nωₙ
            tmp = Γ[ωₙ, :, :]
            for νₙ in (-Nνₙ):(Nνₙ-1)
                tmp[νₙ+Nνₙ+1, νₙ+Nνₙ+1] = tmp[νₙ+Nνₙ+1, νₙ+Nνₙ+1,] + 1.0 / get_symm_f(bubble,ωₙ-simParams.n_iω-1, νₙ, qi) 
            end
            tmp = inv(tmp)
            χ[ωₙ, qi]   = sum(tmp)/(β^2)
            tmpSum = sum(tmp, dims=2) 
            for νₙ in (-Nνₙ):(Nνₙ-1)
                γres[ωₙ, νₙ+Nνₙ+1, qi] = tmpSum[νₙ+Nνₙ+1]./ (get_symm_f(bubble,ωₙ-simParams.n_iω-1, νₙ,  qi) .* (1 - U*χ[ωₙ, qi]))
            end
        end
    end
    return χ, γres
end

function calc_DΓA_Σ_naive(χch, χsp, trilexch, trilexsp, bubble, Σ_loc, FUpDo, qMult, qGrid, kGrid,
                              modelParams::ModelParameters, simParams::SimulationParameters)
    Nν = simParams.n_iν
    kList = collect(kGrid)
    ϵkqList = gen_squareLattice_full_ekq_grid(kList, collect(qGrid))

    Σ_ladder = zeros(Complex{Float64}, Nν, length(kList))
    for (qi,q) in enumerate(qGrid)
        #TODO: factor??
        qiNorm = qMult[qi]/((simParams.Nq-1)^2*8*4)#/(4.0*8.0)
        for (νi,νₙ) in enumerate(0:(Nν-1))
            for (ωi,ωₙ) in enumerate((-simParams.n_iω):simParams.n_iω)
                Σν = get_symm_f(Σ_loc,ωₙ + νₙ)
                #TODO: access trilx nd chi by freq, instead of index
                tmp = (1.5 * trilexsp[ωi, νi+Nν, qi]*(1 + modelParams.U*χsp[ωi, qi])-
                                                     0.5 * trilexch[ωi, νi+Nν, qi]*(1 - modelParams.U*χch[ωi, qi])-1.5+0.5)
                for (νpi, νpₙ) in enumerate((-Nν):(Nν-1))
                    tmp += get_symm_f(bubble,ωₙ, νpₙ, qi)*FUpDo[ωi, νi+Nν, νpi]
                end
                for (ki,k) in enumerate(kGrid)
                    for perm in 1:size(ϵkqList,3)
                        Gν = GF_from_Σ(ωₙ + νₙ, modelParams.β, modelParams.μ, ϵkqList[ki,qi,perm], Σν) 
                        Σ_ladder[νi, ki] += tmp*Gν*qiNorm*modelParams.U/modelParams.β
                    end
                end
            end
        end
    end
    return Σ_ladder
end


function calc_χ_trilex_parallel(Γ::Array{Complex{Float64},3}, bubble::Array{Complex{Float64},3},
                              U::Float64, β::Float64, simParams::SimulationParameters)
    Nω = simParams.n_iω
    Nν = simParams.n_iν
    Nq = size(bubble, 3)
    γres = SharedArray{Complex{Float64}}( 2*Nω+1, 2*Nν, Nq)
    χ = SharedArray{Complex{Float64}}( 2*Nω+1, Nq)    # ωₙ x q (summed over νₙ)

    @sync @distributed for qi in 1:Nq
        χ_ladder = zeros(Complex{Float64}, 2*Nν, 2*Nν)
        for (ωi,ωₙ) in enumerate((-Nω):Nω)
            @inbounds χ_ladder = Γ[ωi, :, :]
            for (νi,νₙ) in enumerate((-Nν):(Nν-1))
                @inbounds χ_ladder[νi, νi] = χ_ladder[νi, νi] + 1.0 / get_symm_χ(bubble,ωₙ, νₙ, qi) 
            end
            @inbounds χ_ladder = inv(χ_ladder)
            @inbounds χ[ωi, qi] = sum(χ_ladder)/(β^2)
            @inbounds tmpSum = sum(χ_ladder, dims=2) 
            for (νi,νₙ) in enumerate((-Nν):(Nν-1))
                @inbounds γres[ωi, νi, qi] = tmpSum[νi] / (get_symm_χ(bubble, ωₙ, νₙ,  qi) * (1 - U*χ[ωi, qi]))
            end
        end
    end
    return sdata(χ), sdata(γres)
end
