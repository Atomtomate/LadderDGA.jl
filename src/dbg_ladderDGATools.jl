@inline GF_from_Σ_ap(n::Int64, β::Float64, μ::Float64, ϵₖ::T, Σ::Complex{T}) where T =
                    1/((π/β)*(2*n + 1)*1im + μ - ϵₖ - Σ)

function calc_bubble(Σ::Array{Complex{Float64},1}, qGrid, 
                              modelParams::ModelParameters, simParams::SimulationParameters)
    #TODO: this is slower, than having this as a parameter. Check again after conversion to module
    #
    Nq   = size(collect(qGrid),1)
    _, kIntGrid = gen_kGrid(simParams.Nint, modelParams.D; min = 0, max = 2π, include_min = false) 
    tsc =  modelParams.D == 3 ? 0.40824829046386301636 : 0.5
    ϵkIntList   = squareLattice_ekGrid(kIntGrid, tsc)
    ϵkqIntList  = gen_squareLattice_ekq_grid(kIntGrid, qGrid, tsc)
    res = zeros(Complex{Float64}, Nq, simParams.n_iν, 2*simParams.n_iω+1)
    @simd for qi in 1:Nq
        @simd for νₙ in 0:simParams.n_iν-1
            Σν = get_symm_f(Σ,νₙ)
            for (ωi, ωₙ) in enumerate((-simParams.n_iω):simParams.n_iω)
                Σων = get_symm_f(Σ,νₙ + ωₙ)
                #res[qi,  νₙ+1, ωi] = 0.0 #[ωi, νₙ+1, qi] = 0.0
                @simd for ki in 1:length(ϵkIntList)
                    @inbounds ϵₖ₂ = ϵkqIntList[ki, qi]
                    Gν = GF_from_Σ(νₙ, modelParams.β, modelParams.μ, ϵkIntList[ki], Σν) 
                    Gνω = GF_from_Σ(νₙ + ωₙ, modelParams.β, modelParams.μ, ϵₖ₂, Σων)
                    @inbounds res[qi,  νₙ+1, ωi] -= Gν*Gνω
                end
            end
        end
    end
    #res = sdata(res)
    res = (2^modelParams.D) * modelParams.β .* res ./ ((2*simParams.Nint)^modelParams.D)
    res = permutedims(res, [3,2,1])
    res = cat(conj.(res[end:-1:1,end:-1:1,:]),res, dims=2)
    #res = convert(Array{Complex{Float64}}, res)
    return res
end


function calc_bubble_ap(Σ::Array{Complex{Float64},1}, qGrid, 
                              modelParams::ModelParameters, simParams::SimulationParameters)
    #TODO: this is slower, than having this as a parameter. Check again after conversion to module
    Nq   = size(collect(qGrid),1)
    _, kIntGrid = gen_kGrid(simParams.Nint, modelParams.D; min = 0, max = 2π, include_min = false) 
    ϵkIntList   = squareLattice_ekGrid(kIntGrid)
    ϵkqIntList  = gen_squareLattice_ekq_grid(kIntGrid, qGrid)

    bubble = Array{Complex{Float64}}(undef,  2*simParams.n_iω+1, simParams.n_iν, Nq)
    setprecision(10000) do
        res = Array{Complex{BigFloat}}(undef, Nq, simParams.n_iν, 2*simParams.n_iω+1)
        for qi in 1:Nq
            for νₙ in 0:simParams.n_iν-1
                Σν = Complex{BigFloat}(get_symm_f(Σ,νₙ))
                for (ωi, ωₙ) in enumerate((-simParams.n_iω):simParams.n_iω)
                    Σων = Complex{BigFloat}(get_symm_f(Σ,νₙ + ωₙ))
                    res[qi,  νₙ+1, ωi] = Complex{BigFloat}(0.0)  # Initialize to 0
                    for ki in 1:length(ϵkIntList)
                        ϵₖ = BigFloat(ϵkIntList[ki])
                        ϵₖ₂ = BigFloat(ϵkqIntList[ki, qi])
                        Gν = Complex{BigFloat}(GF_from_Σ_ap(νₙ, modelParams.β, modelParams.μ, ϵₖ, Σν))
                        Gνω = Complex{BigFloat}(GF_from_Σ_ap(νₙ + ωₙ, modelParams.β, modelParams.μ, ϵₖ₂, Σων))

                        res[qi,  νₙ+1, ωi] -= Complex{BigFloat}(4*Gν*Gνω)
                    end
                end
            end
        end
        res = modelParams.β .* res ./ ((2*simParams.Nint)^2)    # Normalization
        res = permutedims(res, [3,2,1])                         # Reorder ranks to [Bose, Fermi, QPoiubts]
        res = cat(conj.(res[end:-1:1,end:-1:1,:]),res, dims=2)  # Build full bubble term
        bubble = convert(Array{Complex{Float64}}, res)             # Convert from arbitrary precision to Float64
    end
   
    return bubble
end



function calc_DΓA_Σ_impr(χch::Array{Complex{Float64}, 2}, χsp::Array{Complex{Float64}, 2},
                             γch::Array{Complex{Float64}, 3}, γsp::Array{Complex{Float64}, 3},
                             bubble, Σ_loc, FUpDo, qMult, qGrid, kGrid,
                             modelParams::ModelParameters, simParams::SimulationParameters)
    qList = collect(qGrid)
    Nω = simParams.n_iω
    Nν = simParams.n_iν
    Nq = simParams.Nq
    kList = collect(kGrid)
    if modelParams.D == 3
        tsc = 0.40824829046386301636
    elseif modelParams.D == 2
        tsc = 0.5
    end
    ϵkqList = gen_squareLattice_full_ekq_grid(kList, qList, tsc)
    Σ_ladder_tmp = zeros(Complex{Float64}, length(kList), Nν, size(bubble, 1))
    tmp_νp = Array{Complex{Float64}}(undef, size(bubble, 2))

    cut = 0
    if simParams.tail_corrected
        ωmin = Int(floor((Nω-cut)*3/4))
        ωmax = Int(floor(Nω)-cut)
        W1 = build_weights(ωmin, ωmax, [0,1,2,3])
        numin = Int(floor((Nν-cut)*3/4))
        numax = Int(floor(Nν)-cut)
        W2 = build_weights(numin, numax, [0,1,2,3])
    end


    for qi in 1:length(qList)
        qiNorm = qMult[qi]/((2*(Nq-1))^2*8)#/(4.0*8.0)
        for νi in 1:Nν
            νₙ = νi - 1
            for (ωi,ωₙ) in enumerate((-Nω):Nω)
                Σν = get_symm_f(Σ_loc,ωₙ + νₙ)
                for (νpi, νpₙ) in enumerate((-Nν):(Nν-1))
                    tmp_νp[νpi] = bubble[ωi, νpi, qi]*FUpDo[ωi, νi+Nν, νpi]
                end
                tmp_ω_1 = approx_full_sum(tmp_νp[cut:(end-cut)], W2, [1], fast=true)
                tmp_ω_2 = sum(tmp_νp[cut:(end-cut)]) 
                #= println("----") =#
                #= println(tmp_ω_1) =#
                #= println(tmp_ω_2) =#
                #= println("====") =#

                tmp_ω = tmp_ω_1
                tmp_ω +=  (1.5 * γsp[ωi, νi+Nν, qi]*(1 + modelParams.U*χsp[ωi, qi]) -
                               0.5 * γch[ωi, νi+Nν, qi]*(1 - modelParams.U*χch[ωi, qi])-1.5+0.5)
                for ki in 1:length(kList)
                    for perm in 1:size(ϵkqList,3)
                        Gν = GF_from_Σ(ωₙ + νₙ, modelParams.β, modelParams.μ, ϵkqList[ki,qi,perm], Σν) 
                        Σ_ladder_tmp[ki, νi, ωi] += tmp_ω*Gν*qiNorm*modelParams.U/modelParams.β
                    end
                end
            end
        end
    end
    Σ_ladder_tmp = Σ_ladder_tmp[:,:,cut:(end-cut)]
    Σ_ladder_1 = sum((Σ_ladder_tmp), dims=[3])[:,:,1]
    Σ_ladder_2 = [approx_full_sum(Σ_ladder_tmp[i,j,:], W1, [1], fast=true) for j in 1:size(Σ_ladder_tmp,2) for i in 1:size(Σ_ladder_tmp,1)];
    Σ_ladder_2 = reshape(Σ_ladder_2, size(Σ_ladder_1))
    #println(Σ_ladder_1)
    #println(Σ_ladder_2)
    return Σ_ladder_1, Σ_ladder_2, tmp_νp[cut:(end-cut)]
end


function calc_DΓA_Σ_noise(χch::Array{Complex{Float64}, 2}, χsp::Array{Complex{Float64}, 2},
                             γch::Array{Complex{Float64}, 3}, γsp::Array{Complex{Float64}, 3},
                             bubble, Σ_loc, FUpDo, qMult, qGrid, kGrid,
                             modelParams::ModelParameters, simParams::SimulationParameters,
                            noiseFlagArr)
    qList = collect(qGrid)
    Nν = simParams.n_iν
    kList = collect(kGrid)
    if modelParams.D == 3
        tsc = 0.40824829046386301636
    elseif modelParams.D == 2
        tsc = 0.5
    end
    ϵkqList = gen_squareLattice_full_ekq_grid(kList, qList, tsc)
    noiseVarArr = zeros(Complex{Float64}, 7)
    noiseVarArr[1] = mean(γch)/2
    noiseVarArr[2] = mean(γsp)/2
    noiseVarArr[3] = mean(χch)/2
    noiseVarArr[4] = mean(χsp)/2
    noiseVarArr[5] = mean(bubble)/2
    noiseVarArr[6] = mean(FUpDo)/2
    Σν = get_symm_f(Σ_loc,0)
    noiseVarArr[7] = GF_from_Σ(0, modelParams.β, modelParams.μ, ϵkqList[1,1,1], Σν)/4

    rng = MersenneTwister(1234);


    Σ_ladder = SharedArray{Complex{Float64}}(length(kList), Nν)
    #TODO: get rid of simParams
    #Σ_ladder = zeros(Complex{Float64}, Nν, length(kList))
    @sync @distributed for νi in 1:Nν
        νₙ = νi - 1
        for qi in 1:length(qList)
            @inbounds qiNorm = qMult[qi]/((2*(simParams.Nq-1))^2*8)#/(4.0*8.0)
            # approx omega sum: introduce Sigma_ladder tmp, that depends on omega, then use approx_sum
            for (ωi,ωₙ) in enumerate((-simParams.n_iω):simParams.n_iω)
                Σν = get_symm_f(Σ_loc,ωₙ + νₙ)
                #TODO: access trilx nd chi by freq, instead of index
                @inbounds tmp = (1.5 * (1+noiseFlagArr[2]*noiseVarArr[2]*randn(rng, ComplexF64))*γsp[ωi, νi+Nν, qi]*
                                  (1 + (1+noiseFlagArr[4]*noiseVarArr[4]*randn(rng, ComplexF64))*modelParams.U*χsp[ωi, qi])-
                                  0.5 * (1+noiseFlagArr[1]*noiseVarArr[1]*randn(rng, ComplexF64))*γch[ωi, νi+Nν, qi]*
                                  (1 - (1+noiseFlagArr[3]*noiseVarArr[3]*randn(rng, ComplexF64))*modelParams.U*χch[ωi, qi])-
                                  1.5+0.5)
                #approx nu sum: extend tmp by one dimension, then use approx_sum
                for (νpi, νpₙ) in enumerate((-Nν):(Nν-1))
                    @inbounds tmp += (1+noiseFlagArr[5]*noiseVarArr[5]*randn(rng, ComplexF64))*bubble[ωi, νpi, qi] * 
                                     (1+noiseFlagArr[6]*noiseVarArr[6]*randn(rng, ComplexF64))*FUpDo[ωi, νi+Nν, νpi]
                end
                for ki in 1:length(kList)
                    for perm in 1:size(ϵkqList,3)
                        @inbounds Gν = GF_from_Σ(ωₙ + νₙ, modelParams.β, modelParams.μ, ϵkqList[ki,qi,perm], Σν) 
                        @inbounds Σ_ladder[ki, νi] += (1+noiseFlagArr[7]*noiseVarArr[7]*randn(rng, ComplexF64))*tmp*Gν*qiNorm*modelParams.U/modelParams.β
                    end
                end
            end
        end
    end
    return sdata(Σ_ladder)
end
