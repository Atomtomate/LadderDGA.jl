#TODO: kList must be template parameter for dimensions
#TODO: nw and niv grid as parameters? 
#TODO: define GF type that knows about which dimension stores which variable

@inline @fastmath GF_from_Σ(n::Int64, β::Float64, μ::Float64, ϵₖ::Float64, Σ::Complex{Float64}) =
                    1/((π/β)*(2*n + 1)*1im + μ - ϵₖ - Σ)
@inline @fastmath w_from_Σ(n::Int64, β::Float64, μ::Float64, Σ::Complex{Float64}) =
                    ((π/β)*(2*n + 1)*1im + μ - Σ)

function calc_bubble_parallel(Σ::Array{Complex{Float64},1}, qGrid, 
                              modelParams::ModelParameters, simParams::SimulationParameters)
    #TODO: this is slower, than having this as a parameter. Check again after conversion to module
    Nq   = size(collect(qGrid),1)
    _, kIntGrid = gen_kGrid(simParams.Nint, modelParams.D; min = 0, max = 2π, include_min = false) 
    ϵkIntList   = squareLattice_ekGrid(kIntGrid)
    ϵkqIntList  = gen_squareLattice_ekq_grid(kIntGrid, qGrid)

    res = SharedArray{Complex{Float64}}(  2*simParams.n_iω+1, simParams.n_iν, Nq)
    @sync @distributed for qi in 1:Nq
        @simd for ki in 1:length(ϵkIntList)
            @inbounds ϵₖ₂ = ϵkqIntList[ki, qi]
            @simd for νₙ in 0:simParams.n_iν-1
                Σν = get_symm_f(Σ,νₙ)
                Gν = GF_from_Σ(νₙ, modelParams.β, modelParams.μ, ϵkIntList[ki], Σν) 
                @simd for ωₙ in (-simParams.n_iω):simParams.n_iω
                    Σων = get_symm_f(Σ,νₙ + ωₙ)
                    @inbounds res[ωₙ+simParams.n_iω+1, νₙ+1, qi] -= 4*Gν*
                                       GF_from_Σ(νₙ + ωₙ, modelParams.β, modelParams.μ, ϵₖ₂, Σων)
                end
            end
        end
    end
    return modelParams.β*sdata(res)/((2*simParams.Nint)^2)
end


"""
    Solve χ = χ₀ - 1/β² χ₀ Γ χ
    with indices: χ[ω, q] = χ₀[]
"""
function calc_χ_trilex(Γ::Array{Complex{Float64},3}, bubble::Array{Complex{Float64},3},
                              U::Float64, β::Float64, simParams::SimulationParameters)
    Nω = simParams.n_iω
    Nν = simParams.n_iν
    Nq = size(bubble, 3)
    γres = SharedArray{Complex{Float64}}( 2*Nω+1, 2*Nν, Nq)
    χ = SharedArray{Complex{Float64}}( 2*Nω+1, Nq)    # ωₙ x q (summed over νₙ)

    if simParams.tail_corrected
        W = build_weights(Int(floor(Nν/2))-20, Int(floor(Nν/2)), 7)
    end

    @sync @distributed for qi in 1:Nq
        χ_ladder = zeros(Complex{Float64}, 2*Nν, 2*Nν)
        for (ωi,ωₙ) in enumerate((-Nω):Nω)
            @inbounds χ_ladder = Γ[ωi, :, :]
            for (νi,νₙ) in enumerate((-Nν):(Nν-1))
                @inbounds χ_ladder[νi, νi] = χ_ladder[νi, νi] + 1.0 / get_symm_χ(bubble,ωₙ, νₙ, qi) 
            end
            @inbounds χ_ladder = inv(χ_ladder)

            @inbounds χ[ωi, qi] = if simParams.tail_corrected
                    approx_full_sum(χ_ladder, Int(floor(Nν*3/4)), modelParams, simParams, [1,2], W=W)[1]
                else sum(χ_ladder)/(β^2) end
            @time @inbounds tmpSum    = if simParams.tail_corrected
                    approx_full_sum(χ_ladder, Int(floor(Nν*3/4)), modelParams, simParams, [2], W=W)[:,1]
                else sum(χ_ladder, dims=2) end
            for (νi,νₙ) in enumerate((-Nν):(Nν-1))
                @inbounds γres[ωi, νi, qi] = tmpSum[νi] / 
                            (get_symm_χ(bubble, ωₙ, νₙ,  qi) * (1 - U*χ[ωi, qi]))
            end
        end
    end
    return sdata(χ), sdata(γres)
end


#TODO: compute start point according to fortran code
function calc_λ_correction(χ, χloc, qMult, modelParams)
    qMult_tmp = reshape(qMult, 1, (size(qMult)...))
    norm      = sum(qMult)*modelParams.β
    χ_new(λ)  = real(1 ./ (1 ./ χ .+ λ))
    f(λ) = abs(sum(qMult_tmp ./ (1 ./ χ .+ λ[1]))/norm - χloc)
    λ    = Optim.minimizer(Optim.optimize(f, [0.0], BFGS(); autodiff = :forward))[1]
    return λ, χ_new(λ)
end

function calc_DΓA_Σ_parallel(χch::Array{Complex{Float64}, 2}, χsp::Array{Complex{Float64}, 2},
                             γch::Array{Complex{Float64}, 3}, γsp::Array{Complex{Float64}, 3},
                             bubble, Σ_loc, FUpDo, qMult, qGrid, kGrid,
                             modelParams::ModelParameters, simParams::SimulationParameters)
    qList = collect(qGrid)
    Nν = simParams.n_iν
    kList = collect(kGrid)
    ϵkqList = gen_squareLattice_full_ekq_grid(kList, qList)

    Σ_ladder = SharedArray{Complex{Float64}}(length(kList), Nν)
    #TODO: qifactor??
    #Σ_ladder = zeros(Complex{Float64}, Nν, length(kList))
    @sync @distributed for νi in 1:Nν
        νₙ = νi - 1
        for qi in 1:length(qList)
            @inbounds qiNorm = qMult[qi]/((2*(simParams.Nq-1))^2*8)#/(4.0*8.0)
            for (ωi,ωₙ) in enumerate((-simParams.n_iω):simParams.n_iω)
                Σν = get_symm_f(Σ_loc,ωₙ + νₙ)
                #TODO: access trilx nd chi by freq, instead of index
                @inbounds tmp = (1.5 * γsp[ωi, νi+Nν, qi]*(1 + modelParams.U*χsp[ωi, qi])-
                                                     0.5 * γch[ωi, νi+Nν, qi]*(1 - modelParams.U*χch[ωi, qi])-1.5+0.5)
                for (νpi, νpₙ) in enumerate((-Nν):(Nν-1))
                    @inbounds tmp += get_symm_χ(bubble,ωₙ, νpₙ, qi)*FUpDo[ωi, νi+Nν, νpi]
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
