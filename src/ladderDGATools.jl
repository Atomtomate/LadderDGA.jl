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
    #res = zeros(Complex{Float64}, 2*simParams.n_iω+1, simParams.n_iν, Nq)
    @sync @distributed for qi in 1:Nq
        @simd for ki in 1:length(ϵkIntList)
            @inbounds ϵₖ₂ = ϵkqIntList[ki, qi]
            @simd for νₙ in 0:simParams.n_iν-1
                Σν = get_symm_f(Σ,νₙ)
                Gν = GF_from_Σ(νₙ, modelParams.β, modelParams.μ, ϵkIntList[ki], Σν) 
                for (ωi, ωₙ) in enumerate((-simParams.n_iω):simParams.n_iω)
                    Σων = get_symm_f(Σ,νₙ + ωₙ)
                    Gνω = GF_from_Σ(νₙ + ωₙ, modelParams.β, modelParams.μ, ϵₖ₂, Σων)
                    @inbounds res[ωi, νₙ+1, qi] -= 4*Gν*Gνω
                                       
       #= if ωi == 1 =#
       #=     println("$(νₙ) $(collect(qGrid)[qi][1]) $(collect(qGrid)[qi][2]) $(ϵkIntList[ki]) $(ϵₖ₂) $(Gν) $(Gνω)") =#
       #= end =#
                end
            end
        end
    end
    res = modelParams.β .* sdata(res) ./ ((2*simParams.Nint)^2)
    res = cat(conj.(res[end:-1:1,end:-1:1,:]),res, dims=2)
    return res
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
    χ = SharedArray{Complex{Float64}}(2*Nω+1, Nq)    # ωₙ x q (summed over νₙ)

    if simParams.tail_corrected
        numin = Int(floor((Nν/2)*3/4))
        numax = Int(floor(Nν/2))
        W = build_weights(numin, numax, [0,1,2,3,4])
    end

    @sync @distributed for qi in 1:Nq
        χ_tmp = Array{eltype(Γ)}(undef, 2*Nν, 2*Nν)
        for (ωi,ωₙ) in enumerate((-Nω):Nω)
            @inbounds χ_tmp = Γ[ωi, :, :]
            for (νi,νₙ) in enumerate((-Nν):(Nν-1))
                @inbounds χ_tmp[νi, νi] = χ_tmp[νi, νi] + 1.0 / bubble[ωi, νi, qi] 
            end
            @inbounds χ_tmp = inv(χ_tmp)

            @inbounds χ[ωi, qi] = simParams.tail_corrected ? 
                approx_full_sum(χ_tmp, W, modelParams, [1,2]) : sum(χ_tmp)/(modelParams.β^2) 
            tmpSum = Array{eltype(χ_tmp)}(undef, size(χ_tmp,1))
            for i in size(χ_tmp, 1)
                @inbounds tmpSum[i] = simParams.tail_corrected ? 
                approx_full_sum(χ_tmp[i,:], W, modelParams, [1]) : sum(χ_tmp[i,:], dims=1)/(modelParams.β)
            end
            for (νi,νₙ) in enumerate((-Nν):(Nν-1))
                @inbounds γres[ωi, νi, qi] = tmpSum[νi] / 
                            (bubble[ωi, νi, qi]  * (1 - U*χ[ωi, qi]))
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
    start_val = maximum(real(χ[ceil(Int64, size(χ,1)/2), :]))
    λ    = Optim.minimizer(Optim.optimize(f, [start_val], Newton(); autodiff = :forward))[1]
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
            # approx omega sum: introduce Sigma_ladder tmp, that depends on omega, then use approx_sum
            for (ωi,ωₙ) in enumerate((-simParams.n_iω):simParams.n_iω)
                Σν = get_symm_f(Σ_loc,ωₙ + νₙ)
                #TODO: access trilx nd chi by freq, instead of index
                @inbounds tmp = (1.5 * γsp[ωi, νi+Nν, qi]*(1 + modelParams.U*χsp[ωi, qi])-
                                                     0.5 * γch[ωi, νi+Nν, qi]*(1 - modelParams.U*χch[ωi, qi])-1.5+0.5)
                #approx nu sum: extend tmp by one dimension, then use approx_sum
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
