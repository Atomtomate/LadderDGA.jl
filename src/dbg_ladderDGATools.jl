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


