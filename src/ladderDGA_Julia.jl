using Printf 

const PARALLEL = true
const LOAD_FORTRAN = false

using Distributed
using JLD
#TODO: auto load fortran, if dimensions do not match
#TODO: easy: fix inconsistency between bose and fermi grid (remove ν - 1 and inc nω as written i nconfig by one)
#TODO: implement generic indexing: https://docs.julialang.org/en/v1/devdocs/offset-arrays/
#TODO: don't fix type to complex

if PARALLEL
    addprocs(8)
end
@everywhere include("$(@__DIR__)/IO.jl")
@everywhere include("$(@__DIR__)/dispersion.jl")
@everywhere include("$(@__DIR__)/GFTools.jl")
@everywhere include("$(@__DIR__)/ladderDGATools.jl")
@everywhere include("$(@__DIR__)/../test/old_ladderDGATools.jl")

    print("Reading Inputs...")
    modelParams, simParams = readConfig(ARGS[1])
    if LOAD_FORTRAN
        convert_from_fortran()
    end
    vars    = load(simParams.inputVars) 
    G0      = vars["g0"]
    GImp    = vars["gImp"]
    Γch     = vars["GammaCharge"]
    Γsp     = vars["GammaSpin"]
    χDMFTch = vars["chiDMFTCharge"]
    χDMFTsp = vars["chiDMFTSpin"]
    freqBox = vars["freqBox"]
    ωGrid   = (-simParams.n_iω):(simParams.n_iω)
    νGrid   = (-simParams.n_iν):(simParams.n_iν-1)

    print("\rInputs Read. Starting Computation                                          \n")
    Σ_loc = Σ_Dyson(G0, GImp)
    FUpDo = FUpDo_from_χDMFT(0.5 .* (χDMFTch - χDMFTsp), GImp, ωGrid, νGrid, νGrid, modelParams.β)

    #TODO: determine max bosonic freq and cut off there
    χLocch = sum(χDMFTch)/(modelParams.β^3)
    χLocsp = sum(χDMFTsp)/(modelParams.β^3)

    _, kGrid   = reduce_kGrid.(gen_kGrid(simParams.Nk, modelParams.D; min = 0, max = π, include_min = true))
    kList = collect(kGrid)
    qIndices, qGrid  = reduce_kGrid.(gen_kGrid(simParams.Nq, modelParams.D; min = 0, max = π, include_min = true))
    qMultiplicity    =  kGrid_multiplicity(qIndices)

    #TODO: remove ~5s overhead (precompile)
    print("Calculating bubble: ")
    @time bubble = calc_bubble_parallel(Σ_loc, qGrid, modelParams, simParams)
    print("Calculating χ and γ in the charge channel: ")
    @time χch, trilexch = calc_χ_trilex_parallel(Γch, bubble, modelParams.U, modelParams.β, simParams)
    print("Calculating χ and γ in the spin channel: ")
    @time χsp, trilexsp = calc_χ_trilex_parallel(Γsp, bubble, -modelParams.U, modelParams.β, simParams)
    print("Calculating λ correction in the charge channel: ")
    @time λch, χch_λ = calc_λ_correction(χch, χLocch, qMultiplicity, modelParams)
    print("Calculating λ correction in the spin channel: ")
    @time λsp, χsp_λ = calc_λ_correction(χsp, χLocsp, qMultiplicity, modelParams)

    print("TODO: replace chisp by chisp_lambda\n")
    print("Calculating Σ ladder: ")
    @time Σ_ladder = calc_DΓA_Σ_parallel(χch, χsp, trilexch, trilexsp, bubble, Σ_loc, FUpDo,
                                      qMultiplicity, qGrid, kGrid, modelParams, simParams)

    save("res.jld", "kGrid", kGrid, "qGrid", qGrid, "qMult", qMultiplicity,
                    "bubble", bubble, "chi_ch", χch, "chi_sp", χsp, "chi_ch_lambda", χch_λ, "chi_sp_lambda", χsp_λ, 
                    "trilex_ch", trilexch, "trilex_sp", trilexsp,
                    "lambda_ch", λch, "lambda_sp", λsp, "Sigma_ladder", Σ_ladder)
    print("\n\n-----------\n\n")
        for (ki,k) in enumerate(kList)
            for vi in 1:size(Σ_ladder, 2)
                @printf("%d %f %f %f %f \n",  vi, kList[ki][1], kList[ki][2],
                        real(Σ_ladder[ki,vi]), imag(Σ_ladder[ki,vi]))
            end
        end
