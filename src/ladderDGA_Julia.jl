    using Printf 

    const PARALLEL = false
    using Distributed
    using SharedArrays
    #TODO: move to JLD2
    using JLD
    using LinearAlgebra
    using GenericLinearAlgebra
    using Optim
    using TOML
    using Printf
    #using ParquetFiles
    #using DataFrames
    using Query
    #using NLsolve
    #using DelimitedFiles
    using Roots
    include("$(@__DIR__)/Config.jl")
    include("$(@__DIR__)/helpers.jl")
    include("$(@__DIR__)/IO.jl")
    include("$(@__DIR__)/dispersion.jl")
    include("$(@__DIR__)/GFTools.jl")
    include("$(@__DIR__)/ladderDGATools.jl")
    include("$(@__DIR__)/dbg_ladderDGATools.jl")
    include("$(@__DIR__)/GFFit.jl")
    include("$(@__DIR__)/lambdaCorrection.jl")
    #include("$(@__DIR__)/../test/old_ladderDGATools.jl")

    #if PARALLEL
    #    addprocs(960)
    #end
    @everywhere using Distributed
    @everywhere using SharedArrays
    @everywhere using JLD
    @everywhere using LinearAlgebra
    @everywhere using GenericLinearAlgebra
    @everywhere using Optim
    @everywhere using TOML
    @everywhere using Printf
    #@everywhere using ParquetFiles
    #@everywhere using DataFrames
    @everywhere using Query
    #@everywhere using NLsolve
    #@everywhere using DelimitedFiles
    @everywhere using Roots
    @everywhere include("$(@__DIR__)/Config.jl")
    @everywhere include("$(@__DIR__)/helpers.jl")
    @everywhere include("$(@__DIR__)/IO.jl")
    @everywhere include("$(@__DIR__)/dispersion.jl")
    @everywhere include("$(@__DIR__)/GFTools.jl")
    @everywhere include("$(@__DIR__)/ladderDGATools.jl")
    @everywhere include("$(@__DIR__)/dbg_ladderDGATools.jl")
    @everywhere include("$(@__DIR__)/GFFit.jl")
    @everywhere include("$(@__DIR__)/lambdaCorrection.jl")
    #@everywhere include("$(@__DIR__)/../test/old_ladderDGATools.jl")
    export calculate_Σ_ladder

    #TODO: auto load fortran, if dimensions do not match
    #TODO: easy: fix inconsistency between bose and fermi grid (remove ν - 1 and inc nω as written i nconfig by one)
    #TODO: implement generic indexing: https://docs.julialang.org/en/v1/devdocs/offset-arrays/
    #TODO: don't fix type to complex

    const configFile = "./config.toml"#ARGS[1]
    const loadFromBak = false

function calculate_Σ_ladder(configFile)

    print("Reading Inputs...")
    modelParams, simParams, env = readConfig(configFile)#
    if env.loadFortran == "text"
        convert_from_fortran(simParams, env, loadFromBak)
        if env.loadAsymptotics
            readEDAsymptotics(env)
        end
    elseif env.loadFortran == "parquet"
        convert_from_fortran_pq(simParams, env)
        if env.loadAsymptotics
            readEDAsymptotics_parquet(env)
        end
    end
    vars    = load(env.inputVars) 
    G0      = vars["g0"]
    GImp    = vars["gImp"]
    Γch     = vars["GammaCharge"]
    Γsp     = vars["GammaSpin"]
    χDMFTch = vars["chiDMFTCharge"]
    χDMFTsp = vars["chiDMFTSpin"]
    freqBox = vars["freqBox"]
    ωGrid   = (-simParams.n_iω):(simParams.n_iω)
    νGrid   = (-simParams.n_iν):(simParams.n_iν-1)
    if env.loadAsymptotics
        asympt_vars = load(env.asymptVars)
        χchAsympt = asympt_vars["chi_ch_asympt"]
        χspAsympt = asympt_vars["chi_sp_asympt"]
    end

    print("\rInputs Read. Starting Computation                                          \n")
    Σ_loc = Σ_Dyson(G0, GImp)
    FUpDo = FUpDo_from_χDMFT(0.5 .* (χDMFTch - χDMFTsp), GImp, ωGrid, νGrid, νGrid, modelParams.β)

    #TODO: use fit for sum here
    χLocch = sum(χDMFTch)/(modelParams.β^3)
    χLocsp = sum(χDMFTsp)/(modelParams.β^3)
    #= println("-------") =#
    #= println(G0) =#
    #= println(GImp) =#
    #= println(Σ_loc) =#
    #= println("-------") =#

    _, kGrid   = reduce_kGrid.(gen_kGrid(simParams.Nk, modelParams.D; min = 0, max = π, include_min = true))
    kList = collect(kGrid)
    qIndices, qGrid  = reduce_kGrid.(gen_kGrid(simParams.Nq, modelParams.D; min = 0, max = π, include_min = true))
    qMultiplicity    = kGrid_multiplicity(qIndices)

    #TODO: remove ~5s overhead (precompile)
    print("Calculating bubble: ")
    @time bubble = calc_bubble(Σ_loc, qGrid, modelParams, simParams);
    # DEBUG:
    #@time bubble_exact = calc_bubble_ap(Σ_loc, qGrid, modelParams, simParams);
    #bubble_ed = readFortranBubble(env.inputDir*"/chi_bubble", 5, 6, 3)
    # END DEBUG

    print("Calculating naiive χ and γ in the charge channel: ")
    @time χch, trilexch = 
        calc_χ_trilex(Γch, bubble, modelParams, simParams, approx_full_sum_flag = false)
    print("Calculating naiive χ and γ in the spin channel: ")
    @time χsp, trilexsp = 
        calc_χ_trilex(Γsp, bubble, modelParams, simParams, Usign=(-1), approx_full_sum_flag = false)
    print("Calculating λ correction in the charge channel: ")
    @time λch, χch_λ = calc_λ_correction(χch, χLocch, qMultiplicity, modelParams)
    print("Calculating λ correction in the spin channel: ")
    @time λsp, χsp_λ = calc_λ_correction(χsp, χLocsp, qMultiplicity, modelParams)


    print("Calculating χ and γ in the charge channel: ")
    @time χch_impr, trilexch_impr  = 
        calc_χ_trilex(Γch, bubble, modelParams, simParams, approx_full_sum_flag = true)
    print("Calculating χ and γ in the spin channel: ")
    @time χsp_impr, trilexsp_impr = 
        calc_χ_trilex(Γsp, bubble, modelParams, simParams, Usign=-1, approx_full_sum_flag = true)
    print("Calculating λ correction in the charge channel: ")
    @time λch_impr, χch_λ_impr = calc_λ_correction(χch_impr, χLocch, qMultiplicity, modelParams)
    print("Calculating λ correction in the spin channel: ")
    @time λsp_impr, χsp_λ_impr = calc_λ_correction(χsp_impr, χLocsp, qMultiplicity, modelParams)


    if !simParams.chi_only
        print("Calculating naiive Σ ladder: ")
        @time Σ_ladder = calc_DΓA_Σ(χch, χsp, trilexch, trilexsp, bubble, Σ_loc, FUpDo,
                                      qMultiplicity, qGrid, kGrid, modelParams, simParams)
        print("Calculating Σ ladder: ")
        @time Σ_ladder_impr = calc_DΓA_Σ_impr(χch_impr, χsp_impr, trilexch_impr, trilexsp_impr, bubble, Σ_loc, FUpDo,
                                          qMultiplicity, qGrid, kGrid, modelParams, simParams)
    end
    save("chi.jld", "chi_ch", χch, "chi_ch_lambda", χch_λ, "chi_sp", χsp, "chi_sp_lambda", χsp_λ, 
         "chi_ch_impr", χch_impr, "chi_ch_lamdba_impr", χch_λ_impr, "chi_sp_impr", χsp_impr,
         "chi_sp_lambda_impr", χsp_λ_impr, compatible=true)
    save("res.jld", "kGrid", collect(kGrid), "qGrid", collect(qGrid), "qMult", qMultiplicity,
                    "bubble", bubble, "chi_ch", χch, "chi_sp", χsp,
                    "trilex_ch", trilexch, "trilex_sp", trilexsp,
                    "lambda_ch", λch, compress=true, compatible=true)
    #print("\n\n-----------\n\n")
end
calculate_Σ_ladder(configFile)
#

#= const ll = [(x1,x2) for x1 in range(-5,stop=5,length=100) for x2 in range(-5,stop=5,length=100)] =#
#= res = SharedArray{Float64}(length(ll)) =#
#= @sync @distributed for lli in 1:length(ll) =#
#= 	res[lli] = eval_f(ll[lli], G0, χch_impr, χsp_impr, trilexch_impr, trilexsp_impr, bubble, Σ_loc, FUpDo, =#
#= 		qMultiplicity, qGrid, kGrid, modelParams, simParams) =#
#= end =#
#= save("grid.jld", sdata(res)) =#
#if PARALLEL
#    rmprocs()
