    using Printf 

    const PARALLEL = false
    using Distributed
    using SharedArrays
    using LinearAlgebra
    using GenericLinearAlgebra
    using JLD
    using Optim
    using TOML
    using Printf
    using DelimitedFiles
    include("$(@__DIR__)/Config.jl")
    include("$(@__DIR__)/helpers.jl")
    include("$(@__DIR__)/IO.jl")
    include("$(@__DIR__)/dispersion.jl")
    include("$(@__DIR__)/GFTools.jl")
    include("$(@__DIR__)/ladderDGATools.jl")
    include("$(@__DIR__)/GFFit.jl")
    #include("$(@__DIR__)/../test/old_ladderDGATools.jl")

    if PARALLEL
        addprocs(8)
    end
    @everywhere using Distributed
    @everywhere using SharedArrays
    @everywhere using JLD
    @everywhere using LinearAlgebra
    @everywhere using GenericLinearAlgebra
    @everywhere using Optim
    @everywhere using TOML
    @everywhere using Printf
    @everywhere include("$(@__DIR__)/Config.jl")
    @everywhere include("$(@__DIR__)/helpers.jl")
    @everywhere include("$(@__DIR__)/IO.jl")
    @everywhere include("$(@__DIR__)/dispersion.jl")
    @everywhere include("$(@__DIR__)/GFTools.jl")
    @everywhere include("$(@__DIR__)/ladderDGATools.jl")
    @everywhere include("$(@__DIR__)/GFFit.jl")
    #@everywhere include("$(@__DIR__)/../test/old_ladderDGATools.jl")
    export calculate_Σ_ladder

    #TODO: auto load fortran, if dimensions do not match
    #TODO: easy: fix inconsistency between bose and fermi grid (remove ν - 1 and inc nω as written i nconfig by one)
    #TODO: implement generic indexing: https://docs.julialang.org/en/v1/devdocs/offset-arrays/
    #TODO: don't fix type to complex

    const configFile = "config.toml"#ARGS[1]

    #function calculate_Σ_ladder(configFile)

    print("Reading Inputs...")
    modelParams, simParams, env = readConfig(configFile)#
    if env.loadFortran
        convert_from_fortran(simParams, env)
        if env.loadAsymptotics
            readEDAsymptotics(env)
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
        χchED = asympt_vars["chi_ch_ED"]
        χspED = asympt_vars["chi_sp_ED"]
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
        #fit_bubble_test(bubble, modelParams, simParams)

        print("Calculating naiive χ and γ in the charge channel: ")
        @time χch, trilexch  = 
            calc_χ_trilex(Γch, bubble, modelParams.U, modelParams.β, simParams, approx_full_sum_flag = false)
        print("Calculating naiive χ and γ in the spin channel: ")
        @time χsp, trilexsp = 
            calc_χ_trilex(Γsp, bubble, -modelParams.U, modelParams.β, simParams, approx_full_sum_flag = false)
        print("Calculating λ correction in the charge channel: ")
        @time λch, χch_λ = calc_λ_correction(χch, χLocch, qMultiplicity, modelParams)
        print("Calculating λ correction in the spin channel: ")
        @time λsp, χsp_λ = calc_λ_correction(χsp, χLocsp, qMultiplicity, modelParams)
        print("Calculating naiive Σ ladder: ")
        @time Σ_ladder = calc_DΓA_Σ(χch, χsp, trilexch, trilexsp, bubble, Σ_loc, FUpDo,
                                          qMultiplicity, qGrid, kGrid, modelParams, simParams)



        print("Calculating χ and γ in the charge channel: ")
        @time χch_impr, trilexch_impr  = 
            calc_χ_trilex(Γch, bubble, modelParams.U, modelParams.β, simParams,approx_full_sum_flag = true)
        print("Calculating χ and γ in the spin channel: ")
        @time χsp_impr, trilexsp_impr = 
            calc_χ_trilex(Γsp, bubble, -modelParams.U, modelParams.β, simParams, approx_full_sum_flag = true)
        #= print("Calculating λ correction in the charge channel: ") =#
        #= @time λch_impr, χch_λ_impr = calc_λ_correction(χch_impr, χLocch, qMultiplicity, modelParams) =#
        #= print("Calculating λ correction in the spin channel: ") =#
        #= @time λsp_impr, χsp_λ_impr = calc_λ_correction(χsp_impr, χLocsp, qMultiplicity, modelParams) =#
        print("Calculating Σ ladder: ")
        @time Σ_ladder_impr = calc_DΓA_Σ_impr(χch_impr, χsp_impr, trilexch_impr, trilexsp_impr, bubble, Σ_loc, FUpDo,
                                          qMultiplicity, qGrid, kGrid, modelParams, simParams)

        print("TODO: replace chisp by chisp_lambda\n")
        save("res.jld", "kGrid", collect(kGrid), "qGrid", collect(qGrid), "qMult", qMultiplicity,
                        "bubble", bubble, "chi_ch", χch, "chi_sp", χsp, "chi_ch_lambda", χch_λ, "chi_sp_lambda", χsp_λ, 
                        "trilex_ch", trilexch, "trilex_sp", trilexsp,
                        "lambda_ch", λch, "lambda_sp", λsp, "Sigma_ladder", Σ_ladder)
        #print("\n\n-----------\n\n")
    #calculate_Σ_ladder(configFile)
    #
    if PARALLEL
        rmprocs()
    end
