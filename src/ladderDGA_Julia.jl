    using Printf 

    using Distributed
    using SharedArrays
    #TODO: move to JLD2
    using JLD
    using LinearAlgebra
    using GenericLinearAlgebra
    using TOML
    using Printf
    using ForwardDiff
    using Query
    using Roots
    using FFTW
    include("$(@__DIR__)/Config.jl")
    include("$(@__DIR__)/helpers.jl")
    include("$(@__DIR__)/IO.jl")
    include("$(@__DIR__)/dispersion.jl")
    include("$(@__DIR__)/GFTools.jl")
    include("$(@__DIR__)/ladderDGATools.jl")
    include("$(@__DIR__)/dbg_ladderDGATools.jl")
    include("$(@__DIR__)/GFFit.jl")
    include("$(@__DIR__)/lambdaCorrection.jl")

    export calculate_Σ_ladder

    #TODO: auto load fortran, if dimensions do not match
    #TODO: easy: fix inconsistency between bose and fermi grid (remove ν - 1 and inc nω as written i nconfig by one)
    #TODO: implement generic indexing, especially dynamic freq_grids
    #TODO: don't fix type to complex
    #TODO: write macro, to reduce number of function parameters (modelParams, simParams, qGrud, etc)

    const configFile = "./config.toml"#ARGS[1]
    const loadFromBak = false

#function calculate_Σ_ladder(configFile)
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
    println("TODO: check beta consistency, config <-> g0man, chi_dir <-> gamma dir")
    ωGrid   = (-simParams.n_iω):(simParams.n_iω)
    νGrid   = (-simParams.n_iν):(simParams.n_iν-1)
    if env.loadAsymptotics
        asympt_vars = load(env.asymptVars)
        χchAsympt = asympt_vars["chi_ch_asympt"]
        χspAsympt = asympt_vars["chi_sp_asympt"]
    end
    #TODO: unify checks
    if !(simParams.Nq*2-2 == simParams.Nint)
        println(stderr, "WARNING: for FFT, q and integration grids must be related in size!! 2*Nq-2 == Nint")
    end

    Σ_loc = Σ_Dyson(G0, GImp)
    FUpDo = FUpDo_from_χDMFT(0.5 .* (χDMFTch - χDMFTsp), GImp, ωGrid, νGrid, νGrid, modelParams.β)

    χLocsp_ω = sum_freq(χDMFTsp, [2,3], simParams, modelParams)[:,1,1]
    usable_loc_sp = find_usable_interval(real(χLocsp_ω))
    χLocsp = sum_freq(χLocsp_ω[usable_loc_sp], [1], simParams, modelParams)[1]

    χLocch_ω = sum_freq(χDMFTch, [2,3], simParams, modelParams)[:,1,1]
    usable_loc_ch = find_usable_interval(real(χLocch_ω))
    χLocch = sum_freq(χLocch_ω[usable_loc_ch], [1], simParams, modelParams)[1]
    println("\rInputs Read. Starting Computation.
          Found usable intervals for local susceptibility of length 
          sp: $(length(usable_loc_sp))
          ch: $(length(usable_loc_ch)) 
          χLoc_sp = $(χLocsp), χLoc_ch = $(χLocch)")

    println("Setting up and calculating local quantitites: ")
    qIndices, qGrid  = reduce_kGrid.(gen_kGrid(simParams.Nq, modelParams.D; min = 0, max = π, include_min = true))
    qMultiplicity    = kGrid_multiplicity(qIndices)
    qNorm            = 8*(simParams.Nq-1)^(modelParams.D)

    println("Calculating bubble: ")
    if simParams.kInt == "FFT"
        @time bubble = calc_bubble_fft(Σ_loc, modelParams, simParams);
    else
        @time bubble = calc_bubble(Σ_loc, qGrid, modelParams, simParams);
    end

    println("Calculating χ and γ in the spin channel: ")
    @time χsp, trilexsp = 
        calc_χ_trilex(Γsp, bubble, modelParams, simParams, Usign=(-1))
    println("Calculating χ and γ in the charge channel: ")
    @time χch, trilexch = 
        calc_χ_trilex(Γch, bubble, modelParams, simParams, Usign=(+1))
        
    χsp_ω = [sum(χsp[i,:] .* qMultiplicity) for i in 1:size(χsp,1)] ./ (qNorm)
    χch_ω = [sum(χch[i,:] .* qMultiplicity) for i in 1:size(χch,1)] ./ (qNorm)

    if simParams.tail_corrected
        χch_sum = approx_full_sum(χch_ω[usable_loc_ch], [1])[1]/(modelParams.β)
        rhs = 0.25 - χch_sum
        println("Using rhs for tail corrected lambda correction: ", rhs, " = 0.25 - ", real(χch_sum))
    else
        χsp_sum = sum(χsp_ω[usable_loc_sp])/(modelParams.β)
        χch_sum = sum(χch_ω[usable_loc_ch])/(modelParams.β)
        rhs = χLocsp + χLocch - χch_sum
        println("Using rhs for non tail corrected lambda correction: ", real(rhs), " = ", real(χLocch), " + ", real(χLocsp), " - ", real(χch_sum))
    end

    println("Calculating λ correction in the spin channel: ")
    @time λsp, χsp_λ = calc_λ_correction(χsp[usable_loc_sp,:], rhs, qMultiplicity, simParams, modelParams)
    println("Found λsp = ", λsp)

    if !simParams.chi_only
        println("Calculating Σ ladder: ")
        @time Σ_ladder = calc_DΓA_Σ(χch, χsp, trilexch, trilexsp, bubble, Σ_loc, FUpDo,
                                      qMultiplicity, qGrid, modelParams, simParams)
        save("Sigma", Σ_ladder, 
              compress=true, compatible=true)
    end
    save("chi.jld", "chi_ch", χch, "chi_sp", χsp, 
         "chi_sp_lambda", χsp_λ, compress=true, compatible=true)
    #= save("res.jld", "kGrid", collect(kGrid), "qGrid", collect(qGrid), "qMult", qMultiplicity, =#
    #=                 "bubble", bubble, =#
    #=                 "trilex_ch", trilexch, "trilex_sp", trilexsp, =#
    #=                 "lambda_ch", λsp, compress=true, compatible=true) =#
    #print("\n\n-----------\n\n")
#end
#calculate_Σ_ladder(configFile)
