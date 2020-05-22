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
    using ForwardDiff
    #using ParquetFiles
    #using DataFrames
    using Query
    #using NLsolve
    #using DelimitedFiles
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
    #include("$(@__DIR__)/../test/old_ladderDGATools.jl")

    #if PARALLEL
    #end
    @everywhere using Distributed
    @everywhere using SharedArrays
    @everywhere using LinearAlgebra
    @everywhere using GenericLinearAlgebra
    @everywhere using FFTW
    @everywhere include("$(@__DIR__)/Config.jl")
    @everywhere include("$(@__DIR__)/helpers.jl")
    @everywhere include("$(@__DIR__)/GFTools.jl")
    @everywhere include("$(@__DIR__)/ladderDGATools.jl")
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
    ωGrid   = (-simParams.n_iω):(simParams.n_iω)
    νGrid   = (-simParams.n_iν):(simParams.n_iν-1)
    if env.loadAsymptotics
        asympt_vars = load(env.asymptVars)
        χchAsympt = asympt_vars["chi_ch_asympt"]
        χspAsympt = asympt_vars["chi_sp_asympt"]
    end

    Σ_loc = Σ_Dyson(G0, GImp)
    FUpDo = FUpDo_from_χDMFT(0.5 .* (χDMFTch - χDMFTsp), GImp, ωGrid, νGrid, νGrid, modelParams.β)

    #TODO: use fit for sum here
    if simParams.tail_corrected
        χLocsp_ω = [approx_full_sum(χDMFTsp[i,:,:], [1,2]) for i in 1:size(χDMFTsp,1)]/(modelParams.β^2)
        χLocch_ω = [approx_full_sum(χDMFTch[i,:,:], [1,2]) for i in 1:size(χDMFTch,1)]/(modelParams.β^2)
        usable_loc_sp = find_usable_interval(real(χLocsp_ω))
        usable_loc_ch = find_usable_interval(real(χLocch_ω))
        χLocsp = approx_full_sum(χLocsp_ω[usable_loc_sp], [1])[1]/(modelParams.β)
        χLocch = approx_full_sum(χLocch_ω[usable_loc_ch], [1])[1]/(modelParams.β)
    else
        χLocsp_ω = sum(χDMFTsp,dims=[2,3])[:,1,1]/(modelParams.β^2)
        χLocch_ω = sum(χDMFTch,dims=[2,3])[:,1,1]/(modelParams.β^2)
        usable_loc_sp = find_usable_interval(real(χLocsp_ω))
        usable_loc_ch = find_usable_interval(real(χLocch_ω))
        χLocsp = sum(χLocsp_ω[usable_loc_sp])/(modelParams.β)
        χLocch = sum(χLocch_ω[usable_loc_ch])/(modelParams.β)
    end
    println("\rInputs Read. Starting Computation.
          Found usable intervals for local susceptibility of length 
          sp: $(length(usable_loc_sp))
          ch: $(length(usable_loc_ch)) 
          χLoc_sp = $(χLocsp), χLoc_ch = $(χLocch)")

    println("Setting up and calculating local quantitites: ")
    _, kGrid         = reduce_kGrid.(gen_kGrid(simParams.Nk, modelParams.D; min = 0, max = π, include_min = true))
    kList            = collect(kGrid)
    qIndices, qGrid  = reduce_kGrid.(gen_kGrid(simParams.Nq, modelParams.D; min = 0, max = π, include_min = true))
    qNorm            = 8*(simParams.Nq-1)^(modelParams.D)

    #TODO: remove ~5s overhead (precompile)
    println("Calculating bubble: ")
    #Σ_loc = convert(SharedArray,Σ_loc)
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
    #usable_ω_sp = find_usable_interval(real(χsp_ω))
    #usable_ω_ch = find_usable_interval(real(χch_ω))
    if simParams.tail_corrected
        χsp_sum = approx_full_sum(χsp_ω, [1])[1]/(modelParams.β)
        χch_sum = approx_full_sum(χch_ω, [1])[1]/(modelParams.β)
        println(stderr, "TODO: fixed half filling for rhs of lambda correction")
        println(χch_sum)
        rhs = 0.25 - χch_sum
    else
        χsp_sum = sum(χsp_ω[usable_loc_sp])/(modelParams.β)
        χch_sum = sum(χch_ω[usable_loc_ch])/(modelParams.β)
        rhs = χLocsp + χLocch - χch_sum
        #println("rhs = $(rhs) =  $(χLocsp) + $(χLocch) - $(χch_sum)")
    end

    println("Calculating λ correction in the spin channel: ")
    @time λsp, χsp_λ = calc_λ_correction(χsp, rhs, qMultiplicity, usable_loc_sp, modelParams)
    println("Found λsp = ", λsp)
    #= println("Calculating λ correction in the charge channel: ") =#
    #= @time λch, χch_λ = calc_λ_correction(χch, χLocch, qMultiplicity, usable_loc_ch, modelParams) =#
    #= println("Found λch = ", λch) =#

    if !simParams.chi_only
        println("Calculating Σ ladder: ")
        @time Σ_ladder = calc_DΓA_Σ(χch, χsp, trilexch, trilexsp, bubble, Σ_loc, FUpDo,
                                      qMultiplicity, qGrid, kGrid, modelParams, simParams)
        save("Sigma", Σ_ladder, "kGrid", kGrid,
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
