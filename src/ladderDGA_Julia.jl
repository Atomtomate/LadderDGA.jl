
using Distributed
using SharedArrays
#TODO: move to JLD2
using JLD
using LinearAlgebra
using GenericLinearAlgebra
using Combinatorics
using TOML          # used for input
using Printf
using ForwardDiff
using Query
#using Roots         # used to find lambda
using IntervalArithmetic, IntervalRootFinding
using StaticArrays
using PaddedViews   # used to pad fft arrays
using FFTW          # used for convolutions
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
#TODO: implement generic indexing, especially dynamic freq_grids
#TODO: write macro, to reduce number of function parameters (modelParams, simParams, etc)

const loadFromBak = false

#function calculate_Σ_ladder(configFile)
    print("Reading Inputs...")
    modelParams, simParams, env, Γch, Γsp, Σ_loc, FUpDo, χLocch, χLocsp, usable_loc_ch, usable_loc_sp = setup_LDGA(configFile)
    println(size.([Γch, Γsp, Σ_loc, FUpDo])...)
    println("\rInputs Read. Starting Computation.
    Found usable intervals for local susceptibility of length 
          sp: $(length(usable_loc_sp))
          ch: $(length(usable_loc_ch)) 
          χLoc_sp = $(printr_s(χLocsp)), χLoc_ch = $(printr_s(χLocch))")

    println("Setting up and calculating k Grid: ")
    kIndices, kGrid = gen_kGrid(simParams.Nk, modelParams.D) 
    ϵkGrid          = squareLattice_ekGrid(kGrid)
    qIndices, qGrid = reduce_kGrid.(cut_mirror.((kIndices, kGrid)))
    qMultiplicity   = kGrid_multiplicity(qIndices)

    println("Calculating bubble: ")
    @time bubble = calc_bubble_fft(Σ_loc, ϵkGrid, length(qIndices), modelParams, simParams);

    println("Calculating χ and γ: ")
    #@time χsp_f, χch_f, χsp_ω_f, χch_ω_f, trilexsp_f, trilexch_f,  usable_sp_f, usable_ch_f  =
    #           calc_χ_trilex(Γsp, Γch, bubble, qMultiplicity, modelParams, simParams, true)
    @time χsp, χch, χsp_ω, χch_ω, trilexsp, trilexch,  usable_sp, usable_ch =
               calc_χ_trilex(Γsp, Γch, bubble, qMultiplicity, modelParams, simParams)

    #@time χsp2,trilexsp2 = calc_χ_trilex(Γsp, bubble, modelParams, simParams, Usign= -1)
    #@time χch2,trilexch2 = calc_χ_trilex(Γch, bubble, modelParams, simParams, Usign= 1)
    usable_ω = intersect(usable_sp, usable_ch)
          #env.fullSums
        
    println("Found usable intervals for non-local susceptibility of length 
          sp: $(usable_sp), length: $(length(usable_sp))
          ch: $(usable_ch), length: $(length(usable_ch))
          usable: $(usable_ω), length: $(length(usable_ω))")

    if simParams.tail_corrected
        χch_sum = approx_full_sum(χch_ω[usable_ch], [1])[1]/(modelParams.β)
        rhs = 0.5 - real(χch_sum)
        println("Using rhs for tail corrected lambda correction: ", printr_s(rhs), " = 0.5 - ", printr_s(χch_sum))
    else
        χsp_sum = sum(χsp_ω[usable_loc_sp])/(modelParams.β)
        χch_sum = sum(χch_ω[usable_loc_ch])/(modelParams.β)
        rhs = real(χLocsp + χLocch - χch_sum)
        println("Using rhs for non tail corrected lambda correction: 
          ", round(real(rhs),digits=4), " = ", round(real(χLocch),digits=4), 
          " + ", round(real(χLocsp),digits=4), " - ", round(real(χch_sum),digits=4))
    end

    println("Calculating λ correction in the spin channel: ")
    @time λsp, χsp_λ = calc_λ_correction(real(χsp[usable_sp,:]), real(rhs), qMultiplicity, simParams, modelParams)
    println("Found λsp = ", λsp)

    Σ_ladder = nothing
    if !simParams.chi_only
        println("Calculating Σ ladder: ")
        #@time Σ_ladder = calc_DΓA_Σ_fft(χsp_λ, χch, trilexsp, trilexch, bubble, Σ_loc, FUpDo, 
        #                          ϵkGrid, qIndices, usable_ω, modelParams, simParams)
        #save("sigma.jld","Sigma", Σ_ladder, 
        #      compress=true, compatible=true)
    end
#    return bubble, χch, χsp, χsp_λ, usable_sp, usable_ch,trilexch, trilexsp, Σ_ladder#, trilexsp2, trilexch2
#end

