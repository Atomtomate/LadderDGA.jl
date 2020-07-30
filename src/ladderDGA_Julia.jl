
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
using Roots         # used to find lambda
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
println("TODO: check beta consistency, config <-> g0man, chi_dir <-> gamma dir")
ωGrid   = (-simParams.n_iω):(simParams.n_iω)
νGrid   = (-simParams.n_iν):(simParams.n_iν-1)
if env.loadAsymptotics
    asympt_vars = load(env.asymptVars)
    χchAsympt = asympt_vars["chi_ch_asympt"]
    χspAsympt = asympt_vars["chi_sp_asympt"]
end
#TODO: unify checks
(simParams.Nk % 2 != 0) && throw("For FFT, q and integration grids must be related in size!! 2*Nq-2 == Nk")

Σ_loc = Σ_Dyson(G0, GImp)
FUpDo = FUpDo_from_χDMFT(0.5 .* (χDMFTch - χDMFTsp), GImp, ωGrid, νGrid, νGrid, modelParams.β)

println(size(χDMFTsp))
χLocsp_ω = sum_freq(χDMFTsp, [2,3], simParams.tail_corrected, modelParams.β)[:,1,1]
usable_loc_sp = env.fullSums ? (1:length(χLocsp_ω)) : find_usable_interval(real(χLocsp_ω))
χLocsp = sum_freq(χLocsp_ω[usable_loc_sp], [1], simParams.tail_corrected, modelParams.β)[1]

χLocch_ω = sum_freq(χDMFTch, [2,3], simParams.tail_corrected, modelParams.β)[:,1,1]
usable_loc_ch = env.fullSums ? (1:length(χLocch_ω)) : find_usable_interval(real(χLocch_ω))
χLocch = sum_freq(χLocch_ω[usable_loc_ch], [1], simParams.tail_corrected, modelParams.β)[1]

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
           calc_χ_trilex(Γsp, Γch, bubble, qMultiplicity, modelParams, simParams, true)
usable_ω = maximum((first(usable_sp),first(usable_ch))):minimum((last(usable_sp),last(usable_ch)))
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
    rhs = χLocsp + χLocch - χch_sum
    println("Using rhs for non tail corrected lambda correction: 
      ", round(real(rhs),digits=4), " = ", round(real(χLocch),digits=4), 
      " + ", round(real(χLocsp),digits=4), " - ", round(real(χch_sum),digits=4))
end

println("Calculating λ correction in the spin channel: ")
@time λsp, χsp_λ = calc_λ_correction(χsp, usable_sp, rhs, qMultiplicity, simParams, modelParams)
println("Found λsp = ", λsp)

if !simParams.chi_only
    println("Calculating Σ ladder: ")
    @time Σ_ladder = calc_DΓA_Σ_fft(χsp_λ, χch, trilexsp, trilexch, bubble, Σ_loc, FUpDo, 
                                  ϵkGrid, qIndices, usable_ω, modelParams, simParams)
    #@time Σ_ladder_f = calc_DΓA_Σ_fft(χsp_λ_f, χch_f, trilexsp_f, trilexch_f, bubble, Σ_loc, FUpDo, 
    #                                  ϵkGrid, qIndices, 1:size(bubble,1), modelParams, simParams)
Σ_ladder_naive = calc_DΓA_Σ(χsp_λ, χch, trilexsp, trilexch,
                             bubble, Σ_loc, FUpDo, qMultiplicity, qGrid, qIndices, usable_ω,
                             modelParams, simParams)
#= Σ_ladder_naive_f = calc_DΓA_Σ(χsp_λ_f, χch_f, trilexsp_f, trilexch_f, =#
#=                               bubble, Σ_loc, FUpDo, qMultiplicity, qGrid, qIndices, 1:size(bubble,1), =#
#=                              modelParams, simParams) =#
    save("sigma.jld","Sigma", Σ_ladder, 
          compress=true, compatible=true)
end
#save("chi.jld", "chi_ch", χch, "chi_sp", χsp, 
#     "chi_sp_lambda", χsp_λ, compress=true, compatible=true)
#end
#calculate_Σ_ladder(configFile)
