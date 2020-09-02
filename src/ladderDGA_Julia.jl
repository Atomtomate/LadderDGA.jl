using Logging
using Distributed
using SharedArrays
#TODO: move to JLD2
using JLD
using DelimitedFiles
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

logger = ConsoleLogger(stderr, Logging.Info, meta_formatter=Logging.default_metafmt,
              show_limited=true, right_justify=0)
global_logger(logger)


include("$(@__DIR__)/Config.jl")
include("$(@__DIR__)/DataTypes.jl")
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

function calculate_Σ_ladder(configFile)
    @info "Reading Inputs..."
    #TODO: use sturcts of local quantities
    modelParams, simParams, env, Γsp, Γch, GImp, Σ_loc, FUpDo, χLocsp, χLocch, usable_loc_sp, usable_loc_ch = setup_LDGA(configFile, false);
    @debug simParams
    @info """Inputs Read. Starting Computation.
    Found usable intervals for local susceptibility of length 
          sp: $(length(usable_loc_sp))
          ch: $(length(usable_loc_ch)) 
          χLoc_sp = $(printr_s(χLocsp)), χLoc_ch = $(printr_s(χLocch))"""

    @info "Setting up and calculating k Grid: "
    kIndices, kGrid = gen_kGrid(simParams.Nk, modelParams.D);
    ϵkGrid          = squareLattice_ekGrid(kGrid);
    qIndices, qGrid = reduce_kGrid.(cut_mirror.((kIndices, kGrid)));
    qMultiplicity   = kGrid_multiplicity(qIndices);

    GLoc = Gfft_from_Σ(Σ_loc, ϵkGrid, -simParams.n_iω:(simParams.n_iν+simParams.n_iω-1), modelParams);
    GImpSym = store_symm_f(GImp,-simParams.n_iω:(simParams.n_iν+simParams.n_iω-1));
    @info "Calculating bubble: "
    @time bubble = calc_bubble_fft(GLoc, length(qIndices), modelParams, simParams);
    @time bubbleLoc = calc_bubble_fft(GImpSym, 1, modelParams, simParams);

    @info "Calculating χ and γ: "
    #@time χsp_f, χch_f, χsp_ω_f, χch_ω_f, trilexsp_f, trilexch_f,  usable_sp_f, usable_ch_f  =
    #           calc_χ_trilex(Γsp, Γch, bubble, qMultiplicity, modelParams, simParams, true)
    #TODO: data structure for χ trilex
    #TODO: combine return values somehow
    @time χsp, χch, χsp_ω, χch_ω, trilexsp, trilexch, usable_sp, usable_ch =
        calc_χ_trilex(Γsp, Γch, bubble, qMultiplicity, modelParams, simParams);

    @info "TODO: computation of local quantities does not make any sense"
    @time χspLoc, χchLoc, _, _, trilexspLoc, trilexchLoc, _, _ =
        calc_χ_trilex(Γsp, Γch, bubbleLoc, [1], modelParams, simParams);

    #@time χsp2,trilexsp2 = calc_χ_trilex(Γsp, bubble, modelParams, simParams, Usign= -1)
    #@time χch2,trilexch2 = calc_χ_trilex(Γch, bubble, modelParams, simParams, Usign= 1)
    usable_ω = intersect(usable_sp, usable_ch)
        
    @info """Found usable intervals for non-local susceptibility of length 
          sp: $(usable_sp), length: $(length(usable_sp))
          ch: $(usable_ch), length: $(length(usable_ch))
          usable: $(usable_ω), length: $(length(usable_ω))"""

    if simParams.tail_corrected
        χch_sum = sum_freq(χch_ω, [1], simParams.tail_corrected, modelParams.β)[1]
        rhs = 0.5 - real(χch_sum)
        @info "Using rhs for tail corrected lambda correction: " rhs " = 0.5 - " χch_sum
    else
        χsp_sum = sum(χsp_ω[usable_loc_sp])/(modelParams.β)
        χch_sum = sum(χch_ω[usable_loc_ch])/(modelParams.β)
        rhs = real(χLocsp + χLocch - χch_sum)
        @info "Using rhs for non tail corrected lambda correction: " rhs " = "  χLocch " + " χLocsp " - " χch_sum
    end

    @info "Calculating λ correction in the spin channel: "
    @time λsp = calc_λsp_correction(χsp[usable_sp,:], rhs, qMultiplicity, simParams, modelParams)
    @info "Found λsp " λsp

    @info "computing λ corrected χsp, using " simParams.χFillType " as fill value outside usable ω range."
    χsp_λ = zeros(eltype(χsp), size(χsp)...)
    if simParams.χFillType == zero_χ_fill
        χsp_λ[usable_sp,:] =  χ_λ(χsp[usable_sp,:], λsp) 
    elseif simParams.χFillType == lambda_χ_fill
        χsp_λ =  χ_λ(χsp, λsp) 
    else
        χsp_λ =  χsp 
        χsp_λ[usable_sp,:] =  χ_λ(χsp[usable_sp,:], λsp) 
    end

    Σ_ladder = nothing
    Σ_ladderLoc = nothing
    if !simParams.chi_only
        @info "Calculating Σ ladder: "
        @time Σ_ladderLoc = calc_DΓA_Σ_fft(χspLoc, χchLoc, trilexspLoc, trilexchLoc, bubbleLoc, GImpSym, FUpDo, 
                                           [1], [1], usable_ω, 1:simParams.n_iν, 1,
                                           modelParams, simParams, simParams.tail_corrected)
        Σ_ladderLoc = Σ_ladderLoc .+ modelParams.n * modelParams.U/2.0;
        #(-simParams.n_iω):(simParams.n_iω)
        @time Σ_ladder = calc_DΓA_Σ_fft(χsp_λ, χch, trilexsp, trilexch, bubble, GLoc, FUpDo, 
                                  ϵkGrid, qIndices, usable_ω, 1:simParams.n_iν, simParams.Nk,
                                  modelParams, simParams, simParams.tail_corrected)
        Σ_ladder_corrected = Σ_ladder .- Σ_ladderLoc .+ Σ_loc[eachindex(Σ_ladderLoc)]
        #save("sigma.jld","Sigma", Σ_ladder, 
        #      compress=true, compatible=true)
    end
    return bubble, χsp, χsp_λ, χch, usable_sp, usable_ch, trilexsp, trilexch, Σ_ladder, Σ_ladder_corrected
end

