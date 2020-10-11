# ===================== Dependencies ======================
using ArgParse
using Logging
using Distributed
@everywhere using SharedArrays
#TODO: move to JLD2
using JLD
using DelimitedFiles
@everywhere using LinearAlgebra
@everywhere using GenericLinearAlgebra
using Combinatorics
using TOML          # used for input
@everywhere using Printf
using ForwardDiff
@everywhere using Query
using IntervalArithmetic, IntervalRootFinding
using StaticArrays
using PaddedViews   # used to pad fft arrays
using FFTW          # used for convolutions
using ProgressMeter


# ======================= Includes ========================
include("$(@__DIR__)/Config.jl")
include("$(@__DIR__)/DataTypes.jl")
include("$(@__DIR__)/helpers.jl")
include("$(@__DIR__)/IO.jl")
include("$(@__DIR__)/dispersion.jl")
include("$(@__DIR__)/GFTools.jl")
include("$(@__DIR__)/GFFit.jl")
include("$(@__DIR__)/ladderDGATools.jl")
include("$(@__DIR__)/lambdaCorrection.jl")
include("$(@__DIR__)/thermodynamics.jl")


#TODO: this could be a macro modifying the 3 main functions
# ======================== Setup ==========================
if myid() == 1
    # ==================== Argument Parser ====================
    s = ArgParseSettings()
    @add_arg_table s begin
        "--config", "-c"
            help = "configuration file, default `config.toml`"
            arg_type = String
            default = "config.toml"
    end
    @warn "ArgParse not working with jupyter notebooks"
    @warn "Check U and Beta consistency in chi_dir/gamma_dir"
    args = parse_args([], s)
    @info "Reading Inputs..."
    const modelParams, simParams, env, impQ_sp, impQ_ch, GImp_pos, Σ_loc_pos, FUpDo  = setup_LDGA(args["config"], false);

    # ======================== Logger =========================
    const global io = open("run.out","w+")
    #= stream = ((env.logfile == "stderr") || (env.logfile == "stdout")) ? =# 
    #=             (@eval $(Symbol(env.logfile))) : open(env.logfile, "w+") =#
    #=             logger = SimpleLogger(io, Logging.Info) =#
    #logger = SimpleLogger(stream, Base.eval(Logging, Symbol(env.loglevel))) 
                   #   meta_formatter=Logging.default_metafmt,
                   #   show_limited=true, right_justify=0)
    logger = ConsoleLogger(stderr, Logging.Info, 
                      meta_formatter=Logging.default_metafmt,
                      show_limited=true, right_justify=0)
    global_logger(logger)

    # ==================== Progress Meter =====================
    Nsteps = 2* (4) * (2*simParams.n_iω+1)
    const channel = RemoteChannel(()->Channel{Bool}(Nsteps), 1)
    if env.progressbar
        p = Progress(Nsteps)
        @async while take!(channel)
            next!(p)
        end
    end

    const kIndices, kGrid = gen_kGrid(simParams.Nk, modelParams.D);
    const ϵkGrid          = squareLattice_ekGrid(kGrid);
    const qIndices, qGrid, ϵqGrid = reduce_kGrid.(cut_mirror.((kIndices, kGrid, collect(ϵkGrid))));
    const qMultiplicity   = kGrid_multiplicity(qIndices);

    const fft_range = -simParams.n_iω:(simParams.n_iν+simParams.n_iω-1)
    const GImp_sym = store_symm_f(GImp_pos, fft_range)
    const GImp = convert(SharedArray,reshape(GImp_sym, (length(GImp_sym),1)));
    const GLoc_fft_pos = Gfft_from_Σ(Σ_loc_pos, ϵkGrid, fft_range, modelParams);
    const GLoc_fft = convert(SharedArray, GLoc_fft_pos);

    @info """Inputs Read. Starting Computation.
    Found usable intervals for local susceptibility of length 
          sp: $(length(impQ_sp.usable_ω))
          ch: $(length(impQ_ch.usable_ω)) 
          χLoc_sp = $(printr_s(impQ_sp.χ_loc)), χLoc_ch = $(printr_s(impQ_ch.χ_loc))"""

        
    #TODO: wrap progress bar here. put! should no show up in ladderDGATools
    const calc_bubble(G::GνqT, len::Int64) = calc_bubble_int(G, len, modelParams, simParams, channel);
    const calc_χ_trilex(Γr::ΓT, bubble::BubbleT, qMult::Array{Float64,1}, U::Float64) = 
            calc_χ_trilex_int(Γr, bubble, qMult, U, modelParams.β, channel,
                              tc=simParams.tail_corrected, fullRange=simParams.fullChi);

    @warn "Forcing non tail corrected summation in EoM"
    const calc_Σ(Qsp::NonLocalQuantities, Qch::NonLocalQuantities, bubble::BubbleT, G::GνqT, 
                qIndices::qGridT, usable_ω::UnitRange{Int64}, Nk::Int64) = 
                calc_DΓA_Σ_int(Qsp, Qch, bubble, G, FUpDo, qIndices, usable_ω, Nk,
                               modelParams, simParams, simParams.tail_corrected)
end

