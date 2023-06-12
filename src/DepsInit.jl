# ==================================================================================================== #
#                                           DepsInit.jl                                                #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#   Setup after loading the module. All dependencies, precompilation, logging and multi-core           #
#   preperations should be done here.                                                                  #
# -------------------------------------------- TODO -------------------------------------------------- #
#   initialize workers here instead of relying on julia -p                                             #
#   logging to file does not work correctly.                                                           #
# ==================================================================================================== #


# ========================================== Dependencies ============================================
using ArgParse
using Logging, LoggingExtras
using OffsetArrays
using Distributed
using FFTW              # used for convolutions
using Combinatorics     # 
using SpecialFunctions  # e.g. PolyLog in filling
using TOML              # input configuration
using JLD2, FileIO      # input/output files
using LinearAlgebra

# Fortran compatibility
using Printf, DelimitedFiles

using Roots # Roots.jl for μ determination for now, may be replaced by NLsolve

# lDGA related
using Dispersions
using BSE_SC

using TimerOutputs

using Base.Iterators, Base

# ============================================= Includes =============================================
include("LapackWrapper.jl")
include("Config.jl")
include("DataTypes.jl")
include("helpers.jl")
include("parallelization_helpers.jl")
include("IO.jl")
include("GFTools.jl")
include("GFFit.jl")
include("ladderDGATools.jl")
include("ladderDGATools_singleCore.jl")
include("thermodynamics.jl")
include("LambdaCorrection/LambdaCorrection.jl")

# ======================================== Internal Packages =========================================
using .LapackWrapper
using .LambdaCorrection

# =================================== Parallelization Bookkeeping ====================================
global_vars = String[]
wcache = Ref(WorkerCache())

# ==================== Precompilation ====================
# TODO: precompile calc_... for CompleX{Float64}
# TODO: use SnoopCompiler to find bottlenecks
#
# ======================================== Initialization ============================================
function __init__()

    global to = TimerOutput()
    global LOG_BUFFER = IOBuffer()
    global LOG = ""
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--config", "-c"
            help = "configuration file, default `config.toml`"
            arg_type = String
            default = "config.toml"
    end

    args = parse_args([], s)
    io = stdout
    global logger_console = ConsoleLogger(io, Logging.Info, meta_formatter=Logging.default_metafmt, show_limited=true, right_justify=0)
    global logger_file = SimpleLogger(LOG_BUFFER, Logging.Info)
    global logger = global_logger(TeeLogger(logger_console,logger_file))
end
