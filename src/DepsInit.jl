# ==================================================================================================== #
#                                           DepsInit.jl                                                #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
#   Last Edit Date  : 29.08.22                                                                         #
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
using JLD2, FileIO
using FFTW
using Combinatorics
using TOML          # used for input

# Fortran compatibility
using Printf, DelimitedFiles

using NLsolve

# lDGA related
using Dispersions
using BSE_SC

using TimerOutputs

using Base.Iterators

# ============================================= Includes =============================================
include("$(@__DIR__)/LapackWrapper.jl")
include("$(@__DIR__)/Config.jl")
include("$(@__DIR__)/DataTypes.jl")
include("$(@__DIR__)/parallelization_helpers.jl")
include("$(@__DIR__)/helpers.jl")
include("$(@__DIR__)/IO.jl")
include("$(@__DIR__)/GFTools.jl")
include("$(@__DIR__)/GFFit.jl")
include("$(@__DIR__)/ladderDGATools.jl")
include("$(@__DIR__)/ladderDGATools_singleCore.jl")
include("$(@__DIR__)/lambdaCorrection.jl")
include("$(@__DIR__)/LambdaCorrection/LambdaCorrection.jl")
include("$(@__DIR__)/thermodynamics.jl")

# ======================================== Internal Packages =========================================
using .LapackWrapper
using .LambdaCorrection

# =================================== Parallelization Bookkeeping ====================================
global_vars = String[]
wcache = WorkerCache()

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
    @add_arg_table s begin
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
