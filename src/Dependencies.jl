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



s = ArgParseSettings()
@add_arg_table s begin
    "--config", "-c"
        help = "configuration file, default `config.toml`"
        arg_type = String
        default = "config.toml"
end
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
include("$(@__DIR__)/GFFit.jl")
include("$(@__DIR__)/lambdaCorrection.jl")
