using Test
using Distributed
using SharedArrays
using JLD
using Optim
using TOML
using Printf
include("$(@__DIR__)/Config.jl")
include("$(@__DIR__)/helpers.jl")
include("$(@__DIR__)/IO.jl")
include("$(@__DIR__)/dispersion.jl")
include("$(@__DIR__)/GFTools.jl")
include("$(@__DIR__)/ladderDGATools.jl")
include("$(@__DIR__)/GFFit.jl")

@testset "ladderDGA_Julia.jl" begin
    # Write your own tests here.
end


tests = ["IO","ladderDGATools", "GFFit"] 
for t in tests
    print("running $t\n")
    include("$(t).jl")
end
