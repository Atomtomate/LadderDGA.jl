using LadderDGA
using Test
using OffsetArrays

mP_1 = ModelParameters(1.1, 1.2, 1.3, 1.4, 1.6, 1.7)
mP_2 = ModelParameters(1.3, 1.3/2, 5.0, 1.0, 1.6, 1.7)
sP_1 = SimulationParameters(1,2,3,true,nothing,1.5, 1:3,0.1,false)
sP_2 = SimulationParameters(10,10,0,true,nothing,NaN,-20:20,0.0,true)
kG_0 = gen_kGrid("2dsc-1.0",1)
kG_1 = gen_kGrid("2dsc-1.1",4)
kG_2 = gen_kGrid("2dsc-0.25",100)

χ_1 = χT([1.1 1.2 1.1; 2.1 2.3 2.1], 1.0)
χ_2 = χT([-1.1 1.2 1.1; 2.1 2.3 2.1], 1.0, full_range=false, reduce_range_prct=0.0)
sP_grid_s1 = SimulationParameters(4,5,2,true,nothing,1.5,1:3,0.1,false)
sP_grid_s0 = SimulationParameters(4,5,2,false,nothing,1.5,1:3,0.1,false)
ωνgrid_test_s1 = [(i,j - trunc(Int,1*i/2)) for i in -4:4, j in -5:5-1]
ωνgrid_test_s0 = [(i,j - trunc(Int,0*i/2)) for i in -4:4, j in -5:5-1]


@testset "Config" begin
    include("Config.jl")
end

@testset "DataTypes" begin
    include("DataTypes.jl")
end

@testset "Run Helpers" begin
    include("runHelpers.jl")
end

@testset "Helpers" begin
    include("helpers.jl")
end

@testset "GFTools" begin
    include("GFTools.jl")
end

@testset "GFFit" begin
    include("GFFit.jl")
end

@testset "LapackWrapper" begin
    include("LapackWrapper.jl")
end

@testset "thermodynamics" begin
    include("thermodynamics.jl")
end

@testset "ladderDGATools" begin
    include("ladderDGATools.jl")
end

@testset "RPA Tools" begin
    include("RPATools.jl")
end

@testset "parallelization" begin
    include("parallelization_helpers.jl")
end

include("LambdaCorrection/runtests.jl")

@testset "full run" begin
    include("full_run.jl")
end

