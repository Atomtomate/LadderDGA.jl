using LadderDGA
using Test

mP_1 = ModelParameters(1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7)
sP_1 = SimulationParameters(1,2,3,false,nothing,1:3,0.1,false)
kG_1 = gen_kGrid("2dsc-1.0",2)
kG_2 = gen_kGrid("2dsc-0.25",100)

χ_1 = χT([1.1 1.2 1.1; 2.1 2.3 2.1])
χ_2 = χT([-1.1 1.2 1.1; 2.1 2.3 2.1], full_range=false, reduce_range_prct=0.0)


@testset "Config" begin
    include("Config.jl")
end

@testset "DataTypes" begin
    include("DataTypes.jl")
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

@testset "parallelization" begin
    include("parallelization_helpers.jl")
end

@testset "full run" begin
    #include("full_run.jl")
end

include("LambdaCorrection/runtests.jl")
