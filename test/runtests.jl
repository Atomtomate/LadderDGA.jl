using LadderDGA
using Test

mP_1 = ModelParameters(1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7)
sP_1 = SimulationParameters(1,2,3,false,nothing,1:3,0.1,false)
kG_1 = gen_kGrid("2dsc-1.0",2)
kG_2 = gen_kGrid("2dsc-0.25",100)


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

@testset "lambdaCorrection" begin
    include("lambdaCorrection.jl")
end


@testset "full run" begin
    #bubble_f, χch_f, χsp_f, trilexch_f, trilexsp_f, Σ_f = loadFortranData(dir)

    #bubble, χch, χsp, χsp_λ, usable_sp, usable_ch,trilexch, trilexsp, Σ_ladder = calculate_Σ_ladder("../config.toml")
    #@test all(bubble .≈ bubble_f)
    #@test all(χsp[usable_sp,:] .≈ χch_f[usable_sp,:])
    #@test all(χch[usable_ch,:] .≈ χch_f[usable_ch,:])
end
include("LambdaCorrection/runtests.jl")
