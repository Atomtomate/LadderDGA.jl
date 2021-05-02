using LadderDGA
using SeriesAcceleration
using Test

mP_1 = ModelParameters(1.1, 1.2, 1.3, 1.4, 5)
sP_1 = SimulationParameters(1,2,false,1,:nothing,:nothing,:common,:native,false,LadderDGA.zero_χ_fill,[0,1,2,3],[0,1,2,3])
sP_2 = SimulationParameters(1,2,false,1,:richardson,:nothing,:common,:native,false,LadderDGA.zero_χ_fill,[0,1,2,3],[0,1,2,3])

@testset "Config" begin
    include("Config.jl")
end

@testset "Helpers" begin
    include("helpers.jl")
end

@testset "GFFit" begin
    include("GFFit.jl")
end

@testset "ladderDGATools" begin
    include("ladderDGATools.jl")
end

@testset "full run" begin
    #bubble_f, χch_f, χsp_f, trilexch_f, trilexsp_f, Σ_f = loadFortranData(dir)

    #bubble, χch, χsp, χsp_λ, usable_sp, usable_ch,trilexch, trilexsp, Σ_ladder = calculate_Σ_ladder("../config.toml")
    #@test all(bubble .≈ bubble_f)
    #@test all(χsp[usable_sp,:] .≈ χch_f[usable_sp,:])
    #@test all(χch[usable_ch,:] .≈ χch_f[usable_ch,:])
end
