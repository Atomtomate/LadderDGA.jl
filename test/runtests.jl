using FortranFiles
using Test

include("loadFortranData.jl")
include("../src/ladderDGA_Julia.jl")

@testset "full run" begin
    dir = "/home/julian/Hamburg/ladderDGA3D_FFT"
    bubble_f, χch_f, χsp_f, trilexch_f, trilexsp_f, Σ_f = loadFortranData(dir)

    bubble, χch, χsp, χsp_λ, usable_sp, usable_ch,trilexch, trilexsp, Σ_ladder = calculate_Σ_ladder("../config.toml")
    @test all(bubble .≈ bubble_f)
    @test all(χsp[usable_sp,:] .≈ χch_f[usable_sp,:])
    @test all(χch[usable_ch,:] .≈ χch_f[usable_ch,:])
end
