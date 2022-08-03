@testset "ModelParameters" begin
    @test mP_1.U ≈ 1.1
    @test mP_1.μ ≈ 1.2
    @test mP_1.β ≈ 1.3
    @test mP_1.n ≈ 1.4
    @test mP_1.sVk ≈ 1.5
    @test mP_1.Epot_DMFT ≈ 1.6 
    @test mP_1.Ekin_DMFT ≈ 1.7
end

@testset "SimulationParameters" begin
    @test sP_1.n_iω == 1
    @test sP_1.n_iν == 2
    @test sP_2.n_iν_shell == 3
    @test sP_1.shift == false
    @test sP_1.tc_type_f == :nothing
    @test sP_1.tc_type_b == :nothing
    @test sP_1.λc_type == :nothing
    @test sP_1.ωsum_type == :common
    @test sP_1.λ_rhs == :native
    @test sP_1.fullChi == false
    @test sP_1.usable_prct_reduction == 0.1
    @test sP_1.dbg_full_eom_omega == false
    @test sumExtrapolationHelper === nothing
end
