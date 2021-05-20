@testset "ModelParameters" begin
    @test mP_1.U ≈ 1.1
    @test mP_1.μ ≈ 1.2
    @test mP_1.β ≈ 1.3
    @test mP_1.n ≈ 1.4
end

@testset "SimulationParameters" begin
    @test sP_1.n_iω == 1
    @test sP_1.n_iν == 2
    @test sP_1.shift == false
    @test sP_1.tc_type == :nothing
    @test sP_1.λc_type == :nothing
    @test sP_1.ωsum_type == :common
    @test sP_1.λ_rhs == :native
    @test sP_1.fullChi == false
    @test sP_1.χFillType == LadderDGA.zero_χ_fill
    @test sP_1.bosonic_tail_coeffs == [0,1,2,3]
    @test sP_1.fermionic_tail_coeffs == [0,1,2,3]
    @test sP_1.usable_prct_reduction == 0.1
end
