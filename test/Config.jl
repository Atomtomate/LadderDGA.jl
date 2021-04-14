@testset "ModelParameters" begin
    @test mP_1.U ≈ 1.1
    @test mP_1.μ ≈ 1.2
    @test mP_1.β ≈ 1.3
    @test mP_1.n ≈ 1.4
    @test mP_1.D == 5
end

@testset "SimulationParameters" begin
end
