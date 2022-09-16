@testset "χ₀T" begin
end

@testset "χT" begin
    t = LadderDGA.χT(randn(ComplexF64, 3,4));
    t[1,2] = 0.1
    @test t[1,2] ≈ 0.1
end

@testset "γT" begin
    t = LadderDGA.γT(randn(ComplexF64, 3,4,5));
    t[1,2,3] = 0.1
    @test t[1,2,3] ≈ 0.1
end
