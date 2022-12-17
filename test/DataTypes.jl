@testset "MatsubaraFunction" begin
    td = ComplexF64.(reshape(1:12,3,4))
    t = LadderDGA.χT(td);
    @test all(size(t) .== (3,4))
    @test t[1] ≈ 1
    @test t[1,1] ≈ 1
    t[1] = -1
    @test t[1] ≈ -1
    t[1,1] = -11
    @test t[1] ≈ t[1,1]
    @test t[1] ≈ -11
end

@testset "χ₀T" begin
    td = ComplexF64.(reshape(1:24,3,2,4))
    t1 = ComplexF64.([1,2,3])
    t = LadderDGA.χ₀T(td, kG_1, t1, 1.0, 2.0, 1:4, 1, 1)
    @test all(t.data .≈ td)
    @test t[1,1,1] ≈ 1
    t[1,1,1] = -1
    @test t[1,1,1] ≈ -1
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
