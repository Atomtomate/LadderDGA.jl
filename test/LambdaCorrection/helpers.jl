@testset "χ_λ" begin
    χ_1 = [1.0 2.0 1.0]
    χ_2 = [1.0 2.0 1.0] .+ 0.0im
    χ_3 = χT(χ_2, 1.0)
    @test χ_λ(χ_1[1], 0.0) == χ_1[1]
    @test χ_λ(χ_2[1], 0.0) == χ_2[1]
    @test χ_λ(χ_1[1], 1.0) == χ_1[1]/2
    @test χ_λ(χ_2[1], 1.0) == χ_2[1]/2
    tt = χ_λ(χ_3, 1.1)
    χ_λ!(χ_3, 1.1)
    @test all(χ_3.data .≈ tt.data)
    LadderDGA.reset!(χ_3)
    χ_λ!(χ_3, χ_3, 1.1)
    @test all(χ_3.data .≈ tt.data)
    LadderDGA.reset!(χ_3)
end

@testset "Specialized Root Finding" begin
    f(x) = 2x^3 + x - 3  # real root at x = 1
    df(x) = 6x^2 + 1 
    f2(x) = [2x[1]^3 + x[1] - 3,x[2]]  # real root at x = [1,0]
    df2(x) = [6x^2 + 1, 1] 
    @test newton_right(f, df, -1.0) ≈ 1.0
    @test all(newton_right(f2, [-1.0, -1.0]) .≈ [1.0, 0.0])
end
