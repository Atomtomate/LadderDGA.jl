@testset "χ_λ" begin
    χ_1 = [1.0 2.0 1.0]
    χ_3 = χT(χ_1, 1.0)
    @test χ_λ(χ_1[1], 0.0) == χ_1[1]
    @test χ_λ(χ_1[1], 1.0) == χ_1[1]/2
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
    @test newton_right(f, df, 0.0, -1.0) ≈ 1.0
    @test newton_right(f, 0.0, -1.0) ≈ 1.0
    @test newton_right(f, df, -0.9, -1.0) ≈ 1.0
    @test all(newton_right(f2, [0.0, 0.0], [-1.0, -1.0]) .≈ [1.0, 0.0])
end

@testset "sample f" begin
    f1(x) = 2.0*x + 1.1
    f2(x) = x^2 * sin(x)
    @test LadderDGA.LambdaCorrection.linear_approx(f1(1.0), f1(2.0), 1.0, 2.0, 1.5) ≈ f1(1.5) 
    xvals, yvals = sample_f(f2, -1.01, 5.0, feps_abs=1e-5, xeps_abs=1e-8)
    @test all(yvals .≈ f2.(xvals))
end

@testset "newton_secular" begin
    f1(x) = 8.3/(0.1 - x) + 5.4/(2.2 - x) + 1.3
    df1(x) = 8.3/(0.1 -x)^2 + 5.4/(2.2-x)^2
    f4(x) = 2.0*(1/x) - 1.1
    df4(x) = - 2.0/x^2

    @test LadderDGA.LambdaCorrection.newton(f4,df4, 1.2) ≈ 2.0/1.1
    @test LadderDGA.LambdaCorrection.newton_secular(f4,df4,0.01) ≈ 2.0/1.1
    @test LadderDGA.LambdaCorrection.newton_right(f4,df4,1.2,-1.0) ≈ 2.0/1.1
    @test LadderDGA.LambdaCorrection.newton_secular(f4,0.01) ≈ 2.0/1.1
    @test LadderDGA.LambdaCorrection.newton_right(f4,1.2,-1.0) ≈ 2.0/1.1

    res_t1 = 11.569472045821662
    @test LadderDGA.LambdaCorrection.newton_secular(f1,df1,2.2) ≈ res_t1
    @test LadderDGA.LambdaCorrection.newton_right(f1,df1,2.3,2.2) ≈ res_t1
    @test LadderDGA.LambdaCorrection.newton_secular(f1,2.2) ≈ res_t1
    @test LadderDGA.LambdaCorrection.newton_right(f1,2.3,2.2) ≈ res_t1
end
