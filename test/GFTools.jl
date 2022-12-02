@testset "MatsubaraFreq" begin
    @test all(LadderDGA.iν_array(5.0, 0:2) .≈ 1im .* [(2*0+1)*π/5.0, (2*1+1)*π/5.0, (2*2+1)*π/5.0 ])
    @test all(LadderDGA.iν_array(5.0, 0:2) .≈ LadderDGA.iν_array(5.0, 3))
    @test all(LadderDGA.iω_array(5.0, 0:2) .≈ 1im .* [(2*0)*π/5.0, (2*1)*π/5.0, (2*2)*π/5.0 ])
    @test all(LadderDGA.iω_array(5.0, 0:2) .≈ LadderDGA.iω_array(5.0, 3))
end

@testset "AndersomParamHelpers" begin
    @test LadderDGA.Δ([1.0], [1.1], [1.2im])[1] ≈ 1.1^2/(1.2im - 1.0)
end

@testset "G_from_Σ" begin
    mf_t0 = (1im * π / mP_1.β)
    mf_t1 = (3im * π / mP_1.β)
    @test G_from_Σ(1, mP_1.β, 1.2, 1.3, 1.4 + 0.0im) ≈ 1/(mf_t1 + 1.2 - 1.3 - 1.4)
    @test G_from_Σ(mf_t1, 1.2, 1.3, 1.4 + 0.0im) ≈ 1/(mf_t1 + 1.2 - 1.3 - 1.4)
    @test all(isapprox.(G_from_Σ([1.1 + 0.0im], [1.2], [0,1], mP_1, μ = 1.4, Σloc = [0.0, 1.3 + 0.0im]),
                    [1/(mf_t0 + 1.4  - 1.2 - 1.1) 1/(mf_t1 + 1.4 - 1.2 - 1.3)], atol=0.0001))
    @test LadderDGA.Σ_Dyson([1.1 + 0.0im], [1.2 + 0.0im])[1] ≈ 1/1.1 - 1/1.2
end

@testset "filling" begin
    νnGrid = LadderDGA.iν_array(mP_1.β, -50:49)
    G = G_from_Σ(zeros(ComplexF64,length(νnGrid)), LadderDGA.dispersion(kG_2), -50:49, mP_1)
    @test filling(G, νnGrid, kG_1, mP_1.β) > 0.0
end
