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
    @test all(isapprox.(G_from_Σ(OffsetVector([1.1 + 0.0im],0:0), [1.2], 0:0, mP_1, μ = 1.4),
                        [1/(mf_t0 + 1.4  - 1.2 - 1.1)], atol=0.0001))
    @test all(isapprox.(G_from_Σ(OffsetVector([1.1 + 0.0im], 0:0), [1.2], 0:1, mP_1, μ = 1.4, Σloc = OffsetVector([0.0, 1.3 + 0.0im],0:1)).parent,
                    [1/(mf_t0 + 1.4  - 1.2 - 1.1) 1/(mf_t1 + 1.4 - 1.2 - 1.3)], atol=0.0001))
    @test LadderDGA.Σ_Dyson([1.1 + 0.0im], [1.2 + 0.0im])[1] ≈ 1/1.1 - 1/1.2
end


function G_shell_sum_naive(iν_array::Vector{ComplexF64}, β::Float64)::Float64
    real(sum(1 ./ (iν_array) .^ 2))/β + β/4
end

@testset "filling" begin
    νnGrid = LadderDGA.iν_array(11.1, -50:49)
    G = G_from_Σ(OffsetVector(zeros(ComplexF64,length(νnGrid)),-50:49), LadderDGA.dispersion(kG_1), 0:49, mP_1)
    @test G_shell_sum_naive(νnGrid, 11.1) ≈ LadderDGA.G_shell_sum(50, 11.1) rtol=0.01
    @test filling(G, kG_1, 1.1, 1.2, 1.3) ≈ 1.1706318843228878
    @test filling_pos(G, kG_1, 1.1, 1.2, 1.3) ≈ 1.3378494616162329 rtol=0.01
end


