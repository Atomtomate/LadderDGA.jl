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
    @test G_from_Σ(-1im*2/5, 1.0/15.0, -1.0/30.0, -1.0/10.0 + 0*1im) ≈ 2*1im + 1
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


@testset "core and shell sums" begin
    β = 12.34
    for N in [2,11,100]
        for p in [2,3,4]
            s_naive_f = sum((1/(1im*(2*n+1)*π/β))^p for n in 0:N)
            s_naive_b = sum((1/(1im*(2*n+0)*π/β))^p for n in 1:N)
            s_int_f = LadderDGA.core_sum_fermionic(N, β, p)
            s_int_b = LadderDGA.core_sum_bosonic(N, β, p)
            @test s_naive_f ≈ s_int_f
            @test s_naive_b ≈ s_int_b
        end
        s_naive_f = -sum((1/(1im*(2*n+1)*π/β))^2 for n in -N:(N-1))/β - β/4
        naive_s = -2 * (1/β) * LadderDGA.core_sum_fermionic(N-1, β, 2) - β/4
        @test s_naive_f ≈ naive_s
        @test naive_s ≈ 2*LadderDGA.shell_sum_fermionic(N, β, 2)/β
    end
end

@testset "filling" begin
    N = 2000
    β = 12.34
    U = 1.0
    kG_1 = gen_kGrid("2dsc-1.0",10)
    νnGrid = LadderDGA.iν_array(β, -N:(N-1))
    μ = 0.0 
    mP_3 = ModelParameters(U, μ, β, 1.4, 1.6, 1.7)
    G = G_from_Σ(OffsetVector(zeros(ComplexF64,length(νnGrid)),-N:(N-1)), LadderDGA.dispersion(kG_1), 0:(N-1), mP_3)
    @test filling_pos(G, kG_1, U, μ, β, improved_sum=false) ≈ 1.0 atol = 1e-1
    t1 = filling(G, kG_1, U, μ, β)
    t2 = filling_pos(G, kG_1, U, μ, β)
    @test t1 ≈ 1.0 atol = 1e-3
    @test t2 ≈ 1.0 atol = 1e-3
    @test t1 ≈ t2
end


