@testset "MatsubaraFunction" begin
    td = Float64.(reshape(1:12,3,4))
    t = LadderDGA.χT(td, 1.0);
    @test all(size(t) .== (3,4))
    @test t[1] ≈ 1
    @test t[1,1] ≈ 1
    t[1] = -1
    @test t[1] ≈ -1
    t[1,1] = -11
    @test t[1] ≈ t[1,1]
    @test t[1] ≈ -11
    @test t.β ≈ 1.0
end

@testset "χ₀T" begin
    td = ComplexF64.(reshape(1:24,3,2,4))
    t1 = ComplexF64.([1,2,3])
    t = LadderDGA.χ₀T(:local, td, kG_1, 1:4, 1, true, sP_1, mP_1)
    @test_throws ArgumentError LadderDGA.χ₀T(:local, td, kG_1, 1:-1, 1, true, sP_1, mP_1)
    c1, c2, c3 = LadderDGA.χ₀Asym_coeffs(:local, kG_1, mP_1; sVk=sP_1.sVk)
    @test c1 ≈ mP_1.U*mP_1.n/2 - mP_1.μ
    @test c2[1] ≈ c1^2
    @test c3 ≈ c1*c1 + sP_1.sVk + (mP_1.U^2)*(mP_1.n/2)*(1-mP_1.n/2)
    @test all(t.data .≈ td)
    @test t[1,1,1] ≈ 1
    t[1,1,1] = -1
    @test t[1,1,1] ≈ -1
end

@testset "χT" begin
    t = LadderDGA.χT(randn(3,4), 1.0);
    t2 = LadderDGA.χT(randn(3,5), 1.2, tail_c=[0,0,1.0]);
    ωn = collect(2im .* (-100:100) .* π ./ 11.1)
    t3 = zeros(ComplexF64, 4,length(ωn))
    kG = LadderDGA.gen_kGrid("3Dsc-1.1",2)
    t3[1,:] = 1.2 ./ ωn .+ 2.3 ./ ωn .^ 2
    t3[2,:] = 1.2 ./ ωn .+ 2.3 ./ ωn .^ 2
    t3[3,:] = 1.2 ./ ωn .+ 2.3 ./ ωn .^ 2
    t3[4,:] = 1.2 ./ ωn .+ 2.3 ./ ωn .^ 2
    t3[:,101] .= 0.0
    t[1,2] = 0.1
    @test t[1,2] ≈ 0.1
    @test t.tail_c == []
    @test all(t2.tail_c .== [0,0,1.0])
    @test all(t2.indices_ω .== -2:2)

    χ_test  = χT(real.(t3), 1.0, tail_c = [0.0, 0.0, 2.3])
    χ_test2 = χT(real.(t3), 1.0, tail_c = [0.0, 0.0, 0.0])
    @test real(χ_test.data[1,:] .* ωn .^ 1)[end] ≈ 0.0
    @test real(χ_test.data[1,:] .* ωn .^ 2)[end] ≈ 2.3
    @test_throws ArgumentError LadderDGA.update_tail!(χ_test, [0, 1.0, 4.0], ωn)
    LadderDGA.update_tail!(χ_test, [0, 0.0, 4.0], ωn)
    @test real(χ_test.data[1,:] .* ωn .^ 1)[end] ≈ 0.0
    @test real(χ_test.data[1,:] .* ωn .^ 2)[end] ≈ 4.0
    @test all(χ_test.tail_c .== [0,0.0,4.0])
    @test all(isfinite.(χ_test.data))

    χ_test  = χT(real.(t3), 1.0, tail_c = [0.0, 0.0, 2.3])
    test_sum_direct = real.(sum(t3, dims=2)[:,1])
    test_sum_direct_tf = sum(map(x->x^3,real(t3)), dims=2)[:,1]
    @test all(ωn_grid(t2) .≈ 2 .* π .* 1im .* (-2:2) ./ 1.2)
    @test all(sum_ω(χ_test2) .≈ test_sum_direct)
    @test kintegrate(kG, test_sum_direct) ≈ sum_kω(kG, χ_test2)
    @test kintegrate(kG, test_sum_direct) ≈ sum_ωk(kG, χ_test2)
    @test kintegrate(kG, test_sum_direct_tf) ≈ sum_kω(kG, χ_test2, transform=x->x^3)
    @test abs(sum_kω(kG, χ_test) - sum_ωk(kG, χ_test)) < abs(sum_kω(kG, χ_test))/100
end

@testset "γT" begin
    t = LadderDGA.γT(randn(ComplexF64, 3,4,5));
    t[1,2,3] = 0.1
    @test t[1,2,3] ≈ 0.1
end
