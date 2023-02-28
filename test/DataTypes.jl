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
    t = LadderDGA.χ₀T(td, kG_1, 1:4, 1, true, mP_1)
    @test all(t.data .≈ td)
    @test t[1,1,1] ≈ 1
    t[1,1,1] = -1
    @test t[1,1,1] ≈ -1
end

@testset "χT" begin
    t = LadderDGA.χT(randn(3,4), 1.0);
    t2 = LadderDGA.χT(randn(3,5), 1.2, tail_c=[0,0,1.0]);
    ωn = collect(2im .* (-100:100) .* π ./ 11.1)
    t3 = zeros(ComplexF64, 3,length(ωn))
    t3[1,:] = 1.2 ./ ωn .+ 2.3 ./ ωn .^ 2
    t3[2,:] = 1.2 ./ ωn .+ 2.3 ./ ωn .^ 2
    t3[3,:] = 1.2 ./ ωn .+ 2.3 ./ ωn .^ 2
    t3[:,101] .= 0.0
    t[1,2] = 0.1
    @test t[1,2] ≈ 0.1
    @test t.tail_c == []
    @test all(t2.tail_c .== [0,0,1.0])
    @test all(t2.indices_ω .== -2:2)

    χ_test = χT(real.(t3), 1.0, tail_c = [0.0, 0.0, 2.3])
    @test real(χ_test.data[1,:] .* ωn .^ 1)[end] ≈ 0.0
    @test real(χ_test.data[1,:] .* ωn .^ 2)[end] ≈ 2.3
    @test_throws ArgumentError LadderDGA.update_tail!(χ_test, [0, 1.0, 4.0], ωn)
    LadderDGA.update_tail!(χ_test, [0, 0.0, 4.0], ωn)
    @test real(χ_test.data[1,:] .* ωn .^ 1)[end] ≈ 0.0
    @test real(χ_test.data[1,:] .* ωn .^ 2)[end] ≈ 4.0
    @test all(χ_test.tail_c .== [0,0.0,4.0])
    @test all(isfinite.(χ_test.data))

    @test all(ωn_grid(t2) .≈ 2 .* π .* 1im .* (-2:2) ./ 1.2)
end

@testset "γT" begin
    t = LadderDGA.γT(randn(ComplexF64, 3,4,5));
    t[1,2,3] = 0.1
    @test t[1,2,3] ≈ 0.1
end

@testset "χT helpers" begin
    
end
