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
    t2 = LadderDGA.χT(randn(ComplexF64, 3,4), tail_c=[0,0,1.0]);
    ωn = collect(2im .* (-100:100) .* π ./ 11.1)
    t3 = zeros(ComplexF64, 3,length(ωn))
    t3[1,:] = 1.2 ./ ωn .+ 2.3 ./ ωn .^ 2
    t3[2,:] = 1.2 ./ ωn .+ 2.3 ./ ωn .^ 2
    t3[3,:] = 1.2 ./ ωn .+ 2.3 ./ ωn .^ 2
    t[1,2] = 0.1
    @test t[1,2] ≈ 0.1
    @test t.tail_c == []
    @test all(t2.tail_c .== [0,0,1.0])

    χ_test = χT(t3, tail_c = [0.0, 1.2, 2.3])
    @test real(χ_test.data[1,:] .* ωn .^ 1)[end] ≈ 1.2
    @test real(χ_test.data[1,:] .* ωn .^ 2)[end] ≈ 2.3
    LadderDGA.update_tail!(χ_test, [0, 1.2, 4.0], ωn)
    @test real(χ_test.data[1,:] .* ωn .^ 1)[end] ≈ 1.2
    @test real(χ_test.data[1,:] .* ωn .^ 2)[end] ≈ 4.0
    println(χ_test.tail_c)
    @test all(χ_test.tail_c .== [0,1.2,4.0])
    @test all(isfinite.(χ_test.data))
end

@testset "γT" begin
    t = LadderDGA.γT(randn(ComplexF64, 3,4,5));
    t[1,2,3] = 0.1
    @test t[1,2,3] ≈ 0.1
end
