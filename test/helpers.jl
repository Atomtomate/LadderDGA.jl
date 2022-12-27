@testset "auxiliary functions" begin
    @test all(LadderDGA.reduce_range(1:3, 1.0) .== 2:2)
    @test all(LadderDGA.reduce_range(1:4, 1.0) .== 2:3)
    @test all(LadderDGA.reduce_range(1:5, 1.0) .== 3:3)
    @test all(LadderDGA.reduce_range(1:6, 1.0) .== 3:4)
    @test all(LadderDGA.reduce_range(1:10, 0.2) .== 2:9)
    @test all(LadderDGA.reduce_range(1:20, 0.1) .== 2:19)
    @test all(LadderDGA.reduce_range(1:21, 0.1) .== 2:20)
    @test all(LadderDGA.reduce_range(1:20, 0.5) .== 6:15)
    @test all(LadderDGA.reduce_range(1:21, 0.5) .== 6:16)
end

@testset "Index Functions" begin
    @test all(LadderDGA.νnGrid(4, sP_1) .== -7:2)
    @test LadderDGA.q0_index(kG_1) == 1
end
