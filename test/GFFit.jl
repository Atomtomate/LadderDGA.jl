@testset "build fν max" begin
    @test all(LadderDGA.build_fνmax_fast([3,2,1,2,3],1) .== [1, 5, 11])
    @test all(LadderDGA.build_fνmax_fast([3,2,1,2,3],2) .== [5, 11])
    @test all(LadderDGA.build_fνmax_fast([3,2,1,2,3],3) .== [11])
    @test all(LadderDGA.build_fνmax_fast([3,2,1,1,2,3],1) .== [2, 6, 12])
    @test all(LadderDGA.build_fνmax_fast([3,2,1,1,2,3],2) .== [6, 12])
    @test all(LadderDGA.build_fνmax_fast([3,2,1,1,2,3],3) .== [12])
    a = [3 3 3 3 3; 3 2 2 2 3; 3 2 1 2 3; 3 2 2 2 3; 3 3 3 3 3]
    test = [1, 1+8*2, 1+8*2+16*3]
    @test all(LadderDGA.build_fνmax_fast(a, 1) .== test)
    @test all(LadderDGA.build_fνmax_fast(a, 2) .== test[2:3])
    @test all(LadderDGA.build_fνmax_fast(a, 3) .== test[3])
end

@testset "get_sum_helper" begin
#    @test typeof()
end
