@testset "auxiliary functions" begin
    test = [reshape((1:18) .+ i,3,6) for i in 1:8];
    test_fl = LadderDGA.flatten_2D(test)
    res = true
    for i in 1:8
        !(test[i] == reshape(test_fl[i,:],3,6)) && (res = false)
    end
    @test res
end
