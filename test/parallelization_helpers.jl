@testset "partitions" begin
    @test all(all.(LadderDGA.par_partition(1:30, 3) .== [1:10,11:20,21:30]))
end
