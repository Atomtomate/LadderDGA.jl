@testset "sum_νmax" begin
    test2d = reshape(1:80, (10,8))
    test3d = reshape(1:800, (10,8,10))
    @test all(sum_νmax(test2d, 10; dims=1) .== sum(test2d; dims=1))
    @test all(sum_νmax(test2d, 8; dims=2) .== sum(test2d; dims=2)[:,1])
    @test all(sum_νmax(test2d, 8; dims=[1,2]) .== sum(test2d[1:8,1:8]; dims=[1,2]))
    @test all(sum_νmax(test3d, 6; dims=[1,2,3]) .== sum(test3d[1:6,1:6,1:6]; dims=[1,2,3]))
    @test all(sum_νmax(test3d, 6; dims=[2,3]) .== sum(test3d[:,1:6,1:6]; dims=[2,3]))
    @test all(sum_νmax(test3d, 6; dims=[1,3]) .== sum(test3d[1:6,:,1:6]; dims=[1,3]))
    @test all(sum_νmax(test3d, 6; dims=[1,2]) .== sum(test3d[1:6,1:6,:]; dims=[1,2]))
    @test all(sum_νmax(test3d, 6; dims=[1]) .== sum(test3d[1:6,:,:]; dims=[1]))
    @test all(sum_νmax(test3d, 6; dims=[2]) .== sum(test3d[:,1:6,:]; dims=[2]))
    @test all(sum_νmax(test3d, 6; dims=[3]) .== sum(test3d[:,:,1:6]; dims=[3]))
end

