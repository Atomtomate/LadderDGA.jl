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
    ν0_ind_s0 = map(x-> LadderDGA.ν0Index_of_ωIndex(x, sP_grid_s0), 1:2*sP_grid_s0.n_iω+1)
    i_s0 = CartesianIndex.(zip(1:length(ν0_ind_s0),ν0_ind_s0))
    ν0_ind_s1 = map(x-> LadderDGA.ν0Index_of_ωIndex(x, sP_grid_s1), 1:2*sP_grid_s1.n_iω+1)
    i_s1 = CartesianIndex.(zip(1:length(ν0_ind_s1),ν0_ind_s1))
    @test all(map(x->x[2],ωνgrid_test_s0[i_s0]) .== 0)
    @test all(map(x->x[2],ωνgrid_test_s1[i_s1]) .== 0)
end
