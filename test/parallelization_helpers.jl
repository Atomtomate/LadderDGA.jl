@testset "partitions" begin
    @test all(all.(LadderDGA.par_partition(1:30, 3) .== [1:10,11:20,21:30]))
end

@testset "gen_part" begin
    t = LadderDGA.gen_ν_part(0:sP_grid_s1.n_iν-1, sP_grid_s1, 3)
    test_data = randn(ComplexF64, 3, 2*sP_grid_s1.n_iν, 2*sP_grid_s1.n_iω+1)
    @test length(t) == 3
    data_seg1, νn_seg1, ωn_seg1 = LadderDGA.gen_ν_part_slices(test_data, t[1])
    @test all(νn_seg1 .== [0,1])
    @test all(ωn_seg1[1] .== -4:4)
end
