using Test

@setset "cut_mirror" begin
    , kG  = gen_kGrid(4,2)
    , kG2 = gen_kGrid(6,2)
    @test all(expand_mirror(cut_mirror(kG)) .== collect(kG))
    @test all(expand_mirror(cut_mirror(kG2)) .== collect(kG2))
end
