@testset "MatsubaraFreq" begin
    @test all(LadderDGA.iν_array(5.0, 0:2) .≈ 1im .* [(2*0+1)*π/5.0, (2*1+1)*π/5.0, (2*2+1)*π/5.0 ])
    @test all(LadderDGA.iν_array(5.0, 0:2) .≈ LadderDGA.iν_array(5.0, 3))
    @test all(LadderDGA.iω_array(5.0, 0:2) .≈ 1im .* [(2*0)*π/5.0, (2*1)*π/5.0, (2*2)*π/5.0 ])
    @test all(LadderDGA.iω_array(5.0, 0:2) .≈ LadderDGA.iω_array(5.0, 3))
end

@testset "AndersomParamHelpers" begin
    @test LadderDGA.Δ([1.0], [1.1], [1.2im])[1] ≈ 1.1^2/(1.2im - 1.0)
end

@testset "G_from_Σ" begin
    #TODO: obtain gLoc/ΣLoc
    #TODO: @test all(gLoc_new .≈ gLoc)
end
