@testset "λ_result" begin
    # stub not converged and converged m/dm results
    l1_m = λ_result(1.0, 0.0, LadderDGA.LambdaCorrection.mCorrection, false,
                    0.02, NaN, #eps
                    0.0, 0.0,  #ekin
                    0.01, 0.04, 0.01, 0.04, #epot_1, epot_2, pp_1, pp_2
                    nothing, nothing, nothing, 0.0, 0.0, 0., 0.0, 0.0)
    l1_dm = λ_result(1.0, 0.0, LadderDGA.LambdaCorrection.dmCorrection, false,
                    0.02, NaN, 0.0, 0.0, 
                    0.01, 0.04, 0.01, 0.04,
                    nothing, nothing, nothing, 0.0, 0.0, 0.0, 0.0, 0.0)
    l2_m = λ_result(1.0, 0.0, LadderDGA.LambdaCorrection.mCorrection, true,
                    0.02, NaN, 0.0, 0.0, 
                    0.01, 0.02, 0.02, 0.04,
                    nothing, nothing, nothing, 0.0, 0.0, 0.0, 0.0, 0.0)
    l2_dm = λ_result(1.0, 0.0, LadderDGA.LambdaCorrection.dmCorrection, true,
                    0.02, NaN, 0.0, 0.0, 
                    0.04, 0.03, 0.01, 0.02,
                    nothing, nothing, nothing, 0.0, 0.0, 0.0, 0.0, 0.0)
    @test !converged(l1_m)
    @test !converged(l1_dm)
    @test converged(l2_m)
    @test converged(l2_dm)
    @test !sc_converged(l1_m)
    @test !sc_converged(l1_dm)
    @test sc_converged(l2_m)
    @test sc_converged(l2_dm)
end
