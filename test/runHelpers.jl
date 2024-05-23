@testset "RPAHelper" begin
    rpahelper = setup_RPA(("2dsc-1.0", 2), mP_1, sP_1)
    @test typeof(rpahelper) === RPAHelper
    @test rpahelper.sP === sP_1
    @test rpahelper.mP === mP_1
    @test rpahelper.kG.Ns == 2
end
