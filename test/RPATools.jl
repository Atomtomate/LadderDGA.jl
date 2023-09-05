@testset "RPA" begin

    RPAHelper_1 = setup_RPA(("2dsc-1.0",1), mP_2, sP_2)
    @test typeof(RPAHelper_1) === RPAHelper
    @testset "bubble" begin
        bubble_RPA = calc_bubble(:RPA, RPAHelper_1)
        #bubble_RPA_exact = calc_bubble(:RPA_exact, RPAHelper_1)
        #@test typeof()
    end

end