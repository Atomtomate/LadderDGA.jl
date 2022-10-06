@testset "Energies ED" begin
   Ek,Ep = calc_E_ED("test_data/b1u2_ED_data.jld2")
   @test Ek ≈ -0.1044800085350687  atol=0.001
   @test Ep ≈ 2*0.14176922807732420  atol=0.001
end
