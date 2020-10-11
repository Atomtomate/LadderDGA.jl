using Test

@testset "Sereies Convergence" begin
end

@testset "UnitTests" begin
    t_π = [4*(-1)^n / (2*n + 1) for n in 0:4999] # -> π
    t_π2 = [6/n^2 for n in 1:5000] # -> π^2
    @test Shanks.shanks(t_π[1:100]) ≈ π
end
