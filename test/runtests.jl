using Test

@testset "ladderDGA_Julia.jl" begin
    # Write your own tests here.
end


tests = ["IO","ladderDGATools"] 
for t in tests
    print("running $t\n")
    include("$(t).jl")
end
