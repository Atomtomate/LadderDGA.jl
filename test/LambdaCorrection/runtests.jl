@testset "LambdaCorrection" begin
    @testset "helpers" begin
        include("helpers.jl")
    end

    @testset "conditions" begin
        include("conditions.jl")
    end

    @testset "residualCurves" begin
        include("residualCurves.jl")
    end
end
