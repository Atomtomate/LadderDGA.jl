
@testset "input" begin
    #TODO: test readFortranSymmGF, expect:
    #readFortranSymmGF!(zeros(Complex{Float64}, 4), "../ladderDGA/g0mand", true)
        # 4-element Array{Complex{Float64},1}:
        # 1.0 - 0.548892285im 
        # 1.0 - 0.3749619418im
        # 1.0 + 0.3749619418im
        # 1.0 + 0.548892285im
    #readFortranSymmGF!(zeros(Complex{Float64}, 4), "../ladderDGA/g0mand", true)
        # 4-element Array{Complex{Float64},1}:
        # 1.0 + 0.3749619418im
        # 1.0 + 0.548892285im
        # 1.0 + 0.7603915921im
        # 1.0 + 0.987953651im
    # readFortranSymmGF!(zeros(Complex{Float64}, 3), "../ladderDGA/g0mand", true)
        # ERROR: BoundsError
end

