using LadderDGA.LapackWrapper
using LinearAlgebra

function inv_ref!(A::Matrix)
    A,ipiv,_ = LinearAlgebra.LAPACK.getrf!(A)
    LinearAlgebra.LAPACK.getri!(A, ipiv)
end

@testset "inv!" begin
    for T in [Float64, ComplexF64]
        for N in 8:13
            A = randn(T,N,N)
            A2 = deepcopy(A)
            A,ipiv,_ = LinearAlgebra.LAPACK.getrf!(A)
            ipiv2 = deepcopy(ipiv)

            getrf!(A2, ipiv)
            
            @test (all(A .≈ A2))
            @test (all(ipiv .≈ ipiv2))

            LinearAlgebra.LAPACK.getri!(A, ipiv)

            lwork = Int(-1)
            work  = Vector{T}(undef, 10*size(A,1))
            getri!(A2, ipiv2, work)
            @test (all(A .≈ A2))
        end
    end
end
