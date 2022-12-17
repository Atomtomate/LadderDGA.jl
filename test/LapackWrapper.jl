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
            A3 = deepcopy(A)
            A,ipiv,_ = LinearAlgebra.LAPACK.getrf!(A)
            ipiv2 = deepcopy(ipiv)
            ipiv3 = deepcopy(ipiv)
            work2 = _gen_inv_work_arr(A, ipiv3)

            getrf!(A2, ipiv)
            
            @test (all(A .≈ A2))
            @test (all(ipiv .≈ ipiv2))

            LinearAlgebra.LAPACK.getri!(A, ipiv)

            lwork = Int(-1)
            work  = Vector{T}(undef, 10*size(A,1))
            getri!(A2, ipiv2, work)
            @test (all(A .≈ A2))
            inv!(A3, ipiv3, work2)
            @test all(A2 .≈ A3)
        end
    end
end

            
