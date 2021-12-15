module LapackWrapper
    using LinearAlgebra
    import LinearAlgebra.BLAS.@blasfunc
    import LinearAlgebra.LAPACK.getrf!
    import LinearAlgebra.LAPACK.getri!
    import LinearAlgebra.require_one_based_indexing
    import LinearAlgebra.chkstride1
    import LinearAlgebra.checksquare
    const libblastrampoline = "libblastrampoline"

    export getrf!, getri!
    export inv!, _gen_inv_work_arr

    function _gen_inv_work_arr(A::AbstractMatrix{T}, ipiv::Vector{Int}) where T <: Union{Float64,ComplexF64}
        n = size(A,1)
        lwork = Int(-1)
        work  = Vector{T}(undef, 1)
        info  = Ref{Int}()
        ccall((@blasfunc(dgetri_), libblastrampoline), Cvoid,
              (Ref{Int}, Ptr{T}, Ref{Int}, Ptr{Int},
               Ptr{T}, Ref{Int}, Ptr{Int}),
              n, A, n, ipiv, work, lwork, info)
        resize!(work, Int(real(work[1])))
    end

    function getrf!(A::Matrix{Float64}, ipiv::Vector{Int})
        n = size(A,1)
        ccall((@blasfunc(dgetrf_), libblastrampoline), Cvoid,
              (Ref{Int}, Ref{Int}, Ptr{Float64},
               Ref{Int}, Ptr{Int}, Ptr{Int}),
              n, n, A, stride(A,2), ipiv, Ref{Int}())
        A
    end

    function getri!(A::Matrix{Float64}, ipiv::Vector{Int}, work::Vector{Float64})
        info  = Ref{Int}()
        n = size(A,1)
        ccall((@blasfunc(dgetri_), libblastrampoline), Cvoid,
              (Ref{Int}, Ptr{Float64}, Ref{Int}, Ptr{Int},
               Ptr{Float64}, Ref{Int}, Ptr{Int}),
              n, A, n, ipiv, work, length(work), info)
        A
    end

    function getrf!(A::Matrix{ComplexF64}, ipiv::Vector{Int})
        n = size(A,1)
        ccall((@blasfunc(zgetrf_), libblastrampoline), Cvoid,
              (Ref{Int}, Ref{Int}, Ptr{ComplexF64},
               Ref{Int}, Ptr{Int}, Ptr{Int}),
              n, n, A, stride(A,2), ipiv, Ref{Int}())
        A
    end

    function getri!(A::Matrix{ComplexF64}, ipiv::Vector{Int}, work::Vector{ComplexF64})
        info  = Ref{Int}()
        n = size(A,1)
        ccall((@blasfunc(zgetri_), libblastrampoline), Cvoid,
              (Ref{Int}, Ptr{ComplexF64}, Ref{Int}, Ptr{Int},
               Ptr{ComplexF64}, Ref{Int}, Ptr{Int}),
              n, A, n, ipiv, work, length(work), info)
        A
    end

    function inv!(A::Matrix{Float64}, ipiv::Vector{Int}, work::Vector{Float64})
        getrf!(A, ipiv)
        getri!(A, ipiv, work)
    end

    function inv!(A::Matrix{ComplexF64}, ipiv::Vector{Int}, work::Vector{ComplexF64})
        getrf!(A, ipiv)
        getri!(A, ipiv, work)
    end

    function inv!(A::AbstractMatrix{T}, ipiv::Vector{Int}, work::Vector{T}) where T
        @warn "unkown type $(T), falling back to stdlib implementation"
        inv(A)
    end
end
