# ==================================================================================================== #
#                                     LinearizedEliashberg.jl                                          #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#   Construction and solution of Γs in particle-particle notation, constructed from full vertex        #
#   of λ-corrected DΓA
# -------------------------------------------- TODO -------------------------------------------------- #
#   Extend thsi to λ-RPA                                                                               #
# ==================================================================================================== #


# ============================================ Helpers ===============================================
"""
    build_GG(GF::OffsetMatrix, νnGrid::AbstractVector{Int}, kVecs::AbstractVector{NTuple})::Matrix{ComplexF64}

Builds helper array `A`, defined as: ``A^{\\nu}_{k} = G^\\nu_{k} G^{-\\nu}_{-k}``.
Used, for example, by [`build_Γs`](@ref build_Γs).
"""
function build_GG(GF::OffsetMatrix, νnGrid::AbstractVector{Int}, kVecs::AbstractVector{NTuple})::Matrix{ComplexF64}
    res = Array{ComplexF64}(undef, length(νnGrid), length(kVecs))

    for (νi,νn) in enumerate(νnGrid)
        GF_ν = expandKArr(kG, GF[:,νn].parent)
        GF_ν_minus_k_pre = expandKArr(kG, GF[:,-νn-1].parent)
        shift_vec = 2 .* kG.k0 .- gridshape(kG) .- 1
        GF_ν_minus_k = circshift(reverse(GF_ν_minus_k_pre), shift_vec)
        for (ki,k_vec) in enumerate(k_vecs)
            res[νi,ki] = GF_ν[ki]*GF_ν_minus_k[ki]
        end
    end
    return res
end


"""
    build_q_access(kG::KGrid, k_vecs::AbstractVector{NTuple})::Array{Int,2}

Builds helper array `A`, defined as: ``A^{\\nu}_{k} = G^\\nu_{k} G^{-\\nu}_{-k}``.
Used, for example, by [`build_Γs`](@ref build_Γs).
"""
function build_q_access(kG::KGrid, k_vecs::AbstractVector{NTuple})::Array{Int,2}
    q_lookup = build_q_lookup(kG)
    res = Array{Int,2}(undef, length(k_vecs), length(k_vecs))
    for (ki,k_vec) in enumerate(k_vecs)
        for (kpi,kp_vec) in enumerate(k_vecs)
            q_vec = round.(transform_to_first_BZ(kG, k_vec .- kp_vec), digits=6)                   
            res[ki,kpi] = q_lookup[q_vec]
        end
    end
    return res
end
