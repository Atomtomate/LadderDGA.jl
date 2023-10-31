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
function freq_inbounds(ωi,νi,νpi,sP)
    !(any((ωi,νi,νpi) .< 1) || any((ωi,νi,νpi) .> (2*sP.n_iω,2*sP.n_iν-1,2*sP.n_iν-1)))
end

"""
    build_GG(GF::OffsetMatrix, νnGrid::AbstractVector{Int}, kVecs::AbstractVector{NTuple})::Matrix{ComplexF64}

Builds helper array `A`, defined as: ``A^{\\nu}_{k} = G^\\nu_{k} G^{-\\nu}_{-k}``.
Used, for example, by [`build_Γs`](@ref build_Γs).
"""
function build_GG(kG::KGrid, GF::OffsetMatrix, νnGrid::AbstractVector{Int}, k_vecs::AbstractVector)::Matrix{ComplexF64}
    res = Array{ComplexF64}(undef, length(νnGrid), length(k_vecs))

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
function build_q_access(kG::KGrid, k_vecs::AbstractVector)::Array{Int,2}
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


# ======================================== Main Functions ============================================
"""
    calc_λmax_linEliashberg(bubble::χ₀T, χm::χT, χd::χT, γm::γT, γd::γT, h::lDΓAHelper)

Calculates largest and smallest (real) eigen value of ``\\Gamma_{\\mathrm{s},\\uparrow\\downarrow}``.

TODO: fix version, either calculate version1 or 2!!!
TODO: TeX/DOCU..Ca
"""
function calc_λmax_linEliashberg(bubble::χ₀T, χm::χT, χd::χT, γm::γT, γd::γT, h::lDΓAHelper, env; GF=h.gLoc)
    ϕs, ϕt = jldopen(joinpath(env.inputDir, "DMFT_out.jld2"),"r") do f
        f["Φpp_s"], f["Φpp_t"]
    end;
    ϕs = permutedims(ϕs, [2,3,1]);
    ϕt = permutedims(ϕt, [2,3,1]);
    Phi_ud = 0.5 .* (ϕs .+ ϕt);

    lDGAhelper_Ur = deepcopy(h)
    lDGAhelper_Ur.Γ_m[:,:,:] = lDGAhelper_Ur.Γ_m[:,:,:] .- (-lDGAhelper_Ur.mP.U / lDGAhelper_Ur.mP.β^2)
    lDGAhelper_Ur.Γ_d[:,:,:] = lDGAhelper_Ur.Γ_d[:,:,:] .- ( lDGAhelper_Ur.mP.U / lDGAhelper_Ur.mP.β^2)
    χm_star_gen = calc_gen_χ(lDGAhelper_Ur.Γ_m, bubble, lDGAhelper_Ur.kG);
    χd_star_gen = calc_gen_χ(lDGAhelper_Ur.Γ_d, bubble, lDGAhelper_Ur.kG);

    Fm = F_from_χ_star_gen(bubble, χm_star_gen, χm, γm, -h.mP.U);
    Fd = F_from_χ_star_gen(bubble, χd_star_gen, χd, γd,  h.mP.U);
    Γs1, Γs2 = calc_Γs_ud(Fm, Fd, Phi_ud, h, GF)
    λ1L, _, _, _, _, _ = eigs(Γs1; nev=1, which=:LR, tol=1e-18);
    λ1S, _, _, _, _, _ = eigs(Γs1; nev=1, which=:LR, tol=1e-18);
    λ2L, _, _, _, _, _ = eigs(Γs2; nev=1, which=:SR, tol=1e-18);
    λ2S, _, _, _, _, _ = eigs(Γs2; nev=1, which=:SR, tol=1e-18);
    return λ1L,λ1S,λ2L,λ2S
end

"""
    calc_Γs_ud(Fm, Fd, Phi_ud, h::lDΓAHelper)

Calculates the Γs in particle-particle notation from the ladder vertices.  
"""
function calc_Γs_ud(Fm, Fd, Phi_ud, h::lDΓAHelper, GF::OffsetMatrix)
    cut_to_non_nan = true
    max_ν  = cut_to_non_nan ? trunc(Int, h.sP.n_iν/2) : h.sP.n_iν
    νnGrid = -(max_ν-1):(max_ν-2) #-1:0 #-
    kG = h.kG
    k_vecs = collect(Dispersions.gen_sampling(grid_type(kG), grid_dimension(kG), kG.Ns))
    v_full = collect(Dispersions.gen_sampling(grid_type(kG), grid_dimension(kG), kG.Ns))

    νlen = length(νnGrid)
    klen = length(k_vecs)
    ωn = 0; ωi = h.sP.n_iω+1

    Fm_loc = F_from_χ(:m, h);
    Fd_loc = F_from_χ(:d, h);
    Gνk_Gmνmk = build_GG(kG, GF, νnGrid, k_vecs[:])
    qi_access = build_q_access(kG, k_vecs[:]);

    @warn "TODO: currently calculating two versions of Γ_pp, until Fm_{k'-k} question is resolved"
    Γs_ladder1 = Array{ComplexF64, 2}(undef, length(k_vecs)*length(νnGrid), length(k_vecs)*length(νnGrid));
    fill!(Γs_ladder1, NaN + 1im * NaN)

    Γs_ladder2 = Array{ComplexF64, 2}(undef, length(k_vecs)*length(νnGrid), length(k_vecs)*length(νnGrid));
    fill!(Γs_ladder2, NaN + 1im * NaN)

    Fph_ladder_updo  = permutedims(0.5 .* Fd .- 1.5 .* Fm,[3,1,2,4]) .- reshape(0.5 .* Fd_loc .- 0.5 .* Fm_loc, 1, size(Fd_loc)...)
    Fph_ladder_updo2 = permutedims(0.5 .* Fd .- 0.5 .* Fm,[3,1,2,4]) .- reshape(0.5 .* Fd_loc .- 0.5 .* Fm_loc, 1, size(Fd_loc)...)


    for (νi,νn) in enumerate(νnGrid)
        for (νpi,νpn) in enumerate(νnGrid)      
            ωn_ν_minus_νp = trunc(Int, (2*νn+1 - (2*νpn+1))/2)   
            minus_ν       = trunc(Int, -(2*νn+1)/2 - 1)
            ωi_ladder, νi_ladder, νpi_ladder  = Freq_to_OneToIndex(ωn_ν_minus_νp, νpn, minus_ν, h.sP.shift, h.sP.n_iω, h.sP.n_iν)
            ν_plus_νp  = νn + νpn + 1
            ωi_ladder2,νi_ladder2,νpi_ladder2 = Freq_to_OneToIndex(-ν_plus_νp, νpn,  νn, h.sP.shift, h.sP.n_iω, h.sP.n_iν)
            νi_pp  = νn  + h.sP.n_iν+1; νpi_pp  = νpn + h.sP.n_iν+1; ωi_pp  = ωn  + h.sP.n_iω+1

            if freq_inbounds(ωi_ladder,νi_ladder,νpi_ladder,h.sP)
                for (ki,k_vec) in enumerate(k_vecs)
                    for (kpi,kp_vec) in enumerate(k_vecs)
                        G_mG = Gνk_Gmνmk[νpi, kpi]
                        qi = qi_access[ki,kpi]
                        qi2 = qi_access[kpi,ki]
                        Γs_ladder1[ki+length(k_vecs)*(νi-1),kpi+length(k_vecs)*(νpi-1)] = -(Fph_ladder_updo[qi,νi_ladder,νpi_ladder,ωi_ladder]  .- Phi_ud[νi_pp,νpi_pp,ωi_pp]) * G_mG  / (2 * kG.Nk * h.mP.β)
                        Γs_ladder2[ki+length(k_vecs)*(νi-1),kpi+length(k_vecs)*(νpi-1)] = -(Fph_ladder_updo2[qi,νi_ladder,νpi_ladder,ωi_ladder] .- Fm[νi_ladder2,νpi_ladder2,qi2,ωi_ladder2] .- Phi_ud[νi_pp,νpi_pp,ωi_pp]) * G_mG  / (2 * kG.Nk * h.mP.β)    
                    end
                end
            end 
        end
    end
    return Γs_ladder1, Γs_ladder2
end
