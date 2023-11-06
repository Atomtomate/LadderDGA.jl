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
    qi_access = Array{Int,2}(undef, length(k_vecs), length(k_vecs))
    for (ki,k_vec) in enumerate(k_vecs)
        for (kpi,kp_vec) in enumerate(k_vecs)
            q_vec = round.(Dispersions.transform_to_first_BZ(kG, -1 .* k_vec .- kp_vec), digits=6) 
            q_vec = map(x-> x ≈ 0 ? 0.0 : x, q_vec)
            qi_access[ki,kpi] = q_lookup[q_vec]
        end
    end
    return qi_access
end


# ======================================== Main Functions ============================================
"""
    calc_λmax_linEliashberg(bubble::χ₀T, χm::χT, χd::χT, γm::γT, γd::γT, h::lDΓAHelper, env;
                             GF=h.gLoc, max_Nk::Int=h.kG.Ns, χm_star_gen=nothing, χd_star_gen=nothing)

Calculates largest and smallest (real) eigen value of ``\\Gamma_{\\mathrm{s},\\uparrow\\downarrow}``.

TODO: TeX/DOCU...
"""
function calc_λmax_linEliashberg(bubble::χ₀T, χm::χT, χd::χT, γm::γT, γd::γT, h::lDΓAHelper, env; GF=h.gLoc, max_Nk::Int=h.kG.Ns, χm_star_gen=nothing, χd_star_gen=nothing)
    ϕs, ϕt = jldopen(joinpath(env.inputDir, "DMFT_out.jld2"),"r") do f
        f["Φpp_s"], f["Φpp_t"]
    end;
    ϕs = permutedims(ϕs, [2,3,1]);
    ϕt = permutedims(ϕt, [2,3,1]);
    Phi_ud = 0.5 .* (ϕs .+ ϕt);

    if isnothing(χm_star_gen) || isnothing(χd_star_gen)
        lDGAhelper_Ur = deepcopy(h)
        lDGAhelper_Ur.Γ_m[:,:,:] = lDGAhelper_Ur.Γ_m[:,:,:] .- (-lDGAhelper_Ur.mP.U / lDGAhelper_Ur.mP.β^2)
        lDGAhelper_Ur.Γ_d[:,:,:] = lDGAhelper_Ur.Γ_d[:,:,:] .- ( lDGAhelper_Ur.mP.U / lDGAhelper_Ur.mP.β^2)
        χm_star_gen = calc_gen_χ(lDGAhelper_Ur.Γ_m, bubble, lDGAhelper_Ur.kG);
        χd_star_gen = calc_gen_χ(lDGAhelper_Ur.Γ_d, bubble, lDGAhelper_Ur.kG);
    end

    Fm = F_from_χ_star_gen(bubble, χm_star_gen, χm, γm, -h.mP.U);
    Fd = F_from_χ_star_gen(bubble, χd_star_gen, χd, γd,  h.mP.U);
    Γs1 = calc_Γs_ud(Fm, Fd, Phi_ud, h, GF; max_Nk=max_Nk)
    λ1L, _, _, _, _, _ = eigs(Γs1; nev=1, which=:LR, tol=1e-18);
    λ1S, _, _, _, _, _ = eigs(Γs1; nev=1, which=:SR, tol=1e-18);
    return λ1L,λ1S
end

"""
    calc_Γs_ud(Fm, Fd, Phi_ud, h::lDΓAHelper)

Calculates the Γs in particle-particle notation from the ladder vertices.  
"""
function calc_Γs_ud(Fm, Fd, Phi_ud, h::lDΓAHelper, GF::OffsetMatrix; max_Nk::Int=h.kG.Ns)
    cut_to_non_nan = true
    max_ν  = cut_to_non_nan ? trunc(Int, h.sP.n_iν/2) : h.sP.n_iν
    νnGrid = -(max_ν-1):(max_ν-2) #-1:0 #-
    kG, sub_i = build_kGrid_subsample(h.kG, max_Nk)
    println("lDΓA k-grid: ", h.kG, "linearized Eliashberg Eq. k-grid: ", kG)
    
    k_vecs = collect(Dispersions.gen_sampling(grid_type(kG), grid_dimension(kG), kG.Ns))

    νlen = length(νnGrid)
    klen = length(k_vecs)
    ωn = 0; ωi_pp = h.sP.n_iω+1

    Fm_loc = F_from_χ(:m, h);
    Fd_loc = F_from_χ(:d, h);
    Gνk_Gmνmk = build_GG(kG, GF[sub_i,:], νnGrid, k_vecs[:])
    qi_access = build_q_access(kG, k_vecs[:]);

    @warn "TODO: currently calculating two versions of Γ_pp, until Fm_{k'-k} question is resolved"
    Γs_ladder1 = Array{ComplexF64, 2}(undef, length(k_vecs)*length(νnGrid), length(k_vecs)*length(νnGrid));
    fill!(Γs_ladder1, NaN + 1im * NaN)


    Fph_ladder_updo  = permutedims(0.5 .* Fd[:,:,sub_i,:] .- 1.5 .* Fm[:,:,sub_i,:],[3,1,2,4]) .- reshape(0.5 .* Fd_loc .- 0.5 .* Fm_loc, 1, size(Fd_loc)...)

    for (νi,νn) in enumerate(νnGrid)
        νi_pp  = νn + h.sP.n_iν+1;
        for (νpi,νpn) in enumerate(νnGrid)      
            minus_ν_minus_νp = -νn -νpn - 1   # - νn - νpn
            νpi_pp  = νpn + h.sP.n_iν+1;

            ωi_ladder, νi_ladder, νpi_ladder = Freq_to_OneToIndex(minus_ν_minus_νp, νn, νpn, h.sP.shift, h.sP.n_iω, h.sP.n_iν)

            if freq_inbounds(ωi_ladder,νi_ladder,νpi_ladder,h.sP)
                for (kpi,kp_vec) in enumerate(k_vecs)
                    G_mG = Gνk_Gmνmk[νpi, kpi]
                    for (ki,k_vec) in enumerate(k_vecs)
                        qi = qi_access[ki,kpi]
                        Γs_ladder1[ki+length(k_vecs)*(νi-1),kpi+length(k_vecs)*(νpi-1)] = -(Fph_ladder_updo[qi,νi_ladder,νpi_ladder,ωi_ladder]  .- Phi_ud[νi_pp,νpi_pp,ωi_pp]) * G_mG  / (2 * kG.Nk * h.mP.β)
                    end
                end
            end 
        end
    end
    return Γs_ladder1
end


"""
    calc_λmax_linEliashberg_MatrixFree(bubble::χ₀T, χm::χT, χd::χT, γm::γT, γd::γT, h::lDΓAHelper)

This is a slower, but memory efficient versuion of [`calc_λmax_linEliashberg`](@ref calc_λmax_linEliashberg).

TODO: TeX/DOCU...
"""
function calc_λmax_linEliashberg_MatrixFree(bubble::χ₀T, χm::χT, χd::χT, γm::γT, γd::γT, h::lDΓAHelper, env; GF=h.gLoc, max_Nk::Int=h.kG.Ns, χm_star_gen=nothing, χd_star_gen=nothing)

    cut_to_non_nan = true
    max_ν  = cut_to_non_nan ? trunc(Int, h.sP.n_iν/2) : h.sP.n_iν
    νnGrid = -(max_ν-1):(max_ν-2) #-1:0 #-

    ϕs, ϕt = jldopen(joinpath(env.inputDir, "DMFT_out.jld2"),"r") do f
        f["Φpp_s"], f["Φpp_t"]
    end;
    ϕs = permutedims(ϕs, [2,3,1]);
    ϕt = permutedims(ϕt, [2,3,1]);
    Phi_ud = 0.5 .* (ϕs .+ ϕt);

    if isnothing(χm_star_gen) || isnothing(χd_star_gen)
        lDGAhelper_Ur = deepcopy(h)
        lDGAhelper_Ur.Γ_m[:,:,:] = lDGAhelper_Ur.Γ_m[:,:,:] .- (-lDGAhelper_Ur.mP.U / lDGAhelper_Ur.mP.β^2)
        lDGAhelper_Ur.Γ_d[:,:,:] = lDGAhelper_Ur.Γ_d[:,:,:] .- ( lDGAhelper_Ur.mP.U / lDGAhelper_Ur.mP.β^2)

        println("Generation generalized Susceptibility")
        flush(stdout)
        χm_star_gen = calc_gen_χ(lDGAhelper_Ur.Γ_m, bubble, lDGAhelper_Ur.kG);
        χd_star_gen = calc_gen_χ(lDGAhelper_Ur.Γ_d, bubble, lDGAhelper_Ur.kG);
    end

    Fm = F_from_χ_star_gen(bubble, χm_star_gen, χm, γm, -h.mP.U);
    Fd = F_from_χ_star_gen(bubble, χd_star_gen, χd, γd,  h.mP.U);

    kG, sub_i = build_kGrid_subsample(h.kG, max_Nk)
    println("lDΓA k-grid: ", h.kG, "linearized Eliashberg Eq. k-grid: ", kG)
    flush(stdout)
    
    k_vecs = collect(Dispersions.gen_sampling(grid_type(kG), grid_dimension(kG), kG.Ns))

    Fm_loc = F_from_χ(:m, h);
    Fd_loc = F_from_χ(:d, h);
    Gνk_Gmνmk = build_GG(kG, GF[sub_i,:], νnGrid, k_vecs[:])
    qi_access = build_q_access(kG, k_vecs[:]);

    ωi_pp::Int = h.sP.n_iω+1
    n_iν::Int  = h.sP.n_iν
    νlen::Int  = length(νnGrid)
    klen::Int  = length(k_vecs)

    Fph_ladder_updo  = permutedims(0.5 .* Fd[:,:,sub_i,:] .- 1.5 .* Fm[:,:,sub_i,:],[3,1,2,4]) .- reshape(0.5 .* Fd_loc .- 0.5 .* Fm_loc, 1, size(Fd_loc)...)

    function Γs_op(vec)
        res = zeros(ComplexF64, length(vec))
        for (νi,νn) in enumerate(νnGrid)
          νi_pp  = νn + n_iν + 1
          for (νpi,νpn) in enumerate(νnGrid)      
            minus_ν_minus_νp = -νn - νpn - 1     # - νn - νpn
            νpi_pp  = νpn + n_iν + 1
        
            ωi_ladder, νi_ladder, νpi_ladder = Freq_to_OneToIndex(minus_ν_minus_νp, νn,  νpn, h.sP.shift, ωi_pp-1, n_iν)
            if freq_inbounds( ωi_ladder,  νi_ladder,  νpi_ladder, h.sP)
              for (kpi,kp_vec) in enumerate(k_vecs)
                G_mG = Gνk_Gmνmk[νpi, kpi];
                for (ki,k_vec) in enumerate(k_vecs)
                  qi_minus_k_minus_kp = qi_access[ki,kpi]
                  res[ki+klen*(νi-1)] += -vec[kpi+klen*(νpi-1)] * (   Fph_ladder_updo[qi_minus_k_minus_kp,νi_ladder,νpi_ladder,ωi_ladder]
                                                                         .- Phi_ud[νi_pp,νpi_pp,ωi_pp]) * G_mG 
                end
              end
            else
              println("$νn / $νpn out of bounds")
            end 
          end
        end
        return real(res) ./ (2 * kG.Nk * h.mP.β)
    end

    println("building LinearMap")
    flush(stdout)
    Γs_LM = LinearMap{Float64}(Γs_op, klen*νlen, issymmetric = false)

    println("Computing First EV")
    flush(stdout)
    λ1L, _, _, _, _, _ = eigs(Γs_LM; nev=1, which=:LR, tol=1e-18);
    println("Computing Second EV")
    flush(stdout)
    λ1S, _, _, _, _, _ = eigs(Γs_LM; nev=1, which=:SR, tol=1e-18);
    return λ1L,λ1S
end
