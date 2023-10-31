function getG_naive(kG, Σ, kp, νp, μ, β)
    ϵk = LadderDGA.Dispersions.gen_ϵkGrid(LadderDGA.Dispersions.grid_type(kG), [kp], kG.t, kG.tp, kG.tpp)[1]
    Σν = νp > -1 ? Σ[νp] : conj(Σ[-νp-1])
    Gk = 1/(1im * (2*νp+1)*π/β + μ - ϵk - Σν)
    return Gk
end

function GG_naive(kG, Σ, kp, νp, μ, β)
    Gk = getG_naive(kG, Σ, kp, νp, μ, β)
    G_minusk = getG_naive(kG, Σ, -1 .* kp, -νp-1, μ, β)
    #println(Gk, " ,", G_minusk)
    return Gk*G_minusk
end

@testset "lookups" begin
    kG = LadderDGA.gen_kGrid("2Dsc-1.1-1.0-0.9",8)
    gridshape = LadderDGA.Dispersions.gridshape
    νnGrid = -3:2
    β = 5.0
    μ = 0.5 
    Σ_loc = OffsetArray(2*μ .* (0:4) .+ 0.0im, 0:4)
    k_vecs = collect(LadderDGA.Dispersions.gen_sampling(LadderDGA.Dispersions.grid_type(kG), LadderDGA.Dispersions.grid_dimension(kG), kG.Ns))
    v_full = collect(LadderDGA.Dispersions.gen_sampling(LadderDGA.Dispersions.grid_type(kG), LadderDGA.Dispersions.grid_dimension(kG), kG.Ns))

    gLoc = OffsetArray(Array{ComplexF64,2}(undef, length(kG.kMult),length(νnGrid)), 1:length(kG.kMult), νnGrid)
    gLoc_i = Array{ComplexF64,LadderDGA.Dispersions.grid_dimension(kG)}(undef, gridshape(kG)...)
    ϵk_full = LadderDGA.expandKArr(kG, kG.ϵkGrid)[:]
    for νn in νnGrid 
        Σ_loc_i = (νn < 0) ? conj(Σ_loc[-νn-1]) : Σ_loc[νn]
        gLoc_i = reshape(map(ϵk -> G_from_Σ(νn, β, μ, ϵk, Σ_loc_i), ϵk_full), gridshape(kG))
        gLoc[:, νn] = LadderDGA.reduceKArr(kG, gLoc_i)
    end

    q_lookup = LadderDGA.Dispersions.build_q_lookup(kG)
    result   = Array{Int,2}(undef, length(v_full), length(v_full))
    fails = []
    for k in v_full
        for kp in v_full
            q_test = round.(LadderDGA.Dispersions.transform_to_first_BZ(kG, k .- kp), digits=6)
            !(q_test in keys(q_lookup)) && push!(fails, q_test)
        end
    end
    #println("number of fails in q-lookup: ", length(fails))
    @test length(fails) == 0
    #

    test_arr = Array{Bool}(undef, length(νnGrid), length(k_vecs))

    for (νpi,νpn) in enumerate(νnGrid)
        GF_νp = LadderDGA.expandKArr(kG, gLoc[:,νpn].parent)
        GF_νp_minus_k_pre = LadderDGA.expandKArr(kG, gLoc[:,-νpn-1].parent)
        shift_vec = 2 .* kG.k0 .- LadderDGA.gridshape(kG) .- 1
        GF_νp_minus_k = circshift(reverse(GF_νp_minus_k_pre), shift_vec)

        for (kpi,kp_vec) in enumerate(k_vecs)
            G_mG  = GG_naive(kG, Σ_loc, kp_vec, νpn, μ, β)
            G_mG2 = GF_νp[kpi]*GF_νp_minus_k[kpi]
            test_arr[νpi, kpi]  = G_mG ≈ G_mG2
        end
    end

    #println("number of fails in G(k)G(-k) precomputation: ", count(.! test_arr))
    @test count(.! test_arr) == 0
end
