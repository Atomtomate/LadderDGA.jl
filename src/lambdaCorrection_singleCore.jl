function cond_both_int!(F::Vector{Float64}, λ::Vector{Float64}, 
        nlQ_sp::NonLocalQuantities, nlQ_ch::NonLocalQuantities, χsp_tmp::χT, χch_tmp::χT,
        ωindices::UnitRange{Int}, Σ_ladder_ω::OffsetArray{ComplexF64,3,Array{ComplexF64,3}}, 
        Σ_ladder::OffsetArray{ComplexF64,2,Array{ComplexF64,2}}, Kνωq_pre::Vector{ComplexF64},
        G_corr::Matrix{ComplexF64},νGrid::UnitRange{Int},χ_tail::Vector{ComplexF64},Σ_hartree::Float64,
        E_pot_tail::Matrix{ComplexF64},E_pot_tail_inv::Vector{Float64},Gνω::GνqT,
        λ₀::Array{ComplexF64,3}, kG::KGrid, mP::ModelParameters, sP::SimulationParameters, trafo::Function)::Nothing

    λi = trafo(λ)
    χ_λ!(nlQ_sp.χ, χsp_tmp, λi[1])
    χ_λ!(nlQ_ch.χ, χch_tmp, λi[2])
    k_norm::Int = Nk(kG)

    #TODO: unroll 
    calc_Σ_ω!(eom, Σ_ladder_ω, Kνωq_pre, ωindices, nlQ_sp, nlQ_ch, Gνω, λ₀, mP.U, kG, sP)
    Σ_ladder[:,:] = dropdims(sum(Σ_ladder_ω, dims=[3]),dims=3) ./ mP.β .+ Σ_hartree

    lhs_c1 = 0.0
    lhs_c2 = 0.0
    for (ωi,t) in enumerate(χ_tail)
        tmp1 = 0.0
        tmp2 = 0.0
        for (qi,km) in enumerate(kG.kMult)
            χsp_i_λ = real(nlQ_sp.χ[qi,ωi])
            χch_i_λ = real(nlQ_ch.χ[qi,ωi])
            tmp1 += (χch_i_λ + χsp_i_λ) * km
            tmp2 += (χch_i_λ - χsp_i_λ) * km
        end
        lhs_c1 += 0.5*tmp1/k_norm - t
        lhs_c2 += 0.5*tmp2/k_norm
    end
    iωn = 1im .* 2 .* (-sP.n_iω:sP.n_iω) .* π ./ mP.β

    lhs_c1 = lhs_c1/mP.β - mP.Ekin_DMFT*mP.β/12
    lhs_c2 = lhs_c2/mP.β

    #TODO: the next line is expensive: Optimize G_from_Σ
    G_corr[:] = transpose(flatten_2D(G_from_Σ(Σ_ladder.parent, kG.ϵkGrid, νGrid, mP)));
    E_pot = calc_E_pot(kG, G_corr, Σ_ladder.parent, E_pot_tail, E_pot_tail_inv, mP.β)
    rhs_c1 = mP.n/2 * (1 - mP.n/2)
    rhs_c2 = E_pot/mP.U - (mP.n/2) * (mP.n/2)
    F[1] = lhs_c1 - rhs_c1
    F[2] = lhs_c2 - rhs_c2
    return nothing
end

function extended_λ(nlQ_sp::NonLocalQuantities, nlQ_ch::NonLocalQuantities,
            Gνω::GνqT, λ₀::Array{ComplexF64,3},
            kG::KGrid, mP::ModelParameters, sP::SimulationParameters;
            νmax::Int = -1, iterations::Int=400, ftol::Float64=1e-6, x₀ = [0.1, 0.1])
        # --- prepare auxiliary vars ---
    @info "Using DMFT GF for second condition in new lambda correction"

        # general definitions
    Nq::Int = size(nlQ_sp.γ,1)
    Nν::Int = size(nlQ_sp.γ,2)
    Nω::Int = size(nlQ_sp.γ,3)
    EKin::Float64 = mP.Ekin_DMFT
    ωindices::UnitRange{Int} = (sP.dbg_full_eom_omega) ? (1:size(nlQ_ch.χ,2)) : intersect(nlQ_sp.usable_ω, nlQ_ch.usable_ω)
    ωrange_list = (-sP.n_iω:sP.n_iω)[ωindices]
    ωrange::UnitRange{Int} = first(ωrange_list):last(ωrange_list)
    νmax::Int = νmax < 0 ? minimum([sP.n_iν,floor(Int,3*length(ωindices)/8)]) : νmax
    νGrid::UnitRange{Int} = 0:(νmax-1)
    iωn = 1im .* 2 .* (-sP.n_iω:sP.n_iω)[ωindices] .* π ./ mP.β
    iωn[findfirst(x->x ≈ 0, iωn)] = Inf
    χ_tail::Vector{ComplexF64} = EKin ./ (iωn.^2)
    k_norm::Int = Nk(kG)

    # EoM optimization related definitions
    Kνωq_pre::Vector{ComplexF64} = Vector{ComplexF64}(undef, length(kG.kMult))
    Σ_ladder_ω::OffsetArray{ComplexF64,3,Array{ComplexF64,3}} = OffsetArray(Array{ComplexF64,3}(undef,Nq,νmax,length(ωrange)),
                              1:Nq, 0:νmax-1, ωrange)
    Σ_ladder::OffsetArray{ComplexF64,2,Array{ComplexF64,2}} = OffsetArray(Array{ComplexF64,2}(undef,Nq, νmax),
                              1:Nq, 0:νmax-1)

    # preallications
    χsp_tmp::Matrix{ComplexF64}  = deepcopy(nlQ_sp.χ)
    χch_tmp::Matrix{ComplexF64}  = deepcopy(nlQ_ch.χ)
    G_corr::Matrix{ComplexF64} = Matrix{ComplexF64}(undef, Nq, νmax)

    # Therodynamics preallocations
    Σ_hartree::Float64 = mP.n * mP.U/2.0;
    E_pot_tail_c = [zeros(size(kG.ϵkGrid)),
            (mP.U^2 * 0.5 * mP.n * (1-0.5*mP.n) .+ Σ_hartree .* (kG.ϵkGrid .+ Σ_hartree .- mP.μ))]
    tail = [1 ./ (iν_array(mP.β, νGrid) .^ n) for n in 1:length(E_pot_tail_c)]
    E_pot_tail::Matrix{ComplexF64} = sum(E_pot_tail_c[i] .* transpose(tail[i]) for i in 1:length(tail))
    E_pot_tail_inv::Vector{Float64} = sum((mP.β/2)  .* [Σ_hartree .* ones(size(kG.ϵkGrid)), (-mP.β/2) .* E_pot_tail_c[2]])

    rhs_c1 = mP.n/2 * (1 - mP.n/2)
    λsp_min = get_χ_min(real.(χsp_tmp))
    λch_min = get_χ_min(real.(χch_tmp))
    λsp_max = sum(kintegrate(kG,χ_λ(real.(χch_tmp), λch_min + 1e-8), 1)) / mP.β - rhs_c1
    λch_max = sum(kintegrate(kG,χ_λ(real.(χsp_tmp), λsp_min + 1e-8), 1)) / mP.β - rhs_c1
    @info "λsp ∈ [$λsp_min, $λsp_max], λch ∈ [$λch_min, $λch_max]"

    trafo(x) = [((λsp_max - λsp_min)/2)*(tanh(x[1])+1) + λsp_min, ((λch_max-λch_min)/2)*(tanh(x[2])+1) + λch_min]
    println("λ search interval: $(trafo([-Inf, -Inf])) to $(trafo([Inf, Inf]))")

    
    #WARNING: THIS METHOD CHANGES nlQ and chi needs to be reset later!!
    cond_both!(F::Vector{Float64}, λ::Vector{Float64})::Nothing = 
        cond_both_int!(F, λ, 
        nlQ_sp, nlQ_ch, χsp_tmp, χch_tmp,ωindices, Σ_ladder_ω,Σ_ladder, Kνωq_pre,
        G_corr, νGrid, χ_tail, Σ_hartree, E_pot_tail, E_pot_tail_inv, Gνω, λ₀, kG, mP, sP, trafo)
    
    # TODO: test this for a lot of data before refactor of code
    
    δ   = 1.0 # safety from first pole. decrese this if no roots are found
    λs = [λsp_min, λch_min] .+ δ
    λnew = nlsolve(cond_both!, λs, ftol=ftol, iterations=1)
    λnew.zero = trafo(λnew.zero)
    println(λnew)
    nlQ_sp.χ = χsp_tmp
    nlQ_ch.χ = χch_tmp
    
    return λnew, ""
end
