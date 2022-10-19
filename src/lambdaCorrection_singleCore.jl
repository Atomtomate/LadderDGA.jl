#TODO: combine c2_curves and extended_λ
function extended_λ(
        χ_sp::χT, γ_sp::γT, χ_ch::χT, γ_ch::γT,
        Gνω::GνqT, λ₀::Array{ComplexF64,3}, x₀::Vector{Float64},
        kG::KGrid, mP::ModelParameters, sP::SimulationParameters;
        νmax::Int = -1, iterations::Int=400, ftol::Float64=1e-6)
        # --- prepare auxiliary vars ---
    @info "Using DMFT GF for second condition in new lambda correction"

    # general definitions
    Nq, Nν, Nω = size(γ_sp)
    EKin::Float64 = mP.Ekin_DMFT
    ωindices::UnitRange{Int} = (sP.dbg_full_eom_omega) ? (1:size(χ_ch,2)) : intersect(χ_sp.usable_ω, χ_ch.usable_ω)
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
    χsp_tmp::χT = deepcopy(χ_sp)
    χch_tmp::χT = deepcopy(χ_ch)
    G_corr::Matrix{ComplexF64} = Matrix{ComplexF64}(undef, Nq, νmax)

    # Therodynamics preallocations
    Σ_hartree::Float64 = mP.n * mP.U/2.0;
    E_pot_tail_c = [zeros(size(kG.ϵkGrid)),
            (mP.U^2 * 0.5 * mP.n * (1-0.5*mP.n) .+ Σ_hartree .* (kG.ϵkGrid .+ Σ_hartree .- mP.μ))]
    tail = [1 ./ (iν_array(mP.β, νGrid) .^ n) for n in 1:length(E_pot_tail_c)]
    E_pot_tail::Matrix{ComplexF64} = sum(E_pot_tail_c[i] .* transpose(tail[i]) for i in 1:length(tail))
    E_pot_tail_inv::Vector{Float64} = sum((mP.β/2)  .* [Σ_hartree .* ones(size(kG.ϵkGrid)), (-mP.β/2) .* E_pot_tail_c[2]])

    rhs_c1 = mP.n/2 * (1 - mP.n/2)
    λsp_min = get_λ_min(real.(χsp_tmp.data))
    λch_min = get_λ_min(real.(χch_tmp.data))
    λch_min = if λch_min > 10000
        @warn "found λch_min=$λch_min, resetting to -500"
        -500.0
    else
        λch_min
    end
    λch_max_rhs = rhs_c1# - sum(kintegrate(kG,χ_λ(real.(χsp_tmp.data), λsp_min + 1e-8), 1)) / mP.β
    λsp_max_rhs = rhs_c1# - sum(kintegrate(kG,χ_λ(real.(χch_tmp.data), λch_min + 1e-8), 1)) / mP.β 
    λsp_max = calc_λsp_correction(χsp_tmp.data, ωindices, mP.Ekin_DMFT, λsp_max_rhs, kG, mP, sP) + 0.1
    λch_max = calc_λsp_correction(χch_tmp.data, ωindices, mP.Ekin_DMFT, λch_max_rhs, kG, mP, sP) + 0.1
    @info "λsp ∈ [$λsp_min, $λsp_max], λch ∈ [$λch_min, $λch_max]"

    trafo_bak(x) = [((λsp_max - λsp_min)/2)*(tanh(x[1])+1) + λsp_min, ((λch_max-λch_min)/2)*(tanh(x[2])+1) + λch_min]
    trafo(x) = x
    @info "After transformation: λsp ∈ [$(trafo(λsp_min)), $(trafo(λsp_max))], λch ∈ [$(trafo(λch_min)), $(trafo(λch_max))]"

    cond_both!(F::Vector{Float64}, λ::Vector{Float64})::Nothing = 
        cond_both_int!(F, λ, 
        χ_sp, γ_sp, χ_ch, γ_ch, χsp_tmp, χch_tmp, ωindices, Σ_ladder_ω,Σ_ladder, Kνωq_pre,
        G_corr, νGrid, χ_tail, Σ_hartree, E_pot_tail, E_pot_tail_inv, Gνω, λ₀, kG, mP, sP, trafo)
    
    # TODO: test this for a lot of data before refactor of code
    
    δ   = 0.0 # safety from first pole. decrese this if no roots are found
    λs = x₀
    λnew = nlsolve(cond_both!, λs, ftol=ftol, iterations=100)
    λnew.zero = trafo(λnew.zero)
    println(λnew)
    
    χ_sp.data = deepcopy(χsp_tmp.data)
    χ_ch.data = deepcopy(χch_tmp.data)
    return λnew, ""
end
