#TODO: combine c2_curves and extended_λ
function extended_λ(nlQ_sp::NonLocalQuantities, nlQ_ch::NonLocalQuantities,
            Gνω::GνqT, λ₀::Array{ComplexF64,3}, x₀::Vector{Float64},
            kG::KGrid, mP::ModelParameters, sP::SimulationParameters;
            νmax::Int = -1, iterations::Int=400, ftol::Float64=1e-6)
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
    λch_min = if λch_min > 10000
        @warn "found λch_min=$λch_min, resetting to -500"
        -500.0
    else
        λch_min
    end
    λsp_max_rhs = rhs_c1# - sum(kintegrate(kG,χ_λ(real.(χch_tmp), λch_min + 1e-8), 1)) / mP.β 
    λch_max_rhs = rhs_c1# - sum(kintegrate(kG,χ_λ(real.(χsp_tmp), λsp_min + 1e-8), 1)) / mP.β
    λsp_max = calc_λsp_correction(χsp_tmp, ωindices, mP.Ekin_DMFT, λsp_max_rhs, kG, mP, sP) + 0.1
    λch_max = calc_λsp_correction(χch_tmp, ωindices, mP.Ekin_DMFT, λch_max_rhs, kG, mP, sP) + 0.1
    @info "λsp ∈ [$λsp_min, $λsp_max], λch ∈ [$λch_min, $λch_max]"

    trafo_bak(x) = [((λsp_max - λsp_min)/2)*(tanh(x[1])+1) + λsp_min, ((λch_max-λch_min)/2)*(tanh(x[2])+1) + λch_min]
    trafo(x) = x
    @info "After transformation: λsp ∈ [$(trafo(λsp_min)), $(trafo(λsp_max))], λch ∈ [$(trafo(λch_min)), $(trafo(λch_max))]"

    
    #WARNING: THIS METHOD CHANGES nlQ and chi needs to be reset later!!
    cond_both!(F::Vector{Float64}, λ::Vector{Float64})::Nothing = 
        cond_both_int!(F, λ, 
        nlQ_sp, nlQ_ch, χsp_tmp, χch_tmp,ωindices, Σ_ladder_ω,Σ_ladder, Kνωq_pre,
        G_corr, νGrid, χ_tail, Σ_hartree, E_pot_tail, E_pot_tail_inv, Gνω, λ₀, kG, mP, sP, trafo)
    
    # TODO: test this for a lot of data before refactor of code
    
    δ   = 0.0 # safety from first pole. decrese this if no roots are found
    λs = x₀
    λnew = nlsolve(cond_both!, λs, ftol=ftol, iterations=100)
    λnew.zero = trafo(λnew.zero)
    println(λnew)
    nlQ_sp.χ = deepcopy(χsp_tmp)
    nlQ_ch.χ = deepcopy(χch_tmp)
    
    return λnew, ""
end

function c2_curve(NPoints_coarse::Int, NPoints_negative::Int, last_λ::Vector{Float64}, nlQ_sp::NonLocalQuantities, nlQ_ch::NonLocalQuantities,
            Gνω::GνqT, λ₀::Array{ComplexF64,3},
            kG::KGrid, mP::ModelParameters, sP::SimulationParameters; νmax::Int = -1)
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
    λch_min = if λch_min > 1000
        @warn "found positive λch_min=$λch_min, setting to 0!"
        #throw("Vertex input corrupted!")
        -300.0
    else
        λch_min
    end
    @info "λsp/ch min" λsp_min λch_min
    λsp_max = 500.0
    λch_max = maximum([1000.0, 10*abs(λch_min)])
    λch_max2 = 1e12

    λch_range_negative = 10.0.^(range(0,stop=log10(abs(λch_min)+1),length=NPoints_negative+2)) .+ λch_min .- 1
    λch_range_negative_2 = range(maximum([-200,λch_min]),stop=100,length=20)
    λch_range_coarse = range(0,stop=λch_max,length=NPoints_coarse)
    λch_range_large = 10.0.^(range(0,stop=log10(λch_max2-2*λch_max+1),length=6)) .+ 2*λch_max .- 1
    last_λch_range = isfinite(last_λ[2]) ? range(last_λ[2] - abs(last_λ[2]*0.1), stop = last_λ[2] + abs(last_λ[2]*0.1), length=8) : []
    #λch_range_old = 10.0.^(range(0,stop=log10(-λch_min+1),length=NPoints_negative+2)) .+ λch_min .- 1
    
    λch_range = Float64.(sort(unique(union([0], last_λch_range, λch_range_negative, λch_range_negative_2, λch_range_coarse, λch_range_large))))
    # λsp, λch, lhs2_c1,rhs_c1 ,lhs_c2, rhs_c2, Epot_1, Epot_2
    # #TODO: this could be made more memory efficient
    r_χsp = real.(nlQ_sp.χ)



    # c2_curve_clean_res = zeros(6, length(λch_range))
    # for (i,λch_i) in enumerate(λch_range)
    #     χ_λ!(nlQ_ch.χ, χch_tmp, λch_i)
    #     χch_ω = kintegrate(kG, nlQ_ch.χ[:,ωindices], 1)[1,:]
    #     χch_sum = real(sum(subtract_tail(χch_ω, mP.Ekin_DMFT, iωn)))/mP.β - mP.Ekin_DMFT*mP.β/12
    #     rhs = mP.n * (1 - mP.n/2) - χch_sum
    #     λsp_i = calc_λsp_correction(real.(nlQ_sp.χ), ωindices, mP.Ekin_DMFT, rhs, kG, mP, sP)
    #     χ_λ!(nlQ_sp.χ, χsp_tmp, λsp_i)
    #     lhs_c1, rhs_c1, lhs_c2, rhs_c2 = cond_both_int_clean(nlQ_sp, nlQ_ch,
    #         ωindices, Σ_ladder_ω, Σ_ladder, Kνωq_pre, G_corr, νGrid, χ_tail, Σ_hartree,
    #         E_pot_tail, E_pot_tail_inv, Gνω,λ₀, kG, mP, sP)
        

    #     c2_curve_clean_res[:,i] = [λsp_i, λch_i, lhs_c1, rhs_c1, lhs_c2, rhs_c2]
    #     nlQ_sp.χ = deepcopy(χsp_tmp)
    #     nlQ_ch.χ = deepcopy(χch_tmp)
    # end


    c2_curve_res = zeros(6, length(λch_range))
    @timeit to "c2 loop" for (i,λch_i) in enumerate(λch_range)
            λsp_i, lhs_c1, rhs_c1, lhs_c2, rhs_c2 = cond_both_int(λch_i, nlQ_sp, nlQ_ch, χsp_tmp, χch_tmp,
            ωindices, Σ_ladder_ω, Σ_ladder, Kνωq_pre, G_corr, νGrid, χ_tail, Σ_hartree,
            E_pot_tail, E_pot_tail_inv, Gνω,λ₀, kG, mP, sP)

        c2_curve_res[:,i] = [λsp_i, λch_i, lhs_c1, rhs_c1, lhs_c2, rhs_c2]
        nlQ_sp.χ = deepcopy(χsp_tmp)
        nlQ_ch.χ = deepcopy(χch_tmp)
    end
    # println("!!!!!!! c2 curve check!", all(c2_curve_res .≈ c2_curve_clean_res))
    # @info "!!!!!!! c2 curve check!" all(c2_curve_res .≈ c2_curve_clean_res)

    return c2_curve_res
end
