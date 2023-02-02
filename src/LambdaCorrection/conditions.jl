# ==================================================================================================== #
#                                           conditions.jl                                              #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#   lambda-correction conditions for several methods, fixing different physical properties.            #
# -------------------------------------------- TODO -------------------------------------------------- #
#  REFACTOR!!!!!                                                                                       #
# ==================================================================================================== #

"""
    λsp_correction(χsp::χT, rhs::Float64, kG::KGrid, mP::ModelParameters, sP::SimulationParameters)
                        
Calculates ``\\lambda_\\mathrm{m}`` value, by fixing ``\\sum_{q,\\omega} \\chi^{\\lambda_\\mathrm{m}}_{\\uparrow\\downarrow}(q,i\\omega) = \\frac{n}{2}(1-\\frac{n}{2})``.
"""
function λsp_correction(χsp::χT, rhs::Float64, kG::KGrid, 
                        mP::ModelParameters, sP::SimulationParameters)
    χr::Matrix{Float64}    = real.(χsp[:,χsp.usable_ω])
    iωn = (1im .* 2 .* (-sP.n_iω:sP.n_iω)[χsp.usable_ω] .* π ./ mP.β)
    iωn[findfirst(x->x ≈ 0, iωn)] = Inf
    χ_tail::Vector{Float64} = real.(χsp.tail_c[3] ./ (iωn.^2))

    f_c1_int(λint::Float64)::Float64 = f_c1(χr, λint, kG.kMult, χ_tail)/mP.β - χsp.tail_c[3]*mP.β/12 - rhs
    df_c1_int(λint::Float64)::Float64 = df_c1(χr, λint, kG.kMult, χ_tail)/mP.β - χsp.tail_c[3]*mP.β/12 - rhs

    λsp = newton_right(f_c1_int, df_c1_int, get_λ_min(χr))
    return λsp
end


# ============================================== Helpers =============================================
# --------------------------------- clean versions (slow/single core) --------------------------------
function cond_both_int!(F::Vector{Float64}, λ::Vector{Float64}, 
        χsp::χT, γsp::γT, χch::χT, γch::γT, Σ_loc,
        gLoc_rfft::Matrix{ComplexF64},
        λ₀::Array{ComplexF64,3}, kG::KGrid, mP::ModelParameters, sP::SimulationParameters, trafo::Function)::Nothing

    λi = trafo(λ)
    k_norm::Int = Nk(kG)

    ωindices, νGrid, iωn_f = gen_νω_indices(χsp, χch, mP, sP)
    iωn = iωn_f[ωindices]
    iωn[findfirst(x->x ≈ 0, iωn)] = Inf
    χ_tail::Vector{ComplexF64} = χch.tail_c[3] ./ (iωn.^2)

    lhs_c1, lhs_c2 = lhs_int(χsp, χch, λi[1], λi[2], χ_tail, kG.kMult, k_norm, χsp.tail_c[3], mP.β)
    Σ_ladder = calc_Σ(χsp, γsp, χch, γch, λ₀, gLoc_rfft, kG, mP, sP, νmax=sP.n_iν, λsp=λi[1],λch=λi[2]);
    _, GLoc_new = G_from_Σladder(Σ_ladder, Σ_loc, kG, mP, sP; fix_n=false)
    _, E_pot = calc_E(GLoc_new[:,νGrid].parent, Σ_ladder.parent, kG, mP, νmax = last(νGrid)+1)
    rhs_c1 = mP.n/2 * (1 - mP.n/2)
    rhs_c2 = E_pot/mP.U - (mP.n/2) * (mP.n/2)
    F[1] = lhs_c1 - rhs_c1
    F[2] = lhs_c2 - rhs_c2
    return nothing
end

"""
TODO: refactor (especially _par version)
"""
function run_sc(χsp::χT, γsp::γT, χch::χT, γch::γT, λ₀::AbstractArray{ComplexF64,3}, gLoc_rfft_init::GνqT, Σ_loc::Vector{ComplexF64},
                λch::Float64, kG::KGrid, mP::ModelParameters, sP::SimulationParameters;
                maxit::Int=100, mixing::Float64=0.2, conv_abs::Float64=1e-6, update_χ_tail::Bool=false)
    ωindices, νGrid, iωn_f = gen_νω_indices(χsp, χch, mP, sP)
    iωn = iωn_f[ωindices]
    iωn[findfirst(x->x ≈ 0, iωn)] = Inf
    gLoc_rfft = deepcopy(gLoc_rfft_init)
    gLoc_new = nothing
    Σ_ladder_old::OffsetMatrix{ComplexF64, Matrix{ComplexF64}} = OffsetArray(Matrix{ComplexF64}(undef, length(kG.kMult), length(νGrid)), 1:length(kG.kMult), νGrid)
    Σ_ladder::OffsetMatrix{ComplexF64, Matrix{ComplexF64}} = OffsetArray(Matrix{ComplexF64}(undef, length(kG.kMult), length(νGrid)), 1:length(kG.kMult), νGrid)
    E_pot    = Inf
    E_kin    = Inf
    cont = true
    converged = maxit == 0

    χ_λ!(χch, χch, λch)
    rhs_λsp = λsp_rhs(NaN, χsp, χch, kG, mP, sP)
    λsp = λsp_correction(χsp, real(rhs_λsp), kG, mP, sP)
    reset!(χch)
    if !isfinite(λsp)
        @warn "no finite λsp found!"
        cont = false
    end

    μbak = mP.μ
    it = 1
    while cont
        Σ_ladder_old = deepcopy(Σ_ladder)
        Σ_ladder = calc_Σ(χsp, γsp, χch, γch, λ₀, gLoc_rfft, kG, mP, sP, νmax=last(νGrid)+1, λsp=λsp,λch=λch);
        mixing != 0 && it > 1 && (Σ_ladder[:,:] = (1-mixing) .* Σ_ladder .+ mixing .* Σ_ladder_old)
        μnew, gLoc_new = G_from_Σladder(Σ_ladder, Σ_loc, kG, mP, sP; fix_n=true)
        isnan(μnew) && break
        _, gLoc_rfft = G_fft(gLoc_new, kG, mP, sP)
        mP.μ = μnew
        E_kin, E_pot = calc_E(gLoc_new[:,νGrid].parent, Σ_ladder.parent, kG, mP, νmax = last(νGrid)+1)
        if update_χ_tail
            update_tail!(χsp, [0, 0, E_kin], iωn_f)
            update_tail!(χch, [0, 0, E_kin], iωn_f)
            χ_λ!(χch, χch, λch)
            rhs_λsp = λsp_rhs(NaN, χsp, χch, kG, mP, sP)
            λsp = λsp_correction(χsp, real(rhs_λsp), kG, mP, sP)
            reset!(χch)
        end
        if it != 1
            # println("SC it = $it, conv = $(sum(abs.(Σ_ladder .- Σ_ladder_old))/(size(Σ_ladder_old,2)*kG.Nk)), μ = $μnew")
            # ndens = filling_pos(gLoc_new.parent, kG, mP.U, μnew, mP.β)
            # println("  -> check filling: $(round(ndens,digits=4)) =?= $(round(mP.n,digits=4)), λsp = $(round(λsp,digits=4)), λch = $(round(λch,digits=4))")
            if sum(abs.(Σ_ladder .- Σ_ladder_old))/(size(Σ_ladder_old,2)*kG.Nk) < conv_abs  
                converged = true
                cont = false
            end
        end

        (it >= maxit) && (cont = false)
        it += 1
    end

    lhs_c1 = NaN
    lhs_c2 = NaN
    if isfinite(E_kin)
        χ_tail::Vector{ComplexF64} = χsp.tail_c[3]./ (iωn.^2)
        lhs_c1, lhs_c2 = lhs_int(χsp.data, χch.data, λsp, λch, 
                            χ_tail, kG.kMult, Nk(kG), χsp.tail_c[3], mP.β)
    end
    update_tail!(χsp, [0, 0, mP.Ekin_DMFT], iωn_f)
    update_tail!(χch, [0, 0, mP.Ekin_DMFT], iωn_f)
    μnew = mP.μ
    mP.μ = μbak
    converged = converged && all(isfinite.([lhs_c2, lhs_c2]))
    return Σ_ladder, gLoc_new, E_kin, E_pot, μnew, λsp, lhs_c1, lhs_c2, converged
end

function run_sc_par(χsp::χT, γsp::γT, χch::χT, γch::γT, λ₀::AbstractArray{ComplexF64,3}, gLoc_rfft_init::GνqT, Σ_loc::Vector{ComplexF64},
                    λch::Float64, kG::KGrid, mP::ModelParameters, sP::SimulationParameters;
                    maxit::Int=100, mixing::Float64=0.2, conv_abs::Float64=1e-6, update_χ_tail::Bool=false)

    ωindices, νGrid, iωn_f = gen_νω_indices(χsp, χch, mP, sP)
    iωn = iωn_f[ωindices]
    iωn[findfirst(x->x ≈ 0, iωn)] = Inf
    initialize_EoM(gLoc_rfft_init, 
                λ₀, νGrid, kG, mP, sP, 
                χsp = χsp, γsp = γsp,
                χch = χch, γch = γch)
    gLoc_rfft = deepcopy(gLoc_rfft_init)
    gLoc_new = nothing
    Σ_ladder_old::OffsetMatrix{ComplexF64, Matrix{ComplexF64}} = OffsetArray(Matrix{ComplexF64}(undef, length(kG.kMult), length(νGrid)), 1:length(kG.kMult), νGrid)
    Σ_ladder::OffsetMatrix{ComplexF64, Matrix{ComplexF64}} = OffsetArray(Matrix{ComplexF64}(undef, length(kG.kMult), length(νGrid)), 1:length(kG.kMult), νGrid)
    E_pot    = Inf
    E_kin    = Inf
    cont = true
    converged = maxit == 0

    χ_λ!(χch, χch, λch)
    rhs_λsp = λsp_rhs(NaN, χsp, χch, kG, mP, sP)
    λsp = λsp_correction(χsp, real(rhs_λsp), kG, mP, sP)
    reset!(χch)
    if !isfinite(λsp)
        @warn "no finite λsp found!"
        cont = false
    end

    μbak = mP.μ
    it = 1
    while cont
        update_wcaches_G_rfft!(gLoc_rfft)
        Σ_ladder_old[:,:] = deepcopy(Σ_ladder)
        Σ_ladder[:,:] = calc_Σ_par(kG, mP, sP, λsp=λsp, λch=λch, νrange=νGrid);
        mixing != 0 && it > 1 && (Σ_ladder[:,:] = (1-mixing) .* Σ_ladder .+ mixing .* Σ_ladder_old)
        μnew, gLoc_new = G_from_Σladder(Σ_ladder, Σ_loc, kG, mP, sP; fix_n=true)
        isnan(μnew) && break
        _, gLoc_rfft = G_fft(gLoc_new, kG, mP, sP)
        mP.μ = μnew
        E_kin, E_pot = calc_E(gLoc_new[:,νGrid].parent, Σ_ladder.parent, kG, mP, νmax = last(νGrid)+1)
        if update_χ_tail
            update_tail!([0, 0, E_kin])
            update_tail!(χch, [0, 0, E_kin], iωn_f)
            χ_λ!(χch, χch, λch)
            rhs_λsp = λsp_rhs(NaN, χsp, χch, kG, mP, sP)
            λsp = λsp_correction(χsp, real(rhs_λsp), kG, mP, sP)
            reset!(χch)
        end
        if Σ_ladder_old !== nothing
            # println("SC it = $it, conv = $(sum(abs.(Σ_ladder .- Σ_ladder_old))/(size(Σ_ladder_old,2)*kG.Nk)), μ = $μnew")
            # ndens = filling_pos(gLoc_new.parent, kG, mP.U, μnew, mP.β)
            # println("  -> check filling: $(round(ndens,digits=4)) =?= $(round(mP.n,digits=4)), λsp = $(round(λsp,digits=4)), λch = $(round(λch,digits=4))")
            if sum(abs.(Σ_ladder .- Σ_ladder_old))/(size(Σ_ladder_old,2)*kG.Nk) < conv_abs  
                converged = true
                cont = false
            end
        end
        (it >= maxit) && (cont = false)
        it += 1
    end
    lhs_c1 = NaN
    lhs_c2 = NaN
    if isfinite(E_kin)
        if update_χ_tail
            update_tail!(χsp, [0, 0, E_kin], iωn_f)
            update_tail!(χch, [0, 0, E_kin], iωn_f)
        end
        χ_tail::Vector{ComplexF64} = χsp.tail_c[3] ./ (iωn.^2)
        lhs_c1, lhs_c2 = lhs_int(χsp.data, χch.data, λsp, λch, 
                                 χ_tail, kG.kMult, Nk(kG), χsp.tail_c[3], mP.β)
        if update_χ_tail
            update_tail!(χsp, [0, 0, mP.Ekin_DMFT], iωn_f)
            update_tail!(χch, [0, 0, mP.Ekin_DMFT], iωn_f)
        end
    end
    update_tail!([0, 0, mP.Ekin_DMFT])
    update_tail!([0, 0, mP.Ekin_DMFT])
    μnew = mP.μ
    mP.μ = μbak
    return Σ_ladder, gLoc_new, E_kin, E_pot, μnew, λsp, lhs_c1, lhs_c2, converged
end

# ------------------------------------------ fast versions  ------------------------------------------
"""
    lhs_int(χsp::Matrix, χch::Matrix, χ_tail::Vector{ComplexF64}, kMult::Vector{Float64}, k_norm::Int, Ekin::Float64, β::Float64)

Internal function. This calculates the sum over up-up and up-down susceptibilities, used in [`cond_both_int`](@ref cond_both_int), avoiding allocations.

Returns: 
-------------
Tuple{ComplexF64/Float64}[
    lhs_c1, # ``\\sum_{q,\\omega} \\chi_{\\uparrow,\\downarrow}``
    lhs_c2  # ``\\sum_{q,\\omega} \\chi_{\\uparrow,\\uparrow}``
]
"""
function lhs_int(χsp::Matrix, χch::Matrix, λsp::Float64, λch::Float64, 
                 χ_tail::Vector{ComplexF64}, kMult::Vector{Float64}, k_norm::Int, 
                 Ekin::Float64, β::Float64)
    lhs_c1 = 0.0
    lhs_c2 = 0.0
    for (ωi,t) in enumerate(χ_tail)
        tmp1 = 0.0
        tmp2 = 0.0
        for (qi,km) in enumerate(kMult)
            χsp_i_λ = 1 ./ (λsp + 1/real(χsp[qi,ωi]))
            χch_i_λ = 1 ./ (λch + 1/real(χch[qi,ωi]))
            tmp1 += (χch_i_λ + χsp_i_λ) * km
            tmp2 += (χch_i_λ - χsp_i_λ) * km
        end
        lhs_c1 += 0.5*tmp1/k_norm - t
        lhs_c2 += 0.5*tmp2/k_norm
    end
    lhs_c1 = lhs_c1/β - Ekin*β/12
    lhs_c2 = lhs_c2/β
    return lhs_c1, lhs_c2 
end

#TODO: these need to be refactored. many code replications
function f_c1(χ::Matrix, λ::Float64, kMult::Vector{Float64}, 
                 tail::Vector{Float64})::Float64
    res = 0.0
    resi = 0.0
    norm = sum(kMult)
    for (i,ωi) in enumerate(tail)
        resi = 0.0
        for (ki,km) in enumerate(kMult)
            resi += χ_λ(χ[ki,i],λ) * km
        end
        res += resi/norm - ωi
    end
    return res
end 

function df_c1(χ::Matrix{Float64}, λ::Float64, kMult::Vector{Float64}, 
                tail::Vector{Float64})::Float64
    res = 0.0
    resi = 0.0
    norm = sum(kMult)
    for (i,ωi) in enumerate(tail)
        resi = 0.0
        for (ki,km) in enumerate(kMult)
            resi += dχ_λ(χ[ki,i],λ) * km
        end
        res += resi/norm - ωi
    end
    return res
end
