"""
    lhs_int(χ_sp::Matrix, χ_ch::Matrix, χ_tail::Vector{ComplexF64}, kMult::Vector{Float64}, k_norm::Int, Ekin::Float64, β::Float64)

Internal function. This calculates the sum over up-up and up-down susceptibilities, used in [`cond_both_int`](@ref cond_both_int), avoiding allocations.

Returns: 
-------------
Tuple{ComplexF64/Float64}[
    lhs_c1, # ``\\sum_{q,\\omega} \\chi_{\\uparrow,\\downarrow}``
    lhs_c2  # ``\\sum_{q,\\omega} \\chi_{\\uparrow,\\uparrow}``
]
"""
function lhs_int(χ_sp::Matrix, χ_ch::Matrix, χ_tail::Vector{ComplexF64},
                  kMult::Vector{Float64}, k_norm::Int, Ekin::Float64, β::Float64)
    lhs_c1 = 0.0
    lhs_c2 = 0.0
    for (ωi,t) in enumerate(χ_tail)
        tmp1 = 0.0
        tmp2 = 0.0
        for (qi,km) in enumerate(kMult)
            χsp_i_λ = real(χ_sp[qi,ωi])
            χch_i_λ = real(χ_ch[qi,ωi])
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


function cond_both_int(
        λch_i::Float64, 
        χ_sp::χT, γ_sp::γT, χ_ch::χT, γ_ch::γT,
        χsp_tmp::χT, χch_tmp::χT,
        ωindices::UnitRange{Int}, Σ_ladder_ω::OffsetArray{ComplexF64,3,Array{ComplexF64,3}}, 
        Σ_ladder::OffsetArray{ComplexF64,2,Array{ComplexF64,2}}, Kνωq_pre::Vector{ComplexF64},
        G_corr::Matrix{ComplexF64},νGrid::UnitRange{Int},χ_tail::Vector{ComplexF64},Σ_hartree::Float64,
        E_pot_tail::Matrix{ComplexF64},E_pot_tail_inv::Vector{Float64},Gνω::GνqT,
        λ₀::Array{ComplexF64,3}, kG::KGrid, mP::ModelParameters, sP::SimulationParameters)

    k_norm::Int = Nk(kG)

    χ_λ!(χ_ch, χch_tmp, λch_i)
    rhs_c1 = 0.0
    for (ωi,t) in enumerate(χ_tail)
        tmp1 = 0.0
        for (qi,km) in enumerate(kG.kMult)
            χch_i_λ = χ_ch[qi,ωi]
            tmp1 += χch_i_λ * km
        end
        rhs_c1 -= real(tmp1/k_norm - t)
    end
    rhs_c1 = rhs_c1/mP.β + mP.Ekin_DMFT*mP.β/12 + mP.n * (1 - mP.n/2)
    λsp_i = λsp_correction(χ_sp, mP.Ekin_DMFT, real(rhs_c1), kG, mP, sP)
    χ_λ!(χ_sp, χsp_tmp, λsp_i)

    #TODO: unroll 
    calc_Σ_ω!(eom, Σ_ladder_ω, Kνωq_pre, ωindices, χ_sp, γ_sp, χ_ch, γ_ch, Gνω, λ₀, mP.U, kG, sP)
    Σ_ladder[:,:] = dropdims(sum(Σ_ladder_ω, dims=[3]),dims=3) ./ mP.β .+ Σ_hartree

    lhs_c1, lhs_c2 = lhs_int(χ_sp, χ_ch, χ_tail, kG.kMult, k_norm, mP.Ekin_DMFT, mP.β)

    #TODO: the next line is expensive: Optimize G_from_Σ
    G_corr[:] = G_from_Σ(Σ_ladder.parent, kG.ϵkGrid, νGrid, mP);
    E_pot = EPot1(kG, G_corr, Σ_ladder.parent, E_pot_tail, E_pot_tail_inv, mP.β)
    rhs_c1 = mP.n/2 * (1 - mP.n/2)
    rhs_c2 = E_pot/mP.U - (mP.n/2) * (mP.n/2)
    χ_sp.data = deepcopy(χsp_tmp.data)
    χ_ch.data = deepcopy(χch_tmp.data)
    return λsp_i, lhs_c1, rhs_c1, lhs_c2, rhs_c2
end

function cond_both_int!(F::Vector{Float64}, λ::Vector{Float64}, 
        χ_sp::χT, γ_sp::γT, χ_ch::χT, γ_ch::γT,
        χsp_tmp::χT, χch_tmp::χT,
        ωindices::UnitRange{Int}, Σ_ladder_ω::OffsetArray{ComplexF64,3,Array{ComplexF64,3}}, 
        Σ_ladder::OffsetArray{ComplexF64,2,Array{ComplexF64,2}}, Kνωq_pre::Vector{ComplexF64},
        G_corr::Matrix{ComplexF64},νGrid::UnitRange{Int},χ_tail::Vector{ComplexF64},Σ_hartree::Float64,
        E_pot_tail::Matrix{ComplexF64},E_pot_tail_inv::Vector{Float64},Gνω::GνqT,
        λ₀::Array{ComplexF64,3}, kG::KGrid, mP::ModelParameters, sP::SimulationParameters, trafo::Function)::Nothing

    λi = trafo(λ)
    χ_λ!(χ_sp, χsp_tmp, λi[1])
    χ_λ!(χ_ch, χch_tmp, λi[2])
    k_norm::Int = Nk(kG)

    #TODO: unroll 
    calc_Σ_ω!(eom, Σ_ladder_ω, Kνωq_pre, ωindices, χ_sp, γ_sp, χ_ch, γ_ch, Gνω, λ₀, mP.U, kG, sP)
    Σ_ladder[:,:] = dropdims(sum(Σ_ladder_ω, dims=[3]),dims=3) ./ mP.β .+ Σ_hartree

    lhs_c1, lhs_c2 = lhs_int(χ_sp, χ_ch, χ_tail, kG.kMult, k_norm, mP.Ekin_DMFT, mP.β)

    #TODO: the next line is expensive: Optimize G_from_Σ
    G_corr[:] = G_from_Σ(Σ_ladder.parent, kG.ϵkGrid, νGrid, mP)
    E_pot = EPot1(kG, G_corr, Σ_ladder.parent, E_pot_tail, E_pot_tail_inv, mP.β)
    rhs_c1 = mP.n/2 * (1 - mP.n/2)
    rhs_c2 = E_pot/mP.U - (mP.n/2) * (mP.n/2)
    F[1] = lhs_c1 - rhs_c1
    F[2] = lhs_c2 - rhs_c2
    return nothing
end

function λsp_correction(χ_sp::χT, EKin::Float64, 
                             rhs::Float64, kG::KGrid, 
                             mP::ModelParameters, sP::SimulationParameters)
    χr::Matrix{Float64}    = real.(χ_sp[:,χ_sp.usable_ω])
    iωn = (1im .* 2 .* (-sP.n_iω:sP.n_iω)[χ_sp.usable_ω] .* π ./ mP.β)
    iωn[findfirst(x->x ≈ 0, iωn)] = Inf
    χ_tail::Vector{Float64} = real.(EKin ./ (iωn.^2))

    f_c1_int(λint::Float64)::Float64 = f_c1(χr, λint, kG.kMult, χ_tail)/mP.β - EKin*mP.β/12 - rhs
    df_c1_int(λint::Float64)::Float64 = df_c1(χr, λint, kG.kMult, χ_tail)/mP.β - EKin*mP.β/12 - rhs

    λsp = newton_right(f_c1_int, df_c1_int, get_λ_min(χr))
    return λsp
end

function cond_both_sc_int(λch_i::Float64, 
            χ_sp::χT, γ_sp::γT, χ_ch::χT, γ_ch::γT,
            Σ_loc::Vector{ComplexF64},gLoc_rfft::GνqT,
            λ₀::Array{ComplexF64,3}, kG::KGrid, mP::ModelParameters, sP::SimulationParameters;
            maxit=50, mixing=0.2, conv_abs=1e-6)

    k_norm::Int = Nk(kG)
    ωindices = usable_ωindices(sP, χ_sp, χ_ch)
    νmax::Int = minimum([sP.n_iν,floor(Int,3*length(ωindices)/8)])
    iωn = 1im .* 2 .* (-sP.n_iω:sP.n_iω)[ωindices] .* π ./ mP.β
    iωn[findfirst(x->x ≈ 0, iωn)] = Inf
    χ_tail::Vector{ComplexF64} = mP.Ekin_DMFT ./ (iωn.^2)

    χsp_tmp = deepcopy(χ_sp.data)
    χch_tmp = deepcopy(χ_ch.data)
    Σ_ladder = calc_Σ(χ_sp, γ_sp, χ_ch, γ_ch, λ₀, gLoc_rfft, kG, mP, sP);


    μnew, GLoc_old, GLoc_fft_new, gLoc_rfft_new = G_from_Σladder(Σ_ladder[:,0:νmax], Σ_loc, kG, mP, sP)
    GLoc_new = deepcopy(GLoc_old)
    println(" ========   ")
    println("checks: sum(GLoc_new) =$(abs(sum(GLoc_new))), sum(GLoc_old) =$(abs(sum(GLoc_old))), sum(Σ) = $(abs(sum(Σ_ladder)))")
    converged = false
    it     = 1
    E_pot  = NaN
    lhs_c1 = NaN
    lhs_c2 = NaN
    λsp_i  = NaN

    μ_bak = mP.μ
    while !converged
        χ_λ!(χ_ch, χch_tmp, λch_i)
        rhs_λsp = λsp_rhs(NaN, χ_sp, χ_ch, kG, mP, sP)
        λsp_i = λsp_correction(χ_sp, mP.Ekin_DMFT, real(rhs_λsp), kG, mP, sP)
        χ_λ!(χ_sp, χsp_tmp, λsp_i)
        lhs_c1, lhs_c2 = lhs_int(χ_sp.data, χ_ch.data, χ_tail, kG.kMult, k_norm, mP.Ekin_DMFT, mP.β)

        Σ_ladder_old = Σ_ladder
        Σ_ladder = calc_Σ(χ_sp, γ_sp, χ_ch, γ_ch, λ₀, gLoc_rfft, kG, mP, sP);
        Σ_ladder = (1-mixing) .* Σ_ladder .+ mixing .* Σ_ladder_old

        GLoc_old[:,:] = deepcopy(GLoc_new)
        μnew, GLoc_new, GLoc_fft_new, gLoc_rfft = G_from_Σladder(Σ_ladder[:,0:νmax], Σ_loc, kG, mP, sP; fix_n=true)
        mP.μ = μnew
        E_kin, E_pot = calc_E(GLoc_new[:,0:νmax-1].parent, Σ_ladder.parent, kG, mP, νmax = νmax)

        println("SC it=$it, convergence: $(sum(abs.(GLoc_new[:,0:10] .- GLoc_old[:,0:10]))/kG.Nk) with μ = $μnew")

        ndens = filling_pos(GLoc_new.parent, kG, mP.U, μnew, mP.β)
        println("check filling: $ndens =?= $(mP.n)")
        println("checks: sum(gLoc_rfft) =$(abs(sum(gLoc_rfft)))")
        println("checks: sum(GLoc_new) =$(abs(sum(GLoc_new))), sum(GLoc_old) =$(abs(sum(GLoc_old))), sum(Σ) = $(abs(sum(Σ_ladder)))")

        if sum(abs.(GLoc_new[:,0:10] .- GLoc_old[:,0:10]))/kG.Nk < conv_abs 
            converged = true
        end
        if it >= maxit
            break
        end
        χ_sp.data = deepcopy(χsp_tmp)
        χ_ch.data = deepcopy(χch_tmp)
        it += 1
    end
    μ = mP.μ
    mP.μ = μ_bak
    #E_pot = EPot1(kG, GLoc_new, Σ_ladder.parent, E_pot_tail, E_pot_tail_inv, mP.β)
    E_kin, E_pot = calc_E(GLoc_new[:,0:νmax-1].parent, Σ_ladder.parent, kG, mP, νmax = νmax)
    rhs_c1 = mP.n/2 * (1 - mP.n/2)
    rhs_c2 = E_pot/mP.U - (mP.n/2) * (mP.n/2)
    return λsp_i, lhs_c1, rhs_c1, lhs_c2, rhs_c2, μ, converged
end
