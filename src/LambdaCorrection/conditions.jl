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
    λm_correction(χsp::χT, rhs::Float64, kG::KGrid, mP::ModelParameters, sP::SimulationParameters)
                        
Calculates ``\\lambda_\\mathrm{m}`` value, by fixing ``\\sum_{q,\\omega} \\chi^{\\lambda_\\mathrm{m}}_{\\uparrow\\downarrow}(q,i\\omega) = \\frac{n}{2}(1-\\frac{n}{2})``.
"""
function λm_correction(χsp::χT, rhs::Float64, kG::KGrid, 
                        mP::ModelParameters, sP::SimulationParameters)
    χr::Matrix{Float64}    = real.(χsp[:,χsp.usable_ω])
    iωn = (1im .* 2 .* (-sP.n_iω:sP.n_iω)[χsp.usable_ω] .* π ./ mP.β)
    iωn[findfirst(x->x ≈ 0, iωn)] = Inf
    χ_tail::Vector{Float64} = real.(χsp.tail_c[3] ./ (iωn.^2))
    f_c1_int(λint::Float64)::Float64 = f_c1(χr, λint, kG.kMult, χ_tail)/mP.β - χsp.tail_c[3]*mP.β/12 - rhs
    df_c1_int(λint::Float64)::Float64 = df_c1(χr, λint, kG.kMult, χ_tail)/mP.β - χsp.tail_c[3]*mP.β/12 - rhs

    λm = newton_right(f_c1_int, df_c1_int, get_λ_min(χr))
    return λm
end

"""
    λdm_correction(χ_m, γ_m, χ_d, γ_d, Σ_loc, gLoc_rfft, λ₀, kG, mP, sP; 
        maxit_root = 100, atol_root = 1e-8, λd_min_δ = 0.1, λd_max = 500,
        maxit::Int = 50, update_χ_tail=false, mixing=0.2, conv_abs=1e-6, par=false)

Calculates ``\\lambda_\\mathrm{dm}`` and associated quantities like the self-energy.

TODO: full documentation. Pack results into struct

Returns: 
-------------
    Σ_ladder : ladder self-energy
    G_ladder : ladder Green's function obtained from `Σ_ladder`
    E_kin    : kinetic energy, unless `update_χ_tail = true`, this will be not consistent with the susceptibility tail coefficients.
    E_pot    : one-particle potential energy, obtained through galitskii-migdal formula
    μnew:    : chemical potential of `G_ladder`
    λm       : λ-correction for the magnetic channel
    lhs_c1   : check-sum for the Pauli-principle value obtained from the susceptibilities (`λm` fixes this to ``n/2 \cdot (1-n/2)``) 
    E_pot_2  : Potential energy obtained from susceptibilities. `λd` fixes this to `E_pot`
    converged: error flag. False if no `λd` was found. 
    λd       : λ-correction for the density channel.
"""
function λdm_correction(χ_m::χT, γ_m::γT, χ_d::χT, γ_d::γT, Σ_loc::Vector{ComplexF64},
                        gLoc_rfft::GνqT, λ₀::Array{ComplexF64,3},
                        kG::KGrid, mP::ModelParameters, sP::SimulationParameters; 
                        maxit_root = 50, atol_root = 1e-7,
                        νmax::Int = -1, λd_min_δ = 0.1, λd_max = 500,
                        maxit::Int = 50, update_χ_tail=false, mixing=0.2, conv_abs=1e-6, par=false)
    run_f = par ? run_sc_par : run_sc
    function ff(λd_i::Float64)
        _, _, _, E_pot, _, _, _, E_pot_2, _ = run_f(
                    χ_m, γ_m, χ_d, γ_d, λ₀, gLoc_rfft, Σ_loc, λd_i, kG, mP, sP, 
                maxit=maxit, mixing=mixing, conv_abs=conv_abs, update_χ_tail=update_χ_tail)
        rhs_c2 = E_pot/mP.U - (mP.n/2) * (mP.n/2)
        return rhs_c2 - E_pot_2
    end
    λd_min_tmp = get_λ_min(real(χ_d.data)) 
    track = Roots.Tracks()
    println("Bracket for λdm: [",λd_min_tmp + λd_min_δ*abs(λd_min_tmp), ",", λd_max, "]")
    root = try
        find_zero(ff, (λd_min_tmp + λd_min_δ*abs(λd_min_tmp), λd_max), Roots.A42(), maxiters=maxit_root, atol=atol_root, tracks=track)
    catch e
        println("Error: $e")
        println("Track: $track")
        NaN
    end
    root = if isnan(root)
        try
            find_zero(ff, 0.0, maxiters=maxit_root, atol=atol_root, tracks=track)
        catch e
            println("Error: $e")
            println("Track: $track")
            NaN
        end
    end

    if isnan(root) || track.convergence_flag == :not_converged
        println("WARNING: No λd root was found! Track:")
        println(track)
        nothing, nothing, NaN, NaN, NaN, NaN, NaN, NaN, false, NaN
    elseif root < λd_min_tmp
        println("WARNING: λd = $root outside region ($λd_min_tmp)!")

        nothing, nothing, NaN, NaN, NaN, NaN, NaN, NaN, false, NaN
    else
         run_f(χ_m, γ_m, χ_d, γ_d, λ₀, gLoc_rfft, Σ_loc, root, kG, mP, sP, 
               maxit=maxit, mixing=mixing, conv_abs=conv_abs, update_χ_tail=update_χ_tail)..., root
    end
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

    lhs_c1, E_pot_2 = lhs_int(χsp, χch, λi[1], λi[2], χ_tail, kG.kMult, k_norm, χsp.tail_c[3], mP.β)
    Σ_ladder = calc_Σ(χsp, γsp, χch, γch, λ₀, gLoc_rfft, kG, mP, sP, νmax=sP.n_iν, λm=λi[1],λd=λi[2]);
    _, G_ladder = G_from_Σladder(Σ_ladder, Σ_loc, kG, mP, sP; fix_n=false)
    _, E_pot = calc_E(G_ladder[:,νGrid].parent, Σ_ladder.parent, kG, mP, νmax = last(νGrid)+1)
    rhs_c1 = mP.n/2 * (1 - mP.n/2)
    rhs_c2 = E_pot/mP.U - (mP.n/2) * (mP.n/2)
    F[1] = lhs_c1 - rhs_c1
    F[2] = E_pot_2 - rhs_c2
    return nothing
end

"""
TODO: refactor (especially _par version)
"""
function run_sc(χsp::χT, γsp::γT, χch::χT, γch::γT, λ₀::AbstractArray{ComplexF64,3}, gLoc_rfft_init::GνqT, Σ_loc::Vector{ComplexF64},
                λd::Float64, kG::KGrid, mP::ModelParameters, sP::SimulationParameters;
                maxit::Int=100, mixing::Float64=0.2, conv_abs::Float64=1e-6, update_χ_tail::Bool=false)
    ωindices, νGrid, iωn_f = gen_νω_indices(χsp, χch, mP, sP)
    iωn = iωn_f[ωindices]
    iωn[findfirst(x->x ≈ 0, iωn)] = Inf
    gLoc_rfft = deepcopy(gLoc_rfft_init)
    G_ladder = nothing
    Σ_ladder_old::OffsetMatrix{ComplexF64, Matrix{ComplexF64}} = OffsetArray(Matrix{ComplexF64}(undef, length(kG.kMult), length(νGrid)), 1:length(kG.kMult), νGrid)
    Σ_ladder::OffsetMatrix{ComplexF64, Matrix{ComplexF64}} = OffsetArray(Matrix{ComplexF64}(undef, length(kG.kMult), length(νGrid)), 1:length(kG.kMult), νGrid)
    E_pot    = Inf
    E_kin    = Inf
    cont = true
    converged = maxit == 0

    χ_λ!(χch, χch, λd)
    rhs_λsp = λsp_rhs(NaN, χsp, χch, kG, mP, sP)
    λm = λm_correction(χsp, real(rhs_λsp), kG, mP, sP)
    reset!(χch)
    if !isfinite(λm)
        @warn "no finite λm found!"
        cont = false
    end

    μbak = mP.μ
    it = 1
    while cont
        Σ_ladder_old = deepcopy(Σ_ladder)
        Σ_ladder = calc_Σ(χsp, γsp, χch, γch, λ₀, gLoc_rfft, kG, mP, sP, νmax=last(νGrid)+1, λm=λm,λd=λd);
        mixing != 0 && it > 1 && (Σ_ladder[:,:] = (1-mixing) .* Σ_ladder .+ mixing .* Σ_ladder_old)
        μnew, G_ladder = G_from_Σladder(Σ_ladder, Σ_loc, kG, mP, sP; fix_n=true)
        isnan(μnew) && break
        _, gLoc_rfft = G_fft(G_ladder, kG, mP, sP)
        mP.μ = μnew
        E_kin, E_pot = calc_E(G_ladder[:,νGrid].parent, Σ_ladder.parent, kG, mP, νmax = last(νGrid)+1)
        if update_χ_tail
            update_tail!(χsp, [0, 0, E_kin], iωn_f)
            update_tail!(χch, [0, 0, E_kin], iωn_f)
            χ_λ!(χch, χch, λd)
            rhs_λsp = λsp_rhs(NaN, χsp, χch, kG, mP, sP)
            λm = λm_correction(χsp, real(rhs_λsp), kG, mP, sP)
            reset!(χch)
        end
        if it != 1
            # println("SC it = $it, conv = $(sum(abs.(Σ_ladder .- Σ_ladder_old))/(size(Σ_ladder_old,2)*kG.Nk)), μ = $μnew")
            # ndens = filling_pos(G_ladder.parent, kG, mP.U, μnew, mP.β)
            # println("  -> check filling: $(round(ndens,digits=4)) =?= $(round(mP.n,digits=4)), λm = $(round(λm,digits=4)), λd = $(round(λd,digits=4))")
            if sum(abs.(Σ_ladder .- Σ_ladder_old))/(size(Σ_ladder_old,2)*kG.Nk) < conv_abs  
                converged = true
                cont = false
            end
        end

        (it >= maxit) && (cont = false)
        it += 1
    end

    lhs_c1 = NaN
    E_pot_2 = NaN
    if isfinite(E_kin)
        χ_tail::Vector{ComplexF64} = χsp.tail_c[3]./ (iωn.^2)
        lhs_c1, E_pot_2 = lhs_int(χsp.data, χch.data, λm, λd, 
                            χ_tail, kG.kMult, Nk(kG), χsp.tail_c[3], mP.β)
    end
    update_tail!(χsp, [0, 0, mP.Ekin_DMFT], iωn_f)
    update_tail!(χch, [0, 0, mP.Ekin_DMFT], iωn_f)
    μnew = mP.μ
    mP.μ = μbak
    converged = converged && all(isfinite.([lhs_c1, E_pot_2]))
    return Σ_ladder, G_ladder, E_kin, E_pot, μnew, λm, lhs_c1, E_pot_2, converged
end

function run_sc_par(χsp::χT, γsp::γT, χch::χT, γch::γT, λ₀::AbstractArray{ComplexF64,3}, gLoc_rfft_init::GνqT, Σ_loc::Vector{ComplexF64},
                    λd::Float64, kG::KGrid, mP::ModelParameters, sP::SimulationParameters;
                    maxit::Int=100, mixing::Float64=0.2, conv_abs::Float64=1e-6, update_χ_tail::Bool=false)

    ωindices, νGrid, iωn_f = gen_νω_indices(χsp, χch, mP, sP)
    iωn = iωn_f[ωindices]
    iωn[findfirst(x->x ≈ 0, iωn)] = Inf
    initialize_EoM(gLoc_rfft_init, 
                λ₀, νGrid, kG, mP, sP, 
                χsp = χsp, γsp = γsp,
                χch = χch, γch = γch)
    gLoc_rfft = deepcopy(gLoc_rfft_init)
    G_ladder = nothing
    Σ_ladder_old::OffsetMatrix{ComplexF64, Matrix{ComplexF64}} = OffsetArray(Matrix{ComplexF64}(undef, length(kG.kMult), length(νGrid)), 1:length(kG.kMult), νGrid)
    Σ_ladder::OffsetMatrix{ComplexF64, Matrix{ComplexF64}} = OffsetArray(Matrix{ComplexF64}(undef, length(kG.kMult), length(νGrid)), 1:length(kG.kMult), νGrid)
    E_pot    = Inf
    E_kin    = Inf
    cont = true
    converged = maxit == 0

    χ_λ!(χch, χch, λd)
    rhs_λsp = λsp_rhs(NaN, χsp, χch, kG, mP, sP)
    λm = λm_correction(χsp, real(rhs_λsp), kG, mP, sP)
    reset!(χch)
    if !isfinite(λm)
        @warn "no finite λm found!"
        cont = false
    end

    μbak = mP.μ
    it = 1
    while cont
        update_wcaches_G_rfft!(gLoc_rfft)
        Σ_ladder_old[:,:] = deepcopy(Σ_ladder)
        Σ_ladder[:,:] = calc_Σ_par(kG, mP, sP, λm=λm, λd=λd, νrange=νGrid);
        mixing != 0 && it > 1 && (Σ_ladder[:,:] = (1-mixing) .* Σ_ladder .+ mixing .* Σ_ladder_old)
        μnew, G_ladder = G_from_Σladder(Σ_ladder, Σ_loc, kG, mP, sP; fix_n=true)
        isnan(μnew) && break
        _, gLoc_rfft = G_fft(G_ladder, kG, mP, sP)
        mP.μ = μnew
        E_kin, E_pot = calc_E(G_ladder[:,νGrid].parent, Σ_ladder.parent, kG, mP, νmax = last(νGrid)+1)
        if update_χ_tail
            update_tail!([0, 0, E_kin])
            update_tail!(χch, [0, 0, E_kin], iωn_f)
            χ_λ!(χch, χch, λd)
            rhs_λsp = λsp_rhs(NaN, χsp, χch, kG, mP, sP)
            λm = λm_correction(χsp, real(rhs_λsp), kG, mP, sP)
            reset!(χch)
        end
        if Σ_ladder_old !== nothing
            # println("SC it = $it, conv = $(sum(abs.(Σ_ladder .- Σ_ladder_old))/(size(Σ_ladder_old,2)*kG.Nk)), μ = $μnew")
            # ndens = filling_pos(G_ladder.parent, kG, mP.U, μnew, mP.β)
            # println("  -> check filling: $(round(ndens,digits=4)) =?= $(round(mP.n,digits=4)), λm = $(round(λm,digits=4)), λd = $(round(λd,digits=4))")
            if sum(abs.(Σ_ladder .- Σ_ladder_old))/(size(Σ_ladder_old,2)*kG.Nk) < conv_abs  
                converged = true
                cont = false
            end
        end
        (it >= maxit) && (cont = false)
        it += 1
    end
    lhs_c1 = NaN
    E_pot_2 = NaN
    if isfinite(E_kin)
        if update_χ_tail
            update_tail!(χsp, [0, 0, E_kin], iωn_f)
            update_tail!(χch, [0, 0, E_kin], iωn_f)
        end
        χ_tail::Vector{ComplexF64} = χsp.tail_c[3] ./ (iωn.^2)
        lhs_c1, E_pot_2 = lhs_int(χsp.data, χch.data, λm, λd, 
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
    return Σ_ladder, G_ladder, E_kin, E_pot, μnew, λm, lhs_c1, E_pot_2, converged
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
function lhs_int(χsp::Matrix, χch::Matrix, λm::Float64, λd::Float64, 
                 χ_tail::Vector{ComplexF64}, kMult::Vector{Float64}, k_norm::Int, 
                 Ekin::Float64, β::Float64)
    lhs_c1 = 0.0
    lhs_c2 = 0.0
    for (ωi,t) in enumerate(χ_tail)
        tmp1 = 0.0
        tmp2 = 0.0
        for (qi,km) in enumerate(kMult)
            χsp_i_λ = 1 ./ (λm + 1/real(χsp[qi,ωi]))
            χch_i_λ = 1 ./ (λd + 1/real(χch[qi,ωi]))
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

