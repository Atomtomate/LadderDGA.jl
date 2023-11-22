# return trace, Σ_ladder, G_ladder, E_kin, E_pot, μnew, λm, lhs_c1, E_pot_2, converged

"""
    λdm_correction_old(χ_m, γ_m, χ_d, γ_d, [Σ_loc, gLoc_rfft, λ₀, kG, mP, sP] OR [h::lDΓAHelper, λ₀]; 
        maxit_root = 100, atol_root = 1e-8, λd_min_δ = 0.1, λd_max = 500,
        maxit::Int = 50, update_χ_tail=false, mixing=0.2, conv_abs=1e-8, par=false)

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
    lhs_c1   : check-sum for the Pauli-principle value obtained from the susceptibilities (`λm` fixes this to ``n/2 \\cdot (1-n/2)``) 
    E_pot_2  : Potential energy obtained from susceptibilities. `λd` fixes this to `E_pot`
    converged: error flag. False if no `λd` was found. 
    λd       : λ-correction for the density channel.
"""
function λdm_correction_old(χ_m::χT, γ_m::γT, χ_d::χT, γ_d::γT, λ₀::Array{ComplexF64,3}, h::lDΓAHelper;
                        maxit_root = 50, atol_root = 1e-8, νmax::Int = -1, λd_min_δ = 0.05, λd_max = 500,
                        maxit::Int = 50, update_χ_tail=false, mixing=0.2, conv_abs=1e-8, par=false, with_trace=false)
    λdm_correction_old(χ_m, γ_m, χ_d, γ_d, h.Σ_loc, h.gLoc_rfft, h.χloc_m_sum, λ₀, h.kG, h.mP, h.sP; 
                   maxit_root = maxit_root, atol_root = atol_root, νmax = νmax, λd_min_δ = λd_min_δ, λd_max = λd_max,
                   maxit = maxit, update_χ_tail=update_χ_tail, mixing=mixing, conv_abs=conv_abs, par=par, with_trace=with_trace)
end

function λdm_correction_old(χ_m::χT, γ_m::γT, χ_d::χT, γ_d::γT, Σ_loc::OffsetVector{ComplexF64},
                        gLoc_rfft::GνqT, χloc_m_sum::Union{Float64,ComplexF64}, λ₀::Array{ComplexF64,3},
                        kG::KGrid, mP::ModelParameters, sP::SimulationParameters; 
                        maxit_root = 50, atol_root = 1e-8,
                        νmax::Int = -1, λd_min_δ = 0.05, λd_max = 500,
                        maxit::Int = 50, update_χ_tail=false, mixing=0.2, conv_abs=1e-8, par=false, with_trace=false)

    ωindices, νGrid, iωn_f = gen_νω_indices(χ_m, χ_d, mP, sP, full=true)
    iωn = iωn_f[ωindices]
    iωn[findfirst(x->x ≈ 0, iωn)] = Inf
    iωn_m2 = 1 ./ iωn .^ 2
    gLoc_rfft_init = deepcopy(gLoc_rfft)
    par && initialize_EoM(gLoc_rfft_init, χloc_m_sum, λ₀, νGrid, kG, mP, sP, χ_m = χ_m, γ_m = γ_m, χ_d = χ_d, γ_d = γ_d)
    fft_νGrid= sP.fft_range
    G_ladder::OffsetMatrix{ComplexF64, Matrix{ComplexF64}}      = OffsetArray(Matrix{ComplexF64}(undef, length(kG.kMult), length(fft_νGrid)), 1:length(kG.kMult), fft_νGrid) 
    Σ_ladder_work::OffsetMatrix{ComplexF64, Matrix{ComplexF64}} = OffsetArray(Matrix{ComplexF64}(undef, length(kG.kMult), length(νGrid)), 1:length(kG.kMult), νGrid)
    Σ_ladder::OffsetMatrix{ComplexF64, Matrix{ComplexF64}}      = OffsetArray(Matrix{ComplexF64}(undef, length(kG.kMult), length(νGrid)), 1:length(kG.kMult), νGrid)



    trace = with_trace ? Ref(DataFrame(it = Int[], λm = Float64[], λd = Float64[], μ = Float64[], EKin = Float64[], EPot = Float64[], 
        lhs_c1 = Float64[], EPot_c2 = Float64[], cs_d = Float64[], cs_m = Float64[], cs_Σ = Float64[], cs_G = Float64[])) : Ref(nothing)
    function ff_seq(λd_i::Float64)
        copy!(gLoc_rfft_init, gLoc_rfft)
        _, _, _, _, E_pot, _, _, _, E_pot_2, _ = run_sc_old(
                    χ_m, γ_m, χ_d, γ_d, χloc_m_sum, λ₀, gLoc_rfft, Σ_loc, λd_i, kG, mP, sP, 
                maxit=maxit, mixing=mixing, conv_abs=conv_abs, update_χ_tail=update_χ_tail, trace=trace)
        return E_pot - E_pot_2
    end


    function ff_par(λd_i::Float64)
        copy!(gLoc_rfft_init, gLoc_rfft)
            _, E_pot, _, _, _, E_pot_2, _ = run_sc_old!(G_ladder, Σ_ladder_work,  Σ_ladder, gLoc_rfft_init,
                         νGrid, iωn_f, iωn_m2, χ_m, χ_d, Σ_loc, λd_i, kG, mP, sP;
                         maxit=maxit, mixing=mixing, conv_abs=conv_abs, update_χ_tail=update_χ_tail, par=true, trace=trace)
        return E_pot - E_pot_2
    end
    residual_f = par ? ff_par : ff_seq

    λd_min_tmp = get_λ_min(real(χ_d.data)) 
    println("Bracket for λdm: [",λd_min_tmp + λd_min_δ*abs(λd_min_tmp), ",", λd_max, "]")
    root = try
        newton_right(residual_f, 0.0, λd_min_tmp; nsteps=maxit_root, atol=atol_root)
    catch e
        println("Error: $e")
        println("Retrying with initial guess 0!")
        NaN
    end

    if isnan(root) #|| track.convergence_flag == :not_converged
        println("WARNING: No λd root was found!")
        reset!(χ_d)
        return trace, nothing, NaN, NaN, NaN, NaN, NaN, NaN, false, NaN
    elseif root < λd_min_tmp
        println("WARNING: λd = $root outside region ($λd_min_tmp)!")
        reset!(χ_d)
        return trace, nothing, NaN, NaN, NaN, NaN, NaN, NaN, false, NaN
    else
        copy!(gLoc_rfft_init, gLoc_rfft)
        if par
            vars = run_sc_old!(G_ladder, Σ_ladder_work,  Σ_ladder, gLoc_rfft_init, 
                         νGrid, iωn_f, iωn_m2, χ_m, χ_d, Σ_loc, root, kG, mP, sP;
                         maxit=maxit, mixing=mixing, conv_abs=conv_abs, update_χ_tail=update_χ_tail, par=true, trace=trace)
            return trace, Σ_ladder, G_ladder, vars..., root
        else
            vars = run_sc_old(
                    χ_m, γ_m, χ_d, γ_d, χloc_m_sum, λ₀, gLoc_rfft_init, Σ_loc, root, kG, mP, sP, 
                maxit=maxit, mixing=mixing, conv_abs=conv_abs, update_χ_tail=update_χ_tail, trace=trace)
            return trace, vars[2:end]..., root
        end
    end
end


# ============================================== Helpers =============================================
# --------------------------------- clean versions (slow/single core) --------------------------------
"""
TODO: refactor (especially _par version)
"""
function run_sc_old(χ_m::χT, γ_m::γT, χ_d::χT, γ_d::γT, χloc_m_sum::Union{Float64,ComplexF64}, 
                λ₀::AbstractArray{ComplexF64,3}, gLoc_rfft_init::GνqT, Σ_loc::OffsetVector{ComplexF64},
                λd::Float64, kG::KGrid, mP::ModelParameters, sP::SimulationParameters;
    maxit::Int=100, mixing::Float64=0.2, conv_abs::Float64=1e-8, update_χ_tail::Bool=false, trace::Ref=Ref(nothing))
    _, νGrid, iωn_f = gen_νω_indices(χ_m, χ_d, mP, sP)
    gLoc_rfft = deepcopy(gLoc_rfft_init)
    fft_νGrid= sP.fft_range
    G_ladder::OffsetMatrix{ComplexF64, Matrix{ComplexF64}} = OffsetArray(Matrix{ComplexF64}(undef, length(kG.kMult), length(fft_νGrid)), 1:length(kG.kMult), fft_νGrid) 
    Σ_ladder_old::OffsetMatrix{ComplexF64, Matrix{ComplexF64}} = OffsetArray(Matrix{ComplexF64}(undef, length(kG.kMult), length(νGrid)), 1:length(kG.kMult), νGrid)
    Σ_ladder::OffsetMatrix{ComplexF64, Matrix{ComplexF64}}     = OffsetArray(Matrix{ComplexF64}(undef, length(kG.kMult), length(νGrid)), 1:length(kG.kMult), νGrid)
    E_pot    = Inf
    E_pot_2  = NaN
    E_kin    = NaN
    cont     = true
    lhs_c1   = Inf
    converged = maxit == 0


    rhs_λsp = λm_rhs(NaN, χ_m, χ_d,kG, mP, sP; λd=λd)
    λm, val = λm_correction(χ_m, real(rhs_λsp), kG, mP, sP)
    if !isfinite(λm)
        @warn "no finite λm found!"
        cont = false
    end

    μbak = mP.μ
    it = 1
    while cont
        copy!(Σ_ladder_old, Σ_ladder)
        E_pot_bak  = E_pot; 
        lhs_c1_bak = lhs_c1
        Σ_ladder = calc_Σ(χ_m, γ_m, χ_d, γ_d, χloc_m_sum, λ₀, Σ_loc, gLoc_rfft, kG, mP, sP, νmax=last(νGrid)+1, λm=λm,λd=λd);
        mixing != 0 && it > 1 && (Σ_ladder[:,:] = (1-mixing) .* Σ_ladder .+ mixing .* Σ_ladder_old)
        μnew, G_ladder = G_from_Σladder(Σ_ladder, Σ_loc, kG, mP, sP; fix_n=false)
        isnan(μnew) && break
        _, gLoc_rfft = G_fft(G_ladder, kG, sP)
        E_kin, E_pot = calc_E(G_ladder, Σ_ladder, μnew, kG, mP, νmax=last(axes(Σ_ladder,2)))
        if update_χ_tail
            if isfinite(E_kin)
            update_tail!(χ_m, [0, 0, E_kin], iωn_f)
            update_tail!(χ_d, [0, 0, E_kin], iωn_f)
            rhs_λsp = λm_rhs(NaN, χ_m, χ_d,kG, mP, sP, λd=λd)
            λm, val = λm_correction(χ_m, real(rhs_λsp), kG, mP, sP)
            else
                println("Warning: unable to update χ tail: E_kin not finite")
            end
        end

        χ_m_sum    = sum_kω(kG, χ_m, λ=λm)
        χ_d_sum    = sum_kω(kG, χ_d, λ=λd)
        lhs_c1     = real(χ_d_sum + χ_m_sum)/2
        E_pot_2    = (mP.U/2)*real(χ_d_sum - χ_m_sum) + mP.U * (mP.n/2 * mP.n/2)
        if abs(E_pot - E_pot_bak) < conv_abs && abs(lhs_c1 - lhs_c1_bak) < conv_abs
            converged = true
            cont = false
        end
        (it >= maxit) && (cont = false)

        if !isnothing(trace[])
            row = [it, λm, λd, μnew, E_kin, E_pot, lhs_c1, E_pot_2, abs(χ_d_sum), abs(χ_m_sum), abs(sum(Σ_ladder)), abs(sum(G_ladder))]
            push!(trace[], row)
        end
        it += 1
    end

    update_tail!(χ_m, [0, 0, mP.Ekin_DMFT], iωn_f)
    update_tail!(χ_d, [0, 0, mP.Ekin_DMFT], iωn_f)
    μnew = mP.μ
    mP.μ = μbak
    converged = converged && all(isfinite.([lhs_c1, E_pot_2]))
    return trace, Σ_ladder, G_ladder, E_kin, E_pot, μnew, λm, lhs_c1, E_pot_2, converged
end

function run_sc_par_old(χ_m::χT, γ_m::γT, χ_d::χT, γ_d::γT, χloc_m_sum::Union{Float64,ComplexF64}, 
                    λ₀::AbstractArray{ComplexF64,3}, gLoc_rfft_init::GνqT, Σ_loc::OffsetVector{ComplexF64},
                    λd::Float64, kG::KGrid, mP::ModelParameters, sP::SimulationParameters;
                    maxit::Int=100, mixing::Float64=0.2, conv_abs::Float64=1e-8, update_χ_tail::Bool=false, par = true)
    ωindices, νGrid, iωn_f = gen_νω_indices(χ_m, χ_d, mP, sP)
    iωn = iωn_f[ωindices]
    iωn[findfirst(x->x ≈ 0, iωn)] = Inf
    iωn_m2 = 1 ./ iωn .^ 2
    gLoc_rfft_init = deepcopy(gLoc_rfft_init)
    par && initialize_EoM(gLoc_rfft_init, χloc_m_sum, λ₀, νGrid, kG, mP, sP, χ_m = χ_m, γ_m = γ_m, χ_d = χ_d, γ_d = γ_d)
    trace = DataFrame(it = Int[], λm = Float64[], λd = Float64[], μ = Float64[], EKin = Float64[], EPot = Float64[], 
        lhs_c1 = Float64[], EPot_c2 = Float64[], cs_d = Float64[], cs_m = Float64[], cs_Σ = Float64[], cs_G = Float64[])
    fft_νGrid= sP.fft_range
    G_ladder::OffsetMatrix{ComplexF64, Matrix{ComplexF64}} = OffsetArray(Matrix{ComplexF64}(undef, length(kG.kMult), length(fft_νGrid)), 1:length(kG.kMult), fft_νGrid) 
    Σ_ladder_work::OffsetMatrix{ComplexF64, Matrix{ComplexF64}} = OffsetArray(Matrix{ComplexF64}(undef, length(kG.kMult), length(νGrid)), 1:length(kG.kMult), νGrid)
    Σ_ladder::OffsetMatrix{ComplexF64, Matrix{ComplexF64}} = OffsetArray(Matrix{ComplexF64}(undef, length(kG.kMult), length(νGrid)), 1:length(kG.kMult), νGrid)


    E_kin, E_pot, μnew, λm, lhs_c1, E_pot_2, converged = run_sc_old!(G_ladder, Σ_ladder_work,  Σ_ladder, gLoc_rfft_init, 
                         νGrid, iωn_f, iωn_m2, χ_m, χ_d, Σ_loc, λd, kG, mP, sP;
                         maxit=maxit, mixing=mixing, conv_abs=conv_abs, update_χ_tail=update_χ_tail, par=true, trace=Ref(trace))
    return trace, Σ_ladder, G_ladder, E_kin, E_pot, μnew, λm, lhs_c1, E_pot_2, converged
end

function run_sc_old!(G_ladder::OffsetMatrix, Σ_ladder_work::OffsetMatrix,  Σ_ladder::OffsetMatrix, gLoc_rfft::GνqT, 
                 νGrid::AbstractVector{Int}, iωn_f::Vector{ComplexF64}, iωn_m2::Vector{ComplexF64},
                 χ_m::χT, χ_d::χT, Σ_loc::OffsetVector{ComplexF64},
                 λd::Float64, kG::KGrid, mP::ModelParameters, sP::SimulationParameters;
    maxit::Int=100, mixing::Float64=0.2, conv_abs::Float64=1e-8, update_χ_tail::Bool=false, par = true, trace::Ref=Ref(nothing))
    dbgt = TimerOutput()
    E_pot    = Inf
    E_kin    = NaN
    lhs_c1   = Inf
    E_pot_2  = NaN
    cont     = true
    converged= maxit == 0
    μbak     = mP.μ
    it       = 1
    fft_νGrid= sP.fft_range

    λd != 0 && χ_λ!(χ_d, λd)
    rhs_λsp = λm_rhs(NaN, χ_m, χ_d, kG, mP, sP)
    λm, val = λm_correction(χ_m, real(rhs_λsp), kG, mP, sP)
    if !isfinite(λm)
        @warn "no finite λm found!"
        cont = false
    end

    while cont
        E_pot_bak  = E_pot; 
        lhs_c1_bak = lhs_c1
        @timeit dbgt "01" (par && update_wcaches_G_rfft!(gLoc_rfft))
        copy!(Σ_ladder_work, Σ_ladder)
        @timeit dbgt "03" calc_Σ_par!(Σ_ladder, λm=λm, λd=λd)
        (mixing != 0 && it > 1 && (Σ_ladder[:,:] = (1-mixing) .* Σ_ladder .+ mixing .* Σ_ladder_work))
        @timeit dbgt "05" μnew = G_from_Σladder!(G_ladder, Σ_ladder, Σ_loc, kG, mP; fix_n=false)
        isnan(μnew) && break
        @timeit dbgt "06" G_rfft!(gLoc_rfft, G_ladder, kG, fft_νGrid)
        mP.μ = μnew
        # TODO: optimize calc_E (precompute tails) 
        E_kin, E_pot = calc_E(G_ladder, Σ_ladder, μnew, kG, mP, νmax=last(axes(Σ_ladder,2)))
        if update_χ_tail
            if isfinite(E_kin)
                par && update_tail!([0, 0, E_kin])
                update_tail!(χ_d, [0, 0, E_kin], iωn_f)
                update_tail!(χ_m, [0, 0, E_kin], iωn_f)
                rhs_λsp = λm_rhs(NaN, χ_m, χ_d, kG, mP, sP)
                λm, val = λm_correction(χ_m, real(rhs_λsp), kG, mP, sP)
            else
                println("Warning: unable to update χ tail: E_kin not finite")
            end
        end

        χ_m_sum    = sum_kω(kG, χ_m, λ=λm)
        χ_d_sum    = sum_kω(kG, χ_d)
        lhs_c1     = real(χ_d_sum + χ_m_sum)/2
        E_pot_2    = (mP.U/2)*real(χ_d_sum - χ_m_sum) + mP.U * (mP.n/2 * mP.n/2)
        if abs(E_pot - E_pot_bak) < conv_abs
            converged = true
            cont = false
        end
        (it >= maxit) && (cont = false)
        if !isnothing(trace[])
            row = [it, λm, λd, μnew, E_kin, E_pot, lhs_c1, E_pot_2, abs(sum(χ_d)), abs(sum(χ_m)), abs(sum(Σ_ladder)), abs(sum(G_ladder))]
            push!(trace[], row)
        end

        it += 1
    end
    if isfinite(E_kin)
        if update_χ_tail
            update_tail!(χ_m, [0, 0, mP.Ekin_DMFT], iωn_f)
            update_tail!(χ_d, [0, 0, mP.Ekin_DMFT], iωn_f)
            par && update_tail!([0, 0, mP.Ekin_DMFT])
            par && update_tail!([0, 0, mP.Ekin_DMFT])
        end
    end
    λd != 0 && reset!(χ_d)
    μnew = mP.μ
    mP.μ = μbak
    converged = converged && all(isfinite.([lhs_c1, E_pot_2]))
    return E_kin, E_pot, μnew, λm, lhs_c1, E_pot_2, converged
end
