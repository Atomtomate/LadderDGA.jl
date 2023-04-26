# ==================================================================================================== #
#                                           conditions.jl                                              #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#   lambda-correction conditions for several methods, fixing different physical properties.            #
# -------------------------------------------- TODO -------------------------------------------------- #
#  REFACTOR!!!!!                                                                                       #
#  Optimize Calc_E and remainder of run_sc!                                                            #
# ==================================================================================================== #

# =========================================== Interface ==============================================
mutable struct λ_result
    λm::Float64
    λd::Float64
    type::Symbol
    converged::Bool
    EKin::Float64
    EPot_p1::Float64
    EPot_p2::Float64
    PP_p1::Float64
    PP_p2::Float64
    trace::Union{DataFrame,Nothing}
    G_ladder::Union{Nothing,OffsetMatrix}
    Σ_ladder::Union{Nothing, OffsetMatrix}
    μ::Float64
    function λ_result(λm::Float64, λd::Float64, type::Symbol, converged::Bool)
        new(λm, λd, type, converged, NaN, NaN, NaN, NaN, NaN, nothing, nothing, nothing, NaN)
    end
    function λ_result(λm::Float64, λd::Float64, type::Symbol, converged::Bool, 
                      EKin::Float64, EPot_p1::Float64, EPot_p2::Float64, PP_p1::Float64, PP_p2::Float64, 
                      trace::Union{DataFrame,Nothing}, 
                      G_ladder::Union{Nothing, OffsetMatrix}, Σ_ladder::Union{Nothing,OffsetMatrix}, μ::Float64)
        new(λm, λd, type, converged, EKin, EPot_p1, EPot_p2, PP_p1, PP_p2, trace, G_ladder, Σ_ladder, μ)
    end
end

function λ_correction(type::Symbol, χm::χT, γm::γT, χd::χT, γd::γT, λ₀, h::lDΓAHelper; 
                      λ_val_only::Bool=false, λm_rhs_type::Symbol=:native, 
                      verbose::Bool=false, validate_threshold::Float64=1e-8)
    if type == :m
        rhs = λm_rhs(χm, χd, 0.0, h; λ_rhs = λm_rhs_type)
        λm, validation = λm_correction(χm, rhs, h, verbose=verbose, validate_threshold=validate_threshold)
        if λ_val_only
            return λ_result(λm, 0.0, :m, validation)
        else
            error("Full result for λm not imlemented yet")
        end
    else
    end
end

# =============================================== λm =================================================
"""
    λm_correction(χm::χT, rhs::Float64, h::lDΓAHelper; verbose::Bool=false, validate_threshold::Float64=1e-8)
    λm_correction(χm::χT, rhs::Float64, kG::KGrid, mP::ModelParameters, sP::SimulationParameters; verbose::Bool=false, validate_threshold::Float64=1e-8)
                        
Calculates ``\\lambda_\\mathrm{m}`` value, by fixing ``\\sum_{q,\\omega} \\chi^{\\lambda_\\mathrm{m}}_{\\uparrow\\uparrow}(q,i\\omega) = \\frac{n}{2}(1-\\frac{n}{2})``.

Set `verbose` to obtain a trace of the checks.
`validate_threshold` sets the threshold for the `rhs ≈ lhs` condition, set to `Inf` in order to accept any result. 
"""
function λm_correction(χm::χT, rhs::Float64, h::lDΓAHelper; verbose::Bool=false, validate_threshold::Float64=1e-8)
    λm_correction(χm, rhs, h.kG, h.mP, h.sP, validate_threshold=validate_threshold)
end

function λm_correction(χm::χT, rhs::Float64, kG::KGrid, mP::ModelParameters, sP::SimulationParameters; verbose::Bool=false, validate_threshold::Float64=1e-8)
    λm_min = get_λ_min(χm)
    χr::SubArray{Float64,2}    = view(χm,:,χm.usable_ω)
    iωn = (1im .* 2 .* (-sP.n_iω:sP.n_iω)[χm.usable_ω] .* π ./ mP.β)
    ωn2_tail::Vector{Float64} = real.(χm.tail_c[3] ./ (iωn.^2))
    zero_ind = findfirst(x->!isfinite(x), ωn2_tail)
    ωn2_tail[zero_ind] = 0.0

    f_c1(λint::Float64)::Float64 = sum_kω(kG, χr, χm.β, χm.tail_c[3], ωn2_tail; transform=(f(x::Float64)::Float64 = χ_λ(x, λint))) - rhs
    df_c1(λint::Float64)::Float64 = sum_kω(kG, χr, χm.β, χm.tail_c[3], ωn2_tail; transform=(f(x::Float64)::Float64 = dχ_λ(x, λint)))
    λm = newton_right(f_c1, df_c1, 0.0, λm_min)

    check, check2 = if isfinite(validate_threshold) || verbose
        χ_λ!(χm, λm)
        check  = sum_kω(kG, χm)
        check2 = sum_ωk(kG, χm)
        reset!(χm)
        check, check2
    else
        -Inf, Inf
    end
    if verbose
        println("CHECK for rhs = $rhs  : ", check, " => 0 ?=? ", abs(rhs - check), " (sum_kω) ?=? ", abs(rhs - check2), " (sum_ωk).")
    end
    validation = (abs(rhs - check) <= validate_threshold) &&  (abs(rhs - check2) <= validate_threshold) 
    return λm, validation
end

# =============================================== λm =================================================

"""
    λdm_correction(χ_m, γ_m, χ_d, γ_d, [Σ_loc, gLoc_rfft, λ₀, kG, mP, sP] OR [h::lDΓAHelper, λ₀]; 
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
function λdm_correction_new(χ_m::χT, γ_m::γT, χ_d::χT, γ_d::γT, λ₀::Array{ComplexF64,3}, h::lDΓAHelper;
                        maxit_root = 50, atol_root = 1e-8, νmax::Int = -1, λd_min_δ = 0.05, λd_max = 500,
                        maxit::Int = 50, update_χ_tail=false, mixing=0.2, conv_abs=1e-8, par=false, with_trace=false)
    λdm_correction_new(χ_m, γ_m, χ_d, γ_d, h.Σ_loc, h.gLoc_rfft, h.χloc_m_sum, λ₀, h.kG, h.mP, h.sP; 
                   maxit_root = maxit_root, atol_root = atol_root, νmax = νmax, λd_min_δ = λd_min_δ, λd_max = λd_max,
                   maxit = maxit, update_χ_tail=update_χ_tail, mixing=mixing, conv_abs=conv_abs, par=par, with_trace=with_trace)
end
function λdm_correction_new(χ_m::χT, γ_m::γT, χ_d::χT, γ_d::γT, Σ_loc::Vector{ComplexF64},
                        gLoc_rfft::GνqT, χloc_m_sum::Union{Float64,ComplexF64}, λ₀::Array{ComplexF64,3},
                        kG::KGrid, mP::ModelParameters, sP::SimulationParameters; 
                        maxit_root = 50, atol_root = 1e-8,
                        νmax::Int = -1, λd_min_δ = 0.05, λd_max = 500,
                        maxit::Int = 50, update_χ_tail=false, mixing=0.2, conv_abs=1e-8, par=false, verbose::Bool=false)

    ωindices, νGrid, iωn_f = gen_νω_indices(χ_m, χ_d, mP, sP)
    iωn = iωn_f[ωindices]
    iωn[findfirst(x->x ≈ 0, iωn)] = Inf
    iωn_m2 = 1 ./ iωn .^ 2
    gLoc_rfft_init = deepcopy(gLoc_rfft)
    par && initialize_EoM(gLoc_rfft_init, χloc_m_sum, λ₀, νGrid, kG, mP, sP, χ_m = χ_m, γ_m = γ_m, χ_d = χ_d, γ_d = γ_d)
    fft_νGrid= sP.fft_range
    G_ladder::OffsetMatrix{ComplexF64, Matrix{ComplexF64}}      = OffsetArray(Matrix{ComplexF64}(undef, length(kG.kMult), length(fft_νGrid)), 1:length(kG.kMult), fft_νGrid) 
    Σ_ladder_work::OffsetMatrix{ComplexF64, Matrix{ComplexF64}} = OffsetArray(Matrix{ComplexF64}(undef, length(kG.kMult), length(νGrid)), 1:length(kG.kMult), νGrid)
    Σ_ladder::OffsetMatrix{ComplexF64, Matrix{ComplexF64}}      = OffsetArray(Matrix{ComplexF64}(undef, length(kG.kMult), length(νGrid)), 1:length(kG.kMult), νGrid)

    λd_min_tmp = get_λ_min(real(χ_d.data)) 
    verbose && println("Bracket for λdm: [",λd_min_tmp + λd_min_δ*abs(λd_min_tmp), ",", λd_max, "]")
    root = try
        #find_zero(residual_f, (λd_min_tmp + λd_min_δ*abs(λd_min_tmp), λd_max), Roots.A42(), maxiters=maxit_root, atol=atol_root, tracks=track)
        newton_right(residual_f, 0.0, λd_min_tmp; nsteps=maxit_root, atol=atol_root)
    catch e
        println("Error: $e")
        println("Retrying with initial guess 0!")
        NaN
    end

    if isnan(root) #|| track.convergence_flag == :not_converged
        println("WARNING: No λd root was found!")
        reset!(χ_d)
    elseif root < λd_min_tmp
        println("WARNING: λd = $root outside region ($λd_min_tmp)!")
        reset!(χ_d)
    end
end



# =============================================== sc =================================================
function run_sc_new(χm::χT, γm::γT, χd::χT, γd::γT, 
                    gLoc_rfft_init::GνqT, λ₀::Array{ComplexF64,3}, h::lDΓAHelper;
                    maxit::Int=100, mixing::Float64=0.2, conv_abs::Float64=1e-8, trace=true)
    E_pot_1 = Inf
    E_pot_2 = Inf
    rhs_c1  = h.mP.n/2*(1-h.mP.n/2)
    E_kin   = Inf
    μnew    = h.mP.μ
    it      = 1
    done    = false
    converged = false
    _, νGrid, iωn_f = gen_νω_indices(χm, χd, h.mP, h.sP)
    gLoc_rfft = deepcopy(gLoc_rfft_init)
    fft_νGrid= h.sP.fft_range
    Nk = length(h.kG.kMult)
    G_ladder::OffsetMatrix{ComplexF64, Matrix{ComplexF64}} = OffsetArray(Matrix{ComplexF64}(undef, Nk, length(fft_νGrid)), 1:Nk, fft_νGrid) 
    Σ_ladder::OffsetMatrix{ComplexF64, Matrix{ComplexF64}} = OffsetArray(Matrix{ComplexF64}(undef, Nk, length(νGrid)),     1:Nk, νGrid)
    Σ_ladder_work = similar(Σ_ladder)

    traceDF = DataFrame(it = Int[], λm = Float64[], λd = Float64[], μ = Float64[], EKin = Float64[], EPot = Float64[], 
        lhs_c1 = Float64[], EPot_c2 = Float64[], cs_m = Float64[], cs_m2 = Float64[],
        cs_d = Float64[], cs_d2 = Float64[], cs_Σ = Float64[], cs_G = Float64[])

    while !done
        copy!(Σ_ladder_work, Σ_ladder)

        Σ_ladder = calc_Σ(χm, γm, χd, γd, h.χloc_m_sum, λ₀, gLoc_rfft, h.kG, h.mP, h.sP, νmax=last(νGrid)+1) 
        mixing != 0 && it > 1 && (Σ_ladder[:,:] = (1-mixing) .* Σ_ladder .+ mixing .* Σ_ladder_work)
        μnew = G_from_Σladder!(G_ladder, Σ_ladder, h.Σ_loc, h.kG, h.mP; fix_n=true)
        isnan(μnew) && break
        _, gLoc_rfft = G_fft(G_ladder, h.kG, h.sP)
        E_pot_1_old = E_pot_1
        E_kin, E_pot_1 = calc_E(G_ladder[:,νGrid].parent, Σ_ladder.parent, μnew, h.kG, h.mP, νmax = last(νGrid)+1)

        if trace
            χ_m_sum  = sum_kω(h.kG, χm)
            χ_d_sum  = sum_kω(h.kG, χd)
            χ_m_sum2 = sum_ωk(h.kG, χm)
            χ_d_sum2 = sum_ωk(h.kG, χd)
            lhs_c1   = real(χ_d_sum + χ_m_sum)/2
            E_pot_2    = (h.mP.U/2)*real(χ_d_sum - χ_m_sum) + h.mP.U * (h.mP.n/2 * h.mP.n/2)
            row = [it, χm.λ, χd.λ, μnew, E_kin, E_pot_1, lhs_c1, E_pot_2, χ_m_sum, χ_m_sum2, χ_d_sum, χ_d_sum2, abs(sum(Σ_ladder)), abs(sum(G_ladder))]
            push!(traceDF, row)
        end

        # if it != 1 && abs(sum(Σ_ladder .- Σ_ladder_work))/(h.kG.Nk) < conv_abs  
        if abs(E_pot_1 - E_pot_1_old) < conv_abs && abs(rhs_c1 - lhs_c1) < conv_abs
            converged = true
            done = true
        end
        (it >= maxit) && (done = true)

        it += 1
    end

    if isfinite(E_kin)
        χ_m_sum = sum_kω(h.kG, χm)
        χ_d_sum = sum_kω(h.kG, χd)
        lhs_c1  = real(χ_d_sum + χ_m_sum)/2
        E_pot_2    = (h.mP.U/2)*real(χ_d_sum - χ_m_sum) + h.mP.U * (h.mP.n/2 * h.mP.n/2)
    end
    update_tail!(χm, [0, 0, h.mP.Ekin_DMFT], iωn_f)
    update_tail!(χd, [0, 0, h.mP.Ekin_DMFT], iωn_f)
    converged = converged && all(isfinite.([lhs_c1, E_pot_2]))
    return λ_result(χm.λ, χd.λ, :test, converged, E_kin, E_pot_1, E_pot_2, rhs_c1, lhs_c1, 
                    traceDF, G_ladder, Σ_ladder, μnew)
end
