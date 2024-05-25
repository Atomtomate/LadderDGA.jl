# ==================================================================================================== #
#                                      lambda_dm_correction.jl                                         #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#   magnetic channel lambda correction.
# -------------------------------------------- TODO -------------------------------------------------- #
#   Do the debug/verbose prints with printf or logging (as in main module)
# ==================================================================================================== #

"""
    λdm_tsc_correction(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::Array{ComplexF64,3}, h, sP, mP;
                        validation_threshold::Float64 = 1e-8, log_io = devnull
    )


"""
function λdm_tsc_correction(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::λ₀T, h;
                           validation_threshold::Float64 = 1e-8,
                           max_steps_m::Int = 2000, max_steps_dm::Int = 2000, max_steps_sc::Int = 2000,
                           log_io = devnull, tc = true)       

    λd_min = get_λ_min(χd)
    
    function f_c2_sc(λd_i::Float64)
        converged, μ_new, λm_i, G_ladder_it, Σ_ladder_it, χm_it, χd_it = run_tsc(χm, γm, χd, γd, λ₀, λd_i, h; maxit=max_steps_sc, max_steps_m = max_steps_m, conv_abs=validation_threshold, tc = tc)
        #TODO: use Epot_1
        Ekin_1, Epot_1 = calc_E(G_ladder_it, Σ_ladder_it, μ_new, h.kG, h.mP)
        Epot_2 = EPot_p2(χm_it, χd_it, λm_i, λd_i, h.mP.n, h.mP.U, h.kG)
        if !converged
            @warn "internal tsc did not converge"
            return (Epot_1 - Epot_2)*10
        end
        return Epot_1 - Epot_2
    end    
    λd = newton_secular(f_c2_sc, λd_min; nsteps=max_steps_dm, atol=validation_threshold)
    converged, μ_new, λm, G_ladder_it, Σ_ladder_it, χm_it, χd_it = run_tsc(χm, γm, χd, γd, λ₀, λd, h; maxit=max_steps_sc, conv_abs=validation_threshold, tc = tc)

    return λ_result(dm_tscCorrection, χm_it, χd_it, μ_new, G_ladder_it, Σ_ladder_it, λm, λd, converged, h; validation_threshold = validation_threshold, max_steps_m = max_steps_m)
end


function run_tsc(χm_bak::χT, γm::γT, χd_bak::χT, γd::γT, λ₀::λ₀T, λd::Float64, h;
                maxit::Int=100, mixing::Float64=0.2, conv_abs::Float64=1e-8, 
                max_steps_m=1000, tc::Bool = true)
    it        = 1
    λm_i      = 0.0
    converged = false

    μ_it, G_ladder_it, Σ_ladder_it = calc_G_Σ(χm_bak, γm, χd_bak, γd, λ₀, λm_i, λd, h; tc = tc, fix_n = true)
    _, gLoc_rfft = G_fft(G_ladder_it, h.kG, h.sP)
    Σ_ladder_last = deepcopy(Σ_ladder_it)
    iωn_grid = ωn_grid(χm_bak)

    # internal λm-correction stuff
    PP_p1  = h.mP.n / 2 * (1 - h.mP.n / 2)
    
    # in case we encounter NaN/Inf in tail
    χm_int = deepcopy(χm_bak)
    χd_int = deepcopy(χd_bak)

    while it <= maxit && !converged
        χd_sum   = sum_kω(h.kG, χd_int, λ = λd)
        rhs_c1   = h.mP.n * (1 - h.mP.n / 2) - χd_sum
        ωn2_tail = ω2_tail(χm_int)
        λm_i     = λm_correction_val(χm_int, rhs_c1, h.kG, ωn2_tail; max_steps=max_steps_m, eps=conv_abs)

        #TODO: inplace calc_Σ!, this clutters the code -.-
        μ_it, G_ladder_it, Σ_ladder_it = calc_G_Σ(χm_int, γm, χd_int, γd, λ₀, λm_i, λd, h; gLoc_rfft = gLoc_rfft, tc = tc, fix_n = true)

        if !all(isfinite.(Σ_ladder_it)) 
            @error "EoM self-consistency encountered NaN/Inf in self-energy."
            break
        end

        # Check for sc and λm convergence
        χm_sum = sum_kω(h.kG, χm_int, λ = λm_i)
        PP_p2  = real(χd_sum + χm_sum) / 2
        if sum(abs.(Σ_ladder_it .- Σ_ladder_last)) < conv_abs && (abs(PP_p1 - PP_p2) < conv_abs)
            converged = true
        end

        G_rfft!(gLoc_rfft, G_ladder_it, h.kG, h.sP.fft_range)
        copyto!(Σ_ladder_last, Σ_ladder_it)
        Ekin_1, Epot_1 = calc_E(G_ladder_it, Σ_ladder_it, μ_it, h.kG, h.mP)

        if !isfinite(Ekin_1)
            @error "Kinetic energy not finite! Aborting λ-tsc"
            break
        else
            update_tail!(χm_int, [0, 0, Ekin_1], iωn_grid)
            update_tail!(χd_int, [0, 0, Ekin_1], iωn_grid)
        end
        it += 1
    end

    return converged, μ_it, λm_i, G_ladder_it, Σ_ladder_it, χm_int, χd_int
end