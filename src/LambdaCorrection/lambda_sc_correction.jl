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
    λdm_sc_correction(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::λ₀T, h;
                           validation_threshold::Float64 = 1e-8,
                           max_steps_m::Int = 2000, max_steps_dm::Int = 2000, max_steps_sc::Int = 2000,
                           log_io = devnull, tc = true)

Runs partial self-consistency loop (update of propagators in equation of motion) within [`λdm correction`](@ref λdm_correction)
"""
function λm_sc_correction(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::λ₀T, h;
                           validation_threshold::Float64 = 1e-8,
                           max_steps_m::Int = 2000, max_steps_dm::Int = 2000, max_steps_sc::Int = 200,
                           log_io = devnull, tc = true)       

    λd_min = get_λ_min(χd)
    λd  = 0.0
    rhs = λm_rhs(χm, χd, h; λd=λd)
    λm  = λm_correction_val(χm, rhs, h; max_steps=max_steps_m, eps=validation_threshold)
    converged, μ_new, G_ladder_it, Σ_ladder_it = run_sc(χm, γm, χd, γd, λ₀, λm, λd, h; maxit=max_steps_sc, conv_abs=validation_threshold, tc = tc)
    return λ_result(m_scCorrection, χm, χd, μ_new, G_ladder_it, Σ_ladder_it, λm, λd, converged, h; validation_threshold = validation_threshold, max_steps_m = max_steps_m)
end

"""
    λdm_sc_correction(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::λ₀T, h;
                           validation_threshold::Float64 = 1e-8,
                           max_steps_m::Int = 2000, max_steps_dm::Int = 2000, max_steps_sc::Int = 2000,
                           log_io = devnull, tc = true)

Runs partial self-consistency loop (update of propagators in equation of motion) within [`λdm correction`](@ref λdm_correction)
"""
function λdm_sc_correction(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::λ₀T, h;
                           validation_threshold::Float64 = 1e-8,
                           max_steps_m::Int = 2000, max_steps_dm::Int = 2000, max_steps_sc::Int = 2000,
                           log_io = devnull, tc = true)       

    λd_min = get_λ_min(χd)
    ωn2_tail = ω2_tail(χm)

    function f_c2_sc(λd_i::Float64)
        rhs_c1 = λm_rhs(χm, χd, h; λd=λd_i)
        λm_i   = λm_correction_val(χm, rhs_c1, h.kG, ωn2_tail; max_steps=max_steps_m, eps=validation_threshold)
        converged, μ_new, G_ladder_it, Σ_ladder_it = run_sc(χm, γm, χd, γd, λ₀, λm_i, λd_i, h; maxit=max_steps_sc, conv_abs=validation_threshold, tc = tc)
        #TODO: use Epot_1
        Ekin_1, Epot_1 = calc_E(G_ladder_it, Σ_ladder_it, μ_new, h.kG, h.mP)
        Epot_2 = EPot_p2(χm, χd, λm_i, λd_i, h.mP.n, h.mP.U, h.kG)
        return Epot_1 - Epot_2
    end    

    λd = newton_secular(f_c2_sc, λd_min; nsteps=max_steps_dm, atol=validation_threshold)
    rhs = λm_rhs(χm, χd, h; λd=λd)
    λm  = λm_correction_val(χm, rhs, h; max_steps=max_steps_m, eps=validation_threshold)
    converged, μ_new, G_ladder_it, Σ_ladder_it = run_sc(χm, γm, χd, γd, λ₀, λm, λd, h; maxit=max_steps_sc, conv_abs=validation_threshold, tc = tc)
    return λ_result(dm_scCorrection, χm, χd, μ_new, G_ladder_it, Σ_ladder_it, λm, λd, converged, h; validation_threshold = validation_threshold, max_steps_m = max_steps_m)
end


function run_sc(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::λ₀T, λm::Float64, λd::Float64, h;
                maxit::Int=100, mixing::Float64=0.2, conv_abs::Float64=1e-8, tc::Bool = true)
    it      = 1
    converged = false

    μ_it, G_ladder_it, Σ_ladder_it = calc_G_Σ(χm, γm, χd, γd, λ₀, λm, λd, h; tc = tc, fix_n = true)
    _, gLoc_rfft = G_fft(G_ladder_it, h.kG, h.sP)
    Σ_ladder_last = deepcopy(Σ_ladder_it)

    while it <= maxit && !converged
        #TODO: inplace calc_Σ!, this clutters the code -.-
        μ_it, G_ladder_it, Σ_ladder_it = calc_G_Σ(χm, γm, χd, γd, λ₀, λm, λd, h; gLoc_rfft = gLoc_rfft, tc = tc, fix_n = true)

        !all(isfinite.(Σ_ladder_it)) && break
        all(abs.(Σ_ladder_it .- Σ_ladder_last) .< conv_abs) && (converged = true)

        G_rfft!(gLoc_rfft, G_ladder_it, h.kG, h.sP.fft_range)
        copyto!(Σ_ladder_last, Σ_ladder_it)
        it += 1
    end

    return converged, μ_it, G_ladder_it, Σ_ladder_it
end