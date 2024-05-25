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
    λdm_correction(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::Array{ComplexF64,3}, h;
                        validation_threshold::Float64 = 1e-8, max_steps_m::Int = 2000, max_steps_dm::Int = 2000, log_io = devnull
    )

Computes the `λm` and `λd` parameters for the consistency of Pauli principle and potential energie on one- and two-particle level.
Returns a [`λ_result`](@ref λ_result) object.
"""
function λdm_correction(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::λ₀T, h;
                        validation_threshold::Float64 = 1e-8, max_steps_m::Int = 2000, max_steps_dm::Int = 2000, log_io = devnull
    )       
    λm, λd = λdm_correction_val(χm, γm, χd, γd,λ₀, h;
                validation_threshold = validation_threshold, max_steps_m = max_steps_m,
                max_steps_dm = max_steps_dm, log_io = log_io)     
    return λ_result(dmCorrection, χm, γm, χd, γd, λ₀, λm, λd, true, h; validation_threshold = validation_threshold, max_steps_m = max_steps_m)
end

"""
    λdm_correction_val(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::Array{ComplexF64,3}, h;
                        validation_threshold::Float64 = 1e-8, max_steps_m::Int = 2000, max_steps_dm::Int = 2000, log_io = devnull
    )

Computes the `λm` and `λd` parameters for the consistency of Pauli principle and potential energie on one- and two-particle level.
Returns the bare λ-values, usually one should run [`λdm_correction`](@ref λdm_correction), which returns a [`λ_result`](@ref λ_result) object 
that stores additional consistency checks.
"""
function λdm_correction_val(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::λ₀T, h;
                        validation_threshold::Float64 = 1e-8, max_steps_m::Int = 2000, 
                        max_steps_dm::Int = 2000, log_io = devnull)       
    λd_min   = get_λ_min(χd)
    ωn2_tail = ω2_tail(χm)

    function f_c2(λd_i::Float64)
        rhs_c1 = λm_rhs(χm, χd, h; λd=λd_i)
        λm_i   = λm_correction_val(χm, rhs_c1, h.kG, ωn2_tail; max_steps=max_steps_m, eps=validation_threshold)
        μ_new, G_ladder, Σ_ladder = calc_G_Σ(χm, γm, χd, γd, λ₀, λm_i, λd_i, h; tc = true, fix_n = true)
        #TODO: use Epot_1
        Ekin_1, Epot_1 = calc_E(G_ladder, Σ_ladder, μ_new, h.kG, h.mP)
        Epot_2 = EPot_p2(χm, χd, λm_i, λd_i, h.mP.n, h.mP.U, h.kG)
        return Epot_1 - Epot_2
    end
    
    λd  = newton_secular(f_c2, λd_min; nsteps=max_steps_dm, atol=validation_threshold)
    rhs = λm_rhs(χm, χd, h; λd=λd)
    λm  = λm_correction_val(χm, rhs, h; max_steps=max_steps_m, eps=validation_threshold)
    return λm, λd
end