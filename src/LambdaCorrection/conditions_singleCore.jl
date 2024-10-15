# ==================================================================================================== #
#                                           conditions.jl                                              #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#   lambda-correction conditions for several methods, fixing different physical properties.            #
# -------------------------------------------- TODO -------------------------------------------------- #
# ==================================================================================================== #


# =========================================== Interface ==============================================
"""
    λ_correction(type::Symbol, χm::χT, γm::γT, χd::χT, γd::γT, λ₀, h::lDΓAHelper; 
                 λm_rhs_type::Symbol=:native, fix_n::Bool=true, 
                 νmax::Int=eom_ν_cutoff(h), λ_min_δ::Float64 = 0.0001,
                 maxit::Int=100, mixing::Float64=0.2, conv_abs::Float64=1e-8, trace::Bool=false,
                 λ_val_only::Bool=false, verbose::Bool=false, validation_threshold::Float64=1e-8, tc::Bool=true)

Executes λ-correction. 
TODO: finish docu

Arguments
-------------
- **`type`** : `Symbol`, options are `:m`, `:dm`, `:m_sc`, `:dm_sc`, `:m_tsc` and `:dm_tsc`
- **`χm`**   :
- **`γm`**   :
- **`χd`**   :
- **`γd`**   :
"""
function λ_correction(type::Symbol, χm::χT, γm::γT, χd::χT, γd::γT, λ₀, h::lDΓAHelper; 
             λm_rhs_type::Symbol=:native, fix_n::Bool=true, 
             νmax::Int=eom_ν_cutoff(h), λ_min_δ::Float64 = 0.0001,
             maxit::Int=100, mixing::Float64=0.2, conv_abs::Float64=1e-8, trace::Bool=false,
             λ_val_only::Bool=false, log_io = devnull, validation_threshold::Float64=1e-8, tc::ΣTail=default_Σ_tail_correction())

    if type == :m
        if  λ_val_only 
            rhs, PP_p1  = λm_rhs(χm, χd, h; λ_rhs = λm_rhs_type)
            λm_correction_val(χm, rhs, h)
        else
            λm_correction(χm, γm, χd, γd, λ₀, h;
                νmax = νmax, log_io = log_io, 
                fix_n = fix_n, validation_threshold = validation_threshold, tc = tc
            )
        end
    elseif type == :dm
        if λ_val_only
            λdm_correction_val(χm, γm, χd, γd, λ₀, h; fix_n=fix_n,  
                           validation_threshold=validation_threshold, 
                           log_io=log_io, tc=tc)
        else
            λdm_correction(χm, γm, χd, γd, λ₀, h; fix_n=fix_n,  
                           validation_threshold=validation_threshold, 
                           log_io=log_io, tc=tc)
        end
    elseif type == :sc
        @error "λ-correction type '$type' not implemented through wrapper"
    elseif type == :sc_m
        @error "λ-correction type '$type' not implemented through wrapper"
    else
        error("λ-correction type '$type' not recognized!")
    end
end

# ========================================== Conditions ==============================================
"""
    Cond_PauliPrinciple(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::λ₀T, h;
                       λm::Float64 = NaN, λd::Float64 = NaN, λ_rhs::Symbol = :native,
                       νmax::Int = eom_ν_cutoff(h), fix_n::Bool = true, tc::Type{<: ΣTail} = default_Σ_tail_correction(), λ_rhs::Symbol = :native, 
                       validation_threshold::Float64 = 1e-8, max_steps::Int = 2000, verbose::Bool=false, log_io= devnull
         )
    Cond_EPot(...)
    Cond_EKin(...)
    Cond_Tail(...)

Each of the functiosn returns a `Tuple` with the first element being the one-particle
and the second the two-particle quantity of the respective quantity:
- PauliPrinciple: ``\\frac{n}{2} \\left(1 - \\frac{n}{2} \\right), \\sum_{\\omega,q}\\left( \\chi_m + \\chi_d \\right)``
- EPot: ``\\sum_{\\nu,k} G^\\lambda \\Sigma^\\lambda, \\sum_{\\omega,q}\\left( \\chi_m - \\chi_d \\right)``
- EKin: ``\\sum_{\\nu,k} \\epsilon_k G^\\lambda, \\lim_{m \\to \\infty} (i \\omega_m)^2 \\chi_m``
- Tail: ``\\sum_{\\omega} \\chi^\\omega_\\mathrm{DMFT}, \\sum_{\\omega,q} \\chi^{\\lambda_m,\\omega}_{q}``
- Causal: ``\\max \\Im \\Sigma^{\\nu_0}_k, ``\\max \\Im \\Sigma^{\\nu_0}_{k \\approx k_F}``, see also [`estimate_ef`](@ref estimate_ef)
"""
function Cond_PauliPrinciple(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::λ₀T, h;
                       λm::Float64 = NaN, λd::Float64 = NaN, λ_rhs::Symbol = :native,
                       νmax::Int = eom_ν_cutoff(h), fix_n::Bool = true, tc::Type{<: ΣTail} = default_Σ_tail_correction(),
                       validation_threshold::Float64 = 1e-8, max_steps::Int = 2000, verbose::Bool=false, log_io= devnull
         )

         χm_sum = sum_kω(h.kG, χm, λ = λm)
         χd_sum = sum_kω(h.kG, χd, λ = λd)
         rhs,PP_p1 = λm_rhs(χm, χd, h; λ_rhs = λ_rhs, verbose=verbose)
         PP_p2  = real(χd_sum + χm_sum) / 2

    return real(PP_p1), PP_p2
end

function Cond_EPot(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::λ₀T, h; 
                λm::Float64 = NaN, λd::Float64 = NaN, 
                νmax::Int = eom_ν_cutoff(h), fix_n::Bool = true, tc::Type{<: ΣTail} = default_Σ_tail_correction(),
                validation_threshold::Float64 = 1e-8, max_steps_m::Int = 2000,
                max_steps_dm::Int = 2000, log_io = devnull
        )
    μ_new, G_ladder, Σ_ladder = calc_G_Σ(χm, γm, χd, γd, λ₀, λm, λd, h; νmax=νmax, tc = tc, fix_n = fix_n)
    Ekin_p1, Epot_p1 = calc_E(G_ladder, Σ_ladder, μ_new, h.kG, h.mP)
    Epot_p2 = EPot_p2(χm, χd, λm, λd, h.mP.n, h.mP.U, h.kG)
    return Epot_p1, Epot_p2
end

function Cond_EKin(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::λ₀T, h; 
                λm::Float64 = NaN, λd::Float64 = NaN, 
                νmax::Int = eom_ν_cutoff(h), fix_n::Bool = true, tc::Type{<: ΣTail} = default_Σ_tail_correction(),
                validation_threshold::Float64 = 1e-8, max_steps_m::Int = 2000,
                max_steps_dm::Int = 2000, log_io = devnull
        )
    μ_new, G_ladder, Σ_ladder = calc_G_Σ(χm, γm, χd, γd, λ₀, λm, λd, h; νmax=νmax, tc = tc, fix_n = fix_n)
    Ekin_p1, Epot_p1 = calc_E(G_ladder, Σ_ladder, μ_new, h.kG, h.mP)
    Ekin_p2 = χm.tail_c[3]
    return Ekin_p1, Ekin_p2
end


function Cond_Tail(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::λ₀T, h; 
                λm::Float64 = NaN, λd::Float64 = NaN, 
                νmax::Int = eom_ν_cutoff(h), fix_n::Bool = true, tc::Type{<: ΣTail} = default_Σ_tail_correction(),
                validation_threshold::Float64 = 1e-8, max_steps_m::Int = 2000,
                max_steps_dm::Int = 2000, log_io = devnull
        )
        χm_sum = sum_kω(h.kG, χm, λ = λm)
        χm_loc_sum = h.χloc_m_sum
        return χm_loc_sum, χm_sum
end

function Cond_Causal(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::λ₀T, h; 
    λm::Float64 = NaN, λd::Float64 = NaN, 
    νmax::Int = eom_ν_cutoff(h), fix_n::Bool = true, tc::Type{<: ΣTail} = default_Σ_tail_correction(),
    validation_threshold::Float64 = 1e-8, max_steps_m::Int = 2000,
    max_steps_dm::Int = 2000, log_io = devnull
)
    μ_new, G_ladder, Σ_ladder = calc_G_Σ(χm, γm, χd, γd, λ₀, λm, λd, h; νmax=νmax, tc = tc, fix_n = fix_n)
    ef = estimate_ef(Σ_ladder, h.kG, μ_new, h.mP.β)
    SC_2 = any(ef) ? maximum(imag.(Σ_ladder[ef,0])) : NaN
    return maximum(imag.(Σ_ladder[:,0])), SC_2
end

# ============================================== λm ==================================================
include("lambda_m_correction.jl")
# ============================================== λdm =================================================
include("lambda_dm_correction.jl")
# ============================================= λdmsc ================================================
include("lambda_sc_correction.jl")
# ============================================= λdmtsc ===============================================
include("lambda_tsc_correction.jl")

# ====================================== Results Validation ==========================================
function validation_to_string(PP_p1, PP_p2, EPot_p1, EPot_p2, sums_check)
    error("Not implemented yet.")
end

"""
    validate_sums(kG::KGrid, χr::χT[, λr::Float64])

Returns ``\\sum_k \\sum_\\omega \\chi^{\\lambda_r,\\omega}_{r,q} - \\sum_\\omega \\sum_k \\chi^{\\lambda_r,\\omega}_{r,q}``.
"""
function validate_sums(kG::KGrid, χr::χT, λr::Float64)
    χ_λ!(χr, λr)
    res = validate_sums(kG, χr)
    reset!(χr)
    return res
end

function validate_sums(kG::KGrid, χr::χT)
    check1 = sum_kω(kG, χr)
    check2 = sum_ωk(kG, χr)
    return check1 - check2
end

"""
    validate_PP(χm::χT, χd::χT, λm::Float64, λd::Float64, n::Float64)

Returns .
"""
function validate_PP(χm::χT, χd::χT, λm::Float64, λd::Float64, n::Float64)
    error("not implemented yet")
end

"""
    validate_EPot(χm::χT, χd::χT, λm::Float64, n::Float64)

Returns .
"""
function validate_EPot(EPot_P1::Float64, χm::χT, χd::χT, λm::Float64, n::Float64)
    error("not implemented yet")
end

function validate_positivity()
    error("not implemented yet")
end
