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
                 λm_rhs_type::Symbol=:native, fit_μ::Bool=true, 
                 νmax::Int=eom_ν_cutoff(h), λ_min_δ::Float64 = 0.0001,
                 maxit::Int=100, mixing::Float64=0.2, conv_abs::Float64=1e-8, trace::Bool=false,
                 λ_val_only::Bool=false, verbose::Bool=false, validate_threshold::Float64=1e-8, tc::Bool=true)

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
             λm_rhs_type::Symbol=:native, fit_μ::Bool=true, 
             νmax::Int=eom_ν_cutoff(h), λ_min_δ::Float64 = 0.0001,
             maxit::Int=100, mixing::Float64=0.2, conv_abs::Float64=1e-8, trace::Bool=false,
             λ_val_only::Bool=false, verbose::Bool=false, validate_threshold::Float64=1e-8, tc::Bool=true)

    if type == :m
        λm_correction_full(χm, γm, χd, γd, λ₀, h;
                           fit_μ=fit_μ,  
                           νmax=νmax, λ_min_δ=λ_min_δ, verbose=verbose,
                           validate_threshold=validate_threshold, tc=tc)
    elseif type == :dm
        λdm_correction(χm, γm, χd, γd, λ₀, h; 
                       fit_μ=fit_μ,  
                       νmax=νmax, λ_min_δ=λ_min_δ,
                       validate_threshold=validate_threshold, 
                       verbose=verbose, λ_val_only=λ_val_only, tc=tc)
    elseif type == :sc
        run_sc(χm, γm, χd, γd, λ₀, h; maxit=maxit, mixing=mixing, conv_abs=conv_abs, trace=trace)
    elseif type == :sc_m
        run_sc(χm, γm, χd, γd, λ₀, h; type=:m, maxit=maxit, mixing=mixing, conv_abs=conv_abs, trace=trace)
    else
        error("λ-correction type '$type' not recognized!")
    end
end


# ============================================== λm ==================================================
include("lambda_m_correction.jl")
# ============================================== λdm =================================================
include("lambda_dm_correction.jl")
# ============================================= λdmsc ================================================

# ====================================== Results Validation ==========================================
function validation_to_string(PP_p1, PP_p2, EPot_p1, EPot_p2, sums_check)

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

Returns ``\\sum_k \\sum_\\omega \\chi^{\\lambda_r,\\omega}_{r,q} - \\sum_\\omega \\sum_k \\chi^{\\lambda_r,\\omega}_{r,q}``.
"""
function validate_PP(χm::χT, χd::χT, λm::Float64, λd::Float64, n::Float64)
    error("not implemented yet")
end

"""
    validate_EPot(χm::χT, χd::χT, λm::Float64, n::Float64)

Returns ``\\sum_k \\sum_\\omega \\chi^{\\lambda_r,\\omega}_{r,q} - \\sum_\\omega \\sum_k \\chi^{\\lambda_r,\\omega}_{r,q}``.
"""
function validate_EPot(EPot_P1::Float64, χm::χT, χd::χT, λm::Float64, n::Float64)
    error("not implemented yet")
end
