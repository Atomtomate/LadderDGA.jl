# ==================================================================================================== #
#                                       lambda_m_correction.jl                                         #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#   magnetic channel lambda correction.
# -------------------------------------------- TODO -------------------------------------------------- #
#   Do the debug/verbose prints with printf or logging (as in main module)
# ==================================================================================================== #


"""
    λm_rhs(χm::χT, χd::χT, h::RunHelper; λd::Float64=NaN, λ_rhs = :native, verbose=false)
    λm_rhs(imp_density::Float64, χm::χT, χd::χT, λd::Float64, kG::KGrid, mP::ModelParameters, sP::SimulationParameters, λ_rhs = :native)

Helper function for the right hand side of the Pauli principle conditions (λm correction).
`imp_density` can be set to `NaN`, if the rhs (``\\frac{n}{2}(1-\\frac{n}{2})``) should not be error-corrected (not ncessary or usefull when asymptotic improvement are active).
TODO: write down formula, explain imp_density as compensation to DMFT.
"""
function λm_rhs(χm::χT, χd::χT, h::RunHelper; λd::Float64=NaN, λ_rhs = :native, verbose=false)
    imp_density::Float64 = if typeof(h) === RPAHelper
        imp_density = NaN64
    elseif typeof(h) === lDΓAHelper
        imp_density = h.imp_density
    else
        error("RunHelper type not implemented!")
    end
    λm_rhs(imp_density, χm, χd, h.kG, h.mP, h.sP; λd=λd, λ_rhs=λ_rhs, verbose=verbose)
end

function λm_rhs(imp_density::Float64, χm::χT, χd::χT, kG::KGrid, mP::ModelParameters, sP::SimulationParameters; λd::Float64 = NaN, λ_rhs::Symbol = :native, verbose::Bool = false)
    χd.λ != 0 && !isnan(λd) && error("Stopping λ rhs calculation: λd = $λd AND χd.λ = $(χd.λ). Reset χd.λ, or do not provide additional λ-correction for this function.")
    χd_sum = sum_kω(kG, χd, λ = λd)

    verbose && @info "λsp correction infos:"
    rhs, PP_p1 = if (((typeof(sP.χ_helper) != Nothing) && λ_rhs == :native) || λ_rhs == :fixed)
        verbose && @info "  ↳ using n * (1 - n/2) - Σ χd as rhs" # As far as I can see, the factor 1/2 has been canceled on both sides of the equation for the Pauli principle => update output
        rhs_i = mP.n/2 * (1 - mP.n / 2)
        rhs_i*2 - χd_sum,  rhs_i
    else
        !isfinite(imp_density) && throw(ArgumentError("imp_density argument is not finite! Cannot use DMFT rror compensation method"))
        verbose && @info "  ↳ using χupup_DMFT - Σ χd as rhs"
        2 * imp_density - χd_sum, imp_density
    end

    if verbose
        @info """  ↳ Found usable intervals for non-local susceptibility of length 
                     ↳ sp: $(χm.usable_ω), length: $(length(χm.usable_ω))
                     ↳ ch: $(χd.usable_ω), length: $(length(χd.usable_ω))
                   ↳ χd sum = $(χd_sum), rhs = $(rhs)"""
    end
    return rhs, PP_p1
end

"""
    λm_correction(χm::χT,γm::γT,χd::χT,γd::γT,λ₀::λ₀T, h::lDΓAHelper;
                  νmax::Int = eom_ν_cutoff(h), fix_n::Bool = true, tc = default_Σ_tail_correction(), 
                  validation_threshold::Float64 = 1e-8, max_steps::Int = 2000, 
                  verbose=false, log_io= devnull
"""
function λm_correction(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::λ₀T, h;
                       νmax::Int = eom_ν_cutoff(h), fix_n::Bool = true, tc::Type{<: ΣTail} = default_Σ_tail_correction(), λ_rhs::Symbol = :native, 
                       validation_threshold::Float64 = 1e-8, max_steps::Int = 2000, verbose::Bool=false, log_io= devnull
)
    rhs,PP_p1 = λm_rhs(χm, χd, h; λ_rhs = λ_rhs, verbose=verbose)
    λm  = λm_correction_val(χm, rhs, h; max_steps=max_steps, eps=validation_threshold)
    return λ_result(mCorrection, χm, γm, χd, γd, λ₀, λm, χd.λ, true, h; 
            tc = tc, PP_p1 = PP_p1, validation_threshold = validation_threshold, max_steps_m = max_steps)
end

"""
    λm_correction_val(χm::χT, rhs::Float64, h::lDΓAHelper)
    λm_correction_val(χm::χT, rhs::Float64, kG::KGrid, ωn2_tail)
                        
Used internally for performance reasons (because the ``\\lambda_\\mathrm{d}`` correction needs this calculation repeatedly), see [`λm_correction`](@ref λm_correction) for the user-sided version.
Calculates ``\\lambda_\\mathrm{m}`` value, by fixing ``\\sum_{q,\\omega} \\chi^{\\lambda,\\omega}_{\\uparrow\\uparrow}(q,i\\omega) = \\frac{n}{2}(1-\\frac{n}{2})``.
This is only calculates the value and validation numbers and does not return a full `λ_result` object. 


TODO: finish docu

Arguments
-------------
- **`χm`**        :
- **`rhs`**       :
- **`h`**         :
- **`ωn2_tail `** :
- **`verbose`**   :
- **`ωn2_tail `** :
"""
function λm_correction_val(χm::χT, rhs::Float64, h; 
                           max_steps::Int=1000, eps::Float64=1e-8)
    ωn2_tail = ω2_tail(χm)
    λm_correction_val(χm, rhs, h.kG, ωn2_tail; max_steps=max_steps, eps=eps)
end

function λm_correction_val(χm::χT, rhs::Float64, kG::KGrid, ωn2_tail::Vector{Float64}; 
                           max_steps::Int=1000, eps::Float64=1e-8)
    λm_min = get_λ_min(χm)
    χr::SubArray{Float64,2} = view(χm, :, χm.usable_ω)

    f_c1(λint::Float64)::Float64 = sum_kω(kG, χr, χm.β, χm.tail_c[3], ωn2_tail; transform = (f(x::Float64)::Float64 = χ_λ(x, λint))) - rhs
    df_c1(λint::Float64)::Float64 = sum_kω(kG, χr, χm.β, χm.tail_c[3], ωn2_tail; transform = (f(x::Float64)::Float64 = dχ_λ(x, λint)))
    λm = newton_secular(f_c1, df_c1, λm_min; nsteps=max_steps, atol=eps)
    return λm
end

function λm_correction_val2(χm::χT, rhs::Float64, h; 
                           max_steps::Int=1000, eps::Float64=1e-8)
    ωn2_tail = ω2_tail(χm)
    λm_correction_val2(χm, rhs, h.kG, ωn2_tail; max_steps=max_steps, eps=eps)
end
function λm_correction_val2(χm::χT, rhs::Float64, kG::KGrid, ωn2_tail::Vector{Float64}; 
                           max_steps::Int=1000, eps::Float64=1e-8)
    λm_min = get_λ_min(χm)
    χr::SubArray{Float64,2} = view(χm, :, χm.usable_ω)

    f_c1(λint::Float64)::Float64 = sum_kω(kG, χr, χm.β, χm.tail_c[3], ωn2_tail; transform = (f(x::Float64)::Float64 = 1.5 * χ_λ(x, λint))) - rhs
    df_c1(λint::Float64)::Float64 = sum_kω(kG, χr, χm.β, χm.tail_c[3], ωn2_tail; transform = (f(x::Float64)::Float64 = 1.5 * dχ_λ(x, λint)))
    λm = newton_secular(f_c1, df_c1, λm_min; nsteps=max_steps, atol=eps)
    return λm
end
