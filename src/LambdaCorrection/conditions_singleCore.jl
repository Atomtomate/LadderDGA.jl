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
                 νmax::Int=-1, λ_min_δ::Float64 = 0.0001,
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
             νmax::Int=-1, λ_min_δ::Float64 = 0.0001,
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
"""
    λm_correction_val(χm::χT, rhs::Float64, h::lDΓAHelper; verbose::Bool=false, validate_threshold::Float64=1e-8)
    λm_correction_val(χm::χT, rhs::Float64, kG::KGrid, ωn2_tail; verbose::Bool=false, validate_threshold::Float64=1e-8)
                        
Used internally for performance reasons (because the ``\\lambda_\\mathrm{d}`` correction needs this calculation repeatedly), see [`λm_correction`](@ref λm_correction) for the user-sided version.
Calculates ``\\lambda_\\mathrm{m}`` value, by fixing ``\\sum_{q,\\omega} \\chi^{\\lambda_\\mathrm{oftenm}}_{\\uparrow\\uparrow}(q,i\\omega) = \\frac{n}{2}(1-\\frac{n}{2})``.
This is only calculates the value and validation numbers and does not return a full `λ_result` object. 

Set `verbose` to obtain a trace of the checks.
`validate_threshold` sets the threshold for the `rhs ≈ lhs` condition, set to `Inf` in order to accept any result. 

Arguments
-------------
- **`χm`**        :
- **`rhs`**       :
- **`h`**         :
- **`ωn2_tail `** :
- **`verbose`**   :
- **`ωn2_tail `** :
"""
function λm_correction_val(χm::χT, rhs::Float64, h::lDΓAHelper; validate_threshold::Float64 = 1e-8, verbose::Bool = false)
    iωn = (1im .* 2 .* (-h.sP.n_iω:h.sP.n_iω)[χm.usable_ω] .* π ./ h.mP.β)
    ωn2_tail::Vector{Float64} = real.(χm.tail_c[3] ./ (iωn .^ 2))
    zi = findfirst(x -> abs(x) < 1e-10, iωn)
    ωn2_tail[zi] = 0.0
    λm_correction(χm, rhs, h.kG, ωn2_tail, validate_threshold = validate_threshold, verbose = verbose)
end

function λm_correction_val(χm::χT, rhs::Float64, kG::KGrid, ωn2_tail::Vector{Float64}; validate_threshold::Float64 = 1e-8, verbose::Bool = false)
    λm_min = get_λ_min(χm)
    χr::SubArray{Float64,2} = view(χm, :, χm.usable_ω)

    f_c1(λint::Float64)::Float64 = sum_kω(kG, χr, χm.β, χm.tail_c[3], ωn2_tail; transform = (f(x::Float64)::Float64 = χ_λ(x, λint))) - rhs
    df_c1(λint::Float64)::Float64 = sum_kω(kG, χr, χm.β, χm.tail_c[3], ωn2_tail; transform = (f(x::Float64)::Float64 = dχ_λ(x, λint)))
    λm = newton_right(f_c1, df_c1, 0.0, λm_min)

    return λm
end

# ============================================== λdm =================================================
# ============================================= λdmsc ================================================
