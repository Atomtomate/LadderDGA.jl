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
    λm_correction(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::Array{ComplexF64,3}, h::lDΓAHelper;
                       νmax::Int=-1, λ_min_δ::Float64 = 0.0001, λ_val_only::Bool=false, verbose::Bool=false,
                       fit_μ::Bool=true, validation_threshold::Float64=1e-8, tc=true)

"""
function λm_correction(χm::χT,γm::γT,χd::χT,γd::γT,λ₀::Array{ComplexF64,3},h::lDΓAHelper;
                       νmax::Int = eom_ν_cutoff(h), fit_μ::Bool = true, tc = true, validation_threshold::Float64 = 1e-8, verbose::Bool = false
)
    rhs  = λm_rhs(χm, χd, h; λ_rhs = :native)
    λm   = λm_correction_val(χm, rhs, h)
    Σ_ladder = calc_Σ(χm, γm, χd, γd, λ₀, h, νmax = νmax, λm = λm, tc = tc)
    μnew, G_ladder   = G_from_Σladder(Σ_ladder, h.Σ_loc, h.kG, h.mP, h.sP; fix_n = fit_μ)
    χ_m_sum = sum_kω(h.kG, χm, λ = λm)
    χ_d_sum = sum_kω(h.kG, χd)
    PP_p1  = h.mP.n / 2 * (1 - h.mP.n / 2)
    PP_p2  = real(χ_d_sum + χ_m_sum) / 2
    EKin_p1, EPot_p1 = calc_E(G_ladder, Σ_ladder, μnew, h.kG, h.mP)
    EPot_p2 = (h.mP.U / 2) * real(χ_d_sum - χ_m_sum) + h.mP.U * (h.mP.n / 2 * h.mP.n / 2)
    ndens       = filling_pos(G_ladder[:,0:h.sP.n_iν], h.kG, h.mP.U, μnew, h.mP.β, improved_sum=true)
    res = λ_result(λm, χd.λ, mCorrection, true, validation_threshold, NaN, EKin_p1, EPot_p1, EPot_p2, PP_p1, PP_p2, nothing, G_ladder, Σ_ladder, μnew, ndens, h.mP.n)
    if verbose
        n_check, pp_check, epot_check = validate(res)
        ndens_check = filling_pos(G_ladder[:,0:h.sP.n_iν], h.kG, h.mP.U, μnew, h.mP.β, improved_sum=false)
        println("Checking λm correction. Sums check χm: ", round(validate_sums(h.kG, χm, λm),digits=6), ", χd: ", round(validate_sums(h.kG,χd),digits=6))
        println("   density p2 = ", round(ndens,digits=6), "(impr.), " , round(ndens_check,digits=6), "(naive), p1 = ", round(h.mP.n,digits=6), " (check = $n_check)")
        println("   Pauli-Principle p1  = $(round(PP_p1,digits=6))")
        println("   Pauli-Principle p2  = $(round(PP_p2,digits=6))", " (check = $pp_check)")
        println("   Potential Energy p1 = $(round(EPot_p1,digits=6))")
        println("   Potential Energy p2 = $(round(EPot_p2,digits=6))", " (check = $epot_check)")
    end
    return res
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
function λm_correction_val(χm::χT, rhs::Float64, h::lDΓAHelper)
    # pre-compute 1/ω^2 tail. sum_kω can do this internally, but this prevents repeated calculation
    iωn = (1im .* 2 .* (-h.sP.n_iω:h.sP.n_iω)[χm.usable_ω] .* π ./ h.mP.β)
    ωn2_tail::Vector{Float64} = real.(χm.tail_c[3] ./ (iωn .^ 2))
    # exclude 0-frequency
    zi = findfirst(x -> abs(x) < 1e-10, iωn)
    ωn2_tail[zi] = 0.0
    λm_correction_val(χm, rhs, h.kG, ωn2_tail)
end

function λm_correction_val(χm::χT, rhs::Float64, kG::KGrid, ωn2_tail::Vector{Float64})
    λm_min = get_λ_min(χm)
    χr::SubArray{Float64,2} = view(χm, :, χm.usable_ω)

    f_c1(λint::Float64)::Float64 = sum_kω(kG, χr, χm.β, χm.tail_c[3], ωn2_tail; transform = (f(x::Float64)::Float64 = χ_λ(x, λint))) - rhs
    df_c1(λint::Float64)::Float64 = sum_kω(kG, χr, χm.β, χm.tail_c[3], ωn2_tail; transform = (f(x::Float64)::Float64 = dχ_λ(x, λint)))
    λm = newton_secular(f_c1, df_c1, λm_min)

    return λm
end
