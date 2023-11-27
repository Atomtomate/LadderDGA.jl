# ==================================================================================================== #
#                                       lambda_m_correction.jl                                         #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#   magnetic channel lambda correction.
# -------------------------------------------- TODO -------------------------------------------------- #
# ==================================================================================================== #

"""
    λm_correction(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::Array{ComplexF64,3}, h::lDΓAHelper;
                       νmax::Int=-1, λ_min_δ::Float64 = 0.0001, λ_val_only::Bool=false, verbose::Bool=false,
                       fit_μ::Bool=true, validation_threshold::Float64=1e-8, tc=true)

"""
function λm_correction(
    χm::χT,
    γm::γT,
    χd::χT,
    γd::γT,
    λ₀::Array{ComplexF64,3},
    h::lDΓAHelper;
    νmax::Int = -1,
    λ_min_δ::Float64 = 0.0001,
    λ_val_only::Bool = false,
    verbose::Bool = false,
    fit_μ::Bool = true,
    validation_threshold::Float64 = 1e-8,
    tc = true,
)

    νmax = νmax < 0 ? floor(Int, size(γm, γm.axis_types[:ν]) / 2) : νmax
    rhs = λm_rhs(χm, χd, h; λ_rhs = :native)
    λm, validation = λm_correction_val(χm, rhs, h, verbose = verbose, validation_threshold = validation_threshold, λ_min_δ = λ_min_δ)
    Σ_ladder = calc_Σ(χm, γm, χd, γd, λ₀, h, νmax = νmax, λm = λm, tc = tc)
    μnew, G_ladder = G_from_Σladder(Σ_ladder, h.Σ_loc, h.kG, h.mP, h.sP; fix_n = fit_μ)
    EKin1, EPot1 = calc_E(G_ladder, Σ_ladder, μnew, h.kG, h.mP)
    rhs_c1 = h.mP.n / 2 * (1 - h.mP.n / 2)
    χ_m_sum = sum_kω(h.kG, χm, λ = λm)
    χ_d_sum = sum_kω(h.kG, χd)
    lhs_c1 = real(χ_d_sum + χ_m_sum) / 2
    EPot2 = (h.mP.U / 2) * real(χ_d_sum - χ_m_sum) + h.mP.U * (h.mP.n / 2 * h.mP.n / 2)
    ndens = filling_pos(G_ladder, h.mP.U, μnew, h.mP.β)
    λ_result(λm, χd.λ, :m, true, validation_threshold, NaN, EKin1, EPot1, EPot2, rhs_c1, lhs_c1, nothing, G_ladder, Σ_ladder, μnew, ndens)
end


"""
    λm_correction_val(χm::χT, rhs::Float64, h::lDΓAHelper; verbose::Bool=false, validation_threshold::Float64=1e-8, λ_min_δ=1e-4)
    λm_correction_val(χm::χT, rhs::Float64, kG::KGrid, ωn2_tail; verbose::Bool=false, validation_threshold::Float64=1e-8, λ_min_δ=1e-4)
                        
Used internally for performance reasons (because the ``\\lambda_\\mathrm{d}`` correction needs this calculation repeatedly), see [`λm_correction`](@ref λm_correction) for the user-sided version.
Calculates ``\\lambda_\\mathrm{m}`` value, by fixing ``\\sum_{q,\\omega} \\chi^{\\lambda_\\mathrm{oftenm}}_{\\uparrow\\uparrow}(q,i\\omega) = \\frac{n}{2}(1-\\frac{n}{2})``.
This is only calculates the value and validation numbers and does not return a full `λ_result` object. 

Set `verbose` to obtain a trace of the checks.
`validation_threshold` sets the threshold for the `rhs ≈ lhs` condition, set to `Inf` in order to accept any result. 


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
function λm_correction_val(χm::χT, rhs::Float64, h::lDΓAHelper; validation_threshold::Float64 = 1e-8, verbose::Bool = false, λ_min_δ = 1e-4)
    iωn = (1im .* 2 .* (-h.sP.n_iω:h.sP.n_iω)[χm.usable_ω] .* π ./ h.mP.β)
    ωn2_tail::Vector{Float64} = real.(χm.tail_c[3] ./ (iωn .^ 2))
    zi = findfirst(x -> abs(x) < 1e-10, iωn)
    ωn2_tail[zi] = 0.0
    λm_correction(χm, rhs, h.kG, ωn2_tail, validation_threshold = validation_threshold, verbose = verbose, λ_min_δ = λ_min_δ)
end

function λm_correction_val(χm::χT, rhs::Float64, kG::KGrid, ωn2_tail::Vector{Float64}; validation_threshold::Float64 = 1e-8, verbose::Bool = false, λ_min_δ = 1e-4)
    λm_min = get_λ_min(χm)
    χr::SubArray{Float64,2} = view(χm, :, χm.usable_ω)

    f_c1(λint::Float64)::Float64 = sum_kω(kG, χr, χm.β, χm.tail_c[3], ωn2_tail; transform = (f(x::Float64)::Float64 = χ_λ(x, λint))) - rhs
    df_c1(λint::Float64)::Float64 = sum_kω(kG, χr, χm.β, χm.tail_c[3], ωn2_tail; transform = (f(x::Float64)::Float64 = dχ_λ(x, λint)))
    λm = newton_right(f_c1, df_c1, 0.0, λm_min, δ = λ_min_δ)

    check, check2 = if isfinite(validation_threshold) || verbose
        χ_λ!(χm, λm)
        check = sum_kω(kG, χm)
        check2 = sum_ωk(kG, χm)
        reset!(χm)
        check, check2
    else
        -Inf, Inf
    end
    if verbose
        println("CHECK for rhs = $rhs  : ", check, " ?=? ", 0)
        println("sum_kω - PP = ", abs(rhs - check))
        println("sum_ωk - PP = ", abs(rhs - check2))
    end
    validation = (abs(rhs - check) <= validation_threshold) && (abs(rhs - check2) <= validation_threshold)
    return λm, validation
end
