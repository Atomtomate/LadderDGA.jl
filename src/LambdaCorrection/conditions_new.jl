# ==================================================================================================== #
#                                           conditions.jl                                              #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#   lambda-correction conditions for several methods, fixing different physical properties.            #
# -------------------------------------------- TODO -------------------------------------------------- #
#  REFACTOR!!!!!                                                                                       #
#  Pack results of lambda-correction into result struct                                                #
#  Optimize Calc_E and remainder of run_sc!                                                            #
# ==================================================================================================== #

mutable struct λ_result
    λm::Float64
    λd::Float64
    type::Symbol
    EKin::Float64
    EPot::Float64
    residual_EPot::Float64
    residual_PP::Float64
    trace::DataFrame
    G_ladder::OffsetMatrix
    Σ_ladder::OffsetMatrix
    μ::Float64
end

"""
    λm_correction(χ_m::χT, rhs::Float64, kG::KGrid, mP::ModelParameters, sP::SimulationParameters)
                        
Calculates ``\\lambda_\\mathrm{m}`` value, by fixing ``\\sum_{q,\\omega} \\chi^{\\lambda_\\mathrm{m}}_{\\uparrow\\uparrow}(q,i\\omega) = \\frac{n}{2}(1-\\frac{n}{2})``.
"""
function λm_correction(χ_m::χT, rhs::Float64, kG::KGrid, 
                        mP::ModelParameters, sP::SimulationParameters)

    λm_min = get_λ_min(χ_m)
    χr::Matrix{Float64}    = real.(χ_m[:,χ_m.usable_ω])
    iωn = (1im .* 2 .* (-sP.n_iω:sP.n_iω)[χ_m.usable_ω] .* π ./ mP.β)
    iωn[findfirst(x->x ≈ 0, iωn)] = Inf
    χ_tail::Vector{Float64} = real.(χ_m.tail_c[3] ./ (iωn.^2))
    f_c1_int(λint::Float64)::Float64 = f_c1(χr, λint, kG.kMult, χ_tail)/mP.β - χ_m.tail_c[3]*mP.β/12 - rhs
    df_c1_int(λint::Float64)::Float64 = df_c1(χr, λint, kG.kMult, χ_tail)/mP.β - χ_m.tail_c[3]*mP.β/12 - rhs

    λm = newton_right(f_c1_int, df_c1_int, 0.0, λm_min)
    check = sum_kω(kG, χ_m, λ=λm)
    println("CHECK for rhs = $rhs  : ", check, " => 0 ", abs(rhs - check))
    return λm
end

