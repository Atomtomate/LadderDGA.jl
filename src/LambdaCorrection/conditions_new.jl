# ==================================================================================================== #
#                                           conditions.jl                                              #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#   lambda-correction conditions for several methods, fixing different physical properties.            #
# -------------------------------------------- TODO -------------------------------------------------- #
#  REFACTOR!!!!!                                                                                       #
#  Optimize Calc_E and remainder of run_sc!                                                            #
# ==================================================================================================== #

mutable struct λ_result
    λm::Float64
    λd::Float64
    type::Symbol
    converged::Bool
    EKin::Float64
    EPot_p1::Float64
    EPot_p2::Float64
    PP_p1::Float64
    PP_p2::Float64
    trace::Union{DataFrame,Nothing}
    G_ladder::Union{Nothing,OffsetMatrix}
    Σ_ladder::Union{Nothing, OffsetMatrix}
    μ::Float64
    function λ_result(λm::Float64, λd::Float64, type::Symbol, converged::Bool)
        new(λm, λd, type, converged, NaN, NaN, NaN, NaN, NaN, nothing, nothing, nothing, NaN)
    end
    function λ_result(λm::Float64, λd::Float64, type::Symbol, converged::Bool, 
                      EKin::Float64, EPot_p1::Float64, EPot_p2::Float64, PP_p1::Float64, PP_p2::Float64, 
                      trace::Union{DataFrame,Nothing}, 
                      G_ladder::Union{Nothing, OffsetMatrix}, Σ_ladder::Union{Nothing,OffsetMatrix}, μ::Float64)
        new(λm, λd, type, converged, EKin, EPot_p1, EPot_p2, PP_p1, PP_p2, trace, G_ladder, Σ_ladder, μ)
    end
end

function λ_correction(type::Symbol, χm::χT, γm::γT, χd::χT, γd::γT, gLoc_rfft, λ₀, 
                      kG::KGrid, mP::ModelParameters, sP::SimulationParameters; 
                      imp_density::Float64=NaN, λ_val_only::Bool=false, λm_rhs_type::Symbol=:native, 
                      verbose::Bool=false, validate_threshold::Float64=1e-8)
    if type == :m
        rhs = λm_rhs(imp_density, χm, χd, 0.0, kG, mP, sP; λ_rhs = λm_rhs_type)
        λm, validation = λm_correction(χm, rhs, kG, mP, sP, verbose=verbose, validate_threshold=validate_threshold)
        if λ_val_only
            return λ_result(λm, 0.0, :m, validation)
        else
            error("Full result not imlemented yet")
        end
    else
    end
end

"""
    λm_correction(χm::χT, kG::KGrid, mP::ModelParameters, sP::SimulationParameters; rhs::Float64, verbose::Bool=false, validate_threshold::Float64=1e-8)
                        
Calculates ``\\lambda_\\mathrm{m}`` value, by fixing ``\\sum_{q,\\omega} \\chi^{\\lambda_\\mathrm{m}}_{\\uparrow\\uparrow}(q,i\\omega) = \\frac{n}{2}(1-\\frac{n}{2})``.

Set `verbose` to obtain a trace of the checks.
`validate_threshold` sets the threshold for the `rhs ≈ lhs` condition, set to `Inf` in order to accept any result. 
"""
function λm_correction(χm::χT, rhs::Float64, kG::KGrid, mP::ModelParameters, sP::SimulationParameters; verbose::Bool=false, validate_threshold::Float64=1e-8)
    λm_min = get_λ_min(χm)
    χr::SubArray{Float64,2}    = view(χm,:,χm.usable_ω)
    iωn = (1im .* 2 .* (-sP.n_iω:sP.n_iω)[χm.usable_ω] .* π ./ mP.β)
    ωn2_tail::Vector{Float64} = real.(χm.tail_c[3] ./ (iωn.^2))
    zero_ind = findfirst(x->!isfinite(x), ωn2_tail)
    ωn2_tail[zero_ind] = 0.0

    f_c1(λint::Float64)::Float64 = sum_kω(kG, χr, χm.β, χm.tail_c[3], ωn2_tail; transform=(f(x::Float64)::Float64 = χ_λ(x, λint))) - rhs
    df_c1(λint::Float64)::Float64 = sum_kω(kG, χr, χm.β, χm.tail_c[3], ωn2_tail; transform=(f(x::Float64)::Float64 = dχ_λ(x, λint)))
    λm = newton_right(f_c1, df_c1, 0.0, λm_min)

    check, check2 = if isfinite(validate_threshold) || verbose
        χ_λ!(χm, λm)
        check  = sum_kω(kG, χm)
        check2 = sum_ωk(kG, χm)
        reset!(χm)
        check, check2
    else
        -Inf, Inf
    end
    if verbose
        println("CHECK for rhs = $rhs  : ", check, " => 0 ?=? ", abs(rhs - check), " (sum_kω) ?=? ", abs(rhs - check2), " (sum_ωk).")
    end
    validation = (abs(rhs - check) <= validate_threshold) &&  (abs(rhs - check2) <= validate_threshold) 
    return λm, validation
end


function run_sc_new(gLoc_rfft_init::GνqT,
                    χm::χT, γm::γT, χd::χT, γd::γT, λ₀::Array{ComplexF64,3}, Σ_loc::Vector{ComplexF64},
                    kG::KGrid, mP::ModelParameters, sP::SimulationParameters;
                    maxit::Int=100, mixing::Float64=0.2, conv_abs::Float64=1e-8, trace=true)
    E_pot_1 = NaN
    E_pot_2 = NaN
    E_kin   = NaN
    μbak    = mP.μ
    it      = 1
    done    = false
    converged = false
    χ_m_loc_sum = 0.0 
    _, νGrid, iωn_f = gen_νω_indices(χm, χd, mP, sP)
    gLoc_rfft = deepcopy(gLoc_rfft_init)
    fft_νGrid= sP.fft_range
    G_ladder::OffsetMatrix{ComplexF64, Matrix{ComplexF64}} = OffsetArray(Matrix{ComplexF64}(undef, length(kG.kMult), length(fft_νGrid)), 1:length(kG.kMult), fft_νGrid) 
    Σ_ladder::OffsetMatrix{ComplexF64, Matrix{ComplexF64}} = OffsetArray(Matrix{ComplexF64}(undef, length(kG.kMult), length(νGrid)), 1:length(kG.kMult), νGrid)
    Σ_ladder_work = similar(Σ_ladder)

    traceDF = DataFrame(it = Int[], λm = Float64[], λd = Float64[], μ = Float64[], EKin = Float64[], EPot = Float64[], 
        lhs_c1 = Float64[], EPot_c2 = Float64[], cs_m = Float64[], cs_m2 = Float64[],
        cs_d = Float64[], cs_d2 = Float64[], cs_Σ = Float64[], cs_G = Float64[])

    while !done
        copy!(Σ_ladder_work, Σ_ladder)
        Σ_ladder = calc_Σ(χm, γm, χd, γd, χ_m_loc_sum, λ₀, gLoc_rfft, kG, mP, sP, νmax = last(νGrid)+1);
        mixing != 0 && it > 1 && (Σ_ladder[:,:] = (1-mixing) .* Σ_ladder .+ mixing .* Σ_ladder_work)
        μ_i = G_from_Σladder!(G_ladder, Σ_ladder, Σ_loc, kG, mP; fix_n=true)
        isnan(μ_i) && break
        _, gLoc_rfft = G_fft(G_ladder, kG, sP)
        E_kin, E_pot_1 = calc_E(G_ladder[:,νGrid].parent, Σ_ladder.parent, μ_i, kG, mP, νmax = last(νGrid)+1)

        if trace
            χ_m_sum  = sum_kω(kG, χm)
            χ_d_sum  = sum_kω(kG, χd)
            χ_m_sum2 = sum_ωk(kG, χm)
            χ_d_sum2 = sum_ωk(kG, χd)
            lhs_c1   = real(χ_d_sum + χ_m_sum)/2
            E_pot_2  = real(χ_d_sum - χ_m_sum)/2
            row = [it, χm.λ, χd.λ, μ_i, E_kin, E_pot_1, lhs_c1, E_pot_2, χ_m_sum, χ_m_sum2, χ_d_sum, χ_d_sum2, abs(sum(Σ_ladder)), abs(sum(G_ladder))]
            push!(traceDF, row)
        end

        if it != 1 && abs(sum(Σ_ladder .- Σ_ladder_work))/(kG.Nk) < conv_abs  
            converged = true
            done = true
        end
        (it >= maxit) && (done = true)

        it += 1
    end

    if isfinite(E_kin)
        χ_m_sum = sum_kω(kG, χm)
        χ_d_sum = sum_kω(kG, χd)
        lhs_c1  = real(χ_d_sum + χ_m_sum)/2
        E_pot_2 = real(χ_d_sum - χ_m_sum)/2
    end
    update_tail!(χm, [0, 0, mP.Ekin_DMFT], iωn_f)
    update_tail!(χd, [0, 0, mP.Ekin_DMFT], iωn_f)
    μnew = mP.μ
    mP.μ = μbak
    converged = converged && all(isfinite.([lhs_c1, E_pot_2]))
    return λ_result(χm.λ, χd.λ, :test, converged, E_kin, E_pot_1, E_pot_2, mP.n/2*(1-mP.n/2), lhs_c1, 
                    traceDF, G_ladder, Σ_ladder, μnew)
end
