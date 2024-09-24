# ==================================================================================================== #
#                                         ConditionCurves.jl                                           #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#   Functions for sampling solution curves to the various lambda-correction conditions.                #
# -------------------------------------------- TODO -------------------------------------------------- #
#   Most of the functions here are almost exact copies of the _val correction functuons. Refactor.     #
# ==================================================================================================== #


"""
    PPCond_curve(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::λ₀T, h; tc::Bool=true, feps_abs::Float64=1e-8, xeps_abs::Float64=1e-8, maxit::Int=2000, λmax::Float64=30.0)

Samples a curve for the difference between the Pauli principle value on one- and two-particle level.
See also [`sample_f`](@ref sample_f) for a description of the numerical parameters.
"""
function PPCond_curve(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::λ₀T, h; tc::Bool=true, feps_abs::Float64=1e-8, xeps_abs::Float64=1e-8, maxit::Int=2000, λmax::Float64=30.0)
    λm_min = get_λ_min(χm)
    χr::SubArray{Float64,2} = view(χm, :, χm.usable_ω)
    rhs,_ = λm_rhs(χm, χd, h; λd=0.0)
    ωn2_tail = ω2_tail(χm)

    f_c1(λint::Float64)::Float64 = sum_kω(h.kG, χr, χm.β, χm.tail_c[3], ωn2_tail; transform = (f(x::Float64)::Float64 = χ_λ(x, λint))) - rhs
    df_c1(λint::Float64)::Float64 = sum_kω(h.kG, χr, χm.β, χm.tail_c[3], ωn2_tail; transform = (f(x::Float64)::Float64 = dχ_λ(x, λint)))

    sample_f(f_c1, λm_min - 1.0, λmax; feps_abs=feps_abs, xeps_abs=xeps_abs, maxit=maxit)
end


"""
    EPotCond_curve(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::λ₀T, h; tc::Bool=true, feps_abs::Float64=1e-8, xeps_abs::Float64=1e-8, maxit::Int=2000, λmax::Float64=30.0)

Samples a curve for the difference between the potential energy on one- and two-particle level.
TODO: Also returns \\lambda_\\mathrm{m}(\\lambda_\\mathrm{d})``. 
See also [`sample_f`](@ref sample_f) for a description of the numerical parameters.
"""
function EPotCond_curve(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::λ₀T, h; tc::Bool=true, feps_abs::Float64=1e-8, xeps_abs::Float64=1e-8, maxit::Int=2000, λmax::Float64=30.0)
    λd_min::Float64   = get_λ_min(χd)
    ωn2_tail = ω2_tail(χm)
    Nq::Int = length(h.kG.kMult)
    νmax::Int = eom_ν_cutoff(h.sP)

    Kνωq_pre::Vector{ComplexF64} = Vector{ComplexF64}(undef, Nq)
    Σ_ladder = OffsetArray(Matrix{ComplexF64}(undef, Nq, νmax), 1:Nq, 0:νmax-1)
    G_ladder = similar(Σ_ladder)
    iν = iν_array(h.mP.β, collect(axes(Σ_ladder, 2)))
    tc_factor= (tc ? tail_factor(h.mP.U, h.mP.β, h.mP.n, h.Σ_loc, iν) : 0.0 ./ iν)

    function f_c2(λd_i::Float64)
        rhs_c1,_ = λm_rhs(χm, χd, h; λd=λd_i)
        λm_i   = λm_correction_val(χm, rhs_c1, h.kG, ωn2_tail)
        # @timeit to "dbg3" μ_new = calc_G_Σ!(G_ladder, Σ_ladder, Kνωq_pre, tc_factor, χm, γm, χd, γd, λ₀, λm_i, λd_i, h; fix_n = fix_n)
        
        (λm_i != 0) && χ_λ!(χm, λm_i)
        (λd_i != 0) && χ_λ!(χd, λd_i)
        tc_term  = tail_correction_term(sum_kω(h.kG, χm), h.χloc_m_sum, tc_factor)
            calc_Σ!(Σ_ladder, Kνωq_pre, χm, γm, χd, γd, λ₀, tc_term, h.gLoc_rfft, h.kG, h.mP, h.sP)
        (λm_i != 0) && reset!(χm)
        (λd_i != 0) && reset!(χd)
        μ_new = G_from_Σladder!(G_ladder, Σ_ladder, h.Σ_loc, h.kG, h.mP; fix_n = true, μ = h.mP.μ, n = h.mP.n)
        Ekin_1, Epot_1 = calc_E(G_ladder, Σ_ladder, μ_new, h.kG, h.mP)
        Epot_2 = EPot_p2(χm, χd, λm_i, λd_i, h.mP.n, h.mP.U, h.kG)
        return Epot_1 - Epot_2
    end
    
    sample_f(f_c2, λd_min - 1.0, λmax; feps_abs=feps_abs, xeps_abs=xeps_abs, maxit=maxit)
end

"""
    EPotCond_curve(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::λ₀T, h; tc::Bool=true, feps_abs::Float64=1e-8, xeps_abs::Float64=1e-8, maxit::Int=2000, λmax::Float64=30.0)

Samples a curve for the difference between the potential energy on one- and two-particle level.
TODO: Also returns \\lambda_\\mathrm{m}(\\lambda_\\mathrm{d})``. 
See also [`sample_f`](@ref sample_f) for a description of the numerical parameters.
"""
function EPotCond_sc_curve(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::λ₀T, h; 
     method=:sc,
     tc::Bool=true, feps_abs::Float64=1e-8, xeps_abs::Float64=1e-8,
     maxit::Int=2000, maxit_sc::Int=500, mixing::Float64=0.3, λmax::Float64=30.0, sc_conv_abs::Float64=1e-7, verbose::Bool=false, verbose_sc::Bool=false)
    λd_min::Float64   = get_λ_min(χd)
    ωn2_tail = ω2_tail(χm)
    Nq::Int = length(h.kG.kMult)
    νmax::Int = eom_ν_cutoff(h.sP)
    fft_νGrid= h.sP.fft_range

    Kνωq_pre::Vector{ComplexF64} = Vector{ComplexF64}(undef, Nq)
    Σ_ladder = OffsetArray(Matrix{ComplexF64}(undef, Nq, νmax), 1:Nq, 0:νmax-1)
    G_ladder = OffsetArray(Matrix{ComplexF64}(undef, Nq, length(fft_νGrid)), 1:Nq, fft_νGrid) 
    G_ladder_bak = similar(G_ladder)
    iν = iν_array(h.mP.β, collect(axes(Σ_ladder, 2)))
    tc_factor_term = (tc ? tail_factor(h.mP.U, h.mP.β, h.mP.n, h.Σ_loc, iν) : 0.0 ./ iν)

    function f_c2(λd_i::Float64)
        rhs_c1,_ = λm_rhs(χm, χd, h; λd=λd_i)
        λm_i   = λm_correction_val(χm, rhs_c1, h.kG, ωn2_tail)
        verbose && println("running λm=$λm_i, λd=$λd_i")
        converged, μ_new = if method == :sc
            run_sc!(G_ladder, Σ_ladder, G_ladder_bak, Kνωq_pre, tc_factor_term, 
                            χm, γm, χd, γd, λ₀, λm_i, λd_i, h; 
                            maxit=maxit_sc, mixing=mixing, conv_abs=sc_conv_abs, verbose=verbose_sc)
            elseif method == :tsc
                run_tsc!(G_ladder, Σ_ladder, G_ladder_bak, Kνωq_pre, tc_factor_term, 
                            χm, γm, χd, γd, λ₀, λm_i, λd_i, h; 
                            maxit=maxit_sc, mixing=mixing, conv_abs=sc_conv_abs, verbose=verbose_sc)
            else 
                @error "method " method " not recognized. Use :sc or :tsc"
            end
        Ekin_1, Epot_1 = calc_E(G_ladder, Σ_ladder, μ_new, h.kG, h.mP)
        Epot_2 = EPot_p2(χm, χd, λm_i, λd_i, h.mP.n, h.mP.U, h.kG)
        ΔEPot = Epot_1 - Epot_2
        verbose && println(" -> converged = $converged, ΔEPot = $ΔEPot ($Epot_1 - $Epot_2). μ = $μ_new")
        return converged ? ΔEPot : NaN
    end
    
    sample_f(f_c2, λd_min + 1e-4, λmax; feps_abs=feps_abs, xeps_abs=xeps_abs, maxit=maxit)
end

"""
    λm_of_λd_curve(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::λ₀T, h; tc::Bool=true, feps_abs::Float64=1e-8, xeps_abs::Float64=1e-8, maxit::Int=2000, λmax::Float64=30.0)

Samples ``\\lambda_\\mathrm{m}(\\lambda_\\mathrm{d})`` for λ-correcitons of type `mode`. 
See also [`sample_f`](@ref sample_f) for a description of the numerical parameters.
"""
function λm_of_λd_curve(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::λ₀T, h; tc::Bool=true, feps_abs::Float64=1e-8, xeps_abs::Float64=1e-8, maxit::Int=2000, λmax::Float64=30.0)
    λd_min::Float64   = get_λ_min(χd)
    ωn2_tail = ω2_tail(χm)
    Nq::Int = length(h.kG.kMult)
    νmax::Int = eom_ν_cutoff(h.sP)

    Kνωq_pre::Vector{ComplexF64} = Vector{ComplexF64}(undef, Nq)
    Σ_ladder = OffsetArray(Matrix{ComplexF64}(undef, Nq, νmax), 1:Nq, 0:νmax-1)
    G_ladder = similar(Σ_ladder)
    iν = iν_array(h.mP.β, collect(axes(Σ_ladder, 2)))
    tc_factor= (tc ? tail_factor(h.mP.U, h.mP.β, h.mP.n, h.Σ_loc, iν) : 0.0 ./ iν)

    function f_c2(λd_i::Float64)
        rhs_c1,_ = λm_rhs(χm, χd, h; λd=λd_i)
        λm_i   = λm_correction_val(χm, rhs_c1, h.kG, ωn2_tail)

        (λm_i != 0) && χ_λ!(χm, λm_i)
        (λd_i != 0) && χ_λ!(χd, λd_i)
        tc_term  = tail_correction_term(sum_kω(h.kG, χm), h.χloc_m_sum, tc_factor)
            calc_Σ!(Σ_ladder, Kνωq_pre, χm, γm, χd, γd, λ₀, tc_term, h.gLoc_rfft, h.kG, h.mP, h.sP)
        (λm_i != 0) && reset!(χm)
        (λd_i != 0) && reset!(χd)
        μ_new = G_from_Σladder!(G_ladder, Σ_ladder, h.Σ_loc, h.kG, h.mP; fix_n = true, μ = h.mP.μ, n = h.mP.n)
        Ekin_1, Epot_1 = calc_E(G_ladder, Σ_ladder, μ_new, h.kG, h.mP)
        Epot_2 = EPot_p2(χm, χd, λm_i, λd_i, h.mP.n, h.mP.U, h.kG)
        return Epot_1 - Epot_2
    end
    
    sample_f(f_c2, λd_min - 1.0, λmax; feps_abs=feps_abs, xeps_abs=xeps_abs, maxit=maxit)
end
