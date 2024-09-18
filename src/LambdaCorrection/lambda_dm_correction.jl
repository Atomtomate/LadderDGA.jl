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
    λdm_correction_clean(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::Array{ComplexF64,3}, h;
                        validation_threshold::Float64 = 1e-8, max_steps_m::Int = 2000, max_steps_dm::Int = 2000, log_io = devnull
    )

Computes the `λm` and `λd` parameters for the consistency of Pauli principle and potential energie on one- and two-particle level.
Returns a [`λ_result`](@ref λ_result) object.
"""
function λdm_correction_clean(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::λ₀T, h; fix_n::Bool = true,
                        validation_threshold::Float64 = 1e-8, max_steps_m::Int = 2000, max_steps_dm::Int = 2000, log_io = devnull
    )       
    λm, λd = λdm_correction_val_clean(χm, γm, χd, γd,λ₀, h; fix_n = fix_n,
                validation_threshold = validation_threshold, max_steps_m = max_steps_m,
                max_steps_dm = max_steps_dm, log_io = log_io)     
    return λ_result(dmCorrection, χm, γm, χd, γd, λ₀, λm, λd, true, h; validation_threshold = validation_threshold, max_steps_m = max_steps_m)
end

"""
    λdm_correction_val_clean(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::Array{ComplexF64,3}, h;
                        validation_threshold::Float64 = 1e-8, max_steps_m::Int = 2000, max_steps_dm::Int = 2000, log_io = devnull
    )

Computes the `λm` and `λd` parameters for the consistency of Pauli principle and potential energie on one- and two-particle level.
Returns the bare λ-values, usually one should run [`λdm_correction`](@ref λdm_correction), which returns a [`λ_result`](@ref λ_result) object 
that stores additional consistency checks.
"""
function λdm_correction_val_clean(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::λ₀T, h;
                        validation_threshold::Float64 = 1e-8, max_steps_m::Int = 2000, 
                        max_steps_dm::Int = 2000, log_io = devnull, fix_n::Bool = true)
    λd_min   = get_λ_min(χd)
    ωn2_tail = ω2_tail(χm)

    function f_c2(λd_i::Float64)
        rhs_c1,_ = λm_rhs(χm, χd, h; λd=λd_i)
        λm_i   = λm_correction_val(χm, rhs_c1, h.kG, ωn2_tail; max_steps=max_steps_m, eps=validation_threshold)
        μ_new, G_ladder, Σ_ladder = calc_G_Σ(χm, γm, χd, γd, λ₀, λm_i, λd_i, h; tc = true, fix_n = fix_n)
        #TODO: use Epot_1
        Ekin_1, Epot_1 = calc_E(G_ladder, Σ_ladder, μ_new, h.kG, h.mP)
        Epot_2 = EPot_p2(χm, χd, λm_i, λd_i, h.mP.n, h.mP.U, h.kG)
        return Epot_1 - Epot_2
    end
    
    λd  = newton_secular(f_c2, λd_min; nsteps=max_steps_dm, atol=validation_threshold)
    rhs,_ = λm_rhs(χm, χd, h; λd=λd)
    λm  = λm_correction_val(χm, rhs, h; max_steps=max_steps_m, eps=validation_threshold)
    return λm, λd
end

"""
    λdm_correction(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::Array{ComplexF64,3}, h;
                        validation_threshold::Float64 = 1e-8, max_steps_m::Int = 2000, max_steps_dm::Int = 2000, log_io = devnull
    )

Computes the `λm` and `λd` parameters for the consistency of Pauli principle and potential energie on one- and two-particle level.
Returns a [`λ_result`](@ref λ_result) object.
"""
function λdm_correction(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::λ₀T, h; fix_n::Bool = true,
                        validation_threshold::Float64 = 1e-8, max_steps_m::Int = 2000,
                        max_steps_dm::Int = 2000, log_io = devnull, tc::Bool = true
    )       
    λm, λd = λdm_correction_val(χm, γm, χd, γd,λ₀, h; fix_n = fix_n,
                validation_threshold = validation_threshold, max_steps_m = max_steps_m,
                max_steps_dm = max_steps_dm, log_io = log_io, tc=tc)     
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
                        max_steps_dm::Int = 2000, log_io = devnull, fix_n::Bool = true, tc::Bool = true)::Tuple{Float64,Float64}

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
        λm_i   = λm_correction_val(χm, rhs_c1, h.kG, ωn2_tail; max_steps=max_steps_m, eps=validation_threshold)
        # @timeit to "dbg3" μ_new = calc_G_Σ!(G_ladder, Σ_ladder, Kνωq_pre, tc_factor, χm, γm, χd, γd, λ₀, λm_i, λd_i, h; fix_n = fix_n)
        
    (λm_i != 0) && χ_λ!(χm, λm_i)
    (λd_i != 0) && χ_λ!(χd, λd_i)
    tc_term  = tail_correction_term(sum_kω(h.kG, χm), h.χloc_m_sum, tc_factor)
        calc_Σ!(Σ_ladder, Kνωq_pre, χm, γm, χd, γd, λ₀, tc_term, h.gLoc_rfft, h.kG, h.mP, h.sP)
    (λm_i != 0) && reset!(χm)
    (λd_i != 0) && reset!(χd)
    μ_new = G_from_Σladder!(G_ladder, Σ_ladder, h.Σ_loc, h.kG, h.mP; fix_n = fix_n, μ = h.mP.μ, n = h.mP.n)
        #TODO: use Epot_1
        Ekin_1, Epot_1 = calc_E(G_ladder, Σ_ladder, μ_new, h.kG, h.mP)
        Epot_2 = EPot_p2(χm, χd, λm_i, λd_i, h.mP.n, h.mP.U, h.kG)
        return Epot_1 - Epot_2
    end
    
    λd  = newton_secular(f_c2, λd_min; nsteps=max_steps_dm, atol=validation_threshold)
    rhs,_ = λm_rhs(χm, χd, h; λd=λd)
    λm  = λm_correction_val(χm, rhs, h; max_steps=max_steps_m, eps=validation_threshold)
    return λm, λd
end
