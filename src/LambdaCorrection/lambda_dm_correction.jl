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
function λdm_correction_clean(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::λ₀T, h; 
                        νmax::Int = eom_ν_cutoff(h), fix_n::Bool = true, tc::Type{<: ΣTail} = default_Σ_tail_correction(),
                        use_trivial_λmin::Bool = (tc === ΣTail_EoM) , λd_min::Float64 = NaN, λd_max::Float64 = 200.0, λd_δ::Float64 = 1e-2,
                        validation_threshold::Float64 = 1e-8, max_steps_m::Int = 2000, max_steps_dm::Int = 2000, log_io = devnull, RF_Method=Roots.FalsePosition()
    )       
    λm, λd = λdm_correction_val_clean(χm, γm, χd, γd,λ₀, h; νmax=νmax, fix_n = fix_n, tc = tc,
                use_trivial_λmin=use_trivial_λmin, λd_min=λd_min, λd_max=λd_max, λd_δ=λd_δ,
                validation_threshold = validation_threshold, max_steps_m = max_steps_m,
                max_steps_dm = max_steps_dm, log_io = log_io, RF_Method=RF_Method)     
    return λ_result(dmCorrection, χm, γm, χd, γd, λ₀, λm, λd, true, h; tc=tc, validation_threshold = validation_threshold, max_steps_m = max_steps_m, fix_n=fix_n)
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
                        νmax::Int = eom_ν_cutoff(h), fix_n::Bool = true, tc::Type{<: ΣTail} = default_Σ_tail_correction(),
                        use_trivial_λmin::Bool = (tc === ΣTail_EoM), λd_min::Float64 = NaN, λd_max::Float64 = 200.0, λd_δ::Float64 = 1e-2,
                        validation_threshold::Float64 = 1e-8, max_steps_m::Int = 2000, 
                        max_steps_dm::Int = 2000, log_io = devnull, RF_Method=Roots.FalsePosition())
    λd_min::Float64 = if !isnan(λd_min)
        λd_min
    else
        if use_trivial_λmin 
            get_λ_min(χd)
        else
            get_λd_min(χm, γm, χd, γd, λ₀, h)
        end
    end
    ωn2_tail = ω2_tail(χm)
    νGrid = 0:(νmax-1)
    iν = iν_array(h.mP.β, νGrid)

    function f_c2(λd_i::Float64)
        rhs_c1,_ = λm_rhs(χm, χd, h; λd=λd_i)
        λm_i   = λm_correction_val(χm, rhs_c1, h.kG, ωn2_tail; max_steps=max_steps_m, eps=validation_threshold)
        μ_new, G_ladder, Σ_ladder = calc_G_Σ(χm, γm, χd, γd, λ₀, λm_i, λd_i, h; νmax=νmax, tc = tc, fix_n = fix_n)
        !isfinite(μ_new) && error("encountered μ=$μ_new @ λd = $λd_i // λm = $λm_i")

        EPot_tail, EPot_tail_inv = EPot_p1_tail(iν, μ_new, h)
        Epot_1 = EPot_p1(view(G_ladder,:,νGrid), view(Σ_ladder,:,νGrid), EPot_tail, EPot_tail_inv, h.mP.β, h.kG)
        Epot_2 = EPot_p2(χm, χd, λm_i, λd_i, h.mP.n, h.mP.U, h.kG)
        return Epot_1 - Epot_2
    end
    #λd  = newton_right(f_c2, λd_min+10.0, λd_min+1e-3)
    λd  = find_zero(f_c2, (λd_min + λd_δ, λd_max), RF_Method; atol=validation_threshold, maxiters=max_steps_dm)
    rhs,_ = λm_rhs(χm, χd, h; λd=λd)
    λm  = λm_correction_val(χm, rhs, h; max_steps=max_steps_m, eps=validation_threshold)
    return λm, λd
end

"""
    λdm_correction(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::λ₀T, h; 
                        νmax::Int = eom_ν_cutoff(h), fix_n::Bool = true, tc::Type{<: ΣTail} = default_Σ_tail_correction(),
                        use_trivial_λmin::Bool = (tc === ΣTail_EoM), λd_min::Float64 = NaN, λd_max::Float64 = 200.0, λd_δ::Float64 = 1e-2,
                        validation_threshold::Float64 = 1e-8, max_steps_m::Int = 2000,
                        max_steps_dm::Int = 2000, log_io = devnull, RF_Method=Roots.FalsePosition()
    )  

Computes the `λm` and `λd` parameters for the consistency of Pauli principle and potential energie on one- and two-particle level.
Returns a [`λ_result`](@ref λ_result) object.
"""
function λdm_correction(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::λ₀T, h; 
                        νmax::Int = eom_ν_cutoff(h), fix_n::Bool = true, tc::Type{<: ΣTail} = default_Σ_tail_correction(),
                        use_trivial_λmin::Bool = (tc === ΣTail_EoM), λd_min::Float64 = NaN, λd_max::Float64 = 200.0, λd_δ::Float64 = 1e-2,
                        validation_threshold::Float64 = 1e-8, max_steps_m::Int = 2000,
                        max_steps_dm::Int = 2000, log_io = devnull, RF_Method=Roots.FalsePosition()
    )       
    λm, λd = λdm_correction_val(χm, γm, χd, γd, λ₀, h; fix_n = fix_n, tc=tc,
                use_trivial_λmin=use_trivial_λmin, λd_min=λd_min, λd_max=λd_max, λd_δ=λd_δ,
                validation_threshold = validation_threshold, max_steps_m = max_steps_m,
                max_steps_dm = max_steps_dm, log_io = log_io, RF_Method=RF_Method)     
    return λ_result(dmCorrection, χm, γm, χd, γd, λ₀, λm, λd, true, h; νmax=νmax, tc=tc, validation_threshold = validation_threshold, max_steps_m = max_steps_m, fix_n = fix_n)
end

"""
    λdm_correction(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::λ₀T, h; 
                        νmax::Int = eom_ν_cutoff(h), fix_n::Bool = true, tc::Type{<: ΣTail} = default_Σ_tail_correction(),
                        use_trivial_λmin::Bool = (tc === ΣTail_EoM), λd_min::Float64 = NaN, λd_max::Float64 = 200.0, λd_δ::Float64 = 1e-2,
                        validation_threshold::Float64 = 1e-8, max_steps_m::Int = 2000,
                        max_steps_dm::Int = 2000, log_io = devnull, RF_Method=Roots.FalsePosition()
    )::Tuple{Float64,Float64}

    λdm_correction_val(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::λ₀T, h; 
                        νmax::Int = eom_ν_cutoff(h), fix_n::Bool = true,tc::Type{<: ΣTail} = default_Σ_tail_correction(),
                        use_trivial_λmin::Bool = (tc === ΣTail_EoM),  λd_min::Float64 = NaN, λd_max::Float64 = 200.0, λd_δ::Float64 = 1e-2,
                        validation_threshold::Float64 = 1e-8, max_steps_m::Int = 2000, 
                        max_steps_dm::Int = 2000, log_io = devnull, RF_Method=Roots.FalsePosition())::Tuple{Float64,Float64}

Computes the `λm` and `λd` parameters for the consistency of Pauli principle and potential energie on one- and two-particle level.
Returns the bare λ-values, usually one should run [`λdm_correction`](@ref λdm_correction), which returns a [`λ_result`](@ref λ_result) object 
that stores additional consistency checks.
"""
function λdm_correction_val(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::λ₀T, h; 
                        νmax::Int = eom_ν_cutoff(h), fix_n::Bool = true,tc::Type{<: ΣTail} = default_Σ_tail_correction(),
                        use_trivial_λmin::Bool = (tc === ΣTail_EoM),  λd_min::Float64 = NaN, λd_max::Float64 = 200.0, λd_δ::Float64 = 1e-2,
                        validation_threshold::Float64 = 1e-8, max_steps_m::Int = 2000, 
                        max_steps_dm::Int = 2000, log_io = devnull, RF_Method=Roots.FalsePosition())::Tuple{Float64,Float64}
    ωn2_tail = ω2_tail(χm)
    Nq::Int = length(h.kG.kMult)
    
    Kνωq_pre::Vector{ComplexF64} = Vector{ComplexF64}(undef, Nq)
    Σ_ladder = OffsetArray(Matrix{ComplexF64}(undef, Nq, νmax), 1:Nq, 0:νmax-1)
    G_ladder = similar(Σ_ladder)
    
    νGrid = 0:(νmax-1)
    iν = iν_array(h.mP.β, νGrid)
    tc_factor = tail_factor(tc, h.mP.U, h.mP.β, h.mP.n, h.Σ_loc, iν) 

    λd_min::Float64 = if !isnan(λd_min)
        λd_min
    else
        if use_trivial_λmin 
            get_λ_min(χd)
        else
            get_λd_min(χm, γm, χd, γd, λ₀, h)
        end
    end

    function f_c2(λd_i::Float64)
        rhs_c1,_ = λm_rhs(χm, χd, h; λd=λd_i)
        λm_i   = λm_correction_val(χm, rhs_c1, h.kG, ωn2_tail; max_steps=max_steps_m, eps=validation_threshold)
        tc_term  = (tc === ΣTail_EoM) ? h.χ_m_loc : tail_correction_term(sum_kω(h.kG, χm, λ=λm_i), h.χloc_m_sum, tc_factor)
        μ_new = calc_G_Σ!(G_ladder, Σ_ladder, Kνωq_pre, tc_term, χm, γm, χd, γd, λ₀, λm_i, λd_i, h, fix_n=fix_n)
        !isfinite(μ_new) && error("encountered μ=$μ_new @ λd = $λd_i // λm = $λm_i")

        EPot_tail, EPot_tail_inv = EPot_p1_tail(iν, μ_new, h)
        Epot_1 = EPot_p1(view(G_ladder,:,νGrid), view(Σ_ladder,:,νGrid), EPot_tail, EPot_tail_inv, h.mP.β, h.kG)
        Epot_2 = EPot_p2(χm, χd, λm_i, λd_i, h.mP.n, h.mP.U, h.kG)
        return Epot_1 - Epot_2
    end
        
    #λd  = newton_right(f_c2, λd_min+10.0, λd_min+1e-3)
    λd  = find_zero(f_c2, (λd_min + λd_δ, λd_max), RF_Method; atol=validation_threshold, maxiters=max_steps_dm)
    rhs,_ = λm_rhs(χm, χd, h; λd=λd)
    λm  = λm_correction_val(χm, rhs, h; max_steps=max_steps_m, eps=validation_threshold)
    return λm, λd
end
