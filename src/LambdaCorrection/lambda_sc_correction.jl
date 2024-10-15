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
    λm_sc_correction(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::λ₀T, h;
                           validation_threshold::Float64 = 1e-8,
                           max_steps_m::Int = 2000, max_steps_dm::Int = 2000, max_steps_sc::Int = 2000,
                           log_io = devnull, tc = default_Σ_tail_correction())

Runs partial self-consistency loop (update of propagators in equation of motion) within [`λdm correction`](@ref λdm_correction)
"""
function λm_sc_correction(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::λ₀T, h;
                           use_trivial_λmin::Bool=false, λd_min::Float64=NaN,
                           validation_threshold::Float64 = 1e-8, λ_rhs = :native,
                           max_steps_m::Int = 2000, max_steps_dm::Int = 2000, max_steps_sc::Int = 200,
                           log_io = devnull, tc::Type{<: ΣTail} = default_Σ_tail_correction())       

    λd_min::Float64 = if !isnan(λd_min)
        λd_min
    else
        if use_trivial_λmin 
            get_λ_min(χd)
        else
            get_λd_min(χm, γm, χd, γd, λ₀, h)
        end
    end
    λd  = 0.0
    rhs,PP_p1 = λm_rhs(χm, χd, h; λd=λd, λ_rhs = :native)
    λm  = λm_correction_val(χm, rhs, h; max_steps=max_steps_m, eps=validation_threshold)
    converged, μ_new, G_ladder_it, Σ_ladder_it, trace = run_sc(χm, γm, χd, γd, λ₀, λm, λd, h; maxit=max_steps_sc, conv_abs=validation_threshold, tc = tc)
    return λ_result(m_scCorrection, χm, χd, μ_new, G_ladder_it, Σ_ladder_it, λm, λd, converged, h; PP_p1=PP_p1, validation_threshold = validation_threshold, max_steps_m = max_steps_m)
end

"""
    λdm_sc_correction(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::λ₀T, h;
                           validation_threshold::Float64 = 1e-8,
                           max_steps_m::Int = 2000, max_steps_dm::Int = 2000, max_steps_sc::Int = 2000,
                           log_io = devnull, tc = default_Σ_tail_correction())

Runs partial self-consistency loop (update of propagators in equation of motion) within [`λdm correction`](@ref λdm_correction)
"""
function λdm_sc_correction_clean(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::λ₀T, h;
                           use_trivial_λmin::Bool=false, λd_min::Float64=NaN,
                           validation_threshold::Float64 = 1e-8,
                           max_steps_m::Int = 2000, max_steps_dm::Int = 2000, max_steps_sc::Int = 2000,
                           log_io = devnull, tc::Type{<: ΣTail} = default_Σ_tail_correction())       

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

    function f_c2_sc(λd_i::Float64)
        rhs_c1,_ = λm_rhs(χm, χd, h; λd=λd_i)
        λm_i   = λm_correction_val(χm, rhs_c1, h.kG, ωn2_tail; max_steps=max_steps_m, eps=validation_threshold)
        converged, μ_new, G_ladder_it, Σ_ladder_it, _ = run_sc(χm, γm, χd, γd, λ₀, λm_i, λd_i, h; maxit=max_steps_sc, conv_abs=validation_threshold, tc = tc)
        #TODO: use Epot_1
        Ekin_1, Epot_1 = calc_E(G_ladder_it, Σ_ladder_it, μ_new, h.kG, h.mP)
        Epot_2 = EPot_p2(χm, χd, λm_i, λd_i, h.mP.n, h.mP.U, h.kG)
        return Epot_1 - Epot_2
    end    

    λd = newton_secular(f_c2_sc, λd_min; nsteps=max_steps_dm, atol=validation_threshold)
    rhs,PP_p1 = λm_rhs(χm, χd, h; λd=λd)
    λm  = λm_correction_val(χm, rhs, h; max_steps=max_steps_m, eps=validation_threshold)
    converged, μ_new, G_ladder_it, Σ_ladder_it, _ = run_sc(χm, γm, χd, γd, λ₀, λm, λd, h; maxit=max_steps_sc, conv_abs=validation_threshold, tc = tc)
    return λ_result(dm_scCorrection, χm, χd, μ_new, G_ladder_it, Σ_ladder_it, λm, λd, converged, h; PP_p1=PP_p1, validation_threshold = validation_threshold, max_steps_m = max_steps_m)
end


function λdm_sc_correction(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::λ₀T, h;
                           use_trivial_λmin::Bool=false, λd_min::Float64=NaN,
                           validation_threshold::Float64 = 1e-8,
                           max_steps_m::Int = 2000, max_steps_dm::Int = 2000, max_steps_sc::Int = 2000,
                           log_io = devnull, tc::ΣTail=default_Σ_tail_correction(), verbose::Bool=false)       

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
    Nq, Nω = size(χm)
    νmax::Int = eom_ν_cutoff(h.sP)
    fft_νGrid= h.sP.fft_range

    Kνωq_pre    = Vector{ComplexF64}(undef, length(h.kG.kMult))
    G_ladder_it = OffsetArray(Matrix{ComplexF64}(undef, Nq, length(fft_νGrid)), 1:Nq, fft_νGrid) 
    G_ladder_bak = similar(G_ladder_it)
    Σ_ladder_it = OffsetArray(Matrix{ComplexF64}(undef, Nq, νmax), 1:Nq, 0:νmax-1)
    iν = iν_array(h.mP.β, collect(axes(Σ_ladder_it, 2)))
    tc_factor_term = tail_factor(tc, h.mP.U, h.mP.β, h.mP.n, h.Σ_loc, iν)

    function f_c2_sc(λd_i::Float64)
        rhs_c1,_ = λm_rhs(χm, χd, h; λd=λd_i)
        λm_i   = λm_correction_val(χm, rhs_c1, h.kG, ωn2_tail; max_steps=max_steps_m, eps=validation_threshold)
        converged, μ_new = run_sc!(G_ladder_it, Σ_ladder_it, G_ladder_bak, Kνωq_pre, tc_factor_term, 
                χm, γm, χd, γd, λ₀, λm_i, λd_i, h; 
                maxit=max_steps_sc, conv_abs=validation_threshold, verbose=verbose)
        #TODO: use Epot_1
        Ekin_1, Epot_1 = calc_E(G_ladder_it, Σ_ladder_it, μ_new, h.kG, h.mP)
        Epot_2 = EPot_p2(χm, χd, λm_i, λd_i, h.mP.n, h.mP.U, h.kG)
        return Epot_1 - Epot_2
    end    

    λd = newton_secular(f_c2_sc, λd_min; nsteps=max_steps_dm, atol=validation_threshold)
    rhs,PP_p1 = λm_rhs(χm, χd, h; λd=λd)
    λm  = λm_correction_val(χm, rhs, h; max_steps=max_steps_m, eps=validation_threshold)
    converged, μ_new = run_sc!(G_ladder_it, Σ_ladder_it, G_ladder_bak, Kνωq_pre, tc_factor_term, 
                χm, γm, χd, γd, λ₀, λm, λm, h; maxit=max_steps_sc, conv_abs=validation_threshold)
    return λ_result(dm_scCorrection, χm, χd, μ_new, G_ladder_it, Σ_ladder_it, λm, λd, converged, h; PP_p1=PP_p1, validation_threshold = validation_threshold, max_steps_m = max_steps_m)
end

#= 
TODO: refactor run_sc:
function run_sc(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::λ₀T, λm::Float64, λd::Float64, h;
                maxit::Int=500, mixing::Float64=0.3, mixing_start_it::Int=10,
                conv_abs::Float64=1e-8, tc::Bool = true, trace::Bool=false, verbose::Bool=false)
    converged = false
    tr = []

    μ_it, G_ladder_it, Σ_ladder_it = calc_G_Σ(χm, γm, χd, γd, λ₀, λm, λd, h; tc = tc, fix_n = true)
    G_ladder_bak = similar(G_ladder_it)
    _, gLoc_rfft = G_fft(G_ladder_it, h.kG, h.sP)
    Kνωq_pre::Vector{ComplexF64} = Vector{ComplexF64}(undef, length(h.kG.kMult))
    iν = iν_array(h.mP.β, collect(axes(Σ_ladder_it, 2)))
    tc_factor = (tc ? tail_factor(h.mP.U, h.mP.β, h.mP.n, h.Σ_loc, iν) : 0.0 ./ iν)
 
    trace && push!(tr, Σ_ladder_it)

    converged, μ_it = run_sc!(G_ladder_it, Σ_ladder_it, G_ladder_bak, Kνωq_pre, tc_factor, 
                    χm, γm, χd, γd, λ₀, λm, λd, h; gLoc_rfft=gLoc_rfft,
                    maxit=maxit-1, mixing=mixing, mixing_start_it=mixing_start_it,
                    conv_abs=conv_abs, trace=(trace ? tr : nothing), verbose=verbose)

    return converged, μ_it, G_ladder_it, Σ_ladder_it, tr
end 
=#

function run_sc(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::λ₀T, λm::Float64, λd::Float64, h;
                maxit::Int=500, mixing::Float64=0.3, mixing_start_it::Int=10,
                conv_abs::Float64=1e-8, tc::Type{<: ΣTail} = default_Σ_tail_correction(), trace::Bool=false, verbose::Bool=false)
    it      = 2
    converged = false
    tr = []

    μ_it, G_ladder_it, Σ_ladder_it = calc_G_Σ(χm, γm, χd, γd, λ₀, λm, λd, h; tc = tc, fix_n = true)
    G_ladder_bak = similar(G_ladder_it)
    _, gLoc_rfft = G_fft(G_ladder_it, h.kG, h.sP)
 
    trace && push!(tr, Σ_ladder_it)

    while it <= maxit && !converged
        it > mixing_start_it && copy!(G_ladder_bak, G_ladder_it)

        μ_it, G_ladder_it, Σ_ladder_it = calc_G_Σ(χm, γm, χd, γd, λ₀, λm, λd, h; gLoc_rfft = gLoc_rfft, tc = tc, fix_n = true)
        trace && push!(tr, Σ_ladder_it)

        Δit = it > 1 ? sum(abs.(G_ladder_it .- G_ladder_bak))/prod(size(G_ladder_it)) : Inf
        Δit < conv_abs && (converged = true)

        it > mixing_start_it && (@inbounds G_ladder_it[:,:] = (1-mixing) .* G_ladder_it[:,:] .+ mixing .* G_ladder_bak[:,:])
        !all(isfinite.(Σ_ladder_it)) && break
        
        verbose && println("It = $it[λm=$λm/λd=$λd]: Δit=$Δit")
        G_rfft!(gLoc_rfft, G_ladder_it, h.kG, h.sP.fft_range)
        it += 1
    end

    return converged, μ_it, G_ladder_it, Σ_ladder_it, tr
end


function run_sc!(G_ladder_it::OffsetArray, Σ_ladder_it::OffsetArray, G_ladder_bak::OffsetArray, Kνωq_pre, tc_factor, 
                χm::χT, γm::γT, χd::χT, γd::γT, λ₀::λ₀T, λm::Float64, λd::Float64, h;
                maxit::Int=500, mixing::Float64=0.3, mixing_start_it::Int=10,
                conv_abs::Float64=1e-8, trace=nothing, verbose::Bool=false)
    it      = 1
    converged = false

    gLoc_rfft = deepcopy(h.gLoc_rfft)
    μ_it = h.mP.μ
    tc_term  = tail_correction_term(sum_kω(h.kG, χm, λ=λm), h.χloc_m_sum, tc_factor)

    while it <= maxit && !converged
        it > mixing_start_it && copy!(G_ladder_bak, G_ladder_it)

        μ_it = calc_G_Σ!(G_ladder_it, Σ_ladder_it, Kνωq_pre, tc_term, χm, γm, χd, γd, λ₀, λm, λd, h; gLoc_rfft=gLoc_rfft)
        !isnothing(trace) && push!(trace, Σ_ladder_it)

        Δit = it > 1 ? sum(abs.(G_ladder_it .- G_ladder_bak))/prod(size(G_ladder_it)) : Inf
        Δit < conv_abs && (converged = true)

        it > mixing_start_it && (@inbounds G_ladder_it[:,:] = (1-mixing) .* G_ladder_it[:,:] .+ mixing .* G_ladder_bak[:,:])
        !all(isfinite.(Σ_ladder_it)) && break

        verbose && println("It = $it[λm=$λm/λd=$λd]: Δit=$Δit")
        G_rfft!(gLoc_rfft, G_ladder_it, h.kG, h.sP.fft_range)
        it += 1
    end

    return converged, μ_it
end
