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
    λm_tsc_correction(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::Array{ComplexF64,3}, h, sP, mP;
                        validation_threshold::Float64 = 1e-8, log_io = devnull
    )


"""
function λm_tsc_correction(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::λ₀T, h;
                           validation_threshold::Float64 = 1e-8, λ_rhs = :native,
                           max_steps_m::Int = 2000, max_steps_dm::Int = 2000, max_steps_sc::Int = 2000,
                           log_io = devnull, tc::Symbol = default_Σ_tail_correction())       

    λd = 0.0
    converged, μ_new, λm, G_ladder_it, Σ_ladder_it, χm_it, χd_it = run_tsc(χm, γm, χd, γd, λ₀, λd, h; maxit=max_steps_sc, conv_abs=validation_threshold, tc = tc)
    return λ_result(m_tscCorrection, χm_it, χd_it, μ_new, G_ladder_it, Σ_ladder_it, λm, λd, converged, h; validation_threshold = validation_threshold, max_steps_m = max_steps_m)
end


"""
    λdm_tsc_correction(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::Array{ComplexF64,3}, h, sP, mP;
                        validation_threshold::Float64 = 1e-8, log_io = devnull
    )


"""
function λdm_tsc_correction_clean(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::λ₀T, h;
                           use_trivial_λmin::Bool = false,
                           validation_threshold::Float64 = 1e-8,
                           max_steps_m::Int = 2000, max_steps_dm::Int = 2000, max_steps_sc::Int = 2000,
                           log_io = devnull, tc::Symbol = default_Σ_tail_correction())       

    λd_min::Float64   = if use_trivial_λmin 
        get_λ_min(χd)
    else
        get_λd_min(χm, γm, χd, γd, λ₀, h)
    end
    
    function f_c2_sc(λd_i::Float64)
        converged, μ_new, λm, G_ladder_it, Σ_ladder_it, χm_it, χd_it = run_tsc(χm, γm, χd, γd, λ₀, λd_i, h; maxit=max_steps_sc, max_steps_m = max_steps_m, conv_abs=validation_threshold, tc = tc)
        #TODO: use Epot_1
        Ekin_1, Epot_1 = calc_E(G_ladder_it, Σ_ladder_it, μ_new, h.kG, h.mP)
        Epot_2 = EPot_p2(χm_it, χd_it, λm, λd_i, h.mP.n, h.mP.U, h.kG)
        if !converged
            @warn "internal tsc did not converge"
            return (Epot_1 - Epot_2)*10
        end
        return Epot_1 - Epot_2
    end    
    λd = newton_secular(f_c2_sc, λd_min; nsteps=max_steps_dm, atol=validation_threshold)
    converged, μ_new, λm, G_ladder_it, Σ_ladder_it, χm_it, χd_it = run_tsc(χm, γm, χd, γd, λ₀, λd, h; maxit=max_steps_sc, conv_abs=validation_threshold, tc = tc)

    return λ_result(dm_tscCorrection, χm_it, χd_it, μ_new, G_ladder_it, Σ_ladder_it, λm, λd, converged, h; validation_threshold = validation_threshold, max_steps_m = max_steps_m)
end


function run_tsc(χm_bak::χT, γm::γT, χd_bak::χT, γd::γT, λ₀::λ₀T, λd::Float64, h;
                maxit::Int=100, mixing::Float64=0.2, conv_abs::Float64=1e-8, 
                max_steps_m=1000, tc::Symbol = default_Σ_tail_correction())
    it        = 1
    λm        = 0.0
    converged = false

    μ_it, G_ladder_it, Σ_ladder_it = calc_G_Σ(χm_bak, γm, χd_bak, γd, λ₀, λm, λd, h; tc = tc, fix_n = true)
    _, gLoc_rfft = G_fft(G_ladder_it, h.kG, h.sP)
    Σ_ladder_last = deepcopy(Σ_ladder_it)
    iωn_grid = ωn_grid(χm_bak)

    # internal λm-correction stuff
    PP_p1  = h.mP.n / 2 * (1 - h.mP.n / 2)
    
    # in case we encounter NaN/Inf in tail
    χm_it = deepcopy(χm_bak)
    χd_it = deepcopy(χd_bak)

    tail_bak_m = deepcopy(χm_it.tail_c)
    tail_bak_d = deepcopy(χd_it.tail_c)
    while it <= maxit && !converged
        χd_sum   = sum_kω(h.kG, χd_it, λ = λd)
        rhs_c1   = h.mP.n * (1 - h.mP.n / 2) - χd_sum
        ωn2_tail = ω2_tail(χm_it)
        λm       = λm_correction_val(χm_it, rhs_c1, h.kG, ωn2_tail; max_steps=max_steps_m, eps=conv_abs)

        μ_it, G_ladder_it, Σ_ladder_it = calc_G_Σ(χm_it, γm, χd_it, γd, λ₀, λm, λd, h; gLoc_rfft = gLoc_rfft, tc = tc, fix_n = true)

        if !all(isfinite.(Σ_ladder_it)) 
            @error "EoM self-consistency encountered NaN/Inf in self-energy."
            break
        end

        # Check for sc and λm convergence
        χm_sum = sum_kω(h.kG, χm_it, λ = λm)
        PP_p2  = real(χd_sum + χm_sum) / 2
        if sum(abs.(Σ_ladder_it .- Σ_ladder_last)) < conv_abs && (abs(PP_p1 - PP_p2) < conv_abs)
            converged = true
        end

        G_rfft!(gLoc_rfft, G_ladder_it, h.kG, h.sP.fft_range)
        copyto!(Σ_ladder_last, Σ_ladder_it)
        Ekin_1, Epot_1 = calc_E(G_ladder_it, Σ_ladder_it, μ_it, h.kG, h.mP)

        if !isfinite(Ekin_1)
            @error "Kinetic energy not finite! Aborting λ-tsc"
            break
        else
            update_tail!(χm_it, [0, 0, Ekin_1], iωn_grid)
            update_tail!(χd_it, [0, 0, Ekin_1], iωn_grid)
        end
        it += 1
    end
    update_tail!(χm_it, tail_bak_m, iωn_grid)
    update_tail!(χd_it, tail_bak_d, iωn_grid)

    return converged, μ_it, λm, G_ladder_it, Σ_ladder_it, χm_it, χd_it
end


function λdm_tsc_correction(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::λ₀T, h;
                            maxit::Int=500, mixing::Float64=0.3, mixing_start_it::Int=10,
                            conv_abs::Float64=1e-8, tc::Symbol = default_Σ_tail_correction(), trace::Bool=false, verbose::Bool=false)       

    λd_min = get_λ_min(χd)
    Nq, Nω = size(χm)
    νmax::Int = eom_ν_cutoff(h.sP)
    fft_νGrid= h.sP.fft_range

    Kνωq_pre    = Vector{ComplexF64}(undef, length(h.kG.kMult))
    G_ladder_it = OffsetArray(Matrix{ComplexF64}(undef, Nq, length(fft_νGrid)), 1:Nq, fft_νGrid) 
    G_ladder_bak = similar(G_ladder_it)
    Σ_ladder_it = OffsetArray(Matrix{ComplexF64}(undef, Nq, νmax), 1:Nq, 0:νmax-1)
    iν = iν_array(h.mP.β, collect(axes(Σ_ladder_it, 2)))
    tc_factor_term = tail_factor(h.mP.U, h.mP.β, h.mP.n, h.Σ_loc, iν; mode=tc)
    χm_it = deepcopy(χm)
    χd_it = deepcopy(χd)
    
    function f_c2_sc(λd_i::Float64)
        converged, μ_new, λm = run_tsc!(G_ladder_it, Σ_ladder_it, G_ladder_bak, Kνωq_pre, tc_factor_term,
            χm_it, γm, χd_it, γd, λ₀, λd_i, h; maxit=maxit, mixing=mixing, mixing_start_it=mixing_start_it)

        #TODO: use Epot_1
        Ekin_1, Epot_1 = calc_E(G_ladder_it, Σ_ladder_it, μ_new, h.kG, h.mP)
        Epot_2 = EPot_p2(χm_it, χd_it, λm, λd_i, h.mP.n, h.mP.U, h.kG)
        if !converged
            @warn "internal tsc did not converge"
            return (Epot_1 - Epot_2)*10
        end
        return Epot_1 - Epot_2
    end    
    λd = newton_secular(f_c2_sc, λd_min; nsteps=max_steps_dm, atol=validation_threshold)
    converged, μ_new, λm = run_tsc!(G_ladder_it, Σ_ladder_it, G_ladder_bak, Kνωq_pre, tc_factor_term,
            χm_it, γm, χd_it, γd, λ₀, λd, h; maxit=maxit, mixing=mixing, mixing_start_it=mixing_start_it)

    return λ_result(dm_tscCorrection, χm_it, χd_it, μ_new, G_ladder_it, Σ_ladder_it, λm, λd, converged, h)
end

function run_tsc(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::λ₀T, h;
                maxit::Int=500, mixing::Float64=0.3, mixing_start_it::Int=10,
                conv_abs::Float64=1e-8, tc::Bool = true, trace::Bool=false, verbose::Bool=false)
    it      = 2
    converged = false
    tr = []    
    iωn_grid = ωn_grid(χm)
    tail_bak_m = deepcopy(χm.tail_c)
    tail_bak_d = deepcopy(χd.tail_c)
    ωn2_tail = ω2_tail(χm)
    χm_bak = deepcopy(χm)
    χd_bak = deepcopy(χd)
    λm, λd = λdm_correction_val(χm ,γm ,χd, γd ,λ₀, h; validation_threshold = 1e-8, max_steps_m = 2000)

    μ_it, G_ladder_it, Σ_ladder_it = calc_G_Σ(χm, γm, χd, γd, λ₀, λm, λd, h; tc = tc, fix_n = true)
    G_ladder_bak = similar(G_ladder_it)
    _, gLoc_rfft = G_fft(G_ladder_it, h.kG, h.sP)
 
    trace && push!(tr, Σ_ladder_it)
    
    while it <= maxit && !converged
        λm, λd = λdm_correction_val(χm ,γm ,χd, γd ,λ₀, h; validation_threshold = 1e-8, max_steps_m = 2000)
        it > mixing_start_it && copy!(G_ladder_bak, G_ladder_it)

        μ_it, G_ladder_it, Σ_ladder_it = calc_G_Σ(χm, γm, χd, γd, λ₀, λm, λd, h; gLoc_rfft = gLoc_rfft, tc = :full, fix_n = true)
        trace && push!(tr, Σ_ladder_it)

        Δit = it > 1 ? sum(abs.(G_ladder_it .- G_ladder_bak))/prod(size(G_ladder_it)) : Inf
        Δit < conv_abs && (converged = true)

        it > mixing_start_it && (@inbounds G_ladder_it[:,:] = (1-mixing) .* G_ladder_it[:,:] .+ mixing .* G_ladder_bak[:,:])
        !all(isfinite.(Σ_ladder_it)) && break
        
        verbose && println("It = $it[λm=$λm/λd=$λd]: Δit=$Δit")
        G_rfft!(gLoc_rfft, G_ladder_it, h.kG, h.sP.fft_range)
        #TODO: use Ekin_1
        Ekin_1, Epot_1 = calc_E(G_ladder_it, Σ_ladder_it, μ_it, h.kG, h.mP)

        if !isfinite(Ekin_1)
            @error "Kinetic energy not finite! Aborting λ-tsc"
            break
        else
            update_tail!(χm, [0, 0, Ekin_1], iωn_grid)
            update_tail!(χd, [0, 0, Ekin_1], iωn_grid)
        end
        it += 1
    end

    χm = deepcopy(χm_bak)
    χd_bak = deepcopy(χd_bak)
    return converged, λm, λd, μ_it, G_ladder_it, Σ_ladder_it, tr
end

function run_tsc!(G_ladder_it::OffsetArray, Σ_ladder_it::OffsetArray, G_ladder_bak::OffsetArray, Kνωq_pre,  
                χm::χT, γm::γT, χd::χT, γd::γT, λ₀::λ₀T, λd, h;
                maxit::Int=500, mixing::Float64=0.3, mixing_start_it::Int=10,
                conv_abs::Float64=1e-8, trace=nothing, verbose::Bool=false)
    it      = 1
    converged = false
    @assert mixing_start_it > 1

    gLoc_rfft = deepcopy(h.gLoc_rfft)
    μ_it = h.mP.μ
    
    iωn_grid = ωn_grid(χm)
    tail_bak_m = deepcopy(χm.tail_c)
    tail_bak_d = deepcopy(χd.tail_c)
    iν = iν_array(h.mP.β, collect(axes(Σ_ladder_it, 2)))
    λm   = NaN

    while it <= maxit && !converged
        rhs_c1,_ = λm_rhs(χm, χd, h; λd=λd)
        ωn2_tail = ω2_tail(χm)
        λm   = λm_correction_val(χm, rhs_c1, h.kG, ωn2_tail)
        verbose && println("It = $it[λm=$λm/λd=$λd]")
        #λm, λd = λdm_correction_val(χm ,γm ,χd, γd ,λ₀, h; validation_threshold = 1e-8, max_steps_m = 2000)
        tc_factor = tail_factor(h.mP.U, h.mP.β, h.mP.n, h.Σ_loc, iν)
        tc_term  = tail_correction_term(sum_kω(h.kG, χm, λ=λm), h.χloc_m_sum, tc_factor)
        it > mixing_start_it && copy!(G_ladder_bak, G_ladder_it)
        μ_it = calc_G_Σ!(G_ladder_it, Σ_ladder_it, Kνωq_pre, tc_term, χm, γm, χd, γd, λ₀, λm, λd, h; gLoc_rfft=gLoc_rfft)
        !isnothing(trace) && push!(trace, deepcopy(Σ_ladder_it))

        Δit = it > 1 ? sum(abs.(G_ladder_it .- G_ladder_bak))/prod(size(G_ladder_it)) : Inf
        Δit < conv_abs && (converged = true)

        it > mixing_start_it && (@inbounds G_ladder_it[:,:] = (1-mixing) .* G_ladder_it[:,:] .+ mixing .* G_ladder_bak[:,:])
        !all(isfinite.(Σ_ladder_it)) && break

        
        G_rfft!(gLoc_rfft, G_ladder_it, h.kG, h.sP.fft_range)
        #TODO: use Ekin_1
        Ekin_1, Epot_1 = calc_E(G_ladder_it, Σ_ladder_it, μ_it, h.kG, h.mP)
        verbose && println("It = $it[λm=$λm/λd=$λd]: Δit=$Δit // Ekin_1 = $Ekin_1 // cs = $(sum(abs.(χm)))")
        if !isfinite(Ekin_1)
            @error "Kinetic energy not finite! Aborting λ-tsc"
            break
        else
            update_tail!(χm, [0, 0, Ekin_1], iωn_grid)
            update_tail!(χd, [0, 0, Ekin_1], iωn_grid)
        end
        it += 1
    end

    update_tail!(χm, tail_bak_m, iωn_grid)
    update_tail!(χd, tail_bak_d, iωn_grid)

    return converged, λm, λd, μ_it
end
