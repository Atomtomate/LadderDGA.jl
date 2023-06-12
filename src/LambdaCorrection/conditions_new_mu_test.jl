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

# =========================================== Interface ==============================================

"""
    λdm_correction(χm, γm, χd, γd, [Σ_loc, gLoc_rfft, λ₀, kG, mP, sP] OR [h::lDΓAHelper, λ₀]; 
        maxit_root = 100, atol_root = 1e-8, λd_min_δ = 0.1, λd_max = 500,
        maxit::Int = 50, update_χ_tail=false, mixing=0.2, conv_abs=1e-8, par=false)

Calculates ``\\lambda_\\mathrm{dm}`` and associated quantities like the self-energy.

TODO: full documentation. Pack results into struct

Returns: 
-------------
    λdm: `Vector`, containing `λm` and `λd`.
"""
function λdm_correction_no_mu(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::Array{ComplexF64,3}, h::lDΓAHelper;
                        νmax::Int=-1, λ_min_δ::Float64 = 0.15, λ_val_only::Bool=false,
                        sc_max_it::Int = 0, sc_mixing::Float64=0.2, sc_conv::Float64=1e-8,
                        update_χ_tail::Bool=false,
                        validate_threshold::Float64=1e-8, par::Bool=false, verbose::Bool=false, tc::Bool=true)
    λdm_correction_no_mu(χm, γm, χd, γd, h.Σ_loc, h.gLoc_rfft, h.χloc_m_sum, λ₀, h.kG, h.mP, h.sP; 
                   νmax=νmax, λ_min_δ=λ_min_δ, λ_val_only=λ_val_only,
                   sc_max_it=sc_max_it, sc_mixing=sc_mixing, sc_conv=sc_conv,
                   update_χ_tail=update_χ_tail,
                   validate_threshold=validate_threshold, par=par, verbose=verbose, tc=tc)
end

function λdm_correction_no_mu(χm::χT, γm::γT, χd::χT, γd::γT, Σ_loc::OffsetVector{ComplexF64},
                        gLoc_rfft::GνqT, χloc_m_sum::Union{Float64,ComplexF64}, λ₀::Array{ComplexF64,3},
                        kG::KGrid, mP::ModelParameters, sP::SimulationParameters; 
                        νmax::Int = -1, λ_min_δ::Float64 = 0.15, λ_val_only::Bool=false,
                        sc_max_it::Int = 0, sc_mixing::Float64=0.2, sc_conv::Float64=1e-8,
                        update_χ_tail::Bool=false, fit_μ::Bool = false,
                        validate_threshold::Float64=1e-8, par::Bool=false, verbose::Bool=false, tc::Bool=true)

    (χm.λ != 0 || χd.λ != 0) && error("λ parameter already set. Aborting λdm calculation")    
    ωindices, νGrid, iωn_f = gen_νω_indices(χm, χd, mP, sP, full=true)
    if νmax < 1 
        νmax = last(νGrid)+1
    else
        νGrid = νGrid[1:min(length(νGrid),νmax)]
    end

    # --- Preallocations ---
    par && initialize_EoM(gLoc_rfft, χloc_m_sum, λ₀, νGrid, kG, mP, sP, χ_m = χm, γ_m = γm, χ_d = χd, γ_d = γd)
    fft_νGrid = sP.fft_range
    Nq::Int   = length(kG.kMult)
    ωrange::UnitRange{Int}  = -sP.n_iω:sP.n_iω
    G_ladder::OffsetMatrix{ComplexF64, Matrix{ComplexF64}}      = OffsetArray(Matrix{ComplexF64}(undef, Nq, length(fft_νGrid)), 1:Nq, fft_νGrid) 
    Σ_ladder::OffsetMatrix{ComplexF64, Matrix{ComplexF64}}      = OffsetArray(Matrix{ComplexF64}(undef, Nq, length(νGrid)), 1:Nq, νGrid)
    Σ_work   = similar(Σ_ladder)
    Kνωq_pre = par ? nothing : Vector{ComplexF64}(undef, Nq)
    rhs_c1 = mP.n/2 * (1-mP.n/2)
    traceDF = verbose ? DataFrame(it = Int[], λm = Float64[], λd = Float64[], μ = Float64[], n = Float64[], EKin = Float64[], EPot = Float64[], 
        lhs_c1 = Float64[], EPot_c2 = Float64[], cs_m = Float64[], cs_m2 = Float64[],
        cs_d = Float64[], cs_d2 = Float64[], cs_Σ = Float64[], cs_G = Float64[]) : nothing

    # --- Internal root finding function ---
    function residual_vals(λ::MVector{2,Float64})
        χ_λ!(χm,λ[1])
        χ_λ!(χd,λ[2])
        if par
            calc_Σ_par!(Σ_ladder, λm=λ[1], λd=λ[2], tc=tc)
        else
            calc_Σ!(Σ_ladder, Kνωq_pre, χm, γm, χd, γd, χloc_m_sum, λ₀, gLoc_rfft, kG, mP, sP; tc=tc)
        end
        μnew = G_from_Σladder!(G_ladder, Σ_ladder, Σ_loc, kG, mP; fix_n=true)
        E_kin_1, E_pot_1 = calc_E(G_ladder, Σ_ladder, μnew, kG, mP, νmax=last(axes(Σ_ladder,2)))
        n = filling_pos(view(G_ladder, :, 0:last(collect(axes(G_ladder,2)))), kG, mP.U, μnew, mP.β)
        χ_m_sum    = sum_kω(kG, χm)
        χ_d_sum    = sum_kω(kG, χd)
        lhs_c1     = real(χ_d_sum + χ_m_sum)/2
        E_pot_2    = (mP.U/2)*real(χ_d_sum - χ_m_sum) + mP.U * (mP.n/2 * mP.n/2)
        verbose && println("dbg: par = $par: λ=$λ, EPot1 = $E_pot_1, EPot2 = $E_pot_2, PP_1 = $rhs_c1, PP_2 = $lhs_c1")
        reset!(χm)
        reset!(χd)
        return n, E_kin_1, E_pot_1, E_pot_2, lhs_c1 
    end

    function residual_vals_sc(λ::MVector{2,Float64})
        χ_λ!(χm,λ[1])
        χ_λ!(χd,λ[2])
        rhs_c1, lhs_c1, E_pot_1, E_pot_2, E_kin_1, n, converged = run_sc!(νGrid, iωn_f, deepcopy(gLoc_rfft), G_ladder, Σ_ladder, Σ_work, Kνωq_pre, Ref(traceDF),
                χm, γm, χd, γd, λ₀, μ, kG, mP, sP, Σ_loc, χloc_m_sum;
                maxit=sc_max_it, mixing=sc_mixing, conv_abs=sc_conv, update_χ_tail=update_χ_tail)

        reset!(χm)
        reset!(χd)
        return n, E_kin_1, E_pot_1, E_pot_2, lhs_c1 
    end

    function residual_f(λ::MVector{3,Float64})::MVector{3,Float64} 
        n, _, E_pot_1, E_pot_2, lhs_c1 = sc_max_it > 0 ? residual_vals_sc(λ) : residual_vals(λ)
        n_diff::Float64 = n - mP.n
        return MVector{3,Float64}([lhs_c1 - rhs_c1, E_pot_1 - E_pot_2, n_diff])
    end

    # --- actual root finding ---
    λm_min_tmp = get_λ_min(real(χm.data)) 
    λd_min_tmp = get_λ_min(real(χd.data)) 
    start = MVector{3,Float64}([0.0, 0.0, mP.μ])
    min_λ = MVector{3,Float64}([λm_min_tmp + λ_min_δ*abs(λm_min_tmp), λd_min_tmp + λ_min_δ*abs(λd_min_tmp), -Inf])
    root = try
        newton_right(residual_f, start, min_λ, verbose=verbose)
    catch e
        println("Error: $e")
        [NaN, NaN, NaN]
    end
    if any(isnan.(root))
        println("WARNING: No λ root was found!")
    elseif any(root .< min_λ)
        println("WARNING: λ = $root outside region ($min_λ)!")
    end
    if λ_val_only
        return root
    else
        type_str = sc_max_it > 0 ? "_sc" : ""
        type_str = update_χ_tail ? "_tsc" : type_str
        type_str = "dm"*type_str
        n, E_kin_1, E_pot_1, E_pot_2, lhs_c1 = sc_max_it == 0 ? residual_vals(MVector{3,Float64}(root)) : residual_vals_sc(MVector{3,Float64}(root))
        converged = abs(rhs_c1 - lhs_c1) <= validate_threshold && abs(E_pot_1 - E_pot_2) <= validate_threshold
        return λ_result(root[1], root[2], Symbol(type_str), true, converged, E_kin_1, E_pot_1, E_pot_2, rhs_c1, lhs_c1, traceDF, G_ladder, Σ_ladder, root[3], n)
    end
end

# =============================================== sc =================================================
function run_sc(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::Array{ComplexF64,3}, μ::Float64, h::lDΓAHelper;
                type::Symbol=:O, par::Bool=false, λ_min_δ::Float64 = 0.15,
                maxit::Int=100, mixing::Float64=0.2, conv_abs::Float64=1e-8, trace=false, update_χ_tail::Bool=false,
                tc::Bool=true)
    _, νGrid, iωn_f = gen_νω_indices(χm, χd, h.mP, h.sP)
    fft_νGrid= h.sP.fft_range
    Nk = length(h.kG.kMult)
    G_ladder::OffsetMatrix{ComplexF64, Matrix{ComplexF64}} = OffsetArray(Matrix{ComplexF64}(undef, Nk, length(fft_νGrid)), 1:Nk, fft_νGrid) 
    Σ_ladder::OffsetMatrix{ComplexF64, Matrix{ComplexF64}} = OffsetArray(Matrix{ComplexF64}(undef, Nk, length(νGrid)),     1:Nk, νGrid)
    Σ_work   = similar(Σ_ladder)
    Kνωq_pre = Vector{ComplexF64}(undef, Nk)

    traceDF = trace ? DataFrame(it = Int[], λm = Float64[], λd = Float64[], μ = Float64[], n = Float64[], EKin = Float64[], EPot = Float64[], 
        lhs_c1 = Float64[], EPot_c2 = Float64[], cs_m = Float64[], cs_m2 = Float64[],
        cs_d = Float64[], cs_d2 = Float64[], cs_Σ = Float64[], cs_G = Float64[]) : nothing

    if type == :m 
        rhs = λm_rhs(χm, χd, 0.0, h)
        λm, validation = λm_correction(χm, rhs, h)
        λd = 0.0
        χ_λ!(χm, λm)
        χ_λ!(χd, λd)
    elseif type == :dm
        λm, λd = λdm_correction(χm, γm, χd, γd, λ₀, h; λ_min_δ=λ_min_δ, λ_val_only=true,
                                validate_threshold=conv_abs, par=par, verbose=trace, tc=tc)
        χ_λ!(χm, λm)
        χ_λ!(χd, λd)
    end

    par && initialize_EoM(lDGAhelper, λ₀, 0:sP.n_iν-1, χ_m = χm, γ_m = γm, χ_d = χd, γ_d = γd)
    rhs_c1, lhs_c1, E_pot_1, E_pot_2, E_kin, n, μ, sc_converged = run_sc!(νGrid, iωn_f, deepcopy(h.gLoc_rfft), 
                G_ladder, Σ_ladder, Σ_work, Kνωq_pre, Ref(traceDF), χm, γm, χd, γd, λ₀, μ, h.kG, h.mP, h.sP, h.Σ_loc, h.χloc_m_sum;
                maxit=maxit, mixing=mixing, conv_abs=conv_abs, update_χ_tail=update_χ_tail, par=par)
    type != :O  && reset!(χm)
    type == :dm && reset!(χd)


    #TODO: CHECK mu CONVERGENCE
    #filling_pos(view(G_ladder, :, 0:last(fft_grid)), kG, mP.U, μ[1], mP.β)
    converged = all(isfinite.([lhs_c1, E_pot_2])) && abs(rhs_c1 - lhs_c1) <= conv_abs && abs(E_pot_1 - E_pot_2) <= conv_abs
    return λ_result(χm.λ, χd.λ, :sc, sc_converged, converged, E_kin, E_pot_1, E_pot_2, rhs_c1, lhs_c1, 
                    traceDF, G_ladder, Σ_ladder, μ, n)
end


function run_sc!(νGrid::UnitRange{Int}, iωn_f::Vector{ComplexF64}, gLoc_rfft::GνqT, G_ladder::OffsetMatrix{ComplexF64}, 
                 Σ_ladder::OffsetMatrix{ComplexF64}, Σ_work::OffsetMatrix{ComplexF64}, Kνωq_pre::Vector{ComplexF64}, trace::Ref,
                 χm::χT, γm::γT, χd::χT, γd::γT, λ₀::Array{ComplexF64,3}, μ::Float64,
                 kG::KGrid, mP::ModelParameters, sP::SimulationParameters, Σ_loc::OffsetVector{ComplexF64}, χloc_m_sum::Union{Float64,ComplexF64};
                 maxit::Int=100, mixing::Float64=0.2, conv_abs::Float64=1e-8, update_χ_tail::Bool=false, par::Bool=false)
    it      = 1
    done    = false
    converged = false
    fft_νGrid = sP.fft_range
    E_pot_1 = Inf
    χ_m_sum = sum_kω(kG, χm)
    χ_d_sum = sum_kω(kG, χd)
    lhs_c1  = real(χ_d_sum + χ_m_sum)/2
    rhs_c1  = mP.n/2*(1-mP.n/2)
    E_pot_2 = (mP.U/2)*real(χ_d_sum - χ_m_sum) + mP.U * (mP.n/2 * mP.n/2)
    E_kin   = Inf


    while !done
        copy!(Σ_work, Σ_ladder)
        if par
            calc_Σ_par!(Σ_ladder_inplace)
        else
            calc_Σ!(Σ_ladder, Kνωq_pre, χm, γm, χd, γd, χloc_m_sum, λ₀, gLoc_rfft, kG, mP, sP)
        end
        mixing != 0 && it > 1 && (Σ_ladder[:,:] = (1-mixing) .* Σ_ladder .+ mixing .* Σ_work)
        _ = G_from_Σladder!(G_ladder, Σ_ladder, Σ_loc, kG, mP; fix_n=false, μ=μ)
        G_rfft!(gLoc_rfft, G_ladder, kG, fft_νGrid)
        E_pot_1_old = E_pot_1
        E_pot_2_old = E_pot_2
        E_kin, E_pot_1 = calc_E(G_ladder, Σ_ladder, μ, kG, mP, νmax=last(νGrid))

        if update_χ_tail
            if !isfinite(E_kin)
                E_pot_1 = Inf
                lhs_c1  = Inf
                done    = true
            else
                update_tail!(χm, [0, 0, E_kin], iωn_f)
                update_tail!(χd, [0, 0, E_kin], iωn_f)
                χ_m_sum = sum_kω(kG, χm)
                χ_d_sum = sum_kω(kG, χd)
                lhs_c1  = real(χ_d_sum + χ_m_sum)/2
                E_pot_2 = (mP.U/2)*real(χ_d_sum - χ_m_sum) + mP.U * (mP.n/2 * mP.n/2)
            end
        end

        if abs(E_pot_1 - E_pot_1_old) < conv_abs &&  abs(E_pot_2 - E_pot_2_old) < conv_abs
            converged = true
            done = true
        end
        if !isnothing(trace[])
            χ_m_sum2 = sum_ωk(kG, χm)
            χ_d_sum2 = sum_ωk(kG, χd)
            lhs_c1   = real(χ_d_sum + χ_m_sum)/2
            n = filling_pos(view(G_ladder, :, 0:last(fft_νGrid)), kG, mP.U, μ, mP.β)
            row = [it, χm.λ, χd.λ, μ, n, E_kin, E_pot_1, lhs_c1, E_pot_2, χ_m_sum, χ_m_sum2, χ_d_sum, χ_d_sum2, abs(sum(Σ_ladder)), abs(sum(G_ladder))]
            push!(trace[], row)
        end
        (it >= maxit) && (done = true)

        it += 1
    end

    n = filling_pos(view(G_ladder, :, 0:last(fft_νGrid)), kG, mP.U, μ, mP.β)


    update_tail!(χm, [0, 0, mP.Ekin_DMFT], iωn_f)
    update_tail!(χd, [0, 0, mP.Ekin_DMFT], iωn_f)

    return rhs_c1, lhs_c1, E_pot_1, E_pot_2, E_kin, n, converged
end