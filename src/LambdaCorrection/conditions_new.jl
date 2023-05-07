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
mutable struct λ_result
    λm::Float64
    λd::Float64
    type::Symbol
    converged::Bool
    sc_converged::Bool
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
        new(λm, λd, type, true, converged, NaN, NaN, NaN, NaN, NaN, nothing, nothing, nothing, NaN)
    end
    function λ_result(λm::Float64, λd::Float64, type::Symbol, converged::Bool, sc_converged::Bool, 
                      EKin::Float64, EPot_p1::Float64, EPot_p2::Float64, PP_p1::Float64, PP_p2::Float64, 
                      trace::Union{DataFrame,Nothing}, 
                      G_ladder::Union{Nothing, OffsetMatrix}, Σ_ladder::Union{Nothing,OffsetMatrix}, μ::Float64)
        new(λm, λd, type, sc_converged, converged, EKin, EPot_p1, EPot_p2, PP_p1, PP_p2, trace, G_ladder, Σ_ladder, μ)
    end
end

function Base.show(io::IO, m::λ_result)
    compact = get(io, :compact, false)
    cc = m.converged ? "converged" : "NOT converged"
    if !compact
        println(io, "λ-correction (type: $(m.type)), $cc")
        println(io, "λm = $(m.λm), λd = $(m.λd)")
        !isnothing(m.trace) && println(io, "trace: \n", m.trace)
    else
        print(io, "λ-correction (type: $type) result, λm = $(m.λm), λd = $(m.λd) // $converged")
    end
end

function λ_correction(type::Symbol, χm::χT, γm::γT, χd::χT, γd::γT, λ₀, h::lDΓAHelper; 
                      # λm related:
                      λm_rhs_type::Symbol=:native,
                      # λdm related:
                      νmax::Int=-1, λ_min_δ::Float64 = 0.05,
                      # sc_X related:
                      maxit::Int=100, mixing::Float64=0.2, conv_abs::Float64=1e-8, trac::Bool=false,
                      # common options
                      par::Bool=false, verbose::Bool=false, validate_threshold::Float64=1e-8, tc::Bool=true)
    if type == :m
        rhs = λm_rhs(χm, χd, 0.0, h; λ_rhs = λm_rhs_type)
        λm, validation = λm_correction(χm, rhs, h, verbose=verbose, validate_threshold=validate_threshold)
        λ_result(λm, 0.0, :m, validation)
    elseif type == :dm
        λdm_correction(χm, γm, χd, γd, λ₀, h; νmax=νmax, λ_min_δ=λ_min_δ,
                       validate_threshold=validate_threshold, par=par, 
                       verbose=verbose, tc=tc, λ_val_only=λ_val_only)
    elseif type == :sc
        run_sc(χm, γm, χd, γd, λ₀, h; maxit=maxit, mixing=mixing, conv_abs=conv_abs, trace=trace)
    elseif type == :sc_m
        run_sc(χm, γm, χd, γd, λ₀, h; type=:m, maxit=maxit, mixing=mixing, conv_abs=conv_abs, trace=trace)
    else
        error("λ-correction type '$type' not recognized!")
    end
end

# =============================================== λm =================================================
"""
    λm_correction(χm::χT, rhs::Float64, h::lDΓAHelper; verbose::Bool=false, validate_threshold::Float64=1e-8)
    λm_correction(χm::χT, rhs::Float64, kG::KGrid, mP::ModelParameters, sP::SimulationParameters; verbose::Bool=false, validate_threshold::Float64=1e-8)
                        
Calculates ``\\lambda_\\mathrm{m}`` value, by fixing ``\\sum_{q,\\omega} \\chi^{\\lambda_\\mathrm{m}}_{\\uparrow\\uparrow}(q,i\\omega) = \\frac{n}{2}(1-\\frac{n}{2})``.

Set `verbose` to obtain a trace of the checks.
`validate_threshold` sets the threshold for the `rhs ≈ lhs` condition, set to `Inf` in order to accept any result. 
"""
function λm_correction(χm::χT, rhs::Float64, h::lDΓAHelper; validate_threshold::Float64=1e-8, verbose::Bool=false)
    λm_correction(χm, rhs, h.kG, h.mP, h.sP, validate_threshold=validate_threshold, verbose=verbose)
end

function λm_correction(χm::χT, rhs::Float64, kG::KGrid, mP::ModelParameters, sP::SimulationParameters; validate_threshold::Float64=1e-8, verbose::Bool=false)
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

# =============================================== λm =================================================

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
function λdm_correction(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::Array{ComplexF64,3}, h::lDΓAHelper;
                            νmax::Int=-1, λ_min_δ::Float64 = 0.05, λ_val_only::Bool=true,
                            sc_max_it::Int = 0, sc_mixing::Float64=0.2, sc_conv::Float64=1e-8,
                            validate_threshold::Float64=1e-8, par::Bool=false, verbose::Bool=false, tc::Bool=true)
    λdm_correction(χm, γm, χd, γd, h.Σ_loc, h.gLoc_rfft, h.χloc_m_sum, λ₀, h.kG, h.mP, h.sP; 
                       νmax=νmax, λ_min_δ=λ_min_δ, λ_val_only=λ_val_only,
                       sc_max_it=sc_max_it, sc_mixing=sc_mixing, sc_conv=sc_conv,
                       validate_threshold=validate_threshold, par=par, verbose=verbose, tc=tc)
end

function λdm_correction(χm::χT, γm::γT, χd::χT, γd::γT, Σ_loc::OffsetVector{ComplexF64},
                        gLoc_rfft::GνqT, χloc_m_sum::Union{Float64,ComplexF64}, λ₀::Array{ComplexF64,3},
                        kG::KGrid, mP::ModelParameters, sP::SimulationParameters; 
                        νmax::Int = -1, λ_min_δ::Float64 = 0.05, λ_val_only::Bool=true,
                        sc_max_it::Int = 0, sc_mixing::Float64=0.2, sc_conv::Float64=1e-8,
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
        χ_m_sum    = sum_kω(kG, χm)
        χ_d_sum    = sum_kω(kG, χd)
        lhs_c1     = real(χ_d_sum + χ_m_sum)/2
        E_pot_2    = (mP.U/2)*real(χ_d_sum - χ_m_sum) + mP.U * (mP.n/2 * mP.n/2)
        verbose && println("dbg: par = $par: cs Σ = $(abs(sum(Σ_ladder))), cs G = $(abs(sum(G_ladder))), cs χm = $(abs(sum(χm))), cs χd = $(abs(sum(χd))), EPot1 = $E_pot_1, EPot2 = $E_pot_2")
        reset!(χm)
        reset!(χd)
        return E_pot_1, E_pot_2, lhs_c1 
    end

    function residual_vals_sc(λ::MVector{2,Float64})
        χ_λ!(χm,λ[1])
        χ_λ!(χd,λ[2])
        rhs_c1, lhs_c1, E_pot_1, E_pot_2, E_kin_1, μnew, converged = run_sc!(νGrid, iωn_f, deepcopy(gLoc_rfft), G_ladder, Σ_ladder, Σ_work, Kνωq_pre, Ref(nothing),
                χm, γm, χd, γd, λ₀,kG, mP, sP, Σ_loc, χloc_m_sum;
                maxit=sc_max_it, mixing=sc_mixing, conv_abs=sc_conv)

        reset!(χm)
        reset!(χd)
        return E_pot_1, E_pot_2, lhs_c1 
    end

    function residual_f(λ::MVector{2,Float64})::MVector{2,Float64} 
        E_pot_1, E_pot_2, lhs_c1 = sc_max_it > 0 ? residual_vals_sc(λ) : residual_vals(λ)
        return MVector{2,Float64}([lhs_c1 - rhs_c1, E_pot_1 - E_pot_2])
    end

    # --- actual root finding ---
    λm_min_tmp = get_λ_min(real(χm.data)) 
    λd_min_tmp = get_λ_min(real(χd.data)) 
    start = MVector{2,Float64}([0.0, 0.0])
    min_λ = MVector{2,Float64}([λm_min_tmp, λd_min_tmp] .+ λ_min_δ)
    root = try
        newton_right(residual_f, start, min_λ, verbose=verbose)
    catch e
        println("Error: $e")
        [NaN, NaN]
    end
    if any(isnan.(root))
        println("WARNING: No λ root was found!")
    elseif any(root .< min_λ)
        println("WARNING: λ = $root outside region ($min_λ)!")
    end
    if λ_val_only
        return root
    else
        μnew, E_kin_1, E_pot_1, E_pot_2, lhs_c1 = residual_vals(MVector{2,Float64}(root))
        converged = abs(rhs_c1 - lhs_c1) <= validate_threshold && abs(E_pot_1 - E_pot_2) <= validate_threshold
        return λ_result(root[1], root[2], :dm, true, converged, E_kin_1, E_pot_1, E_pot_2, rhs_c1, lhs_c1, nothing, G_ladder, Σ_ladder, μnew)
    end
end

# =============================================== sc =================================================
function run_sc(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::Array{ComplexF64,3}, h::lDΓAHelper;
                type::Symbol=:O, par::Bool=false, λ_min_δ::Float64 = 0.05,
                maxit::Int=100, mixing::Float64=0.2, conv_abs::Float64=1e-8, trace=false,
                tc::Bool=true)
    _, νGrid, iωn_f = gen_νω_indices(χm, χd, h.mP, h.sP)
    fft_νGrid= h.sP.fft_range
    Nk = length(h.kG.kMult)
    G_ladder::OffsetMatrix{ComplexF64, Matrix{ComplexF64}} = OffsetArray(Matrix{ComplexF64}(undef, Nk, length(fft_νGrid)), 1:Nk, fft_νGrid) 
    Σ_ladder::OffsetMatrix{ComplexF64, Matrix{ComplexF64}} = OffsetArray(Matrix{ComplexF64}(undef, Nk, length(νGrid)),     1:Nk, νGrid)
    Σ_work   = similar(Σ_ladder)
    Kνωq_pre = Vector{ComplexF64}(undef, Nk)

    traceDF = trace ? DataFrame(it = Int[], λm = Float64[], λd = Float64[], μ = Float64[], EKin = Float64[], EPot = Float64[], 
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
    rhs_c1, lhs_c1, E_pot_1, E_pot_2, E_kin, μnew, sc_converged = run_sc!(νGrid, iωn_f, deepcopy(h.gLoc_rfft), 
                G_ladder, Σ_ladder, Σ_work, Kνωq_pre, Ref(traceDF), χm, γm, χd, γd, λ₀, h.kG, h.mP, h.sP, h.Σ_loc, h.χloc_m_sum;
                maxit=maxit, mixing=mixing, conv_abs=conv_abs)
    type != :O  && reset!(χm)
    type == :dm && reset!(χd)

    converged = all(isfinite.([lhs_c1, E_pot_2])) && abs(rhs_c1 - lhs_c1) <= conv_abs && abs(E_pot_1 - E_pot_2) <= conv_abs
    return λ_result(χm.λ, χd.λ, :sc, sc_converged, converged, E_kin, E_pot_1, E_pot_2, rhs_c1, lhs_c1, 
                    traceDF, G_ladder, Σ_ladder, μnew)
end


function run_sc!(νGrid::UnitRange{Int}, iωn_f::Vector{ComplexF64}, gLoc_rfft::GνqT, G_ladder::OffsetMatrix{ComplexF64}, 
                 Σ_ladder::OffsetMatrix{ComplexF64}, Σ_work::OffsetMatrix{ComplexF64}, Kνωq_pre::Vector{ComplexF64}, trace::Ref,
                 χm::χT, γm::γT, χd::χT, γd::γT, λ₀::Array{ComplexF64,3}, 
                 kG::KGrid, mP::ModelParameters, sP::SimulationParameters, Σ_loc::OffsetVector{ComplexF64}, χloc_m_sum::Union{Float64,ComplexF64};
                 maxit::Int=100, mixing::Float64=0.2, conv_abs::Float64=1e-8)
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
    μnew    = mP.μ

    while !done
        copy!(Σ_work, Σ_ladder)
        calc_Σ!(Σ_ladder, Kνωq_pre, χm, γm, χd, γd, χloc_m_sum, λ₀, gLoc_rfft, kG, mP, sP)
        mixing != 0 && it > 1 && (Σ_ladder[:,:] = (1-mixing) .* Σ_ladder .+ mixing .* Σ_work)
        μnew = G_from_Σladder!(G_ladder, Σ_ladder, Σ_loc, kG, mP; fix_n=true)
        isnan(μnew) && break
        G_rfft!(gLoc_rfft, G_ladder, kG, fft_νGrid)
        E_pot_1_old = E_pot_1
        E_kin, E_pot_1 = calc_E(G_ladder, Σ_ladder, μnew, kG, mP, νmax=last(νGrid))

        if abs(E_pot_1 - E_pot_1_old) < conv_abs
            converged = true
            done = true
        end
        if !isnothing(trace[])
            χ_m_sum2 = sum_ωk(kG, χm)
            χ_d_sum2 = sum_ωk(kG, χd)
            lhs_c1   = real(χ_d_sum + χ_m_sum)/2
            row = [it, χm.λ, χd.λ, μnew, E_kin, E_pot_1, lhs_c1, E_pot_2, χ_m_sum, χ_m_sum2, χ_d_sum, χ_d_sum2, abs(sum(Σ_ladder)), abs(sum(G_ladder))]
            push!(trace[], row)
        end
        (it >= maxit) && (done = true)

        it += 1
    end
    return rhs_c1, lhs_c1, E_pot_1, E_pot_2, E_kin, μnew, converged
end
