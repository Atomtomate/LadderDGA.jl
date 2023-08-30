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
    trace::Vector{DataFrame}
    G_ladder::Union{Nothing, OffsetMatrix}
    Σ_ladder::Union{Nothing, OffsetMatrix}
    μ::Float64
    n::Float64
    function λ_result(λm::Float64, λd::Float64, type::Symbol, converged::Bool)
        new(λm, λd, type, true, converged, NaN, NaN, NaN, NaN, NaN, DataFrame[], nothing, nothing, NaN, NaN)
    end
    function λ_result(λm::Float64, λd::Float64, type::Symbol, converged::Bool, sc_converged::Bool, 
                      EKin::Float64, EPot_p1::Float64, EPot_p2::Float64, PP_p1::Float64, PP_p2::Float64, 
                      trace::Union{Vector{DataFrame},DataFrame,Nothing}, 
                      G_ladder::Union{Nothing, OffsetMatrix}, Σ_ladder::Union{Nothing,OffsetMatrix}, 
                      μ::Float64, n::Float64)
        trace_int = if typeof(trace) === Nothing
            DataFrame[]
        elseif typeof(trace) === DataFrame
            DataFrame[trace]
        else
            trace
        end
        new(λm, λd, type, sc_converged, converged, EKin, EPot_p1, EPot_p2, PP_p1, PP_p2, trace_int, G_ladder, Σ_ladder, μ, n)
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
                      νmax::Int=-1, λ_min_δ::Float64 = 0.0001,
                      # sc_X r, delete_G_Σ::Bool=trueelated:
                      maxit::Int=100, mixing::Float64=0.2, conv_abs::Float64=1e-8, trace::Bool=false,
                      # common options
                      par::Bool=false, λ_val_only::Bool=false, verbose::Bool=false, validate_threshold::Float64=1e-8, tc::Bool=true)
    if type == :m
        λm_correction_full(χm, γm, χd, γd, λ₀, h;
                           νmax=νmax, λ_min_δ=λ_min_δ, verbose=verbose,
                           validate_threshold=validate_threshold)
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


function λ_correction(type::Symbol, χm::χT, χd::χT, h::RPAHelper; 
    # common options
    verbose::Bool=false, validate_threshold::Float64=1e-8)
    if type == :m
        λm_correction_RPA(χm, χd, h; verbose=verbose, validate_threshold=validate_threshold)
    else
    error("RPA: λ-correction type '$type' not recognized!")
    end
end

function λm_correction_RPA(χm::χT, χd::χT, h::RPAHelper; verbose::Bool=false, validate_threshold::Float64=1e-8)

    kG:: KGrid = h.kG
    rhs = h.mP.n*(1-0.5*h.mP.n) - sum_kω(kG, χd, χd.β, 0.0, zeros(Float64, length(χd.usable_ω)); transform=nothing)
    
    λm_min = get_λ_min(χm)
    χr::SubArray{Float64,2}    = view(χm,:,χm.usable_ω)
    iωn = (1im .* 2 .* (-h.sP.n_iω:h.sP.n_iω)[χm.usable_ω] .* π ./ h.mP.β)

    f_c1(λint::Float64)::Float64  = sum_kω(kG, χr, χm.β, 0.0, zeros(Float64, length(χd.usable_ω)); transform=(f(x::Float64)::Float64 = χ_λ(x, λint))) - rhs
    df_c1(λint::Float64)::Float64 = sum_kω(kG, χr, χm.β, 0.0, zeros(Float64, length(χd.usable_ω)); transform=(f(x::Float64)::Float64 = dχ_λ(x, λint)))
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
        println("CHECK for rhs = $rhs  : ", check, " ?=? ", 0)
        println("sum_kω - PP = ", abs(rhs - check))
        println("sum_ωk - PP = ", abs(rhs - check2))
    end
    validation = (abs(rhs - check) <= validate_threshold) &&  (abs(rhs - check2) <= validate_threshold) 
    return λm, validation


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
    zi = findfirst(x->abs(x)<1e-10, iωn)
    ωn2_tail[zi] = 0.0

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
        println("CHECK for rhs = $rhs  : ", check, " ?=? ", 0)
        println("sum_kω - PP = ", abs(rhs - check))
        println("sum_ωk - PP = ", abs(rhs - check2))
    end
    validation = (abs(rhs - check) <= validate_threshold) &&  (abs(rhs - check2) <= validate_threshold) 
    return λm, validation
end

function λm_correction_full(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::Array{ComplexF64,3}, h::lDΓAHelper;
                            νmax::Int=-1, λ_min_δ::Float64 = 0.0001, λ_val_only::Bool=false, verbose::Bool=false,
                            fit_μ::Bool=true, validate_threshold::Float64=1e-8)

        νmax = νmax < 0 ? floor(Int, size(γm, γm.axis_types[:ν])/2) : νmax
        rhs = λm_rhs(χm, χd, h; λ_rhs = :native)
        λm, validation = λm_correction(χm, rhs, h, verbose=verbose, validate_threshold=validate_threshold)
        Σ_ladder = calc_Σ(χm, γm, χd, γd, λ₀, h, νmax=νmax, λm=λm);
        μnew, G_ladder = G_from_Σladder(Σ_ladder, h.Σ_loc, h.kG, h.mP, h.sP; fix_n=fit_μ)
        EKin1, EPot1 = calc_E(G_ladder, Σ_ladder, μnew, h.kG, h.mP)
        rhs_c1  = h.mP.n/2 * (1-h.mP.n/2)
        χ_m_sum = sum_kω(h.kG, χm, λ=λm)
        χ_d_sum = sum_kω(h.kG, χd)
        lhs_c1  = real(χ_d_sum + χ_m_sum)/2
        EPot2   = (h.mP.U/2)*real(χ_d_sum - χ_m_sum) + h.mP.U * (h.mP.n/2 * h.mP.n/2)
        λ_result(λm, χd.λ, :m, validation, true, EKin1, EPot1, EPot2, rhs_c1, lhs_c1, nothing, G_ladder, Σ_ladder, μnew, h.mP.n)
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
                        νmax::Int=-1, λ_min_δ::Float64 = 0.0001, λ_val_only::Bool=false,
                        sc_max_it::Int = 0, sc_mixing::Float64=0.2, sc_conv::Float64=1e-8,
                        update_χ_tail::Bool=false, fit_μ::Bool=true, μ::Float64=h.mP.μ,
                        validate_threshold::Float64=1e-8, par::Bool=false, verbose::Bool=false, tc::Bool=true)
    λdm_correction(χm, γm, χd, γd, h.Σ_loc, h.gLoc_rfft, h.χloc_m_sum, λ₀, h.kG, h.mP, h.sP; 
                   νmax=νmax, λ_min_δ=λ_min_δ, λ_val_only=λ_val_only,
                   sc_max_it=sc_max_it, sc_mixing=sc_mixing, sc_conv=sc_conv,
                   update_χ_tail=update_χ_tail, fit_μ=fit_μ, μ=μ,
                   validate_threshold=validate_threshold, par=par, verbose=verbose, tc=tc)
end

function λdm_correction(χm::χT, γm::γT, χd::χT, γd::γT, Σ_loc::OffsetVector{ComplexF64},
                        gLoc_rfft::GνqT, χloc_m_sum::Union{Float64,ComplexF64}, λ₀::Array{ComplexF64,3},
                        kG::KGrid, mP::ModelParameters, sP::SimulationParameters; 
                        νmax::Int = -1, λ_min_δ::Float64 = 0.0001, λ_val_only::Bool=false,
                        sc_max_it::Int = 0, sc_mixing::Float64=0.2, sc_conv::Float64=1e-8,
                        update_χ_tail::Bool=false, fit_μ::Bool=true, μ::Float64=mP.μ, 
                        λinit::Vector{Float64}=[0.0,0.0],
                        validate_threshold::Float64=1e-8, par::Bool=false, verbose::Bool=false, tc::Bool=true)

    (χm.λ != 0 || χd.λ != 0) && error("λ parameter already set. Aborting λdm calculation")    
    ωindices, νGrid, iωn_f = gen_νω_indices(χm, χd, mP, sP)
    if νmax < 1 
        νmax = last(νGrid)+1
    else
        νGrid = νGrid[1:min(length(νGrid),νmax)]
    end

    # --- Preallocations ---
    par && initialize_EoM(gLoc_rfft, χloc_m_sum, λ₀, νGrid, kG, mP, sP, χ_m = χm, γ_m = γm, χ_d = χd, γ_d = γd)
    fft_νGrid = sP.fft_range
    Nq::Int   = length(kG.kMult)
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
        (λ[1] > 1e5 || λ[2] > 1e5) && return NaN, NaN, NaN, NaN, NaN, NaN, false
        χ_λ!(χm, λ[1])
        χ_λ!(χd, λ[2])
        if par
            calc_Σ_par!(Σ_ladder, λm=λ[1], λd=λ[2], tc=tc)
        else
            calc_Σ!(Σ_ladder, Kνωq_pre, χm, γm, χd, γd, χloc_m_sum, λ₀, gLoc_rfft, kG, mP, sP; tc=tc)
        end
        μ = G_from_Σladder!(G_ladder, Σ_ladder, Σ_loc, kG, mP; fix_n=fit_μ, μ=μ)
        E_kin_1, E_pot_1 = calc_E(G_ladder, Σ_ladder, μ, kG, mP)
        n = filling_pos(view(G_ladder, :, 0:νmax-1), kG, mP.U, μ, mP.β)
        χ_m_sum    = sum_kω(kG, χm)
        χ_d_sum    = sum_kω(kG, χd)
        lhs_c1     = real(χ_d_sum + χ_m_sum)/2
        E_pot_2    = (mP.U/2)*real(χ_d_sum - χ_m_sum) + mP.U * (mP.n/2 * mP.n/2)
        verbose && println("dbg: par = $par: λ=$λ, EPot1 = $E_pot_1, EPot2 = $E_pot_2, PP_1 = $rhs_c1, PP_2 = $lhs_c1, μ = $μ")
        reset!(χm)
        reset!(χd)
        return n, μ, E_kin_1, E_pot_1, E_pot_2, lhs_c1, true
    end

    function residual_vals_sc(λ::MVector{2,Float64})
        χ_λ!(χm,λ[1])
        χ_λ!(χd,λ[2])
        rhs_c1, lhs_c1, E_pot_1, E_pot_2, E_kin, n, μ, converged = run_sc!(νGrid, iωn_f, deepcopy(gLoc_rfft), G_ladder, Σ_ladder, Σ_work, Kνωq_pre, Ref(traceDF),
                χm, γm, χd, γd, λ₀, mP.μ, kG, mP, sP, Σ_loc, χloc_m_sum;
                maxit=sc_max_it, mixing=sc_mixing, conv_abs=sc_conv, update_χ_tail=update_χ_tail)

        reset!(χm)
        reset!(χd)
        return n, μ, E_kin, E_pot_1, E_pot_2, lhs_c1, converged 
    end

    function residual_f(λ::MVector{2,Float64})::MVector{2,Float64} 
        n, μ, _, E_pot_1, E_pot_2, lhs_c1, converged = sc_max_it > 0 ? residual_vals_sc(λ) : residual_vals(λ)
        return  MVector{2,Float64}([lhs_c1 - rhs_c1, E_pot_1 - E_pot_2])
    end

    # --- actual root finding ---
    λm_min_tmp = get_λ_min(real(χm.data)) 
    λd_min_tmp = get_λ_min(real(χd.data)) 
    start = MVector{2,Float64}(λinit)
    min_λ = MVector{2,Float64}([λm_min_tmp + λ_min_δ*abs(λm_min_tmp), λd_min_tmp + λ_min_δ*abs(λd_min_tmp)])
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

    type_str = sc_max_it > 0 ? "_sc" : ""
    type_str = update_χ_tail ? "_tsc" : type_str
    type_str = "dm"*type_str
    if all(isfinite.(root))
        n, μ, E_kin, E_pot_1, E_pot_2, lhs_c1, sc_converged = sc_max_it == 0 ? residual_vals(MVector{2,Float64}(root)) : residual_vals_sc(MVector{2,Float64}(root))
        converged = abs(rhs_c1 - lhs_c1) <= validate_threshold && abs(E_pot_1 - E_pot_2) <= validate_threshold
        if λ_val_only
            return root[1], root[2], converged
        else
            return λ_result(root[1], root[2], Symbol(type_str), true, converged, E_kin, E_pot_1, E_pot_2, rhs_c1, lhs_c1, traceDF, G_ladder, Σ_ladder, μ, n)
        end
    else
        if λ_val_only
            return root[1], root[2], false
        else
            return λ_result(root[1], root[2], Symbol(type_str), true, false, NaN, NaN, NaN, NaN, NaN, traceDF, nothing, nothing, NaN, NaN)
        end
    end
end

# =============================================== sc =================================================
#
"""
function run_sc(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::Array{ComplexF64,3}, μ::Float64, h::lDΓAHelper;
                type::Symbol=:fix, par::Bool=false, λ_min_δ::Float64 = 0.15, νmax::Int=-1,
                maxit::Int=100, mixing::Float64=0.2, conv_abs::Float64=1e-8, trace=false, verbose::Bool=false, update_χ_tail::Bool=false, fit_μ::Bool=true,
                tc::Bool=true, type=:fix, λm::Float64=0.0, λd::Float64=0.0)

  - *type* : 
    - :fix     : `λm` and `λd` (see above) are used, values are not changed for successive iterations.
    - :pre_dm  : `λm` and `λd` (see above) are used, after the first iteration, lDΓA_dm conditions are used in each iteration.
    - :pre_m   : `λm` (see above) is used, after the first iteration, lDΓA_m conditions is used in each iteration.
    - :dm      : Same as `:fix`, but initial values are obtained from lDΓA_dm conditions.
    - :m       : Same as `:fix`, but initial values are obtained from lDΓA_m condition.
"""

function run_sc(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::Array{ComplexF64,3}, μ::Float64, h::lDΓAHelper;
                par::Bool=false, λ_min_δ::Float64 = 0.0001, νmax::Int=-1,
                maxit::Int=100, mixing::Float64=0.2, conv_abs::Float64=1e-8, trace=false, verbose::Bool=false, update_χ_tail::Bool=false, fit_μ::Bool=true,
                tc::Bool=true, type=:fix, λm::Float64=0.0, λd::Float64=0.0)
    !fit_μ && @warn "Not fitting μ can lead to unphysical results!"
    _, νGrid, iωn_f = gen_νω_indices(χm, χd, h.mP, h.sP)
    if νmax < 1 
        νmax = last(νGrid)+1
    else
        νGrid = νGrid[1:min(length(νGrid),νmax)]
    end

    fft_νGrid= h.sP.fft_range
    Nk = length(h.kG.kMult)
    G_ladder::OffsetMatrix{ComplexF64, Matrix{ComplexF64}} = OffsetArray(Matrix{ComplexF64}(undef, Nk, length(fft_νGrid)), 1:Nk, fft_νGrid) 
    Σ_ladder::OffsetMatrix{ComplexF64, Matrix{ComplexF64}} = OffsetArray(Matrix{ComplexF64}(undef, Nk, length(νGrid)),     1:Nk, νGrid)
    Σ_work   = similar(Σ_ladder)
    Kνωq_pre = Vector{ComplexF64}(undef, Nk)

    traceDF = trace ? DataFrame(it = Int[], λm = Float64[], λd = Float64[], μ = Float64[], n = Float64[], EKin = Float64[], EPot = Float64[], 
        lhs_c1 = Float64[], EPot_c2 = Float64[], cs_m = Float64[], cs_m2 = Float64[],
        cs_d = Float64[], cs_d2 = Float64[], cs_Σ = Float64[], cs_G = Float64[]) : nothing

    λm, λd,validation = if type in [:pre_m] 
        rhs = λm_rhs(χm, χd, h)
        λm, validation = λm_correction(χm, rhs, h)
        λd = 0.0
        λm, λd, validation
    elseif type in [:pre_dm]
        λdm_correction(χm, γm, χd, γd, λ₀, h; νmax=νmax, λ_min_δ=λ_min_δ, λ_val_only=true,
                    validate_threshold=conv_abs, fit_μ=fit_μ, μ=μ, par=par, verbose=verbose, tc=tc)
    elseif type in [:fix, :dm, :m]
        λm, λd, true
    else
        @warn "Unrecognized type = $type in run_sc"
        0.0, 0.0, true
    end

    par && initialize_EoM(lDGAhelper, λ₀, 0:sP.n_iν-1, χ_m = χm, γ_m = γm, χ_d = χd, γ_d = γd)

    λm, λd, rhs_c1, lhs_c1, E_pot_1, E_pot_2, E_kin, n, μ, sc_converged = if validation
        χ_λ!(χm, λm)
        χ_λ!(χd, λd)
        λm_int, λd_int, rhs_c1, lhs_c1, E_pot_1, E_pot_2, E_kin, n, μ, sc_converged = run_sc!(iωn_f, deepcopy(h.gLoc_rfft), 
                    G_ladder, Σ_ladder, Σ_work, Kνωq_pre, Ref(traceDF), χm, γm, χd, γd, λ₀, μ, h;
                    maxit=maxit, mixing=mixing, conv_abs=conv_abs, update_χ_tail=update_χ_tail, 
                    fit_μ=fit_μ, par=par, type=type)
        reset!(χm)
        reset!(χd)
        λm_int, λd_int, rhs_c1, lhs_c1, E_pot_1, E_pot_2, E_kin, n, μ, sc_converged 
    else
        NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, false
    end

    #TODO: CHECK mu CONVERGENCE
    #filling_pos(view(G_ladder, :, 0:last(fft_grid)), kG, mP.U, μ[1], mP.β)
    converged = sc_converged && all(isfinite.([lhs_c1, E_pot_2])) && abs(rhs_c1 - lhs_c1) <= conv_abs && abs(E_pot_1 - E_pot_2) <= conv_abs
    return λ_result(λm, λd, type, sc_converged, converged, E_kin, E_pot_1, E_pot_2, rhs_c1, lhs_c1, 
                    traceDF, G_ladder, Σ_ladder, μ, n)
end


#TODO: docu, :m, :dm, :pre, :fix
function run_sc!(iωn_f::Vector{ComplexF64}, gLoc_rfft::GνqT, G_ladder::OffsetMatrix{ComplexF64}, 
                 Σ_ladder::OffsetMatrix{ComplexF64}, Σ_work::OffsetMatrix{ComplexF64}, Kνωq_pre::Vector{ComplexF64}, trace::Ref,
                 χm::χT, γm::γT, χd::χT, γd::γT, λ₀::Array{ComplexF64,3}, μ::Float64, h::lDΓAHelper;
                 maxit::Int=100, mixing::Float64=0.2, conv_abs::Float64=1e-8, update_χ_tail::Bool=false, fit_μ::Bool=true,
                 par::Bool=false, type::Symbol=:fix)
    it      = 1
    χTail_sc = 1
    done    = false
    χTail_sc_done = false
    converged = false
    fft_νGrid = h.sP.fft_range
    E_pot_1 = Inf
    χ_m_sum = sum_kω(h.kG, χm)
    χ_d_sum = sum_kω(h.kG, χd)
    lhs_c1  = real(χ_d_sum + χ_m_sum)/2
    rhs_c1  = h.mP.n/2*(1-h.mP.n/2)
    E_pot_2 = (h.mP.U/2)*real(χ_d_sum - χ_m_sum) + h.mP.U * (h.mP.n/2 * h.mP.n/2)
    E_kin   = Inf
    νmax    = size(Σ_ladder,2)
    λm      = 0.0
    λd      = 0.0
    Nd = 4


    while !χTail_sc_done
        while !done
            λm,λd,μ,validation = if it == 1 && type in [:pre_dm, :dm, :pre_m, :m]
                λm,λd,validation = χm.λ, χd.λ, true  
                reset!(χm)
                reset!(χd)
                λm,λd,μ,validation
            else
                if type in [:pre_dm, :dm]
                    try
                        res = λdm_correction(χm, γm, χd, γd, h.Σ_loc, gLoc_rfft, h.χloc_m_sum, λ₀, h.kG, h.mP, h.sP;
                                        νmax=νmax, λ_val_only=false, sc_max_it=0, update_χ_tail=false, fit_μ=fit_μ, μ=μ, λinit=[λm,λd],verbose=false)
                        println("DBG: λm=",round(res.λm,digits=Nd),
                                      ", λd=",round(res.λd,digits=Nd), 
                                      ", μ=",round(res.μ,digits=Nd), 
                                      ", n=",round(res.n,digits=Nd), 
                                      " // EPot_p1: ",round(res.EPot_p1,digits=Nd),
                                      " , p2: ",round(res.EPot_p2,digits=Nd),
                                      " :: ", round(res.PP_p1,digits=Nd), 
                                      " // ", round(res.PP_p2,digits=Nd))
                        res.λm, res.λd, res.μ, res.converged   
                    catch e
                        done = true
                        res = λdm_correction(χm, γm, χd, γd, h.Σ_loc, gLoc_rfft, h.χloc_m_sum, λ₀, h.kG, h.mP, h.sP;
                                        νmax=νmax, λ_val_only=false, sc_max_it=0, update_χ_tail=false, fit_μ=false, μ=μ, λinit=[λm,λd],verbose=false)
                        res.λm, res.λd,0.0,false
                    end
                elseif type == [:pre_m, :m]
                    rhs = λm_rhs(χm, χd, h)
                    λm, validation = λm_correction(χm, rhs, h)
                    λm, 0.0, μ, validation
                else
                    λm, λd, μ, true
                end
            end
                       
            if !isfinite(λm) || !isfinite(λd) || !validation
                println("ERROR: (run: β = $(h.mP.β), U = $(h.mP.U), n = $(h.mP.n))\n
    Nν = $(h.sP.n_iν), Nω = $(h.sP.n_iω), Nk = $(h.kG.Ns)\n
    λm = $λm or λd = $λd not finite, OR internal λ correction did not find root!")
                done = true
                break
            end

            copy!(Σ_work, Σ_ladder)
            if par
                calc_Σ_par!(Σ_ladder_inplace, λm=λm, λd=λd)
            else
                (λm != 0) && χ_λ!(χm, λm)
                (λd != 0) && χ_λ!(χd, λd)
                calc_Σ!(Σ_ladder, Kνωq_pre, χm, γm, χd, γd, h.χloc_m_sum, λ₀, gLoc_rfft, h.kG, h.mP, h.sP)
                (λm != 0) && reset!(χm)
                (λd != 0) && reset!(χd)
            end

            mixing != 0 && it > 1 && (Σ_ladder[:,:] = (1-mixing) .* Σ_ladder .+ mixing .* Σ_work)
            μ = G_from_Σladder!(G_ladder, Σ_ladder, h.Σ_loc, h.kG, h.mP; fix_n=fit_μ, μ=μ)
            if isnan(μ) 
                break
            end
            E_pot_1_old = E_pot_1
            E_pot_2_old = E_pot_2
            E_kin, E_pot_1 = calc_E(G_ladder, Σ_ladder, μ, h.kG, h.mP)
            G_rfft!(gLoc_rfft, G_ladder, h.kG, fft_νGrid)


            if type in [:m, :dm, :pre_m, :pre_dm] 
                χ_m_sum = sum_kω(h.kG, χm, λ=λm)
                χ_d_sum = sum_kω(h.kG, χd, λ=λd)
                lhs_c1  = real(χ_d_sum + χ_m_sum)/2
                E_pot_2 = (h.mP.U/2)*real(χ_d_sum - χ_m_sum) + h.mP.U * (h.mP.n/2 * h.mP.n/2)
            end


            if abs(E_pot_1 - E_pot_1_old) < conv_abs && abs(E_pot_2 - E_pot_2_old) < conv_abs
                converged = true
                done = true
            end
            if !isnothing(trace[])
                χ_m_sum2 = sum_ωk(h.kG, χm, λ=λm)
                χ_d_sum2 = sum_ωk(h.kG, χd, λ=λd)
                λm_log = (λm != 0 ? λm : χm.λ)
                λd_log = (λd != 0 ? λd : χd.λ)
                lhs_c1 = real(χ_d_sum + χ_m_sum)/2
                n = filling(G_ladder, h.kG, h.mP.U, μ, h.mP.β)
                row = [it, λm_log, λd_log, μ, n, E_kin, E_pot_1, lhs_c1, E_pot_2, χ_m_sum, χ_m_sum2, χ_d_sum, χ_d_sum2, abs(sum(Σ_ladder)), abs(sum(G_ladder))]
                push!(trace[], row)
            end
            (it >= maxit) && (done = true)
            it += 1
        end

        if update_χ_tail
            println("in χ tail update # $χTail_sc")
            if !isfinite(E_kin)
                E_pot_1 = NaN
                lhs_c1  = NaN
                done    = true
            else
                update_tail!(χm, [0, 0, E_kin], iωn_f)
                update_tail!(χd, [0, 0, E_kin], iωn_f)
            end
            χ_m_sum = sum_kω(h.kG, χm, λ=λm)
            χ_d_sum = sum_kω(h.kG, χd, λ=λd)
            lhs_c1  = real(χ_d_sum + χ_m_sum)/2
            E_pot_2 = (h.mP.U/2)*real(χ_d_sum - χ_m_sum) + h.mP.U * (h.mP.n/2 * h.mP.n/2)

            if χTail_sc >= 20 || abs(E_pot_1 - E_pot_2) < conv_abs && abs(lhs_c1 - h.mP.n/2 * (1 - h.mP.n/2)) < conv_abs
                χTail_sc_done = true
            else
                it = 1
                done = false
            end
        else
            χTail_sc_done = true
        end
        χTail_sc += 1
    end


    n = filling_pos(view(G_ladder, :, 0:νmax-1), h.kG, h.mP.U, μ, h.mP.β)


    update_tail!(χm, [0, 0, h.mP.Ekin_DMFT], iωn_f)
    update_tail!(χd, [0, 0, h.mP.Ekin_DMFT], iωn_f)

    return λm, λd, rhs_c1, lhs_c1, E_pot_1, E_pot_2, E_kin, n, μ, converged
end
