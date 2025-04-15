# ==================================================================================================== #
#                                           ResultType.jl                                              #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#   Type and helpers for λ-correction results.                                                         # 
# -------------------------------------------- TODO -------------------------------------------------- #
# ==================================================================================================== #


# ======================================== Correction Type ===========================================
abstract type CorrectionMethod end
abstract type mCorrection <: CorrectionMethod end
abstract type dmCorrection <: CorrectionMethod end
abstract type m_scCorrection <: CorrectionMethod end
abstract type dm_scCorrection <: CorrectionMethod end
abstract type m_tscCorrection <: CorrectionMethod end
abstract type dm_tscCorrection <: CorrectionMethod end

# ========================================== Result Type =============================================
# --------------------------------------------- Setup ------------------------------------------------
# -------------------------------------------- Results -----------------------------------------------
"""
    λ_result

Containes result and auxilliary information of λ correction, is returned by [`λ_correction`](@ref λ_correction), should not be constructed explicitly!

Fields
-------------
- **`λm`**           : `Float64`, Magnetic channel correction parameter.
- **`λd`**           : `Float64`, Density channel correction parameter.
- **`type`**         : `CorrectionMethod`, Type of λ correction: 
    - `:m`, only magnetic channel
    - `:dm`, magnetic and density channel 
    - `:m_sc`, only magnetic channel, partial self-consistency in EoM 
    - `:dm_sc`, magnetic and density channel, partial self-consistency in EoM 
    - `:m_tsc`, only magnetic channel, partial self-consistency in EoM, updated kinetic energy term in susceptibility tail
    - `:dm_tsc`, magnetic and density channel, partial self-consistency in EoM, updated kinetic energy term in susceptibility tail
- **`sc_converged`** : `Bool`, convergence parameter for self-consistency methods. Always `True` for non self-consist methods. See also [`sc_converged`](@ref sc_converged)
- **`eps_abs`**      : `Float64`, Threshold for convergence. Convergence is assumed when the potential energies and Pauli principle values (depending on method both or one) are equal up to this value.
- **`sc_eps_abs`**   : `Float64`, Threshold for sc convergence. Convergence is assumed when the potential energies and Pauli principle values (for 1- and 2-particle quantities) individually change by less than `sc_eps_abs`.
- **`EKin`**         : `Float64`, 2-Particle kinetic energy
- **`EPot_p1`**      : `Float64`, 1-Particle potential energy, ``G^\\mathbf{\\lambda}_\\mathrm{ladder} \\Sigma^\\mathbf{\\lambda}_\\mathrm{ladder}``, see [`calc_E`](@ref calc_E)
- **`EPot_p2`**      : `Float64`, 2-Particle potential energy, ``\\frac{U}{2} \\sum_{\\omega,q} (\\chi^{\\lambda_\\mathrm{d},\\omega}_{\\mathrm{d},q} - \\chi^{\\lambda_\\mathrm{m},\\omega}_{\\mathrm{m},q} + U\\frac{n^2}{2}``
- **`PP_p1`**        : `Float64`, 1-Particle Pauli principle, ``\\frac{n}{2} (1 - \\frac{n}{2})``
- **`PP_p2`**        : `Float64`, 2-Particle Pauli principle, ``\\frac{1}{2} \\sum_{\\omega,q} (\\chi^{\\lambda_\\mathrm{d},\\omega}_{\\mathrm{d},q} + \\chi^{\\lambda_\\mathrm{m},\\omega}_{\\mathrm{m},q}`` 
- **`trace`**        : `DataFrame/Nothing`, intermediate values of `λ_result` (`Σ_ladder` and `G_ladder` are only stored as checksums) for each self-consistency iteration.
- **`G_ladder`**     : `Nothing/OffsetMatrix`, Green's function after covnergence of λ-correction 
- **`Σ_ladder`**     : `Nothing/OffsetMatrix`, self-energy after λ-correction 
- **`μ`**            : `Float64`, chemical potential after λ-correction
- **`n`**            : `Float64`, electron density. This is used as check, μ should have been internally adjustet to keep this value fixed (i.e. `n ≈ n_dmft`)
- **`n_dmft`**       : `Float64`, input electron density
"""
mutable struct λ_result{T<:CorrectionMethod}
    λm::Float64
    λd::Float64
    sc_converged::Bool
    eps_abs::Float64
    sc_eps_abs::Float64
    EKin_p1::Float64
    EKin_p2::Float64
    EPot_p1::Float64
    EPot_p2::Float64
    PP_p1::Float64
    PP_p2::Float64
    trace::Union{Nothing,DataFrame}
    G_ladder::Union{Nothing,OffsetMatrix}
    Σ_ladder::Union{Nothing,OffsetMatrix}
    μ::Float64
    n::Float64
    n_dmft::Float64
    χm_sum::Float64
    χm_loc_sum::Float64 
    #ΣTail_mode

    # TODO: this constructor is a placeholder in case I decide to move additional checks here (e.g. validate_X and include verbose flag)
    function λ_result(λm::Float64,λd::Float64, type::Type{T},
                      sc_converged::Bool,eps_abs::Float64,sc_eps_abs::Float64,
                      EKin_p1::Float64,EKin_p2::Float64,EPot_p1::Float64,EPot_p2::Float64,PP_p1_val::Float64,PP_p2_val::Float64,
                      trace::Union{DataFrame,Nothing},
                      G_ladder::Union{Nothing,OffsetMatrix},Σ_ladder::Union{Nothing,OffsetMatrix},μ::Float64,n::Float64,n_dmft::Float64,
                      χm_sum::Float64, χm_loc_sum::Float64 
    ) where {T<:CorrectionMethod}
        new{T}(λm,λd,sc_converged,eps_abs,sc_eps_abs,EKin_p1,EKin_p2,EPot_p1,EPot_p2,PP_p1_val,PP_p2_val,trace,
                G_ladder,Σ_ladder,μ,n,n_dmft,χm_sum,χm_loc_sum
        )
    end
end

"""
    λ_result(χm::χT,γm::γT,χd::χT, γd::γT, λ₀::Array{ComplexF64,3}, λm, λd, h; 
                  validation_threshold::Float64 = 1e-8, max_steps_m::Int = 2000)
    λ_result(χm::χT,γm::γT,χd::χT, γd::γT, λ₀::Array{ComplexF64,3}, μ_new, G_ladder, Σ_ladder, λm, λd, h; 
                  validation_threshold::Float64 = 1e-8, max_steps_m::Int = 2000)
                  
Constructs λ_result object, runs all checks and stores them.
"""
function λ_result(type, χm::χT,γm::γT,χd::χT, γd::γT, λ₀::Array{ComplexF64,3}, 
                  λm::Float64, λd::Float64, sc_converged::Bool, h::RunHelper; 
                  νmax::Int = eom_ν_cutoff(h), tc::Type{<: ΣTail}=default_Σ_tail_correction(), PP_p1_val = h.mP.n * (1 - h.mP.n / 2) / 2,
                  validation_threshold::Float64 = 1e-8, max_steps_m::Int = 2000, fix_n::Bool=true)
    if isfinite(λm) && isfinite(λd)
        μ_new, G_ladder, Σ_ladder = calc_G_Σ(χm, γm, χd, γd, λ₀, λm, λd, h; tc = tc, fix_n = fix_n)
        λ_result(type, χm, χd, μ_new, G_ladder, Σ_ladder, λm, λd, sc_converged, h; PP_p1_val = PP_p1_val, validation_threshold = validation_threshold, max_steps_m = max_steps_m)
    else
        NaN, nothing, nothing
        λ_result(λm, λd, type, sc_converged, validation_threshold, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 
                    nothing, nothing, nothing, NaN, NaN, h.mP.n, NaN, h.χloc_m_sum)
    end
end

function λ_result(type, χm::χT, χd::χT, μ_new::Float64, G_ladder, Σ_ladder, 
                  λm::Float64, λd::Float64, sc_converged::Bool, h::RunHelper; 
                  PP_p1_val = h.mP.n * (1 - h.mP.n / 2) / 2, νFitRange=0:last(axes(Σ_ladder, 2)),
                  validation_threshold::Float64 = 1e-8, max_steps_m::Int = 2000)

    Ekin_p1, Epot_p1 = calc_E(G_ladder, Σ_ladder, μ_new, h.kG, h.mP)
    if type === m_tscCorrection || type === dm_tscCorrection
        error("$type not implemented")
    end
    χm_sum = sum_kω(h.kG, χm, λ = λm)
    χd_sum = sum_kω(h.kG, χd, λ = λd)
    PP_p1_val  = PP_p1_val
    PP_p2_val  = real(χd_sum + χm_sum) / 2
    Ekin_p2 = χm.tail_c[3]
    Epot_p2 = EPot_p2(χm, χd, λm, λd, h.mP.n, h.mP.U, h.kG)
    ndens = filling_pos(G_ladder[:,νFitRange], h.kG, h.mP.U, μ_new, h.mP.β, improved_sum=true)
    return λ_result(λm, λd, type, sc_converged, validation_threshold, NaN, Ekin_p1, Ekin_p2, Epot_p1, Epot_p2, PP_p1_val, PP_p2_val, 
                    nothing, G_ladder, Σ_ladder, μ_new, ndens, h.mP.n, χm_sum, h.χloc_m_sum)
end

# ---------------------------------------- IO and Helpers --------------------------------------------
"""
    EPot_diff(result::λ_result)

Difference between potential energies on one- and two particle level (may be negative if `EPot_p2 > EPot_p1`).
"""
EPot_diff(result::λ_result) = result.EPot_p1 - result.EPot_p2

"""
    PP_diff(result::λ_result)

Difference between Pauli principle on one- and two particle level (may be negative if `PP_p2 > PP_p1`).
"""
PP_diff(result::λ_result) = result.PP_p1 - result.PP_p2

"""
    n_diff(result::λ_result)

Difference between density before and after λ-correction (this should always be close to 0!) 
"""
n_diff(result::λ_result) = result.n_dmft - result.n

"""
    tail_diff(result::λ_result)

Difference ``\\sum_{\\omega,q}\\chi^{\\lambda,\\omega}_{m,q}`` and ``\\sum_{\\omega}\\chi^{\\lambda,\\omega}_{m, DMFT}``
"""
tail_diff(result::λ_result) = result.χm_sum - result.χm_loc_sum 


"""
    converged(r::λ_result, eps_abs::Float64=1e-6)

Checks convergences for appropriate parameters of method.
"""
function converged(r::λ_result{T}) where {T}
    if T in [mCorrection, m_scCorrection, m_tscCorrection]
        abs(PP_diff(r)) <= r.eps_abs
    elseif T in [dmCorrection, dm_scCorrection, dm_tscCorrection]
        abs(PP_diff(r)) <= r.eps_abs && abs(EPot_diff(r)) <= r.eps_abs
    else
        error("Unrecognized λ-correction type!")
    end
end

"""
    validation(r::λ_result)

Returns `Tuple` with check for (density, Pauli-principle, potential energy), both checked between one- and two-particle level against `λ_result.eps_abs`.
"""
function validate(r::λ_result)
    n_check    = abs(n_diff(r))    < r.eps_abs
    PP_check   = abs(PP_diff(r))   < r.eps_abs
    EPot_check = abs(EPot_diff(r)) < r.eps_abs
    n_check, PP_check, EPot_check
end

"""
    sc_converged(r::λ_result)

Checks for self-consistency convergence. Returns `true` if method does not invlove a self-consistency loop.
"""
sc_converged(r::λ_result) = r.sc_converged

function Base.show(io::IO, m::λ_result{T}) where {T}
    width = 80
    compact = get(io, :compact, false)
    cc = converged(m) ? @green("converged") : @red("NOT converged")
    if !compact
        rdiff_n = 100 * abs(m.n -  m.n_dmft)/abs(m.n + m.n_dmft)
        rdiff_pp = 100 * abs(m.PP_p1 -  m.PP_p2)/abs(m.PP_p1 + m.PP_p2)
        rdiff_Ep = 100 * abs(m.EPot_p1 - m.EPot_p2)/abs(m.EPot_p1 + m.EPot_p2)
        rdiff_Ek = 100 * abs(m.EKin_p1 - m.EKin_p2)/abs(m.EKin_p1 + m.EKin_p2)
        rdiff_tail = 100 * abs(m.χm_sum - m.χm_loc_sum)/abs(m.χm_sum + m.χm_loc_sum)

        rdiff_n_s = abs(m.n -  m.n_dmft) < m.eps_abs ? @green(@sprintf("Δ = %2.4f%%",rdiff_n)) : @red(@sprintf("Δ = %2.4f%%",rdiff_n)) 
        rdiff_pp_s = abs(m.PP_p1 -  m.PP_p2) < m.eps_abs ? @green(@sprintf("Δ = %2.4f%%",rdiff_pp)) : @red(@sprintf("Δ = %2.4f%%",rdiff_pp)) 
        rdiff_Ep_s = abs(m.EPot_p1 - m.EPot_p2) < m.eps_abs ? @green(@sprintf("Δ = %2.4f%%",rdiff_Ep)) : @red(@sprintf("Δ = %2.4f%%",rdiff_Ep))
        rdiff_Ek_s = abs(m.EKin_p1 - m.EKin_p2) < m.eps_abs ? @green(@sprintf("Δ = %2.4f%%",rdiff_Ek)) : @red(@sprintf("Δ = %2.4f%%",rdiff_Ek))
        rdiff_tail_s = abs(m.χm_sum - m.χm_loc_sum) < m.eps_abs ? @green(@sprintf("Δ = %2.4f%%",rdiff_tail)) : @red(@sprintf("Δ = %2.4f%%",rdiff_tail))
        tprint(Panel(
            @sprintf("λm = %3.8f, λd = %3.8f, μ = %3.8f\n", m.λm, m.λd, m.μ) * 
            @sprintf("n      =  %3.8f,  n DMFT  =  %3.8f,  %s\n", m.n,  m.n_dmft, rdiff_n_s) * 
            @sprintf("PP_1   =  %3.8f,  PP_2    =  %3.8f,  %s\n", m.PP_p1, m.PP_p2, rdiff_pp_s) *
            @sprintf("Epot_1 =  %3.8f,  Epot_2  =  %3.8f,  %s\n", m.EPot_p1, m.EPot_p2, rdiff_Ep_s) *
            @sprintf("Ekin_1 =  %3.8f,  Ekin_2  = %3.8f,  %s\n", m.EKin_p1 , m.EKin_p2, rdiff_Ek_s) *
            @sprintf("χ_m    =  %3.8f,  χ_m_loc = %3.8f,  %s\n", m.χm_sum , m.χm_loc_sum, rdiff_tail_s),
            title="λ-correction (type: $(T)), $cc"
        ))
        !isnothing(m.trace) && println(io, "trace: \n", m.trace)
    else
        print(io, "λ-correction (type: $cc) result, λm = $(m.λm), λd = $(m.λd), μ = $(m.μ) // $converged")
    end
end
