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
- **`n`**            : `Float64`, electron density. This is used as check, μ should have been internally adjustet to keep this value fixed  
"""
mutable struct λ_result{T<:CorrectionMethod}
    λm::Float64
    λd::Float64
    sc_converged::Bool
    eps_abs::Float64
    sc_eps_abs::Float64
    EKin::Float64
    EPot_p1::Float64
    EPot_p2::Float64
    PP_p1::Float64
    PP_p2::Float64
    trace::Union{Nothing,DataFrame}
    G_ladder::Union{Nothing,OffsetMatrix}
    Σ_ladder::Union{Nothing,OffsetMatrix}
    μ::Float64
    n::Float64

    function λ_result(
        λm::Float64,
        λd::Float64,
        type::Type{T},
        sc_converged::Bool,
        eps_abs::Float64,
        sc_eps_abs::Float64,
        EKin::Float64,
        EPot_p1::Float64,
        EPot_p2::Float64,
        PP_p1::Float64,
        PP_p2::Float64,
        trace::Union{DataFrame,Nothing},
        G_ladder::Union{Nothing,OffsetMatrix},
        Σ_ladder::Union{Nothing,OffsetMatrix},
        μ::Float64,
        n::Float64,
    ) where {T<:CorrectionMethod}
        new{T}(
            λm,
            λd,
            sc_converged,
            eps_abs,
            sc_eps_abs,
            EKin,
            EPot_p1,
            EPot_p2,
            PP_p1,
            PP_p2,
            trace,
            G_ladder,
            Σ_ladder,
            μ,
            n,
        )
    end
end

# ---------------------------------------- IO and Helpers --------------------------------------------
"""
    EPot_diff(result::λ_result)

Difference between potential energies on one- and two particle level.
"""
EPot_diff(result::λ_result) = result.EPot_p1 - result.EPot_p2

"""
    PP_diff(result::λ_result)

Difference between Pauli principle on one- and two particle level.
"""
PP_diff(result::λ_result) = result.PP_p1 - result.EPot_p2

"""
    converged(r::λ_result, eps_abs::Float64=1e-6)

Checks convergences for appropriate parameters of method.
"""
function converged(r::λ_result{T}, eps_abs::Float64 = 1e-6) where {T}
    if T in [mCorrection, m_scCorrection, m_tscCorrection]
        abs(PP_diff(r)) <= r.eps_abs
    elseif T in [dmCorrection, dm_scCorrection, dm_tscCorrection]
        abs(PP_diff(r)) <= r.eps_abs && abs(EPot_diff(r)) <= r.eps_abs
    else
        error("Unrecognized λ-correction type!")
    end
end

"""
    sc_converged(r::λ_result)

Checks for self-consistency convergence. Returns `true` if method does not invlove a self-consistency loop.
"""
sc_converged(r::λ_result) = r.sc_converged

function Base.show(io::IO, m::λ_result{T}) where {T}
    compact = get(io, :compact, false)
    cc = converged(m) ? "converged" : "NOT converged"
    if !compact
        println(io, "λ-correction (type: $(T)), $cc")
        println(io, "λm = $(m.λm), λd = $(m.λd), μ = $(m.μ)")
        println(io, "Epot_1 = $(m.EPot_p1), Epot_2 = $(m.EPot_p2)")
        !isnothing(m.trace) && println(io, "trace: \n", m.trace)
    else
        print(io, "λ-correction (type: $type) result, λm = $(m.λm), λd = $(m.λd), μ = $(m.μ) // $converged")
    end
end
