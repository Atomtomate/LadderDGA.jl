# ==================================================================================================== #
#                                        ladderDGATools.jl                                             #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe, Jan Frederik Weißler                                              #
# ----------------------------------------- Description ---------------------------------------------- #
#   ladder DΓA related functions                                                                       #
# -------------------------------------------- TODO -------------------------------------------------- #
#   Cleanup, complete refactor??                                                                       #
# ==================================================================================================== #



# ========================================== Correction Term =========================================

"""
    Σ_hartree(mP::ModelParameters)

Calculates the hartree term of the self energy

`\\Sigma_{hartree}=\\frac{U\\cdot n}{2}`.

Arguments
-------------
- **`mP`** : ModelParameters

Returns
-------
Float64 : Hartree term 
"""
function Σ_hartree(mP::ModelParameters)
    return mP.n * mP.U/2.0
end

"""
    correction_term(mP::ModelParameters, kG::KGrid, χm::χT, χ_m_sum::Union{Float64,ComplexF64}, grid::AbstractArray{Int64,1})

Calculates the so called tail correcion term of the ladder self energy. The purpose of this term is to enforce the limit

`\\lim_{n\\rightarrow\\infty}i\\nu_n\\Sigma_{\\mathbf{q}}^{\\nu_n}=U^2\\frac{n}{2}\\left(1-\\frac{n}{2} \\right )`.

This can be archived by adding the term
    * RPA: `-\\frac{U^2}{i\\nu}\\sum_{\\omega,\\mathbf{q}}\\left( \\chi_{m,\\mathbf{q}}^{\\omega}-\\chi_{0,\\mathbf{q}}^{\\omega}\\right )`
    * ladder-DGA: `-\\frac{U^2}{i\\nu}\\left(\\sum_{\\omega,\\mathbf{q}}\\chi_{m,\\mathbf{q}}^{\\omega}-\\chi_{m,loc} \\right )`
from the ladder self energy.

Arguments
-------------
- **`mP`**         : ModelParameters
- **`kG`**         : KGrid
- **`χm`**         : χT
- **`χ_m_sum`**    : Union{Float64,ComplexF64}. RPA: `\\sum_{\\omega,\\mathbf{q}}\\chi_{0,\\mathbf{q}}^{\\omega}`, lDGA: 'χ_m_sum'.
- **`grid`**       : AbstractArray{Int64,1}
"""
#function correction_term(mP::ModelParameters, kG::KGrid, χm::χT, χ_m_sum::Union{Float64,ComplexF64}, grid::AbstractArray{Int64,1})
#    return - (mP.U) .* (sum_kω(kG, χm) - χ_m_sum) ./ iν_array(mP.β, collect(grid))
#end

"""
    calc_λ0(χ₀::χ₀T, h::lDΓAHelper)
    calc_λ0(χ₀::χ₀T, Fr::FT, h::lDΓAHelper)
    calc_λ0(χ₀::χ₀T, Fr::FT, χ::χT, γ::γT, mP::ModelParameters, sP::SimulationParameters)

Correction term, TODO: documentation
"""
function calc_λ0(χ₀::χ₀T, h::lDΓAHelper)
    F_m = F_from_χ(:m, h)
    calc_λ0(χ₀, F_m, h)
end

function calc_λ0(χ₀::χ₀T, Fr::FT, h::lDΓAHelper)::λ₀T
    calc_λ0(χ₀, Fr, h.χ_m_loc, h.γ_m_loc, h.mP, h.sP)
end

function calc_λ0(χ₀::χ₀T, Fr::FT, χ::χT, γ::γT, mP::ModelParameters, sP::SimulationParameters; improved_sums::Bool = true)::λ₀T
    #TODO: store nu grid in sP?
    Niν = size(Fr, 1)
    Nq = size(χ₀.data, χ₀.axis_types[:q])
    ω_range = 1:size(χ₀.data, χ₀.axis_types[:ω])
    λ0 = λ₀T(undef, size(χ₀.data, χ₀.axis_types[:q]), Niν, length(ω_range))

    if improved_sums && typeof(sP.χ_helper) <: BSE_Asym_Helpers
        #TODO: decide what to do about the warning for ignoring the diagonal part
        @suppress begin
            λ0[:] = calc_λ0_impr(:m, -sP.n_iω:sP.n_iω, Fr, χ₀.data, χ₀.asym, view(γ.data, 1, :, :), view(χ.data, 1, :), mP.U, mP.β, sP.χ_helper)
        end
    else
        #TODO: this is not well optimized, but also not often executed
        @warn "Using plain summation for λ₀, check Σ_ladder tails!"
        fill!(λ0, 0.0)
        for ωi in ω_range
            for νi = 1:Niν
                #TODO: export realview functions?
                v1 = view(Fr, νi, :, ωi)
                for qi = 1:Nq
                    v2 = view(χ₀.data, qi, (sP.n_iν_shell+1):(size(χ₀.data, 2)-sP.n_iν_shell), ωi)
                    λ0[qi, :, ωi] = λ0[qi, :, ωi] .+ v1 .* v2 ./ mP.β^2
                end
            end
        end
    end
    return λ0
end

function calc_Σ_ω!(eomf::Function, Σ_ω::OffsetMatrix{ComplexF64}, Kνωq_pre::Vector{ComplexF64},
                   χm::χT, γm::γT, χd::χT, γd::γT, λ₀::λ₀T, 
                   Gνω::GνqT, U::Float64, kG::KGrid, sP::SimulationParameters,
)
    νdim = ndims(Gνω) > 2 ? length(gridshape(kG)) + 1 : 2 # TODO: this is a fallback for gIm
    fill!(Σ_ω, zero(ComplexF64))
    ω_axis = χm.indices_ω
    for (ωi, ωn) in enumerate(ω_axis)
        νZero = ν0Index_of_ωIndex(ωi, sP)
        νlist = νZero:(sP.n_iν*2)
        length(νlist) > size(Σ_ω, 2) && (νlist = νlist[1:size(Σ_ω, 2)])
        for (νii, νi) in enumerate(νlist)
            for qi = axes(Σ_ω,1)
                @inbounds Kνωq_pre[qi] = eomf(U, γm[qi, νi, ωi], γd[qi, νi, ωi], χm[qi, ωi], χd[qi, ωi], λ₀[qi, νi, ωi])
            end
            #TODO: find a way to not unroll this!
            conv_tmp_add!(view(Σ_ω, :, νii - 1), kG, Kνωq_pre, selectdim(Gνω, νdim, (νii - 1) + ωn))
        end
    end
end


function calc_Σ!(Σ_ladder::OffsetMatrix{ComplexF64}, Kνωq_pre::Vector{ComplexF64},
                 χm::χT, γm::γT, χd::χT, γd::γT, λ₀::λ₀T,
                 tc_term::Union{Float64,Matrix{ComplexF64}}, Gνω::GνqT, kG::KGrid, mP::ModelParameters, sP::SimulationParameters;
)::Nothing
    ΣH = Σ_hartree(mP)
    calc_Σ_ω!(eom, Σ_ladder, Kνωq_pre, χm, γm, χd, γd, λ₀, Gνω, mP.U, kG, sP)
    Σ_ladder.parent[:, :] = Σ_ladder.parent[:, :] ./ mP.β .+ tc_term .+ ΣH 
    return nothing
end


"""
    calc_Σ(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::λ₀T, h::lDΓAHelper;
           νmax=eom_ν_cutoff(h), λm::Float64=0.0, λd::Float64=0.0, tc::Symbol=default_Σ_tail_correction())
    calc_Σ(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::λ₀T, gLoc_rfft, h; 
           νmax::Int=eom_ν_cutoff(h), λm::Float64 = 0.0, λd::Float64 = 0.0, tc::Symbol = default_Σ_tail_correction())
    calc_Σ(χm::χT, γm::γT, χd::χT, γd::γT, χ_m_sum::Union{Float64,ComplexF64}, λ₀::λ₀T,
           Gνω::GνqT, kG::KGrid, mP::ModelParameters, sP::SimulationParameters; 
           νmax=eom_ν_cutoff(sP), λm::Float64=0.0, λd::Float64=0.0, tc::Symbol=default_Σ_tail_correction())
                
Calculates the self-energy from ladder quantities.

This is the single core variant, see [`calc_Σ_par`](@ref calc_Σ_par) for the parallel version.
"""
function calc_Σ(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::λ₀T, h; νmax::Int=eom_ν_cutoff(h), λm::Float64 = 0.0, λd::Float64 = 0.0, tc::Symbol = default_Σ_tail_correction())
    calc_Σ(χm, γm, χd, γd, h.χloc_m_sum, λ₀, h.Σ_loc, h.gLoc_rfft, h.kG, h.mP, h.sP, νmax=νmax, λm = λm, λd = λd, tc = tc)
end

function calc_Σ(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::λ₀T, gLoc_rfft, h; νmax::Int=eom_ν_cutoff(h), λm::Float64 = 0.0, λd::Float64 = 0.0, tc::Symbol = default_Σ_tail_correction())
    calc_Σ(χm, γm, χd, γd, h.χloc_m_sum, λ₀, h.Σ_loc, gLoc_rfft, h.kG, h.mP, h.sP, νmax=νmax, λm = λm, λd = λd, tc = tc)
end

function calc_Σ(χm::χT,γm::γT,χd::χT,γd::γT, χ_m_sum::Union{Float64,ComplexF64}, λ₀::λ₀T,
                Σ_loc::OffsetVector{ComplexF64}, Gνω::GνqT, kG::KGrid, mP::ModelParameters, sP::SimulationParameters;
                νmax::Int = eom_ν_cutoff(sP), λm::Float64 = 0.0, λd::Float64 = 0.0, tc::Symbol = default_Σ_tail_correction(),
)
    χm.λ != 0 && λm != 0 && error("Stopping self energy calculation: λm = $λm AND χm.λ = $(χm.λ)")
    χd.λ != 0 && λd != 0 && error("Stopping self energy calculation: λd = $λd AND χd.λ = $(χd.λ)")
    Nq, Nω = size(χm)
    ωrange::UnitRange{Int} = -sP.n_iω:sP.n_iω

    Kνωq_pre::Vector{ComplexF64} = Vector{ComplexF64}(undef, Nq)
    Σ_ladder = OffsetArray(Matrix{ComplexF64}(undef, Nq, νmax), 1:Nq, 0:νmax-1)

    λm != 0.0 && χ_λ!(χm, λm)
    λd != 0.0 && χ_λ!(χd, λd)

    iν = iν_array(mP.β, collect(axes(Σ_ladder, 2)))
    tc_factor = tail_factor(mP.U, mP.β, mP.n, Σ_loc, iν; mode=tc)
    tc_term  = tail_correction_term(sum_kω(kG, χm), χ_m_sum, tc_factor)
    calc_Σ!(Σ_ladder, Kνωq_pre, χm, γm, χd, γd, λ₀, tc_term, Gνω, kG, mP, sP)

    λm != 0.0 && reset!(χm)
    λd != 0.0 && reset!(χd)
    return Σ_ladder
end


"""
    calc_Σ_parts(χm::χT,γm::γT,χd::χT,γd::γT,h::lDΓAHelper,λ₀::AbstractArray{ComplexF64,3};λm::Float64=0.0, λd::Float64=0.0)
    calc_Σ_parts(χm::χT,γm::γT, χd::χT, γd::γT, χ_m_sum::Union{Float64,ComplexF64}, λ₀::λ₀T,
                 Gνω::GνqT, kG::KGrid,mP::ModelParameters, sP::SimulationParameters;
                 λm::Float64=0.0, λd::Float64=0.0)

Calculates the ``lD\\GammaA`` self-energy (see also [`calc_Σ`](@ref calc_Σ)),
but split into `7` contributions from: `χm`, `γm`, `χd`, `γd`, `U`, `Fm` + `Σ_hartree`, `tail_correction`.

"""
function calc_Σ_parts(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::λ₀T, h::lDΓAHelper; 
                      tc::Symbol = default_Σ_tail_correction(), λm::Float64 = 0.0, λd::Float64 = 0.0)
    calc_Σ_parts(χm, γm, χd, γd, h.χloc_m_sum, λ₀, h.Σ_loc, h.gLoc_rfft, h.kG, h.mP, h.sP; tc = tc, λm = λm, λd = λd)
end

function calc_Σ_parts(χm::χT, γm::γT, χd::χT, γd::γT, χ_m_sum::Union{Float64,ComplexF64}, λ₀::λ₀T,
                      Σ_loc::OffsetVector{ComplexF64}, Gνω::GνqT, kG::KGrid, mP::ModelParameters, sP::SimulationParameters; 
                      tc::Symbol = default_Σ_tail_correction(), λm::Float64 = 0.0, λd::Float64 = 0.0)
    Σ_hartree = mP.n * mP.U / 2.0
    Nq = size(χm, χm.axis_types[:q])

    Kνωq_pre::Vector{ComplexF64} = Vector{ComplexF64}(undef, length(kG.kMult))
    Σ_ladder_i = OffsetArray(Matrix{ComplexF64}(undef, Nq, sP.n_iν), 1:Nq, 0:sP.n_iν-1)
    Σ_ladder = OffsetArray(Array{ComplexF64,3}(undef, Nq, sP.n_iν, 7), 1:Nq, 0:sP.n_iν-1, 1:7)

    λm != 0.0 && χ_λ!(χm, λm)
    λd != 0.0 && χ_λ!(χd, λd)

    iν = iν_array(mP.β, collect(axes(Σ_ladder, 2)))
    tc_factor = tail_factor(mP.U, mP.β, mP.n, Σ_loc, iν; mode=tc)
    
    tc_term  = tail_correction_term(sum_kω(kG, χm), χ_m_sum, tc_factor)
    calc_Σ_ω!(eom_χ_m, Σ_ladder_i, Kνωq_pre, χm, γm, χd, γd, λ₀, Gνω, mP.U, kG, sP)
    Σ_ladder.parent[:, :, 1] = Σ_ladder_i ./ mP.β
    calc_Σ_ω!(eom_γ_m, Σ_ladder_i, Kνωq_pre, χm, γm, χd, γd, λ₀, Gνω, mP.U, kG, sP)
    Σ_ladder.parent[:, :, 2] = Σ_ladder_i ./ mP.β
    calc_Σ_ω!(eom_χ_d, Σ_ladder_i, Kνωq_pre, χm, γm, χd, γd, λ₀, Gνω, mP.U, kG, sP)
    Σ_ladder.parent[:, :, 3] = Σ_ladder_i ./ mP.β
    calc_Σ_ω!(eom_γ_d, Σ_ladder_i, Kνωq_pre, χm, γm, χd, γd, λ₀, Gνω, mP.U, kG, sP)
    Σ_ladder.parent[:, :, 4] = Σ_ladder_i ./ mP.β
    calc_Σ_ω!(eom_rest_01, Σ_ladder_i, Kνωq_pre, χm, γm, χd, γd, λ₀, Gνω, mP.U, kG, sP)
    Σ_ladder.parent[:, :, 5] = Σ_ladder_i ./ mP.β
    calc_Σ_ω!(eom_rest, Σ_ladder_i, Kνωq_pre, χm, γm, χd, γd, λ₀, Gνω, mP.U, kG, sP)
    Σ_ladder.parent[:, :, 6] = Σ_ladder_i ./ mP.β .+ Σ_hartree
    for qi in axes(Σ_ladder, 1)
        Σ_ladder.parent[qi, :, 7] .= tc_term[1,:]
    end
    λm != 0.0 && reset!(χm)
    λd != 0.0 && reset!(χd)

    return Σ_ladder
end


# ==================================== Self energy Tail Correction ===================================
"""
    tail_factor(U::Float64, β::Float64, n::Float64, Σ_loc::OffsetVector{ComplexF64}, iν::Vector{ComplexF64}; 
                              δ::Real=min(0.01, 1 ./ length(iν)))

Calculates the tail factor for [`tail_correction_term`](@ref tail_correction_term).
"""
function tail_factor(U::Float64, β::Float64, n::Float64, Σ_loc::OffsetVector{ComplexF64}, iν::Vector{ComplexF64};
                mode::Symbol=default_Σ_tail_correction(),  δ::Real = 0.05)::Vector{ComplexF64}
    tc_factor = if mode == :full
        - U^2 ./ iν
    elseif mode == :exp_step
        Σlim = U^2 * (n/2) * (1 - n/2)
        DMFT_dff = -imag(Σ_loc[0:length(iν)-1]) .* imag(iν) .- Σlim
        - U^2 .* exp.(-(DMFT_dff) .^ 2 ./ δ) ./ iν
    elseif mode == :plain
        0.0 ./ iν
    else
        @error "Tail factor " mode "not implemented!"
    end
    return tc_factor
end

"""
    tail_correction_term(χm_nl::Float64, χm_loc::Float64, tail_factor::Vector{ComplexF64})

    tail_correction_term(U::Float64, β::Float64, n::Float64, χm_nl::Float64, χm_loc::Float64,
                              Σ_loc::OffsetVector{ComplexF64}, iν::Vector{ComplexF64}; 
                              δ::Real=10.0*length(iν))

Calculates correction term for high frequency behavior of self energy.

``w_\\nu = e^{(-\\Delta^2_{\\nu}/\\delta)}`` with ``\\Delta_{\\nu} = \\nu \\cdot \\Sigma^\\nu_\\mathrm{DMFT} - U^2 \\frac{n}{2} (1 - \\frac{n}{2})``.
See also [`tail_factor`](@ref tail_factor).
"""
function tail_correction_term(χm_nl::Float64, χm_loc::Float64, tail_factor::Vector{ComplexF64})::Matrix{ComplexF64}
    return reshape((χm_nl - χm_loc) .* tail_factor, 1, length(tail_factor))
end

function tail_correction_term(U::Float64, β::Float64, n::Float64, χm_nl::Float64, χm_loc::Float64, 
                              Σ_loc::OffsetVector{ComplexF64}, iν::Vector{ComplexF64}; 
                              δ::Real = min(0.001, 1 ./ 10.0 * length(iν)))

    tf = tail_factor(U, β, n, Σ_loc, iν, δ = δ)
    return tail_correction_term(χm_nl, χm_loc, tf)
end
