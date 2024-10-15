
# ==================================================================================================== #
#                                        SelfEnergy.jl                                                 #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#   ladder DΓA related functions forthe calculation of the self energy                                 #
# -------------------------------------------- TODO -------------------------------------------------- #
#   Documentation                                                                                      #
# ==================================================================================================== #


# ==================================== Self Energy Tail Correction ===================================
abstract type ΣTail end

"""
    ExpStep{δ} <: ΣTail

Value type for Σ-tail. `δ` is the fade-in parameter. 
    See also [`tail_factor`](@ref tail_factor).
"""
struct ΣTail_ExpStep{δ} <: ΣTail
end

abstract type ΣTail_Full <: ΣTail end

abstract type ΣTail_Plain <: ΣTail end


"""
    tail_factor(U::Float64, β::Float64, n::Float64, Σ_loc::OffsetVector{ComplexF64}, iν::Vector{ComplexF64};
                mode::ΣTail=default_Σ_tail_correction())::Vector{ComplexF64}

Calculates the tail factor for [`tail_correction_term`](@ref tail_correction_term).
"""
function tail_factor(::Type{ΣTail_Plain}, U::Float64, β::Float64, n::Float64, Σ_loc::OffsetVector{ComplexF64}, iν::Vector{ComplexF64};)::Vector{ComplexF64}
    return 0.0 ./ iν
end
function tail_factor(::Type{ΣTail_Full}, U::Float64, β::Float64, n::Float64, Σ_loc::OffsetVector{ComplexF64}, iν::Vector{ComplexF64};)::Vector{ComplexF64}
    return - U^2 ./ iν
end

function tail_factor(::Type{ΣTail_ExpStep{δ}}, U::Float64, β::Float64, n::Float64, Σ_loc::OffsetVector{ComplexF64}, iν::Vector{ComplexF64})::Vector{ComplexF64} where δ

    Σlim = U^2 * (n/2) * (1 - n/2)
    DMFT_dff = -imag(Σ_loc[0:length(iν)-1]) .* imag(iν) .- Σlim
    tc_factor = - U^2 .* exp.(-(DMFT_dff) .^ 2 ./ δ) ./ iν

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
function tail_correction_term(χm_nl::Float64, χm_loc::Float64, tf::Vector{ComplexF64})::Matrix{ComplexF64}
    return reshape((χm_nl - χm_loc) .* tf, 1, length(tf))
end

function tail_correction_term(U::Float64, β::Float64, n::Float64, χm_nl::Float64, χm_loc::Float64, 
                              Σ_loc::OffsetVector{ComplexF64}, iν::Vector{ComplexF64}; 
                              mode::ΣTail=default_Σ_tail_correction())

    tf = tail_factor(mode, U, β, n, Σ_loc, iν)
    return tail_correction_term(χm_nl, χm_loc, tf)
end


# ======================================== Equation of Motion ========================================
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
           νmax=eom_ν_cutoff(h), λm::Float64=0.0, λd::Float64=0.0, tc::ΣTail=default_Σ_tail_correction())
    calc_Σ(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::λ₀T, gLoc_rfft, h; 
           νmax::Int=eom_ν_cutoff(h), λm::Float64 = 0.0, λd::Float64 = 0.0, tc::ΣTail = default_Σ_tail_correction())
    calc_Σ(χm::χT, γm::γT, χd::χT, γd::γT, χ_m_sum::Union{Float64,ComplexF64}, λ₀::λ₀T,
           Gνω::GνqT, kG::KGrid, mP::ModelParameters, sP::SimulationParameters; 
           νmax=eom_ν_cutoff(sP), λm::Float64=0.0, λd::Float64=0.0, tc::ΣTail=default_Σ_tail_correction())
                
Calculates the self-energy from ladder quantities.

This is the single core variant, see [`calc_Σ_par`](@ref calc_Σ_par) for the parallel version.
"""
function calc_Σ(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::λ₀T, h; νmax::Int=eom_ν_cutoff(h), λm::Float64 = 0.0, λd::Float64 = 0.0, tc::Type{<: ΣTail}=default_Σ_tail_correction()) 
    calc_Σ(χm, γm, χd, γd, h.χloc_m_sum, λ₀, h.Σ_loc, h.gLoc_rfft, h.kG, h.mP, h.sP, νmax=νmax, λm = λm, λd = λd, tc = tc)
end

function calc_Σ(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::λ₀T, gLoc_rfft, h; νmax::Int=eom_ν_cutoff(h), λm::Float64 = 0.0, λd::Float64 = 0.0, tc::Type{<: ΣTail}=default_Σ_tail_correction())
    calc_Σ(χm, γm, χd, γd, h.χloc_m_sum, λ₀, h.Σ_loc, gLoc_rfft, h.kG, h.mP, h.sP, νmax=νmax, λm = λm, λd = λd, tc = tc)
end

function calc_Σ(χm::χT,γm::γT,χd::χT,γd::γT, χ_m_sum::Union{Float64,ComplexF64}, λ₀::λ₀T,
                Σ_loc::OffsetVector{ComplexF64}, Gνω::GνqT, kG::KGrid, mP::ModelParameters, sP::SimulationParameters;
                νmax::Int = eom_ν_cutoff(sP), λm::Float64 = 0.0, λd::Float64 = 0.0, tc::Type{<: ΣTail} = default_Σ_tail_correction(),
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
    tc_factor = tail_factor(tc, mP.U, mP.β, mP.n, Σ_loc, iν)
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
                      tc::Type{<: ΣTail}=default_Σ_tail_correction(), λm::Float64 = 0.0, λd::Float64 = 0.0)
    calc_Σ_parts(χm, γm, χd, γd, h.χloc_m_sum, λ₀, h.Σ_loc, h.gLoc_rfft, h.kG, h.mP, h.sP; tc = tc, λm = λm, λd = λd)
end

function calc_Σ_parts(χm::χT, γm::γT, χd::χT, γd::γT, χ_m_sum::Union{Float64,ComplexF64}, λ₀::λ₀T,
                      Σ_loc::OffsetVector{ComplexF64}, Gνω::GνqT, kG::KGrid, mP::ModelParameters, sP::SimulationParameters; 
                      tc::Type{<: ΣTail}=default_Σ_tail_correction(), λm::Float64 = 0.0, λd::Float64 = 0.0)
    Σ_hartree = mP.n * mP.U / 2.0
    Nq = size(χm, χm.axis_types[:q])

    Kνωq_pre::Vector{ComplexF64} = Vector{ComplexF64}(undef, length(kG.kMult))
    Σ_ladder_i = OffsetArray(Matrix{ComplexF64}(undef, Nq, sP.n_iν), 1:Nq, 0:sP.n_iν-1)
    Σ_ladder = OffsetArray(Array{ComplexF64,3}(undef, Nq, sP.n_iν, 7), 1:Nq, 0:sP.n_iν-1, 1:7)

    λm != 0.0 && χ_λ!(χm, λm)
    λd != 0.0 && χ_λ!(χd, λd)

    iν = iν_array(mP.β, collect(axes(Σ_ladder, 2)))
    tc_factor = tail_factor(tc, mP.U, mP.β, mP.n, Σ_loc, iν)
    
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

