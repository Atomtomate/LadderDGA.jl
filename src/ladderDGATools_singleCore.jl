# ==================================================================================================== #
#                                        ladderDGATools.jl                                             #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#   ladder DΓA related functions                                                                       #
# -------------------------------------------- TODO -------------------------------------------------- #
#   Cleanup, complete refactor??                                                                       #
# ==================================================================================================== #



# ========================================== Correction Term =========================================
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

function calc_λ0(χ₀::χ₀T, Fr::FT, h::lDΓAHelper)
    calc_λ0(χ₀, Fr, h.χ_m_loc, h.γ_m_loc, h.mP, h.sP)
end

function calc_λ0(χ₀::χ₀T, Fr::FT, χ::χT, γ::γT, mP::ModelParameters, sP::SimulationParameters; improved_sums::Bool = true)
    #TODO: store nu grid in sP?
    Niν = size(Fr, 1)
    Nq = size(χ₀.data, χ₀.axis_types[:q])
    ω_range = 1:size(χ₀.data, χ₀.axis_types[:ω])
    λ0 = Array{ComplexF64,3}(undef, size(χ₀.data, χ₀.axis_types[:q]), Niν, length(ω_range))

    if improved_sums && typeof(sP.χ_helper) <: BSE_Asym_Helpers
        λ0[:] = calc_λ0_impr(:m, -sP.n_iω:sP.n_iω, Fr, χ₀.data, χ₀.asym, view(γ.data, 1, :, :), view(χ.data, 1, :), mP.U, mP.β, sP.χ_helper)
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

#TODO: THIS NEEDS CLEANUP!
function conv_tmp!(res::AbstractVector{ComplexF64}, kG::KGrid, arr1::Vector{ComplexF64}, GView::AbstractArray{ComplexF64,N})::Nothing where {N}
    if Nk(kG) == 1
        res[:] = arr1 .* GView
    else
        expandKArr!(kG, kG.cache1, arr1)
        mul!(kG.cache1, kG.fftw_plan, kG.cache1)
        for i in eachindex(kG.cache1)
            kG.cache1[i] *= GView[i]
        end
        kG.fftw_plan \ kG.cache1
        Dispersions.conv_post!(kG, res, kG.cache1)
    end
    return nothing
end

function calc_Σ_ω!(eomf::Function, Σ_ω::OffsetArray{ComplexF64,3}, Kνωq_pre::Vector{ComplexF64}, χm::χT, γm::γT, χd::χT, γd::γT, 
                  Gνω::GνqT, λ₀::AbstractArray{ComplexF64,3}, U::Float64, kG::KGrid, sP::SimulationParameters)

    νdim = ndims(Gνω) > 2 ? length(gridshape(kG)) + 1 : 2 # TODO: this is a fallback for gIm
    fill!(Σ_ω, zero(ComplexF64))
    for (ωi, ωn) in enumerate(axes(Σ_ω, 3))
        νZero = ν0Index_of_ωIndex(ωi, sP)
        νlist = νZero:(sP.n_iν*2)
        length(νlist) > size(Σ_ω, 2) && (νlist = νlist[1:size(Σ_ω, 2)])
        for (νii, νi) in enumerate(νlist)
            for qi = 1:size(Σ_ω, 1)
                Kνωq_pre[qi] = eomf(U, γm[qi, νi, ωi], γd[qi, νi, ωi], χm[qi, ωi], χd[qi, ωi], λ₀[qi, νi, ωi])
            end
            #TODO: find a way to not unroll this!
            conv_tmp!(view(Σ_ω, :, νii - 1, ωn), kG, Kνωq_pre, selectdim(Gνω, νdim, (νii - 1) + ωn))
        end
    end
end

function calc_Σ_ω!(eomf::Function, Σ_ladder::OffsetMatrix{ComplexF64}, Kνωq_pre::Vector{ComplexF64}, 
                   χm::χT, γm::γT, χd::χT, γd::γT, Gνω::GνqT, λ₀::AbstractArray{ComplexF64,3}, U::Float64,
                   kG::KGrid, sP::SimulationParameters)
    νdim = ndims(Gνω) > 2 ? length(gridshape(kG)) + 1 : 2 # TODO: this is a fallback for gIm
    fill!(Σ_ladder, zero(ComplexF64))
    ω_axis = χm.indices_ω
    for (ωi, ωn) in enumerate(ω_axis)
        νZero = ν0Index_of_ωIndex(ωi, sP)
        νlist = νZero:(sP.n_iν*2)
        length(νlist) > size(Σ_ladder, 2) && (νlist = νlist[1:size(Σ_ladder, 2)])
        for (νii, νi) in enumerate(νlist)
            for qi = 1:size(Σ_ladder, 1)
                Kνωq_pre[qi] = eomf(U, γm[qi, νi, ωi], γd[qi, νi, ωi], χm[qi, ωi], χd[qi, ωi], λ₀[qi, νi, ωi])
            end
            #TODO: find a way to not unroll this!
            conv_tmp_add!(view(Σ_ladder, :, νii - 1), kG, Kνωq_pre, selectdim(Gνω, νdim, (νii - 1) + ωn))
        end
    end
end

function calc_Σ!(Σ_ladder::OffsetMatrix{ComplexF64}, Kνωq_pre::Vector{ComplexF64}, χm::χT, γm::γT, χd::χT, γd::γT, 
                 χ_m_sum::Union{Float64,ComplexF64}, λ₀::AbstractArray{_eltype,3}, tc_factor::Vector, Gνω::GνqT, 
                 kG::KGrid, mP::ModelParameters, sP::SimulationParameters; tc::Bool = true)::Nothing
    Σ_hartree = mP.n * mP.U / 2.0
    calc_Σ_ω!(eom, Σ_ladder, Kνωq_pre, χm, γm, χd, γd, Gνω, λ₀, mP.U, kG, sP)
    tail_correction = (tc ? tail_correction_term(sum_kω(kG, χm), χ_m_sum, tc_factor) : 0.0)
    Σ_ladder.parent[:, :] = Σ_ladder.parent[:, :] ./ mP.β .+ tail_correction .+ Σ_hartree
    return nothing
end

function calc_Σ!(Σ_ladder::OffsetMatrix{ComplexF64}, Σ_ladder_ω::OffsetArray{ComplexF64,3}, Kνωq_pre::Vector{ComplexF64},
                χm::χT, γm::γT, χd::χT, γd::γT, χ_m_sum::Union{Float64,ComplexF64}, λ₀::AbstractArray{_eltype,3},
                tc_factor::Vector, Gνω::GνqT, kG::KGrid, mP::ModelParameters, sP::SimulationParameters; tc::Bool = true,
)::Nothing

    Σ_hartree = mP.n * mP.U / 2.0
    calc_Σ_ω!(eom, Σ_ladder_ω, Kνωq_pre, χm, γm, χd, γd, Gνω, λ₀, mP.U, kG, sP)
    sum!(Σ_ladder, Σ_ladder_ω)
    tail_correction = (tc ? tail_correction_term(sum_kω(kG, χm), χ_m_sum, tc_factor) : 0.0)
    Σ_ladder.parent[:, :] = Σ_ladder.parent[:, :] ./ mP.β .+ reshape(tail_correction, 1, length(tail_correction)) .+ Σ_hartree
    return nothing
end

"""
    calc_Σ(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::AbstractArray{_eltype,3}, h::lDΓAHelper;
                νmax::Int = h.sP.n_iν, λm::Float64=0.0, λd::Float64=0.0, tc::Bool=true)
    calc_Σ(χm::χT, γm::γT, χd::χT, γd::γT, χ_m_sum::Union{Float64,ComplexF64}, λ₀::AbstractArray{_eltype,3},
                Gνω::GνqT, kG::KGrid, mP::ModelParameters, sP::SimulationParameters; 
                νmax::Int = sP.n_iν, λm::Float64=0.0, λd::Float64=0.0, tc::Bool=true)
                
Calculates the self-energy from ladder quantities.

This is the single core variant, see [`calc_Σ_par`](@ref calc_Σ_par) for the parallel version.
"""
function calc_Σ(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::AbstractArray{_eltype,3}, h::lDΓAHelper; νmax::Int = h.sP.n_iν, λm::Float64 = 0.0, λd::Float64 = 0.0, tc::Bool = true)
    calc_Σ(χm, γm, χd, γd, h.χloc_m_sum, λ₀, h.Σ_loc, h.gLoc_rfft, h.kG, h.mP, h.sP, νmax = νmax, λm = λm, λd = λd, tc = tc)
end

function calc_Σ(χm::χT, γm::γT, χd::χT, γd::γT, χ_m_sum::Union{Float64,ComplexF64}, λ₀::AbstractArray{_eltype,3},
                Σ_loc::OffsetVector{ComplexF64}, Gνω::GνqT, kG::KGrid, mP::ModelParameters, sP::SimulationParameters;
                νmax::Int = sP.n_iν, λm::Float64 = 0.0, λd::Float64 = 0.0, tc::Bool = true)
    χm.λ != 0 && λm != 0 && error("Stopping self energy calculation: λm = $λm AND χm.λ = $(χm.λ)")
    χd.λ != 0 && λd != 0 && error("Stopping self energy calculation: λd = $λd AND χd.λ = $(χd.λ)")
    Nq, Nω = size(χm)
    ωrange::UnitRange{Int} = -sP.n_iω:sP.n_iω

    Kνωq_pre::Vector{ComplexF64} = Vector{ComplexF64}(undef, length(kG.kMult))
    Σ_ladder = OffsetArray(Array{Complex{Float64},2}(undef, Nq, νmax), 1:Nq, 0:νmax-1)

    λm != 0.0 && χ_λ!(χm, λm)
    λd != 0.0 && χ_λ!(χd, λd)

    iν = iν_array(mP.β, collect(axes(Σ_ladder, 2)))
    tc_factor_term = (tc ? tail_factor(mP.U, mP.β, mP.n, Σ_loc, iν) : 0.0 ./ iν)
    calc_Σ!(Σ_ladder, Kνωq_pre, χm, γm, χd, γd, χ_m_sum, λ₀, tc_factor_term, Gνω, kG, mP, sP, tc = tc)

    λm != 0.0 && reset!(χm)
    λd != 0.0 && reset!(χd)
    return Σ_ladder
end





"""
    calc_Σ_parts(χm::χT,γm::γT,χd::χT,γd::γT,h::lDΓAHelper,λ₀::AbstractArray{ComplexF64,3};λm::Float64=0.0, λd::Float64=0.0)
    calc_Σ_parts(χm::χT,γm::γT, χd::χT, γd::γT, χ_m_sum::Union{Float64,ComplexF64}, λ₀::AbstractArray{_eltype,3},
                 Gνω::GνqT, kG::KGrid,mP::ModelParameters, sP::SimulationParameters;
                 λm::Float64=0.0, λd::Float64=0.0)

Calculates the ``lD\\GammaA`` self-energy (see also [`calc_Σ`](@ref calc_Σ)),
but split into `7` contributions from: `χm`, `γm`, `χd`, `γd`, `U`, `Fm` + `Σ_hartree`, `tail_correction`.

"""
function calc_Σ_parts(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::AbstractArray{_eltype,3}, h::lDΓAHelper; 
                      tc::Bool = true, λm::Float64 = 0.0, λd::Float64 = 0.0)
    calc_Σ_parts(χm, γm, χd, γd, h.χloc_m_sum, λ₀, h.Σ_loc, h.gLoc_rfft, h.kG, h.mP, h.sP; tc = tc, λm = λm, λd = λd)
end

function calc_Σ_parts(χm::χT, γm::γT, χd::χT, γd::γT, χ_m_sum::Union{Float64,ComplexF64}, λ₀::AbstractArray{_eltype,3},
                      Σ_loc::OffsetVector{ComplexF64}, Gνω::GνqT, kG::KGrid, mP::ModelParameters, sP::SimulationParameters; 
                      tc::Bool = true, λm::Float64 = 0.0, λd::Float64 = 0.0)
    Σ_hartree = mP.n * mP.U / 2.0
    Nq, Nω = size(χm)
    ωrange::UnitRange{Int} = -sP.n_iω:sP.n_iω
    ωindices::UnitRange{Int} = (sP.dbg_full_eom_omega) ? (1:Nω) : intersect(χm.usable_ω, χd.usable_ω)

    Kνωq_pre::Vector{ComplexF64} = Vector{ComplexF64}(undef, length(kG.kMult))
    Σ_ladder_ω = OffsetArray(Array{Complex{Float64},3}(undef, Nq, sP.n_iν, length(ωrange)), 1:Nq, 0:sP.n_iν-1, ωrange)
    Σ_ladder = OffsetArray(Array{Complex{Float64},3}(undef, Nq, sP.n_iν, 7), 1:Nq, 0:sP.n_iν-1, 1:7)

    λm != 0.0 && χ_λ!(χm, λm)
    λd != 0.0 && χ_λ!(χd, λd)

    iν = iν_array(mP.β, collect(axes(Σ_ladder, 2)))
    tc_term = (tc ? tail_correction_term(mP.U, mP.β, mP.n, sum_kω(kG, χm), χ_m_sum, Σ_loc, iν) : 0.0 ./ iν)
    calc_Σ_ω!(eom_χ_m, Σ_ladder_ω, Kνωq_pre, χm, γm, χd, γd, Gνω, λ₀, mP.U, kG, sP)
    Σ_ladder.parent[:, :, 1] = dropdims(sum(Σ_ladder_ω, dims = [3]), dims = 3) ./ mP.β
    calc_Σ_ω!(eom_γ_m, Σ_ladder_ω, Kνωq_pre, χm, γm, χd, γd, Gνω, λ₀, mP.U, kG, sP)
    Σ_ladder.parent[:, :, 2] = dropdims(sum(Σ_ladder_ω, dims = [3]), dims = 3) ./ mP.β
    calc_Σ_ω!(eom_χ_d, Σ_ladder_ω, Kνωq_pre, χm, γm, χd, γd, Gνω, λ₀, mP.U, kG, sP)
    Σ_ladder.parent[:, :, 3] = dropdims(sum(Σ_ladder_ω, dims = [3]), dims = 3) ./ mP.β
    calc_Σ_ω!(eom_γ_d, Σ_ladder_ω, Kνωq_pre, χm, γm, χd, γd, Gνω, λ₀, mP.U, kG, sP)
    Σ_ladder.parent[:, :, 4] = dropdims(sum(Σ_ladder_ω, dims = [3]), dims = 3) ./ mP.β
    calc_Σ_ω!(eom_rest_01, Σ_ladder_ω, Kνωq_pre, χm, γm, χd, γd, Gνω, λ₀, mP.U, kG, sP)
    Σ_ladder.parent[:, :, 5] = dropdims(sum(Σ_ladder_ω, dims = [3]), dims = 3) ./ mP.β
    calc_Σ_ω!(eom_rest, Σ_ladder_ω, Kνωq_pre, χm, γm, χd, γd, Gνω, λ₀, mP.U, kG, sP)
    Σ_ladder.parent[:, :, 6] = dropdims(sum(Σ_ladder_ω, dims = [3]), dims = 3) ./ mP.β .+ Σ_hartree
    for qi = 1:size(Σ_ladder, 1)
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
function tail_factor(U::Float64, β::Float64, n::Float64, Σ_loc::OffsetVector{ComplexF64}, iν::Vector{ComplexF64}; δ::Real = min(0.001, 1 ./ length(iν)))
    Σlim = U^2 * n / 2 * (1 - n / 2)
    DMFT_dff = -imag(Σ_loc[0:length(iν)-1]) .* imag(iν) .- Σlim
    return -2 * U .* exp.(-(DMFT_dff) .^ 2 ./ δ) ./ iν
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
function tail_correction_term(χm_nl::Float64, χm_loc::Float64, tail_factor::Vector{ComplexF64})
    return reshape((χm_nl - χm_loc) .* tail_factor, 1, length(tail_factor))
end

function tail_correction_term(U::Float64, β::Float64, n::Float64, χm_nl::Float64, χm_loc::Float64, 
                              Σ_loc::OffsetVector{ComplexF64}, iν::Vector{ComplexF64}; 
                              δ::Real = min(0.001, 1 ./ 10.0 * length(iν)))

    tf = tail_factor(U, β, n, Σ_loc, iν, δ = δ)
    return tail_correction_term(χm_nl, χm_loc, tf)
end
