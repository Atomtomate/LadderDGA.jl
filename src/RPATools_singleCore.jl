# ==================================================================================================== #
#                                           RPATools_singleCore.jl                                     #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Jan Frederik Weißler                                                             #
# ----------------------------------------- Description ---------------------------------------------- #
#   Functions to calculate RPA specific quantities                                                     #
# -------------------------------------------- TODO -------------------------------------------------- #
#   Unit tests                                                                                         #
#   Refacture λ0 calculation: reduce rank by the dimension of the fermionic matsubara frequency        #
# ==================================================================================================== #

function calc_bubble(type::Symbol, h::RPAHelper)
    if type == :RPA
        calc_bubble(type, h.gLoc_fft, h.gLoc_rfft, h.kG, h.mP, h.sP)
    elseif type == :RPA_exact
        error("not implemented yet")
    else
        throw(ArgumentError("Unkown type in bubble calculation"))
    end
end

function calc_χγ(type::Symbol, h::RPAHelper, χ₀::χ₀T)
    calc_χγ(type, χ₀, h.kG, h.mP, h.sP)
end

function calc_χγ(type::Symbol, χ₀::χ₀T, kG::KGrid, mP::ModelParameters, sP::SimulationParameters)
    s = if type == :d
        1
    elseif type == :m
        -1
    else
        error("Unkown type")
    end
    χ₀_qω = dropdims(sum(χ₀.data, dims = χ₀.axis_types[:ν]), dims = χ₀.axis_types[:ν]) ./ mP.β^2
    Nq = length(kG.kMult)
    Nω = size(χ₀.data, χ₀.axis_types[:ω])
    Nν = 2 * sP.n_iν + 1
    γ = ones(ComplexF64, Nq, Nν, Nω)
    χ = real(χ₀_qω ./ (1 .+ s * mP.U .* χ₀_qω))

    @warn "Set kinetic energy to zero. You should implement the kinetic energy in the rpa case to enable asymptotic sums!"
    return χT(χ, mP.β, full_range = true; tail_c = [0.0, 0.0, 0.0], kG = kG), γT(γ)
end

"""
    calc_χγ(type::Symbol, χ₀::χ₀RPA_T, kG::KGrid, mP::ModelParameters)

    This function corresponds to the following mappings
    
        χ: BZ × 2πN/β → R, (q, ω)↦ χ₀(q,ω) / ( 1 + U_r⋅χ₀(q,ω) )
        
        γ: BZ × π(2N + 1)/β × 2πN/β → C, (q, ν, ω)↦ 1

        where ...
            ... U_r is the Hubbard on-site interaction parameter multiplied by +1 if type = d and -1 if type = m.
            ... χ₀ is the RPA bubble term
            ... ν is a fermionic matsubara frequency
            ... ω is a bosonic matsubara frequency
            ... N is the set of natural numbers
            ... β is the inverse temperature
            ... q is a point in reciprocal space
"""
function calc_χγ(type::Symbol, χ₀::χ₀RPA_T, mP::ModelParameters, sP::SimulationParameters)
    s = if type == :d
        1
    elseif type == :m
        -1
    else
        error("Unkown type")
    end
    Nν = 2 * sP.n_iν
    Nω = size(χ₀.data, χ₀.axis_types[:ω])
    Nq = size(χ₀.data, χ₀.axis_types[:q])

    γ = ones(ComplexF64, Nq, Nν, Nω)
    χ = real( χ₀ ./ (1 .+ s * mP.U .* χ₀) )
    return χT(χ, χ₀.β, full_range=true; tail_c=[0.0, 0.0, χ₀.e_kin]), γT(γ)
end


"""
calc_λ0(χ₀::χ₀RPA_T, helper::RPAHelper)
calc_λ0(χ₀::χ₀RPA_T, sP::SimulationParameters, mP::ModelParameters)

This function corresponds to the following mapping
    
    λ0: BZ × π(2N + 1)/β × 2πN/β → C, (q, ν, ω)↦ -U χ₀(q,ω)

    where ...
        ... U is the Hubbard on-site interaction parameter
        ... χ₀ is the RPA bubble term
    
TODO: λ0 is constant in the fermionic matsubara frequency. This should be refactured.
"""
function calc_λ0(χ₀::χ₀RPA_T, helper::RPAHelper)
    calc_λ0(χ₀, helper.sP, helper.mP)
end

function calc_λ0(χ₀::χ₀RPA_T, sP::SimulationParameters, mP::ModelParameters)
    Nν = 2 * sP.n_iν                        # total number of fermionic matsubara frequencies
    Nω = size(χ₀.data, χ₀.axis_types[:ω])   # total number of bosonic matsubara frequencies
    Nq = size(χ₀.data, χ₀.axis_types[:q])   # number of sample points in the sampled reduced reciprocal lattice space
    
    λ0 = zeros(ComplexF64, Nq, Nν, Nω)
    for νi in 1:Nν
        λ0[:,νi,:] = -mP.U * χ₀
    end
    return λ0
end

function calc_λ0(χ₀::χ₀T, helper::RPAHelper)
    calc_λ0(χ₀, helper.sP, helper.mP)
end

function calc_λ0(χ₀::χ₀T, sP::SimulationParameters, mP::ModelParameters)
    λ0 = -mP.U .* deepcopy(χ₀.data)
    return λ0
end

function calc_Σ_rpa!(Σ_ladder::OffsetMatrix{ComplexF64}, Kνωq_pre::Vector{ComplexF64}, work_kG_exp::Array{ComplexF64,N},
                     χm::χT, χd::χT, χ_0_sum::Union{Float64,ComplexF64}, λ₀::AbstractArray{_eltype,3},
                     Gνω::GνqT, kG::KGrid,
                     mP::ModelParameters, sP::SimulationParameters; tc::Bool=true
) ::Nothing where N
    calc_Σ_ω_rpa!(eom_rpa, Σ_ladder, Kνωq_pre, work_kG_exp, χm, χd, Gνω, λ₀, mP.U, kG, sP)
    tail_correction = (tc ? correction_term(mP, kG, χm, χ_0_sum, collect(axes(Σ_ladder)[2])) : zero(iν_array(mP.β, collect(axes(Σ_ladder)[2]))))
    Σ_ladder.parent[:,:] = Σ_ladder.parent[:,:] ./ mP.β .+ reshape(tail_correction, 1, length(tail_correction)) .+ Σ_hartree(mP)
    return nothing
end

function calc_Σ_ω_rpa!(eomf::Function, Σ_ladder::OffsetMatrix{ComplexF64}, Kνωq_pre::Vector{ComplexF64}, work_kG_exp::Array{ComplexF64,N},
                       χm::χT, χd::χT,
                       Gνω::GνqT, λ₀::AbstractArray{ComplexF64,3}, U::Float64, kG::KGrid,
                       sP::SimulationParameters) where N

    νdim = ndims(Gνω) > 2 ? length(gridshape(kG))+1 : 2 # TODO: this is a fallback for gIm. RPA: ndims(Gνω) > 2 is expected, todo: remove inline if.
    fill!(Σ_ladder, zero(ComplexF64))
    ω_axis = χm.indices_ω
    for (ωi,ωn) in enumerate(ω_axis)
        νZero = ν0Index_of_ωIndex(ωi, sP)
        νlist = νZero:(sP.n_iν*2)
        length(νlist) > size(Σ_ladder,2) && (νlist = νlist[1:size(Σ_ladder,2)])
        
        # evaluate EoM for each element of the brillouin zone
        for qi in 1:size(Σ_ladder,1)
            Kνωq_pre[qi] = eomf(U, χm[qi,ωi], χd[qi,ωi], λ₀[qi,begin,ωi]) # note: independent of ν
        end
        expandKArr!(kG, work_kG_exp, Kνωq_pre)
        kG.fftw_plan * work_kG_exp # apply fourier transform to real/position space. technical: kG.fftw_plan is inplace plan!

        # evaluate cross correlation / perform q-integration for each fermionic matsubara frequency
        for (νii,νi) in enumerate(νlist)
            #TODO: find a way to not unroll this!
            copy!(kG.cache1, work_kG_exp)
            conv_tmp_add_rpa!(view(Σ_ladder,:,νii-1), kG, selectdim(Gνω,νdim,(νii-1) + ωn))
        end
    end
end


"""
    conv_tmp_add_rpa!(res::AbstractVector{ComplexF64}, kG::KGrid, arr1::Vector{ComplexF64}, GView::AbstractArray{ComplexF64,N})::Nothing where N

Expect both input arrays already in fouriertransformed.
"""
function conv_tmp_add_rpa!(res::AbstractVector{ComplexF64}, kG::KGrid, GView::AbstractArray{ComplexF64,N})::Nothing where N
    if Nk(kG) == 1 
        error("Nk(kG) == 1. That was unexpected!")
    else
        for i in eachindex(kG.cache1)
            kG.cache1[i] *= GView[i]
        end
        kG.fftw_plan \ kG.cache1
        Dispersions.conv_post_add!(kG, res, kG.cache1)
    end
    return nothing
end