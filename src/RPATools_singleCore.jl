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

function calc_bubble_RPA(kG::KGrid, mP::ModelParameters, sP::SimulationParameters)
    #kG.kGrid .*
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
    χ₀_qω = dropdims(sum(χ₀.data, dims=χ₀.axis_types[:ν]), dims=χ₀.axis_types[:ν]) ./ mP.β^2
    Nq = length(kG.kMult)
    Nω = size(χ₀.data, χ₀.axis_types[:ω])
    Nν = 2 * sP.n_iν + 1
    γ = ones(ComplexF64, Nq, Nν, Nω)
    χ = real(χ₀_qω ./ (1 .+ s * mP.U .* χ₀_qω))
    return χT(χ, mP.β, full_range=true; tail_c=[0.0, 0.0, 0.0]), γT(γ)
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
    for νi in 1 : Nν
        λ0[ :, νi, :] = -mP.U * χ₀
    end
    return λ0
end
