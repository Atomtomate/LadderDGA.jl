# ==================================================================================================== #
#                                           RPATools_singleCore.jl                                     #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Jan Frederik Weißler                                                             #
# ----------------------------------------- Description ---------------------------------------------- #
#   Functions to calculate RPA specific quantities                                                     #
# -------------------------------------------- TODO -------------------------------------------------- #
#   Unit tests, Proper implementation of the triangular vertex in calc_χγ                              #
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

TBW
"""
function calc_χγ(type::Symbol, χ₀::χ₀RPA_T, kG::KGrid, mP::ModelParameters)
    s = if type == :d
        1
    elseif type == :m
        -1
    else
        error("Unkown type")
    end
    Nq = length(kG.kMult)
    Nω = size(χ₀.data, χ₀.axis_types[:ω])
    Nν = 1
    @warn "Sum over fermionic Matsubara frequencies was performed analytically. Triangular vertex is constructed using one fermionic Matsubara frequency. This needs to be refactured!"
    γ = ones(ComplexF64, Nq, Nν, Nω)
    χ = real(χ₀ ./ (1 .+ s * mP.U .* χ₀))
    return χT(χ, χ₀.β, full_range=true; tail_c=[0.0, 0.0, χ₀.e_kin]), γT(γ)
end