# ==================================================================================================== #
#                                             Common.jl                                                #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#   Common functions for LambdaCorrection sub module
# -------------------------------------------- TODO -------------------------------------------------- #
# ==================================================================================================== #

# ========================================== χ-λ-transform ===========================================
"""
    χ_λ(χ::[Float64,ComplexF64,AbstractArray,χT], λ::Float64)

Computes the λ-corrected susceptibility:  ``\\chi^{\\lambda,\\omega}_q = \\frac{1}{1 / \\chi^{\\lambda,\\omega}_q + \\lambda}``.
The susceptibility ``\\chi`` can be either given element wise, or as χT See also [`χT`](@ref χT) in LadderDGA.jl.
"""
Base.@assume_effects :total χ_λ(χ::Float64, λ::Float64)::Float64 = χ / (λ * χ + 1)

function χ_λ(χ::χT, λ::Float64)::χT
    χ_new = χT(deepcopy(χ.data), χ.β, tail_c = χ.tail_c)
    χ_λ!(χ_new, χ, λ)
    return χ_new
end

function χ_λ(χ::AbstractArray{Float64}, λ::Float64)
    χ_new = similar(χ)
    χ_λ!(χ_new, χ, λ)
    return χ_new
end

"""
    χ_λ!(χ_destination::[AbstractArray,χT], [χ::[AbstractArray,χT], ] λ::Float64)

Inplace version of [`χ_λ`](@ref χ_λ). If the second argument is omitted, results are stored
in the input `χ`.
"""
function χ_λ!(χ_new::χT, χ::χT, λ::Float64)::Nothing
    χ_λ!(χ_new.data, χ.data, λ)
    χ_new.λ = χ.λ + λ
    χ_new.transform! = χ_λ!
    return nothing
end

function χ_λ!(res::AbstractArray, χ::AbstractArray, λ::Float64)::Nothing
    λ == 0.0 && return nothing
    !isfinite(λ) && println("WARNING. SKIPPING λ correction because $λ is not finite!") && return nothing
    for i in eachindex(res)
        res[i] = χ_λ(χ[i], λ)
    end
end

χ_λ!(χ::χT, λ::Float64)::Nothing = χ_λ!(χ, χ, λ)


"""
    dχ_λ(χ::[Float64,ComplexF64,AbstractArray], λ::Float64)

First derivative of [`χ_λ`](@ref χ_λ).
"""
Base.@assume_effects :total dχ_λ(χ::Float64, λ::Float64)::Float64 = -χ_λ(χ, λ)^2
dχ_λ(χ::AbstractArray, λ::Float64) = map(χi -> -((1.0 / χi) + λ)^(-2), χ)

"""
    reset!(χ::χT)

Resets the λ-correction of the `χ` struct.
"""
function reset!(χ::χT)
    if χ.λ != 0
        χ.transform!(χ, -χ.λ)
        χ.λ = 0
        χ.transform! = (f!(χ, λ) = nothing)
    end
end


# ========================================== lDGA Wrappers ===========================================
"""
    calc_G_Σ(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::AbstractArray{ComplexF64,3}, 
             λm::Float64, λd::Float64,
             h::RunHelper, sP::SimulationParameters, mP::ModelParameters; 
             tc::Bool = true, fix_n::Bool = true
)

Returns `μ_new`, `G_ladder`, `Σ_ladder` with λ correction according to function parameters.
"""
function calc_G_Σ(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::AbstractArray{ComplexF64,3}, 
                  λm::Float64, λd::Float64, h::RunHelper; 
                  gLoc_rfft = h.gLoc_rfft, tc::Bool = true, fix_n::Bool = true
)
    Σ_ladder = calc_Σ(χm, γm, χd, γd, λ₀, gLoc_rfft, h; λm = λm, λd = λd, tc = tc)
    μ_new, G_ladder = G_from_Σladder(Σ_ladder, h.Σ_loc, h.kG, h.mP, h.sP; fix_n = fix_n)
    return μ_new, G_ladder, Σ_ladder
end