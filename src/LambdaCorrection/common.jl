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
    χ_new = χT(deepcopy(χ.data), χ.β, usable_ω = χ.usable_ω, tail_c = χ.tail_c)
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

# ============================================== λmin ================================================
"""
    get_λ_min(χr::AbstractArray{Float64,2})::Float64

Computes the smallest possible ``\\lambda``-correction parameter (i.e. first divergence of ``\\chi(q)``),
given as ``\\lambda_\\text{min} = - \\min_{q}(1 / \\chi^{\\omega_0}_q)``.
"""
function get_λ_min(χr::AbstractArray{Float64,2})::Float64
    nh = ω0_index(χr)
    -minimum(1 ./ view(χr, :, nh))
end


"""
    get_λd_min(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::Array{ComplexF64,3}, h::lDΓAHelper; 
                    λd_max::Float64=0.0, Δλ::Float64 = 1e-1, dΣ0_threshold::Float64=4.0)::Float64

This function tries to estimate a minimum value of `λd` that is more accurate for advanced 
λ-corection types, than [`get_λ_min`](@ref get_λ_min).

The minium value is given if either ``\\exists_k : \\In(\\Sigma^{\\nu_0}_k) > 0`` or the
derivative exeeds `dΣ0_threshold`.
Derivative and minimum backoff distance from the `get_λ_min` value are given by `Δλ`.
"""
function get_λd_min(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::Array{ComplexF64,3}, h::lDΓAHelper; 
                    λd_max::Float64=0.0, Δλ::Float64 = 1e-1, dΣ0_threshold::Float64=4.0)::Float64

    Nq = length(h.kG.kMult)

    Kνωq_pre::Vector{ComplexF64} = Vector{ComplexF64}(undef, Nq)
    Σ_ladder = OffsetArray(Matrix{ComplexF64}(undef, Nq, 1), 1:Nq, 0:0)
    return get_λd_min!(Σ_ladder, Kνωq_pre, χm, γm, χd, γd, λ₀, h; λd_max=λd_max, Δλ=Δλ, dΣ0_threshold=dΣ0_threshold)
end

function get_λd_min!(Σ_ladder, Kνωq_pre::Vector{ComplexF64},
                     χm::χT, γm::γT, χd::χT, γd::γT, λ₀::Array{ComplexF64,3}, h::lDΓAHelper; 
                    λd_max::Float64=0.0, Δλ::Float64 = 1e-1, dΣ0_threshold::Float64=4.0)::Float64
    nh = ω0_index(χd)
    λd_min0 = -minimum(1 ./ view(χd, :, nh)) + Δλ
    ωn2_tail = ω2_tail(χm)
    
    λd_max = (λd_max - λd_min0) < 5 ? λd_max + 5 : λd_max
    λd_max_result = λd_max
    
    λd_grid = reverse(LinRange(λd_min0, λd_max, ceil(Int, (λd_max-λd_min0)/Δλ)))
    Σ0_λ_i::Float64 = NaN
    Σ0_λ_im::Float64 = NaN
    dΣ0_λ_last::Float64 = 0.0
    
    for (i,λd_i) in enumerate(λd_grid)
        rhs_c1,_ = λm_rhs(χm, χd, h; λd=λd_i)
        λm_i =  λm_correction_val(χm, rhs_c1, h.kG, ωn2_tail)
        (λm_i != 0) && χ_λ!(χm, λm_i)
        (λd_i != 0) && χ_λ!(χd, λd_i)
        calc_Σ!(Σ_ladder, Kνωq_pre, χm, γm, χd, γd, λ₀, 0.0, h.gLoc_rfft, h.kG, h.mP, h.sP)
        (λm_i != 0) && reset!(χm)
        (λd_i != 0) && reset!(χd)
        Σ0_λ_i = maximum(imag(Σ_ladder[:,0]))
        if i > 1
            dΣ0_λ_last = abs((Σ0_λ_im - Σ0_λ_i)/(λd_grid[i-1] - λd_grid[i]))
            Σ0_λ_im = Σ0_λ_i
        end
        if i > 2 && (Σ0_λ_i > 0 || dΣ0_λ_last > dΣ0_threshold)
            λd_max_result = λd_i
            break
        end
    end
    return λd_max_result
end



# ========================================== lDGA Wrappers ===========================================
"""
    calc_G_Σ(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::AbstractArray{ComplexF64,3}, 
             λm::Float64, λd::Float64,
             h::RunHelper, sP::SimulationParameters, mP::ModelParameters; 
             tc::Bool = :exp_step, fix_n::Bool = true
)

Returns `μ_new`, `G_ladder`, `Σ_ladder` with λ correction according to function parameters.
See also [`tail_factor`](@ref tail_factor) for details about the tail correction.
"""
function calc_G_Σ(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::AbstractArray{ComplexF64,3}, 
                  λm::Float64, λd::Float64, h::RunHelper; 
                  gLoc_rfft::GνqT = h.gLoc_rfft, tc::Symbol=default_Σ_tail_correction(), fix_n::Bool = true
)
    Σ_ladder = calc_Σ(χm, γm, χd, γd, λ₀, gLoc_rfft, h; λm = λm, λd = λd, tc = tc)
    μ_new, G_ladder = G_from_Σladder(Σ_ladder, h.Σ_loc, h.kG, h.mP, h.sP; fix_n = fix_n)
    return μ_new, G_ladder, Σ_ladder
end

"""
    calc_G_Σ!(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::AbstractArray{ComplexF64,3}, 
             λm::Float64, λd::Float64,
             h::RunHelper, sP::SimulationParameters, mP::ModelParameters; 
             tc::Symbol=default_Σ_tail_correction(), fix_n::Bool = true
)

Returns `μ_new`; overrides `G_ladder`, `Σ_ladder` and `Kνωq_pre`.
See [`calc_Σ!`](@ref calc_Σ!) and [`calc_G_Σ`](@ref calc_G_Σ).
"""
function calc_G_Σ!(G_ladder::OffsetMatrix{ComplexF64}, Σ_ladder::OffsetMatrix{ComplexF64}, Kνωq_pre::Vector{ComplexF64},
                    tc_term::Matrix{ComplexF64},
                    χm::χT, γm::γT, χd::χT, γd::γT, λ₀::Array{ComplexF64,3}, 
                  λm::Float64, λd::Float64, h::lDΓAHelper; 
                  gLoc_rfft::GνqT = h.gLoc_rfft, fix_n::Bool = true
)::Float64
    (λm != 0) && χ_λ!(χm, λm)
    (λd != 0) && χ_λ!(χd, λd)
        calc_Σ!(Σ_ladder, Kνωq_pre, χm, γm, χd, γd, λ₀, tc_term, gLoc_rfft, h.kG, h.mP, h.sP)
    (λm != 0) && reset!(χm)
    (λd != 0) && reset!(χd)
    μ_new = G_from_Σladder!(G_ladder, Σ_ladder, h.Σ_loc, h.kG, h.mP; fix_n = fix_n, μ = h.mP.μ, n = h.mP.n)
    return μ_new
end