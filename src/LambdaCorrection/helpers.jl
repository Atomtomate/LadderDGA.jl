# ==================================================================================================== #
#                                            helpers.jl                                                #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#   λ-Correction related helper functions.                                                             #
# -------------------------------------------- TODO -------------------------------------------------- #
#   bisection root finding method is not tested properly.                                              #
#   use full version of bisection: `https://arxiv.org/pdf/1702.05542.pdf`                              #
# ==================================================================================================== #

# ========================================== χ-λ-transform ===========================================
"""
    χ_λ(χ::[Float64,ComplexF64,AbstractArray,χT], λ::Float64)

Computes the λ-corrected susceptibility:  ``\\chi^{\\lambda,\\omega}_q = \\frac{1}{1 / \\chi^{\\lambda,\\omega}_q + \\lambda}``.
The susceptibility ``\\chi`` can be either given element wise, or as χT See also [`χT`](@ref χT) in LadderDGA.jl.
"""
χ_λ(χ::T, λ::Float64) where T <: Union{ComplexF64, Float64} = χ/(λ*χ + 1)

function χ_λ(χ::χT, λ::Float64)::χT 
    χ_new = χT(deepcopy(χ.data), χ.β, tail_c=χ.tail_c)
    χ_λ!(χ_new, χ, λ)
    return χ_new 
end

function χ_λ(χ::AbstractArray, λ::Float64)
    χ_new = similar(χ)
    χ_λ!(χ_new, χ, λ)
    return χ_new
end

"""
    χ_λ!(χ_destination::[AbstractArray,χT], [χ::[AbstractArray,χT], ] λ::Float64)

Inplace version of [`χ_λ`](@ref χ_λ). If the second argument is omitted, results are stored
in the input `χ`.
"""
function χ_λ!(χ_new::χT, χ::χT, λ::Float64)
    χ_λ!(χ_new.data, χ.data, λ)
    χ_new.λ = χ.λ + λ
    return nothing 
end

function χ_λ!(χ_λ::AbstractArray, χ::AbstractArray, λ::Float64) 
    λ == 0.0 && return nothing
    !isfinite(λ) && throw(ArgumentError("λ = $λ is not finite!"))
    for i in eachindex(χ_λ)
        χ_λ[i] = χ[i] ./ ((λ .* χ[i]) .+ 1)
    end
end

χ_λ!(χ::χT, λ::Float64) = χ_λ!(χ, χ, λ)


"""
    dχ_λ(χ::[Float64,ComplexF64,AbstractArray], λ::Float64)

First derivative of [`χ_λ`](@ref χ_λ).
"""
dχ_λ(χ::T, λ::Float64) where T <: Union{Float64, ComplexF64} = -χ_λ(χ, λ)^2
dχ_λ(χ::AbstractArray, λ::Float64) = map(χi -> - ((1.0 / χi) + λ)^(-2), χ)

function reset!(χ::χT)
    if χ.λ != 0
        χ_λ!(χ, -χ.λ) 
    end
end

# ===================================== Specialized Root Finding =====================================
# -------------------------------------------- Bisection ---------------------------------------------
"""
    bisect(λl::T, λm::T, λr::T, Fm::T)::Tuple{T,T} where T <: Union{Float64, Vector{Float64}}

WARNING: Not properly tested!
Bisection root finding algorithm. This is a very crude adaption of the 1D case. 
The root may therefore lie outside the given region and the search space has to
be corrected using [`correct_margins`](@ref correct_margins).

Returns: 
-------------
(Vector of) new interval borders, according to `Fm`.

Arguments:
-------------
- **`λl`** : (Vector of) left border(s) of bisection area
- **`λm`** : (Vector of) central border(s) of bisection area
- **`λr`** : (Vector of) right border(s) of bisection area
- **`Fm`** : (Vector of) Poincare-Miranda condition (s)

"""
function bisect(λl::Float64, λm::Float64, λr::Float64, Fm::Float64)::Tuple{Float64,Float64}
    Fm > 0 ? (λm,λr) : (λl,λm)
end

function bisect(λl::Vector{Float64}, λm::Vector{Float64}, λr::Vector{Float64},
        Fm::Vector{Float64})::Tuple{Vector{Float64},Vector{Float64}}
    λl_1, λr_1 = bisect(λl[1], λm[1], λr[1], Fm[1])
    λl_2, λr_2 = bisect(λl[2], λm[2], λr[2], Fm[2])
    ([λl_1, λl_2], [λr_1, λr_2])
end

"""
    correct_margins(λl::T, λm::T, λr::T, Fm::T, Fr::T)::Tuple{T,T} where T <: Union{Float64, Vector{Float64}}

Helper method for [`bisect`](@ref bisect).
"""
function correct_margins(λl::Float64, λr::Float64, Fl::Float64, Fr::Float64)::Tuple{Float64,Float64}
    Δ = 2 .* (λr .- λl)
    Fr > 0 && (λr = λr + Δ)
    Fl < 0 && (λl = λl - Δ)
    λl, λr
end

function correct_margins(λl::Vector{Float64}, λr::Vector{Float64},
                         Fl::Vector{Float64},Fr::Vector{Float64})::Tuple{Vector{Float64},Vector{Float64}}
    λl_1, λr_1 = correct_margins(λl[1], λr[1], Fl[1], Fr[1])
    λl_2, λr_2 = correct_margins(λl[2], λr[2], Fl[2], Fr[2])
    ([λl_1, λl_2], [λr_1, λr_2])
end

# ------------------------------------------- Newton Right -------------------------------------------
"""
    newton_right(f::Function, df::Function, start::[Float64,Vector{Float64}; nsteps=5000, atol=1e-11)

WARNING: Not properly tested!
This is an adaption of the traditional Newton root finding algorithm, searching 
only to the right of `start`.
"""
function newton_right(f::Function, df::Function, start::Float64; nsteps=5000, atol=1e-11)
    done = false
    δ = 0.1
    x0 = start + δ
    xi = x0
    i = 1
    while !done
        fi = f(xi)
        dfii = 1 / df(xi)
        xlast = xi
        xi = x0 - dfii * fi
        (norm(xi-x0) < atol) && break
        if xi < start            # only ever search to the right!
            δ  = δ/2.0
            x0  = start + δ      # reset with smaller delta
            xi = x0
        else
            x0 = xi
        end
        (i >= nsteps ) && (done = true)
        i += 1
    end
    return xi
end

function newton_right(f::Function, start::Vector{Float64}; nsteps=500, atol=1e-6)::Vector{Float64}
    done = false
    δ = 0.1 .* ones(length(start))
    x0 = start .+ δ
    xi = x0
    i = 1
    cache = FiniteDiff.JacobianCache(xi)
    while !done
        fi = f(xi)
        dfii = inv(FiniteDiff.finite_difference_jacobian(f, xi, cache))
        xlast = xi
        xi = x0 - dfii * fi
        (norm(xi-x0) < atol) && break
        reset_test = xi .< start
        if any(reset_test)        # only ever search to the right!
            δ  = [reset_test[i] ? δ[i]/2.0 : δ[i] for i in 1:length(start)]  
            x0 = start .+ δ      # reset with larger delta
            xi = x0
        else
            x0 = xi
        end
        (i >= nsteps ) && (done = true)
        i += 1
    end
    return xi
end

# ============================================== misc. ===============================================
"""
    get_λ_min(χr::AbstractArray{Float64,2})::Float64

Computes the smallest possible ``\\lambda``-correction parameter (i.e. first divergence of ``\\chi(q)``),
given as ``\\lambda_\\text{min} = - \\min_{q}(1 / \\chi^{\\omega_0}_q)``.
"""
function get_λ_min(χr::AbstractArray{Float64,2})::Float64
    nh  = ω0_index(χr)
    -minimum(1 ./ view(χr,:,nh))
end

"""
λsp_rhs([imp_density::Float64, ]χ_m::χT, χ_d::χT, λd::Float64, kG::KGrid, mP::ModelParameters, sP::SimulationParameters, λ_rhs = :native)

Helper function for the right hand side of the Pauli principle conditions (old λ correction).
TODO: write down formula, explain imp_density as compensation to DMFT.
"""
function λsp_rhs(imp_density::Float64, χ_m::χT, χ_d::χT, λd::Float64, kG::KGrid, mP::ModelParameters, sP::SimulationParameters, λ_rhs = :native; verbose=false)
    χ_d.λ != 0 && λd != 0 && error("Stopping λ rhs calculation: λd = $λd AND χ_d.λ = $(χ_d.λ)")
    usable_ω = intersect(χ_m.usable_ω, χ_d.usable_ω)
    #!(χ_d.tail_c[3] ≈ mP.Ekin_DMFT) && @warn "2nd moment not Ekin DMFT"
    iωn = 1im .* 2 .* (-sP.n_iω:sP.n_iω)[usable_ω] .* π ./ mP.β
    χ_λ!(χ_d, λd)
    χ_d_ω = kintegrate(kG, χ_d[:,usable_ω], 1)[1,:]
    λd != 0 && reset!(χ_d)
    χ_d_sum = real(sum(subtract_tail(χ_d_ω, χ_d.tail_c[3], iωn, 2)))/mP.β - χ_d.tail_c[3] * mP.β/12

    verbose && @info "λsp correction infos:"
    rhs = if (( (typeof(sP.χ_helper) != Nothing) && λ_rhs == :native) || λ_rhs == :fixed)
        verbose && @info "  ↳ using n/2 * (1 - n/2) - Σ χ_d as rhs"
        mP.n * (1 - mP.n/2) - χ_d_sum
    else
        verbose && @info "  ↳ using χupup_DMFT - Σ χ_d as rhs"
        2*imp_density - χ_d_sum
    end

    if verbose
    @info """  ↳ Found usable intervals for non-local susceptibility of length 
                 ↳ sp: $(χ_m.usable_ω), length: $(length(χ_m.usable_ω))
                 ↳ ch: $(χ_d.usable_ω), length: $(length(χ_d.usable_ω))
                 ↳ total: $(usable_ω), length: $(length(usable_ω))
               ↳ χ_d sum = $(χ_d_sum), rhs = $(rhs)"""
    end
    return rhs
end

"""
    λ_seach_range(χ::Matrix{Float64}; λ_max_default = 50)

Calculates reasonable interval for the search of the ``\\lambda``-correction parameter. 

The interval is chosen with ``\\lambda_\\text{min}``, such that all unphysical poles are excluded and
``\\lambda_\\text{max} = \\lambda_\\text{default} / \\max_{q,\\omega} \\chi(q,\\omega)``. The `λ_max_default` parameter may need to be
adjusted, depending on the model, since in principal arbitrarily large maximum values are possible.
"""
function λ_seach_range(χ::Matrix{Float64}; λ_max_default = 50)
    λ_min = get_λ_min(χ)
    λ_max = λ_max_default / maximum(χ)
    if λ_min > 1000
        @warn "found very large λ_min ( = $λ_min). This indicates problems with the susceptibility."
    end
    return λ_min, λ_max
end

"""
    gen_νω_indices(χ_m::χT, χ_d::χT, sP::SimulationParameters)

Internal helper to generate usable bosonic and fermionic ranges. Also returns the ``c_1/x^2`` tail. 
"""
function gen_νω_indices(χ_m::χT, χ_d::χT, mP::ModelParameters, sP::SimulationParameters)
    ωindices = usable_ωindices(sP, χ_m, χ_d)
    νmax::Int = minimum([sP.n_iν,floor(Int,3*length(ωindices)/8)])
    νGrid    = 0:νmax-1
    iωn_f = collect(2im .* (-sP.n_iω:sP.n_iω) .* π ./ mP.β)
    # iωn = iωn_f[ωindices]
    # iωn[findfirst(x->x ≈ 0, iωn)] = Inf
    # χ_tail::Vector{ComplexF64} = χ_d.tail_c[3] ./ (iωn.^2)
    return ωindices, νGrid, iωn_f #χ_tail
end
