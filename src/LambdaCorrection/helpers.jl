# ==================================================================================================== #
#                                            helpers.jl                                                #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
#   Last Edit Date  : 02.10.22                                                                         #
# ----------------------------------------- Description ---------------------------------------------- #
#   λ-Correction related helper functions.                                                             #
# -------------------------------------------- TODO -------------------------------------------------- #
#   bisection root finding method is not tested properly.                                                   #
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
    χ_new = χT(χ.data)
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
in the input data structure.
"""
function χ_λ!(χ_new::χT, χ::χT, λ::Float64)::χT
    χ_λ!(χ_new.data, χ.data, λ)
    χ_new.λ = χ.λ + λ
    return χ_new 
end

χ_λ!(χ_λ::AbstractArray, χ::AbstractArray, λ::Float64) = χ_λ[:] = χ ./ ((λ .* χ) .+ 1)
χ_λ!(χ::χT, λ::Float64) = χ_λ!(χ, χ, λ)


"""
    dχ_λ(χ::[Float64,ComplexF64,AbstractArray], λ::Float64)

First derivative of [`χ_λ`](@ref χ_λ).
"""
dχ_λ(χ::T, λ::Float64) where T <: Union{Float64, ComplexF64} = -χ_λ(χ, λ)^2
dχ_λ(χ::AbstractArray, λ::Float64) = map(χi -> - ((1.0 / χi) + λ)^(-2), χ)


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
        if xi < start               # only ever search to the right!
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
    nh  = ceil(Int64, size(χr,2)/2)
    -minimum(1 ./ view(χr,:,nh))
end
