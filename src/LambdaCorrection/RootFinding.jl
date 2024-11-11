# ==================================================================================================== #
#                                          RootFinding.jl                                              #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#   Root finding methods, specifically adapted for the lDGA methods.                                   #
# -------------------------------------------- TODO -------------------------------------------------- #
# ==================================================================================================== #


# ============================================ Bisection =============================================
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
    Fm > 0 ? (λm, λr) : (λl, λm)
end

function bisect(λl::Vector{Float64}, λm::Vector{Float64}, λr::Vector{Float64}, Fm::Vector{Float64})::Tuple{Vector{Float64},Vector{Float64}}
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

function correct_margins(λl::Vector{Float64}, λr::Vector{Float64}, Fl::Vector{Float64}, Fr::Vector{Float64})::Tuple{Vector{Float64},Vector{Float64}}
    λl_1, λr_1 = correct_margins(λl[1], λr[1], Fl[1], Fr[1])
    λl_2, λr_2 = correct_margins(λl[2], λr[2], Fl[2], Fr[2])
    ([λl_1, λl_2], [λr_1, λr_2])
end


# ============================================ Newton Right ==========================================
# ------------------------------------------------ 1D  -----------------------------------------------
"""
    newton_right(f::Function, [df::Function,] start::[Float64,Vector{Float64},MVector{Float64}], min::[Float64,Vector{Float64},MVector{Float64}]; nsteps=5000, atol=1e-11)

Computes root of function `f` but under the condition that each compontent of the root is larger than the corresponding component of the start vector.
This algorithm also assumes, that `f` is stricly monotonically decreasing in each component.
`nsteps` sets the maximum number of newton-steps, `atol` sets the convergence tolerance.
`df` can be omitted. In this case it is approximated using finite differences.

This is a legacy method. For better convergence performance and reliability please consider using [`newton_secular`](@ref newton_secular).
"""
function newton_right(f::Function, df::Function, start::Float64, min::Float64; nsteps::Int = 500, atol::Float64=1e-8, δ::Float64=1e-4, verbose::Bool=false)::Float64
    done  = false
    xlast = start + δ
    xi    = xlast
    i     = 1
    while !done
        fi = f(xi)
        dfii = 1 / df(xi)
        xi = xlast - dfii * fi
        # Found solution in the correct interval
        (norm(fi) < atol) && (xi > min) && (done = true)
        # only ever search to the right! bisect instead
        if xi < min
            xi = norm(xlast - (min + δ))/2 +  (min + δ)
        else
            xlast = xi
        end
        (i >= nsteps) && (done = true)
        verbose && println("i = $i, xi = $xi, f(xi) = $fi")
        i += 1
    end
    println("nsteps = ", i-1)
    return xi
end

function newton_right(f::Function, start::Float64, min::Float64; nsteps::Int=100, atol::Float64=1e-8, δ::Float64=1e-4, verbose::Bool=false)::Float64
    df(x) = FiniteDiff.finite_difference_derivative(f, x)
    newton_right(f, df, start, min; nsteps = nsteps, atol = atol, δ = δ, verbose=verbose)
end


# ------------------------------------------------ 2D  -----------------------------------------------
#
function newton_right(f::Function, start::Vector{Float64}, min::Vector{Float64}; nsteps = 500, atol::Float64=1e-8, δ::Float64=1e-4, verbose::Bool=false)::Vector{Float64}
    N = length(start)
    newton_right(f, convert(MVector{N,Float64}, start), convert(MVector{N,Float64}, min), nsteps=nsteps, atol=atol, verbose=verbose, reset_backoff=δ)
end

function newton_right(
    f::Function, start::MVector{N,Float64}, min::MVector{N,Float64};
    nsteps::Int = 500, atol::Float64 = 1e-8, verbose::Bool = false, max_reset::Int = 5, reset_backoff::Float64 = 1.0,
)::Vector{Float64} where {N}
    done = false
    xi_last::MVector{N,Float64} = deepcopy(start)
    xi::MVector{N,Float64} = deepcopy(xi_last)
    i::Int = 1
    reset_i::Int = 1
    bisection_fail::Bool = false
    fi::MVector{N,Float64} = similar(start)
    dfii::MMatrix{N,N,Float64,N * N} = Matrix{Float64}(undef, N, N)

    cache = FiniteDiff.JacobianCache(xi)
    while !done
        fi[:] = f(xi)
        dfii[:, :] = inv(FiniteDiff.finite_difference_jacobian(f, xi, cache))
        copyto!(xi_last, xi)
        xi[:] = xi - dfii * fi
        verbose && println("i=$i : xi = $xi, fi = $fi. $(norm(fi)) ?<? $(atol)")
        for l = 1:N
            if xi[l] < min[l]  # resort to bisection if min value is below left side
                verbose && println("i=$i, l=$l: xi[l] = $(xi[l]) < min[l] = $(min[l]), bisection to $(min[l] + abs(xi_last[l] - min[l])/2)")
                xi[l] = min[l] + abs(xi_last[l] - min[l]) / 2
                if abs(xi_last[l] - min[l]) / 2 < 1e-8
                    println("WARNING: Bisection step is smaller than 1e-8. This indicates faulty data or no solution.")
                    bisection_fail = true
                end
            end
        end
        if bisection_fail || !all(isfinite.(fi)) # reset, if NaNs are encountered
            bisection_fail = false
            if reset_i >= max_reset
                error("λ root-finding returned NaN after $i iterations and $max_reset resets. min values are $min")
            end
            reset_i += 1
            xi[:] = start .+ reset_i * reset_backoff .* abs.(min)
        end
        (abs(norm(fi)) < atol) && break
        verbose && println("i=$i, after reset, xi = $xi")
        (i >= nsteps) && (done = true)
        i += 1
    end
    return xi
end

# ========================================== Newton SecularEq ========================================
# ------------------------------------------------ 1D  -----------------------------------------------
"""
    newton(f::Function, df::Function, xi::Float64; nsteps::Int = 500, atol::Float64 = 1e-10)::Float64

Normal Newton method, used for example by [`newton_transformed`](@ref newton_transformed) and [`newton_secular`](@ref newton_secular).

`xi` is the initial guess, for functions with multiple roots, the result will depend on this guess.
"""
function newton(f::Function, df::Function, xi::Float64; nsteps::Int = 500, atol::Float64=1e-8)::Float64
    done  = false
    i     = 1
    while !done
        fi = f(xi)
        dfii = 1 / df(xi)
        xi = xi - dfii * fi
        (norm(fi) < atol || i >= nsteps) && (done = true)
        i += 1
    end
    return xi
end

Base.@assume_effects :total newton_secular_transform(x::Float64,p::Float64)::Float64 = sqrt(x) + p
Base.@assume_effects :total newton_secular_transform_df(x::Float64,p::Float64)::Float64 = 1 /(2*sqrt(x))

"""
    newton_secular(f::Function, [df::Function,], pole::Float64)

Computes largest root of function `f`, assuming it corresponds to a secular equaiton ``f(x) = 1 + \\sum_j \\frac{b_j}{d_j - x}``.
Adapted from Example 2,  https://doi.org/10.48550/arXiv.2204.02326
Given the largest pole ``x_p`` we transform the input according to ``w(x_i) = \\frac{1}{x} + x_p`` and then 
procede with the modified Newton algorithm (using the chain rule):
``
x_{(n+1)} = x_{(n)} + f(w(x_i)) \\cdot  (f'(w(x_i)))^{-1} (w'(x_i))^{-1}
``

For debugging purposes, there are also [`newton_secular_trace`](@ref newton_secular_trace) and [`trace_f`](@ref trace_f) available.

 
Arguments:
-------------
- **`f`**      : function, with structure as given above.
- **`df`**     : derivative function, will be constructed by finite differences, if not provided.
- **`xp`**     : largest pole, it is guaranteed, that there is exactly one root larger than this, which will be returned by the algorithm. 
- **`nsteps`** : maximum number of steps
- **`atol`**   : convergence criterion, i.e. ``|f(x_0)| < `` `atol` will return root `x0`.
"""
function newton_secular(f::Function, df::Function, xp::Float64; nsteps::Int = 500, atol::Float64=1e-8)::Float64
    done::Bool  = false
    xi::Float64 = 1.0
    xi_tf::Float64 = NaN
    i::Int         = 1
    while !done
        xi_tf = newton_secular_transform(xi,xp)
        fi = f(xi_tf)
        dfii = 1 / (df(xi_tf)*newton_secular_transform_df(xi, xp))
        xi = xi - dfii * fi
        # Found solution in the correct interval
        (norm(fi) < atol || i >= nsteps) && (done = true)
        i += 1
    end
    i >= nsteps && !done && @warn "Newton did not converge!"
    return xi_tf#inv_newton_secular_transform(xi,xp)
end

function newton_secular(f::Function, xp::Float64; nsteps::Int = 500, atol::Float64=1e-8)::Float64
    df(x) = FiniteDiff.finite_difference_derivative(f, x)
    newton_secular(f, df, xp; nsteps = nsteps, atol = atol)
end

"""
    newton_secular_trace(f::Function, df::Function, xp::Float64; nsteps::Int = 500, atol::Float64 = 1e-10)::Float64

This is the same as [`newton_secular`](@ref newton_secular), but also returns a trace of the intermediate values `(xi,xi_tf,fi,dfii)`.

"""
function newton_secular_trace(f::Function, df::Function, xp::Float64; nsteps::Int = 500, atol::Float64=1e-8)
    done  = false
    xi    = 1.0
    xi_tf = NaN
    trace = Vector{Vector}(undef, 0)
    i     = 1
    while !done
        xi_tf = newton_secular_transform(xi,xp)
        fi = f(xi_tf)
        dx_xi_tf = df(xi_tf)
        dx_xi_tf_inner = newton_secular_transform_df(xi, xp)
        dfii = 1 / (dx_xi_tf*dx_xi_tf_inner)
        xi = xi - dfii * fi
        push!(trace, [xi, xi_tf, fi, dx_xi_tf, dx_xi_tf_inner, 1 / (dx_xi_tf*dx_xi_tf_inner)])
        (norm(fi) < atol || i >= nsteps) && (done = true)
        i += 1
    end
    return xi_tf, trace
end

function newton_secular_trace(f::Function, xp::Float64; nsteps::Int = 500, atol::Float64=1e-8)
    df(x) = FiniteDiff.finite_difference_derivative(f, x)
    newton_secular_trace(f, df, xp; nsteps = nsteps, atol = atol)
end
