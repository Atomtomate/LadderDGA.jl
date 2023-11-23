# ==================================================================================================== #
#                                            helpers.jl                                                #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#   λ-Correction related helper functions.                                                             #
# -------------------------------------------- TODO -------------------------------------------------- #
#   bisection root finding method is not tested properly.                                              #
# ==================================================================================================== #

# ========================================== χ-λ-transform ===========================================
"""
    χ_λ(χ::[Float64,ComplexF64,AbstractArray,χT], λ::Float64)

Computes the λ-corrected susceptibility:  ``\\chi^{\\lambda,\\omega}_q = \\frac{1}{1 / \\chi^{\\lambda,\\omega}_q + \\lambda}``.
The susceptibility ``\\chi`` can be either given element wise, or as χT See also [`χT`](@ref χT) in LadderDGA.jl.
"""
Base.@assume_effects :total χ_λ(χ::Float64, λ::Float64)::Float64 = χ/(λ*χ + 1)

function χ_λ(χ::χT, λ::Float64)::χT 
    χ_new = χT(deepcopy(χ.data), χ.β, tail_c=χ.tail_c)
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
dχ_λ(χ::AbstractArray, λ::Float64) = map(χi -> - ((1.0 / χi) + λ)^(-2), χ)

function reset!(χ::χT)
    if χ.λ != 0
        χ.transform!(χ, -χ.λ) 
        χ.λ = 0
        χ.transform! = (f!(χ,λ) = nothing)
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
    newton_right(f::Function, [df::Function,] start::[Float64,Vector{Float64},MVector{Float64}], min::[Float64,Vector{Float64},MVector{Float64}]; nsteps=5000, atol=1e-11)

Computes root of function `f` but under the condition that each compontent of the root is larger than the corresponding component of the start vector.
This algorithm also assumes, that `f` is stricly monotonically decreasing in each component.
`nsteps` sets the maximum number of newton-steps, `atol` sets the convergence tolerance.
`df` can be omitted. In this case it is approximated using finite differences.
"""
Base.@assume_effects :total function newton_right(f::Function, df::Function, start::Float64, min::Float64; nsteps::Int=100, atol::Float64=1e-13)::Float64
    done = false
    δ = 1e-4
    x0 = start
    xi = x0
    i = 1
    while !done
        fi = f(xi)
        dfii = 1 / df(xi)
        xi = x0 - dfii * fi
        # Found solution in the correct interval
        (norm(fi) < atol) && (xi > min) && break
        if xi < min                              # only ever search to the right!
            x0 = min + δ + abs(min - x0)/2       # do bisection instead
        else
            x0 = xi
        end
        (i >= nsteps ) && (done = true)
        i += 1
    end
    return xi
end

Base.@assume_effects :total function newton_right(f::Function, start::Float64, min::Float64; nsteps::Int=100, atol::Float64=1e-13)::Float64
    df(x) = FiniteDiff.finite_difference_derivative(f, x)
    newton_right(f, df, start, min; nsteps=nsteps, atol=atol)
end

Base.@assume_effects :total function newton_right(f::Function, start::Vector{Float64}, min::Vector{Float64}; nsteps=500, atol=1e-8)::Vector{Float64}
    N = length(start)
    newton_right(f, convert(MVector{N,Float64}, start), convert(MVector{N,Float64}, min), nsteps=nsteps, atol=atol)
end

Base.@assume_effects :total function newton_right(f::Function, start::MVector{N,Float64}, min::MVector{N,Float64}; 
                      nsteps::Int=500, atol::Float64=1e-8, verbose::Bool=false,
                      max_reset::Int=5, reset_backoff::Float64=1.0)::Vector{Float64} where N
    done = false
    xi_last::MVector{N,Float64} = deepcopy(start)
    xi::MVector{N,Float64}      = deepcopy(xi_last)
    i::Int = 1
    reset_i::Int = 1
    bisection_fail::Bool = false
    fi::MVector{N,Float64}      = similar(start)
    dfii::MMatrix{N,N,Float64,N*N}= Matrix{Float64}(undef, N, N)

    cache = FiniteDiff.JacobianCache(xi)
    while !done
        fi[:]     = f(xi)
        dfii[:,:] = inv(FiniteDiff.finite_difference_jacobian(f, xi, cache))
        copyto!(xi_last, xi)
        xi[:]     = xi - dfii * fi
        verbose && println("i=$i : xi = $xi, fi = $fi. $(norm(fi)) ?<? $(atol)")
        for l in 1:N
            if xi[l] < min[l]  # resort to bisection if min value is below left side
                verbose && println("i=$i, l=$l: xi[l] = $(xi[l]) < min[l] = $(min[l]), bisection to $(min[l] + abs(xi_last[l] - min[l])/2)")
                xi[l] = min[l] + abs(xi_last[l] - min[l])/2
                if abs(xi_last[l] - min[l])/2 < 1e-8
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
    λm_rhs(χ_m::χT, χ_d::χT, λd::Float64, h::lDΓAHelper; λ_rhs = :native, verbose=false)
    λm_rhs(imp_density::Float64, χ_m::χT, χ_d::χT, λd::Float64, kG::KGrid, mP::ModelParameters, sP::SimulationParameters, λ_rhs = :native)

Helper function for the right hand side of the Pauli principle conditions (λm correction).
`imp_density` can be set to `NaN`, if the rhs (``\\frac{n}{2}(1-\\frac{n}{2})``) should not be error-corrected (not ncessary or usefull when asymptotic improvement are active).
TODO: write down formula, explain imp_density as compensation to DMFT.
"""
function λm_rhs(χ_m::χT, χ_d::χT, h::lDΓAHelper; λd::Float64=NaN, λ_rhs = :native, verbose=false)
    λm_rhs(h.imp_density, χ_m, χ_d, h.kG, h.mP, h.sP; λd=λd, λ_rhs=λ_rhs, verbose=verbose)
end

function λm_rhs(imp_density::Float64, χ_m::χT, χ_d::χT, kG::KGrid, mP::ModelParameters, sP::SimulationParameters; λd::Float64=NaN, λ_rhs = :native, verbose=false)
    χ_d.λ != 0 && !isnan(λd) && error("Stopping λ rhs calculation: λd = $λd AND χ_d.λ = $(χ_d.λ). Reset χ_d.λ, or do not provide additional λ-correction for this function.")
    χ_d_sum = sum_kω(kG, χ_d, λ=λd)

    verbose && @info "λsp correction infos:"
    rhs = if (( (typeof(sP.χ_helper) != Nothing) && λ_rhs == :native) || λ_rhs == :fixed)
        verbose && @info "  ↳ using n * (1 - n/2) - Σ χ_d as rhs" # As far as I can see, the factor 1/2 has been canceled on both sides of the equation for the Pauli principle => update output
        mP.n * (1 - mP.n/2) - χ_d_sum
    else
        !isfinite(imp_density) && throw(ArgumentError("imp_density argument is not finite! Cannot use DMFT rror compensation method"))
        verbose && @info "  ↳ using χupup_DMFT - Σ χ_d as rhs"
        2*imp_density - χ_d_sum
    end

    if verbose
    @info """  ↳ Found usable intervals for non-local susceptibility of length 
                 ↳ sp: $(χ_m.usable_ω), length: $(length(χ_m.usable_ω))
                 ↳ ch: $(χ_d.usable_ω), length: $(length(χ_d.usable_ω))
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
function gen_νω_indices(χ_m::χT, χ_d::χT, mP::ModelParameters, sP::SimulationParameters; full=false)
    ωindices = usable_ωindices(sP, χ_m, χ_d)
    νmax::Int = !full ? minimum([sP.n_iν,floor(Int,3*length(ωindices)/8)]) : sP.n_iν
    νGrid    = 0:νmax-1
    iωn_f = collect(2im .* (-sP.n_iω:sP.n_iω) .* π ./ mP.β)
    # iωn = iωn_f[ωindices]
    # iωn[findfirst(x->x ≈ 0, iωn)] = Inf
    # χ_tail::Vector{ComplexF64} = χ_d.tail_c[3] ./ (iωn.^2)
    return ωindices, νGrid, iωn_f #χ_tail
end
