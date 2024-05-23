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

function reset!(χ::χT)
    if χ.λ != 0
        χ.transform!(χ, -χ.λ)
        χ.λ = 0
        χ.transform! = (f!(χ, λ) = nothing)
    end
end

# ===================================== Specialized Root Finding =====================================

# ======================================== tesselated curve. =========================================
# ---------------------------------------------- 1D --------------------------------------------------
"""
    linear_approx(f1::T2, f2::T2, x1::T1, x2::T1, xm::T1) where {T1, T2}

Linear approximation of function sampled at `f1 = f(x1)` and `f2 = f(x2)` at point `xm`, `x1` <= `xm` <= `x2`.
"""
Base.@assume_effects :total function linear_approx(f1::T1, f2::T1, x1::T2, x2::T2, xm::T2) where {T1, T2}
    m = (f2 - f1)/(x2-x1)
    b = f1 - m*x1
    return m*xm + b
end

"""
    sample_f(f::Function, xmin::T, xmax::T; feps_abs::Float64=1e-8, xeps_abs::Float64=1e-8, maxit::Int=1000) where T

Sample a function ``f: \\mathbb{R} \\to \\mathbb{R}`` over the interval ``[xmin, xmax]`` by repeatedly bisecting intervals, that cannot be approximated linearly. 

``x`` values will be sampled with a distance of at least `ϵ`. ``\\delta`` is the bisection criterion.
i.e. if ``|f(x_i) L_f(x_i)| | < `` `feps_abs` for a proposed bisection point, the interval is supposed to be converged.
Algorithm will stop bisection after `maxit` samples
"""
function sample_f(f::Function, xmin::T, xmax::T; feps_abs::Float64=1e-8, xeps_abs::Float64=1e-8, maxit::Int=1000) where T
    fxmin = f(xmin)
    fxmax = f(xmax)
    FT = typeof(fxmin)

    # PriorityQueue ensures largest intervals are bisected first
    todo = PriorityQueue{Tuple{T,FT,T,FT},T,Base.Order.ReverseOrdering}(Base.Order.Reverse)
    done = Stack{Tuple{T,FT}}()

    # Initialize todo with global intgerval
    push!(todo, (xmin, fxmin, xmax, fxmax) => xmax-xmin)
    push!(done, (xmax, fxmax))
    it=2

    while !isempty(todo)
        # TODO: this needs to be replaced by popfirst! for future versions of DataStructures
        x1, f1, x2, f2 = dequeue!(todo)

        if it == maxit
            push!(done, (x1,f1))
        else
            # compare true sample point with lin. approx. bisect if above threshold
            xm      = (x2-x1)/2 + x1
            fm_test = linear_approx(f1, f2, x1, x2, xm)
            fm      = f(xm)

            # println(" ============================================= ")
            # println("it = $it: $fm vs $(f(xm))")
            # println(abs(fm_test - f(xm)) , " OR ", x2 - x1)
            # println(abs(fm_test - f(xm)) < feps_abs, " OR ", x2 - x1 <= xeps_abs)
            # println("[[$x1, $x2] => xm = ", xm)
            # println(" ============================================= ")

            if abs(fm_test - fm) <= feps_abs || abs(xm - x1) <= xeps_abs
                push!(done, (xm, fm))
            else
                it += 1
                # Bisect and set length of intervals as priority
                push!(todo, (x1, f1, xm, fm) => xm-x1)
                push!(todo, (xm, fm, x2, f2) => x2-xm)
            end
        end
    end
    done = collect(done)
    lx = map(first, done)
    fx = map(x->x[2], done)
    ii = sortperm(lx)
    return lx[ii], fx[ii]
end

# ============================================== misc. ===============================================
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
    λm_rhs(χ_m::χT, χ_d::χT, λd::Float64, h::lDΓAHelper; λ_rhs = :native, verbose=false)
    λm_rhs(imp_density::Float64, χ_m::χT, χ_d::χT, λd::Float64, kG::KGrid, mP::ModelParameters, sP::SimulationParameters, λ_rhs = :native)

Helper function for the right hand side of the Pauli principle conditions (λm correction).
`imp_density` can be set to `NaN`, if the rhs (``\\frac{n}{2}(1-\\frac{n}{2})``) should not be error-corrected (not ncessary or usefull when asymptotic improvement are active).
TODO: write down formula, explain imp_density as compensation to DMFT.
"""
function λm_rhs(χ_m::χT, χ_d::χT, h::lDΓAHelper; λd::Float64 = NaN, λ_rhs = :native, verbose = false)
    λm_rhs(h.imp_density, χ_m, χ_d, h.kG, h.mP, h.sP; λd = λd, λ_rhs = λ_rhs, verbose = verbose)
end

function λm_rhs(imp_density::Float64, χ_m::χT, χ_d::χT, kG::KGrid, mP::ModelParameters, sP::SimulationParameters; λd::Float64 = NaN, λ_rhs = :native, verbose = false)
    χ_d.λ != 0 && !isnan(λd) && error("Stopping λ rhs calculation: λd = $λd AND χ_d.λ = $(χ_d.λ). Reset χ_d.λ, or do not provide additional λ-correction for this function.")
    χ_d_sum = sum_kω(kG, χ_d, λ = λd)

    verbose && @info "λsp correction infos:"
    rhs = if (((typeof(sP.χ_helper) != Nothing) && λ_rhs == :native) || λ_rhs == :fixed)
        verbose && @info "  ↳ using n * (1 - n/2) - Σ χ_d as rhs" # As far as I can see, the factor 1/2 has been canceled on both sides of the equation for the Pauli principle => update output
        mP.n * (1 - mP.n / 2) - χ_d_sum
    else
        !isfinite(imp_density) && throw(ArgumentError("imp_density argument is not finite! Cannot use DMFT rror compensation method"))
        verbose && @info "  ↳ using χupup_DMFT - Σ χ_d as rhs"
        2 * imp_density - χ_d_sum
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
    gen_νω_indices(χ_m::χT, χ_d::χT, sP::SimulationParameters)

Internal helper to generate usable bosonic and fermionic ranges. Also returns the ``c_1/x^2`` tail. 
"""
function gen_νω_indices(χ_m::χT, χ_d::χT, mP::ModelParameters, sP::SimulationParameters; full = false)
    ωindices = usable_ωindices(sP, χ_m, χ_d)
    νmax::Int = !full ? minimum([sP.n_iν, floor(Int, 3 * length(ωindices) / 8)]) : sP.n_iν
    νGrid = 0:νmax-1
    iωn_f = collect(2im .* (-sP.n_iω:sP.n_iω) .* π ./ mP.β)
    # iωn = iωn_f[ωindices]
    # iωn[findfirst(x->x ≈ 0, iωn)] = Inf
    # χ_tail::Vector{ComplexF64} = χ_d.tail_c[3] ./ (iωn.^2)
    return ωindices, νGrid, iωn_f #χ_tail
end
