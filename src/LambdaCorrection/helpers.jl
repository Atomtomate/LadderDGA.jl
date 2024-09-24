# ==================================================================================================== #
#                                            helpers.jl                                                #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#   λ-Correction related helper functions.                                                             #
# -------------------------------------------- TODO -------------------------------------------------- #
#   bisection root finding method is not tested properly.                                              #
# ==================================================================================================== #


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
function sample_f(f::Function, xmin::T, xmax::T; feps_abs::Float64=1e-8, xeps_abs::Float64=1e-8, maxit::Int=1000, nan_backoff::Float64=1e-4) where T
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

            if isnan(fm)
                it += 1
                # function cal lfailed, resubmit with slightly shifted edges
                push!(todo, (x1+nan_backoff, f1, x2+nan_backoff, f2) => x2-x1)
            elseif abs(fm_test - fm) <= feps_abs || abs(xm - x1) <= xeps_abs
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
    gen_νω_indices(χ_m::χT, χ_d::χT, sP::SimulationParameters)

Internal helper to generate usable bosonic and fermionic ranges. Also returns the ``c_1/x^2`` tail. 
"""
function gen_νω_indices(χm::χT, χd::χT, mP::ModelParameters, sP::SimulationParameters; full = false)
    ωindices = usable_ωindices(sP, χm, χd)
    νmax::Int = !full ? minimum([sP.n_iν, floor(Int, 3 * length(ωindices) / 8)]) : sP.n_iν
    νGrid = 0:νmax-1
    iωn_f = ωn_grid(χm)
    # iωn = iωn_f[ωindices]
    # iωn[findfirst(x->x ≈ 0, iωn)] = Inf
    # χ_tail::Vector{ComplexF64} = χ_d.tail_c[3] ./ (iωn.^2)
    return ωindices, νGrid, iωn_f #χ_tail
end
