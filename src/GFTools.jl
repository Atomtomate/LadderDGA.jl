# ==================================================================================================== #
#                                           GFTools.jl                                                 #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#   Green's function and Matsubara frequency related functions                                         #
# -------------------------------------------- TODO -------------------------------------------------- #
#   Documentation                                                                                      #
#   unify G_from_Σ APU                                                                                 #
#   come up with a type for MatsubraraFreuqncies                                                       #
#   This file could be a separate module                                                               #
#   Most functions in this files are not used in this project.                                         #
#   Test and optimize functions                                                                        #
#   Rename subtrac_tail and make it more general for arbitrary tails (GF should know its tail)         #
# ==================================================================================================== #



# =================================== Matsubara Frequency Helpers ====================================

"""
    iν_array(β::Real, grid::AbstractArray{Int64,1})::Vector{ComplexF64}
    iν_array(β::Real, size::Int)::Vector{ComplexF64}

Computes list of fermionic Matsubara frequencies.
If length `size` is given, the grid will have indices `0:size-1`.
Bosonic arrays can be generated with [`iω_array`](@ref iω_array).

Returns: 
-------------
Vector of fermionic Matsubara frequencies, given either a list of indices or a length. 
"""
iν_array(β::Real, grid::AbstractArray{Int64,1})::Vector{ComplexF64} =
    ComplexF64[1.0im * ((2.0 * el + 1) * π / β) for el in grid]
iν_array(β::Real, size::Int)::Vector{ComplexF64} = iν_array(β, 0:(size-1))

"""
    iω_array(β::Real, grid::AbstractArray{Int64,1})::Vector{ComplexF64}
    iω_array(β::Real, size::Int)::Vector{ComplexF64}

Computes list of bosonic Matsubara frequencies.
If length `size` is given, the grid will have indices `0:size-1`.
Fermionic arrays can be generated with [`iν_array`](@ref iν_array).

Returns: 
-------------
Vector of bosonic Matsubara frequencies, given either a list of indices or a length. 
"""
iω_array(β::Real, grid::AbstractArray{Int64,1})::Vector{ComplexF64} =
    ComplexF64[1.0im * ((2.0 * el) * π / β) for el in grid]
iω_array(β::Real, size::Int)::Vector{ComplexF64} = iω_array(β, 0:(size-1))

# =================================== Anderson Parameters Helpers ====================================
"""
    Δ(ϵₖ::Vector{Float64}, Vₖ::Vector{Float64}, νₙ::Vector{ComplexF64})::Vector{ComplexF64}

Computes hybridization function ``\\Delta(i\\nu_n) = \\sum_k \\frac{|V_k|^2}{\\nu_n - \\epsilon_k}`` from Anderson parameters (for example obtained through exact diagonalization).

Returns: 
-------------
Hybridization function  over list of given fermionic Matsubara frequencies.

Arguments:
-------------
- **`ϵₖ`** : list of bath levels
- **`Vₖ`** : list of hopping amplitudes
- **`νₙ`** : Vector of fermionic Matsubara frequencies, see also: [`iν_array`](@ref iν_array).

"""
Δ(ϵₖ::Vector{Float64}, Vₖ::Vector{Float64}, νₙ::Vector{ComplexF64})::Vector{ComplexF64} =
    [sum((Vₖ .* conj.(Vₖ)) ./ (ν .- ϵₖ)) for ν in νₙ]

# ===================================== Dyson Equations Helpers ======================================

"""
    G_from_Σ(ind::Int64, β::Float64, μ::Float64, ϵₖ::Float64, Σ::ComplexF64)
    G_from_Σ(mf::ComplexF64, μ::Float64, ϵₖ::Float64, Σ::ComplexF64)

Computes Green's function according to ``[\\frac{(2 n + 1)\\pi i}{\\beta} + \\mu - \\epsilon_k - \\Sigma(k,i\\nu_n)]^{-1}``, where ``\\epsilon_k`` and ``\\Sigma(k,i\\nu_n)`` are given as single values. Convenience wrappers for full grids are provided below.

Arguments:
-------------
- **`ind`** : Matsubara frequency index
- **`mf`**  : Matsubara frequency
- **`β`**   : Inverse temperature (only needs to be set, if index instead of frequency is given)
- **`μ`**   : Chemical potential
- **`ϵₖ`**  : Dispersion relation at fixed `k`, see below for convenience wrappers.
- **`Σ`**   : Self energy at fixed frequency (and potentially fixed `k`), see below for convenience wrappers.
"""
G_from_Σ(ind::Int64, β::Float64, μ::Float64, ϵₖ::Float64, Σ::ComplexF64)::ComplexF64 =
    G_from_Σ(1im * (2 * ind + 1) * π / β, μ, ϵₖ, Σ)
G_from_Σ(mf::ComplexF64, μ::Float64, ϵₖ::Float64, Σ::ComplexF64)::ComplexF64 = 1 / (mf + μ - ϵₖ - Σ)


"""
    G_from_Σ(Σ::OffsetVector{ComplexF64}, ϵkGrid::Vector{Float64}, range::UnitRange{Int}, mP::ModelParameters;
                  μ = mP.μ, Σloc::OffsetVector{ComplexF64} = OffsetVector(ComplexF64[], 0:-1),
            )
    G_from_Σ!(res::OffsetMatrix{ComplexF64}, Σ::OffsetVector{ComplexF64}, ϵkGrid::Vector{Float64}, range::UnitRange{Int},
                mP::ModelParameters; μ = mP.μ, Σloc::OffsetVector{ComplexF64} = OffsetVector(ComplexF64[], 0:-1),
            )::Nothing

    mP::ModelParameters; μ = mP.μ, Σloc::OffsetVector{ComplexF64} = OffsetVector(ComplexF64[], 0:-1),

    G_from_Σ(Σ::AbstractVector{ComplexF64}, ϵkGrid::Vector{Float64}, range::AbstractVector{Int}, mP::ModelParameters; μ = mP.μ,  Σloc::AbstractArray = nothing) 
    G_from_Σ!(res::Matrix{ComplexF64}, Σ::AbstractVector{ComplexF64}, ϵkGrid::Vector{Float64}, range::AbstractVector{Int}, mP::ModelParameters; μ = mP.μ,  Σloc::AbstractVector = nothing) 

#TODO: unify API, redo documentation

Computes Green's function from self energy `Σ` and dispersion `ϵkGrid` over given frequency indices `range`.
Optionally, a different chemical potential `μ` can be provided.
When the non-local self energy is used, one typically wants to extend the usefull range of frequencies by
attaching the tail of the local self energy in the high frequency regime. This is done by providing a
`range` larger than the array size of `Σ` and in addition setting `Σloc` (the size of `Σloc` must be as large or larger than `range`). 
The inplace version stores the result in `res`.
"""
function G_from_Σ(Σ::OffsetVector{ComplexF64}, ϵkGrid::Vector{Float64}, range::UnitRange{Int}, mP::ModelParameters;
                  μ = mP.μ, Σloc::OffsetVector{ComplexF64} = OffsetVector(ComplexF64[], 0:-1),
)
    res = OffsetMatrix{ComplexF64}(Array{ComplexF64,2}(undef, length(ϵkGrid), length(range)), 1:length(ϵkGrid), range)
    G_from_Σ!(res, Σ, ϵkGrid, range, mP, μ = μ, Σloc = Σloc)
    return res
end

function G_from_Σ!(res::OffsetMatrix{ComplexF64}, Σ::OffsetVector{ComplexF64}, ϵkGrid::Vector{Float64}, range::UnitRange{Int},
                   mP::ModelParameters; μ = mP.μ, Σloc::OffsetVector{ComplexF64} = OffsetVector(ComplexF64[], 0:-1),
)::Nothing
    first(range) != 0 && error("G_from_Σ only implemented for range == 0:νmax!")
    for ind in range
        Σi = ind ∈ axes(Σ, 1) ? Σ[ind] : Σloc[ind]
        for (ki, ϵk) in enumerate(ϵkGrid)
            @inbounds res[ki, ind] = G_from_Σ(ind, mP.β, μ, ϵk, Σi)
        end
    end
    return nothing
end

function G_from_Σ!(res::OffsetMatrix{ComplexF64}, Σ::OffsetMatrix{ComplexF64}, ϵkGrid::Vector{Float64}, range::UnitRange{Int},
                   mP::ModelParameters; μ = mP.μ, Σloc::OffsetVector{ComplexF64} = OffsetVector(ComplexF64[], 0:-1),
)::Nothing
    length(ϵkGrid) != size(Σ,1)
    first(range) != 0 && error("G_from_Σ only implemented for range == 0:νmax!")
    for (ki, ϵk) in enumerate(ϵkGrid)
        for ind in range
            Σi = ind ∈ axes(Σ,2) ? Σ[ki,ind] : Σloc[ind]
            @inbounds res[ki, ind] = G_from_Σ(ind, mP.β, μ, ϵk, Σi)
        end
    end
    return nothing
end

function G_from_Σ(Σ::OffsetMatrix{ComplexF64}, ϵkGrid::Vector{Float64}, range::UnitRange{Int},
                      mP::ModelParameters; μ = mP.μ, Σloc::OffsetVector{ComplexF64} = OffsetVector(ComplexF64[], 0:-1),
)
    first(range) != 0 && error("G_from_Σ only implemented for range == 0:νmax!")
    last(range) > last(axes(Σloc,1)) && last(range) > last(axes(Σ,2)) && error("ν-range = $(range) is larger than non-local and local self-energy ranges")
    res = OffsetMatrix{ComplexF64}(Array{ComplexF64,2}(undef, size(Σ,1), length(range)), 1:size(Σ,1), range)
    G_from_Σ!(res, Σ, ϵkGrid, range, mP; μ=μ, Σloc=Σloc)
    return res
end


# ======================================= Self Energy helpers ========================================
"""
    attach_Σloc(Σ_ladder::OffsetMatrix, Σ_loc::OffsetVector; 
                ν_first::Int=last(axis(Σ_ladder,2))+1, ν_last::Int=last(axes(Σloc,1)))

Attach the local self energy tail, starting at `ν_first` up to `ν_last` to the ladder self energy.
#TODO: attach this smoothely by also considering derivatives
"""
function attach_Σloc(Σ_ladder::OffsetMatrix, Σ_loc::OffsetVector; 
                    ν_first::Int=last(axes(Σ_ladder,2))+1, ν_last::Int=last(axes(Σ_loc,1)))
    Nk = size(Σ_ladder,1)
    new_ν_grid = first(axes(Σ_ladder,2)):ν_last 
    res = OffsetMatrix{ComplexF64}(Matrix{ComplexF64}(undef, Nk, length(new_ν_grid)), 1:Nk, new_ν_grid)
    for ki in axes(Σ_ladder,1)
        res[ki,begin:ν_first-1] = Σ_ladder[ki,begin:ν_first-1]
        res[ki,ν_first:ν_last] = Σ_loc[ν_first:ν_last]
    end
    return res
end

function Σ_from_Gladder(Gladder::AbstractMatrix{ComplexF64}, kG::KGrid, μ::Float64, β::Float64)
    res = similar(Gladder)
    Σ_from_Gladder!(res, Gladder, kG, μ, β)
    return res
end

function Σ_from_Gladder!(res::AbstractMatrix, Gladder::OffsetMatrix, kG::KGrid, μ::Float64, β::Float64)
    νn_list = axes(Gladder, 2)
    length(kG.ϵkGrid) != size(res,2)
    for νn in νn_list
        for ki in axes(Gladder, 1)
            res[ki, νn] = μ + 1im * (2 * νn + 1) * π / β - kG.ϵkGrid[ki] - 1 / Gladder[ki, νn]
        end
    end
end

# =============================================== G, Σ ===============================================
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
                  νmax::Int = eom_ν_cutoff(h),
                  gLoc_rfft::GνqT = h.gLoc_rfft, tc = default_Σ_tail_correction(), fix_n::Bool = true
)
    Σ_ladder = calc_Σ(χm, γm, χd, γd, λ₀, gLoc_rfft, h; λm = λm, λd = λd, νmax=νmax, tc = tc)
    μ_new, G_ladder = G_from_Σladder(Σ_ladder, h.Σ_loc, h.kG, h.mP, h.sP; fix_n = fix_n)
    return μ_new, G_ladder, Σ_ladder
end

"""
    calc_G_Σ!(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::AbstractArray{ComplexF64,3}, 
             λm::Float64, λd::Float64,
             h::RunHelper, sP::SimulationParameters, mP::ModelParameters; 
             tc::Type{<: ΣTail} = default_Σ_tail_correction(), fix_n::Bool = true
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

# =============================================== GLoc ===============================================
"""
    G_from_Σladder(Σ_ladder::AbstractMatrix{ComplexF64}, Σloc::Vector{ComplexF64}, kG::KGrid, mP::ModelParameters, sP::SimulationParameters; 
                   fix_n::Bool=false, μ=mP.μ, improved_sum_filling::Bool=true, νRange = sP.fft_range, n = mP.n, νFitRange=0:last(axes(Σ_ladder, 2)) )
    G_from_Σladder!(G_new::OffsetMatrix{ComplexF64}, Σ_ladder::OffsetMatrix{ComplexF64}, Σloc::AbstractVector{ComplexF64}, kG::KGrid, mP::ModelParameters; 
                    fix_n::Bool=false, μ=mP.μ, improved_sum_filling::Bool=true, n = mP.n, νFitRange=0:last(axes(Σ_ladder, 2)) )

Computes Green's function from lDΓA self-energy. This is the Greensfunction used in eq. (8) of Stobbe, J., & Rohringer, G. (2022). Consistency of potential energy in the dynamical vertex approximation. Physical Review B, 106(20), 205101.

The resulting frequency range is given by default as `νRange = sP.fft_range`, if less frequencies are available from `Σ_ladder`, `Σloc` is used instead.
TODO: documentation for arguments
TODO: fit function computes loads of unnecessary frequencies
"""
function G_from_Σladder(Σ_ladder::AbstractMatrix{ComplexF64}, Σloc::OffsetVector{ComplexF64}, kG::KGrid, mP::ModelParameters, sP::SimulationParameters;
                        fix_n::Bool = false, μ = mP.μ, improved_sum_filling::Bool = true, n = mP.n, νRange = sP.fft_range, νFitRange=0:last(axes(Σ_ladder, 2))
    )
        
        G_new = OffsetMatrix(Matrix{ComplexF64}(undef, size(Σ_ladder, 1), length(νRange)), 1:size(Σ_ladder, 1), νRange)
        μ = G_from_Σladder!(G_new, Σ_ladder, OffsetVector(Σloc[0:last(νRange)], 0:last(νRange)), kG, mP,
                fix_n = fix_n, μ = μ, improved_sum_filling = improved_sum_filling, n = n, νFitRange=νFitRange   
            )
        return μ, G_new
    end

function G_from_Σladder!(G_new::OffsetMatrix{ComplexF64}, Σ_ladder::OffsetMatrix{ComplexF64}, Σloc::OffsetVector{ComplexF64}, kG::KGrid, mP::ModelParameters;
                        fix_n::Bool = false, μ = mP.μ, improved_sum_filling::Bool = true, n = mP.n, νFitRange=0:last(axes(Σ_ladder, 2))
)::Float64
    νRange = 0:last(axes(G_new, 2))
    length(νRange) < 10 && @warn "fixing ν range with only $(length(νRange)) frequencies!"
    function fμ(μ::Float64)
        G_from_Σ!(G_new, Σ_ladder, kG.ϵkGrid, νRange, mP, μ = μ, Σloc = Σloc)
        filling_pos(view(G_new, :, νFitRange), kG, mP.U, μ, mP.β; improved_sum = improved_sum_filling) -  n
    end

    function fμ_fallback(μ::Float64)
        G_from_Σ!(G_new, Σ_ladder, kG.ϵkGrid, νRange, mP, μ = μ, Σloc = Σloc)
        filling_pos(view(G_new, :, νFitRange), kG, mP.U, μ, mP.β; improved_sum = false) - n
    end

    μ_bak = μ
    μ = if fix_n
        try
            find_zero(fμ, μ_bak, atol = 1e-8) #nlsolve(fμ, [last_μ])
        catch e
            @warn "improved ($improved_sum_filling) μ determination failed with $(typeof(e)). Falling back to naive summation!"
            try
                find_zero(fμ_fallback, μ_bak, atol = 1e-8) #nlsolve(fμ, [last_μ])
            catch e_int
                G_from_Σ!(G_new, Σ_ladder, kG.ϵkGrid, νRange, mP, μ = μ, Σloc = Σloc)
                filling_pos(view(G_new, :, νFitRange), kG, mP.U, μ, mP.β; improved_sum = false) - n
                @warn "μ determination failed with: $(typeof(e)), fallback failed with $(typeof(e_int))"
                return NaN
            end
        end
    else
        μ
    end
    if !isnan(μ)
        G_from_Σ!(G_new, Σ_ladder, kG.ϵkGrid, νRange, mP, μ = μ, Σloc = Σloc)
    end
    # write complex conjugate in reverse order to negative indices. TODO: find clean way to do this
    ind_neg = first(axes(G_new, 2)):-1
    ind = -1 .* reverse(ind_neg) .- 1
    G_new[:, ind_neg] = conj.(reverse(view(G_new, :, ind), dims=2))
    return μ
end


"""
    Σ_Dyson(GBath::Vector{ComplexF64}, GImp::Vector{ComplexF64})::Vector{ComplexF64}
    Σ_Dyson!(Σ::Vector{ComplexF64}, GBath::Vector{ComplexF64}, GImp::Vector{ComplexF64})::Vector{ComplexF64}

Calculates ``\\Sigma = 1 / G_\\text{bath} - 1 / G_\\text{imp}``.
"""
function Σ_Dyson(GBath::Vector{ComplexF64}, GImp::Vector{ComplexF64})::Vector{ComplexF64}
    Σ = similar(GImp)
    Σ_Dyson!(Σ, GBath, GImp)
    return Σ
end

function Σ_Dyson(GBath::OffsetVector{ComplexF64}, GImp::Vector{ComplexF64})::Vector{ComplexF64}
    Σ = similar(GImp)
    Σ_Dyson!(Σ, GBath.parent, GImp)
    return Σ
end

function Σ_Dyson!(Σ::AbstractVector{ComplexF64}, GBath::Vector{ComplexF64}, GImp::Vector{ComplexF64})
    Σ[:] = 1 ./ GBath .- 1 ./ GImp
    return nothing
end


# ============================================= Filling ==============================================
"""
    filling(G::Vector{ComplexF64}, [kG::KGrid, ] β::Float64)
    filling(G::Vector, U::Float64, μ::Float64, β::Float64, [shell::Float64])

Computes filling of (non-) local Green's function.

If `U`, `μ` and `β` are provided, asymptotic corrections are used. The shell sum can be precomputed using [`shell_sum_fermionic`](@ref shell_sum_fermionic)
If `G` is defined only over positive Matsubara frequencies [`filling_pos`](@ref filling_pos) can be used.
"""
function filling(G::AbstractVector{ComplexF64}, β::Float64)
    n = 2 * sum(G) / β + 1
    imag(n) > 1e-8 && throw("Error: Imaginary part of filling is larger than 10^-8")
    return real(n)
end

function filling(G::AbstractMatrix{ComplexF64}, kG::KGrid, β::Float64)
    n = 2 * sum(kintegrate(kG, G, 1)) / β + 1
    imag(n) > 1e-8 && throw("Error: Imaginary part of filling is larger than 10^-8")
    return real(n)
end

function filling(G::AbstractVector{ComplexF64}, U::Float64, μ::Float64, β::Float64, shell::Float64)::Float64
    2 * (real(sum(G)) / β + 0.5 + μ * shell) / (1 + U * shell)
end

function filling(G::AbstractVector{ComplexF64}, U::Float64, μ::Float64, β::Float64)
    N = floor(Int, length(G) / 2)
    shell = 2 * real(shell_sum_fermionic(N, β, 2) / β)
    filling(G[0:end], U, μ, β, shell)
end

function filling(G::AbstractMatrix{ComplexF64}, kG::KGrid, U::Float64, μ::Float64, β::Float64)
    first(axes(G,2)) >= 0 && @warn "executing filling over unkown ν-axis. Assuming -ν:ν (use fillin_pos instead)"
    filling(kintegrate(kG, G, 1)[1, :], U, μ, β)
end

"""
    filling_pos(G::Vector, U::Float64, μ::Float64, β::Float64[, shell::Float64, improved_sum::Bool=true])::Float64
    filling_pos(G::AbstractMatrix{ComplexF64},kG::KGrid,U::Float64,μ::Float64,β::Float64; improved_sum::Bool = true)::Float64
    filling_pos(G::AbstractMatrix{ComplexF64},kG::KGrid,)::Float64

Returns filling from `G` only defined over positive Matsubara frequencies. 
See [`filling`](@ref filling) for further documentation.
"""
function filling_pos(G::AbstractVector{ComplexF64}, U::Float64, μ::Float64, β::Float64, shell::Float64)::Float64
    sG = sum(G)
    2 * (real(sG + conj(sG)) / β + 0.5 + μ * shell) / (1 + U * shell)
end

function filling_pos(G::AbstractVector{ComplexF64}, U::Float64, μ::Float64, β::Float64; improved_sum::Bool = true)::Float64
    if !improved_sum
        sG = sum(G)
        2 * real(sG + conj(sG)) / β + 1
    else
        N = length(G) #floor(Int, length(G) / 2)
        shell = 2 * real(shell_sum_fermionic(N+1, β, 2) / β)
        filling_pos(G, U, μ, β, shell)
    end
end

function filling_pos(G::AbstractMatrix{ComplexF64},kG::KGrid,U::Float64,μ::Float64,β::Float64; improved_sum::Bool = true)::Float64
    if improved_sum
        filling_pos(view(kintegrate(kG, G, 1), 1, :), U, μ, β, improved_sum = improved_sum)
    else
        filling_pos(G ,kG, β)
    end
end

function filling_pos(G::AbstractMatrix{ComplexF64},kG::KGrid, β::Float64)::Float64
    GInt = view(kintegrate(kG, G, 1), 1, :)
    sG = sum(GInt)
    2 * real(sG + conj(sG)) / β + 1
end

"""
    shell_sum_fermionic(N::Int, β::Float64, power::Int)::Float64

Calculate ``\\frac{1}{\\beta} \\sum_{n \\in \\Omega_\\mathrm{shell}} \\frac{1}{(i \\nu_n)^power}``
`N-1` is the largest frequency index (i.e. ``\\sum_{n=-N}^(N-1) \nu_n`` is in the core region)
"""
shell_sum_fermionic(N::Int, β::Float64, power::Int) = (β / (2 * π * 1im))^(power) * zeta(power, N + 0.5)

"""
    core_sum_fermionic(N::Int, β::Float64, power::Int) 

Fast evaluation of ``\\sum_{n=0}^N \\frac{1}{(\\pi i (2n+1) / \\beta)^l}``
"""
core_sum_fermionic(N::Int, β::Float64, power::Int) =
    (β / (2 * π * 1im))^(power) * (zeta(power, 0.5) - zeta(power, N + 1.5))

"""
    core_sum_bosonic(N::Int, β::Float64, power::Int) 

Fast evaluation of ``\\sum_{n=1}^N \\frac{1}{(\\pi i (2n) / \\beta)^l}``
"""
core_sum_bosonic(N::Int, β::Float64, power::Int) = (β / (2 * π * 1im))^(power) * (zeta(power) - zeta(power, N + 1.0))

# =============================== Frequency Tail Modification Helpers ================================

"""
    subtract_tail(inp::AbstractArray{T,1}, c::Float64, iω::Array{ComplexF64,1}, power::Int) where T <: Number

subtract the ``\\frac{c}{(i\\omega)^\\text{power}}`` high frequency tail from input array `inp`.
"""
function subtract_tail(inp::AbstractVector, c::Float64, iω::Vector{ComplexF64}, power::Int)
    res = Array{eltype(inp),1}(undef, length(inp))
    subtract_tail!(res, inp, c, iω, power)
    return res
end

"""
    subtract_tail!(outp::AbstractArray{T,1}, inp::AbstractArray{T,1}, c::Float64, iω::Array{ComplexF64,1}, power::Int) where T <: Number

subtract the c/(iω)^power high frequency tail from `inp` and store in `outp`. See also [`subtract_tail`](@ref subtract_tail)
"""
function subtract_tail!(outp::AbstractVector, inp::AbstractVector, c::Float64, iω::Vector{ComplexF64}, power::Int)
    for n = eachindex(inp)
        if iω[n] != 0
            outp[n] = inp[n] - (c / (iω[n]^power))
        else
            outp[n] = inp[n]
        end
    end
end

"""
    ω_tail(ωindices::AbstractVector{Int}, coeffs::AbstractVector{Float64}, sP::SimulationParameters) 
    ω_tail(χ_sp::χT, χ_ch::χT, coeffs::AbstractVector{Float64}, β::Float64, sP::SimulationParameters) 

    
"""
function ω_tail(χ_sp::χT, χ_ch::χT, coeffs::AbstractVector{Float64}, sP::SimulationParameters)
    ωindices::UnitRange{Int} = (sP.dbg_full_eom_omega) ? (1:size(χ_sp, 2)) : intersect(χ_sp.usable_ω, χ_ch.usable_ω)
    ω_tail(ωindices, coeffs, χ_sp.β, sP)
end

function ω_tail(ωindices::AbstractArray{Int}, coeffs::AbstractVector{Float64}, β::Float64, sP::SimulationParameters)
    iωn = (1im .* 2 .* (-sP.n_iω:sP.n_iω)[ωindices] .* π ./ β)
    iωn[findfirst(x -> x ≈ 0, iωn)] = Inf
    for (i, ci) in enumerate(coeffs)
        χ_tail::Vector{Float64} = real.(ci ./ (iωn .^ i))
    end
end


# =================================== Estimation of Charge Order =====================================
"""
    BubbleDiff(χ₀::χ₀T, χ₀Loc::χ₀T; qList::Vector{Int}=eachindex(χ₀.data,χ₀.axis_types[:q]))

Computes ``\\chi^{-1,q,\\nu, \\omega_0}_0 - \\chi^{-1}_{0,\\nu,\\mathrm{loc}}`` for all 
q-vectors in `qList` (default: all).
Used to estimate charge order, see also `script_04`.
"""
function BubbleDiff(χ₀::χ₀T, χ₀Loc::χ₀T; qList::AbstractVector=axes(χ₀.data,χ₀.axis_types[:q]))
    ω₀LocInd = ω0_index(χ₀Loc[1,:,:])
    ω₀Ind    = ω0_index(χ₀[1,:,:])
    Nν       = size(core(χ₀), χ₀.axis_types[:ν])

    ev_list = Matrix{ComplexF64}(undef, length(qList), Nν)

    for qi in qList
        ev_list[qi, :] = 1 ./ core(χ₀)[qi,:,ω₀Ind] - 1 ./ core(χ₀Loc)[1,:,ω₀LocInd]
    end
    return ev_list
end


# ==================================== Fermi Surface Estimation ======================================
function lin_fit(ν::Vector{Float64}, Σ::Vector{Float64})
    m = (Σ[2] - Σ[1]) / (ν[2] - ν[1])
    return Σ[1] - m * ν[1]
end

function zero_freq(ν, Σ)
    return Σ[1]
end

"""
    fermi_surface_connected(ef_ind::BitVector, kG::KGrid, D::Int)

Checks for connected fermi surface of `kG` dimensions, given a `BitVector` of points on the fermi surface.
Returns `< 0` if fermi surface is not connected, `== 0` if it is exactly a line, `> 0` if the line is multiple points thick.
"""
function fermi_surface_connected(ef_ind::BitVector, kG::KGrid)
    D = typeof(kG).parameters[2]
    shift_indices = filter(x -> !all(iszero.(x)), collect(Base.product([[xi for xi in [-1, 0, 1]] for Di = 1:D]...))[:])
    ef_ind_exp = convert.(Bool, LadderDGA.expandKArr(kG, convert.(Float64, ef_ind)))
    kernel = sum([circshift(ef_ind_exp, s) for s in shift_indices])[ef_ind_exp]
    sum(kernel .- 2)
end


"""
    estimate_ef(Σ_ladder::OffsetMatrix, kG::KGrid, μ::Float64, β::Float64; ν0_estimator::Function=lin_fit, relax_zero_condition::Float64=10.0)

Estimate fermi surface of `Σ_ladder`, using extrapolation to ``\\nu = 0`` with the function `ν0_estimator` and the condition ``\\lim_{\\nu \\to 0} \\Sigma (\\nu, k_f) = \\mu - \\epsilon_{k_f}``.

"""
function estimate_ef(Σ_ladder::OffsetMatrix, kG::KGrid, μ::Float64, β::Float64; 
                     ν0_estimator::Function = lin_fit, relax_zero_condition::Float64 = 10.0,
)
    νGrid = [(2 * n + 1) * π / β for n = 0:1]
    s_r0 = [ν0_estimator(νGrid, real.(Σ_ladder[i, 0:1])) for i = axes(Σ_ladder, 1)]
    ekf = μ .- kG.ϵkGrid
    ek_diff = ekf .- s_r0
    min_diff = minimum(abs.(ekf .- s_r0))
    return abs.(ek_diff) .< relax_zero_condition * kG.Ns * min_diff
end

"""
    estimate_connected_ef(Σ_ladder::OffsetMatrix, kG::KGrid, μ::Float64, β::Float64; ν0_estimator::Function=lin_fit)

Estimates connected fermi surface. See also [`estimate_ef`](@ref estimate_ef) and [`fermi_surface_connected`](@ref fermi_surface_connected).
Returns fermi surface indices and `relax_zero_condition` (values substantially larger than `1` indicate appearance of fermi arcs).
"""
function estimate_connected_ef(Σ_ladder::OffsetMatrix, kG::KGrid, μ::Float64, β::Float64;
                               ν0_estimator::Function = lin_fit,
)
    ef = nothing
    rc_res = 0.0
    for rc = 0.1:0.1:20.0
        ef = estimate_ef(Σ_ladder, kG, μ, β; ν0_estimator = ν0_estimator, relax_zero_condition = rc)
        conn = fermi_surface_connected(ef.parent, kG)
        rc_res = rc
        conn >= 0 && break
    end
    return ef, rc_res
end
