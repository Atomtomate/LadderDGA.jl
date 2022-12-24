# ==================================================================================================== #
#                                           GFTools.jl                                                 #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#   Green's function and Matsubara frequency related functions                                         #
# -------------------------------------------- TODO -------------------------------------------------- #
#   Documentation                                                                                      #
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
iν_array(β::Real, grid::AbstractArray{Int64,1})::Vector{ComplexF64} = ComplexF64[1.0im*((2.0 *el + 1)* π/β) for el in grid]
iν_array(β::Real, size::Int)::Vector{ComplexF64} = iν_array(β, 0:(size-1))

"""
    iω_array(β::Real, grid::AbstractArray{Int64,1})::Vector{ComplexF64}
    iω_array(β::Real, size::Int)::Vector{ComplexF64}

Computes list of bosonic Matsubara frequencies.
If length `size` is given, the grid will have indices `0:size-1`.
Fermionic arrays can be generated with [`iν_array`](iν_array).

Returns: 
-------------
Vector of bosonic Matsubara frequencies, given either a list of indices or a length. 
"""
iω_array(β::Real, grid::AbstractArray{Int64,1})::Vector{ComplexF64}  = ComplexF64[1.0im*((2.0 *el)* π/β) for el in grid]
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
- **`νₙ`** : Vector of fermionic Matsubara frequencies, see also: [`iν_array`](iν_array).

"""
Δ(ϵₖ::Vector{Float64}, Vₖ::Vector{Float64}, νₙ::Vector{ComplexF64})::Vector{ComplexF64} = [sum((Vₖ .* conj.(Vₖ)) ./ (ν .- ϵₖ)) for ν in νₙ]

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
G_from_Σ(ind::Int64, β::Float64, μ::Float64, ϵₖ::Float64, Σ::ComplexF64)::ComplexF64 = G_from_Σ(1im*(2*ind + 1)*π/β, μ, ϵₖ, Σ)
G_from_Σ(mf::ComplexF64        , μ::Float64, ϵₖ::Float64, Σ::ComplexF64)::ComplexF64 = 1/(mf + μ - ϵₖ - Σ)
                    

"""
    G_from_Σ(Σ::AbstractVector{ComplexF64}, ϵkGrid::Vector{Float64}, range::AbstractVector{Int}, mP::ModelParameters; μ = mP.μ,  Σloc::AbstractArray = nothing) 
    G_from_Σ!(res::Matrix{ComplexF64}, Σ::AbstractVector{ComplexF64}, ϵkGrid::Vector{Float64}, range::AbstractVector{Int}, mP::ModelParameters; μ = mP.μ,  Σloc::AbstractVector = nothing) 

Computes Green's function from self energy `Σ` and dispersion `ϵkGrid` over given frequency indices `range`.
Optionally, a different chemical potential `μ` can be provided.
When the non-local self energy is used, one typically wants to extend the usefull range of frequencies by
attaching the tail of the local self energy in the high frequency regime. This is done by providing a
`range` larger than the array size of `Σ` and in addition setting `Σloc` (the size of `Σloc` must be as large or larger than `range`). 
The inplace version stores the result in `res`.
"""
function G_from_Σ(Σ::Vector{ComplexF64}, ϵkGrid::Vector{Float64}, range::AbstractVector{Int}, mP::ModelParameters; μ = mP.μ,  Σloc::Vector{ComplexF64} = ComplexF64[]) 
    res = Array{ComplexF64,2}(undef, length(ϵkGrid), length(range))
    G_from_Σ!(res, Σ, ϵkGrid, range, mP, μ = μ,  Σloc = Σloc)
    return res
end

function G_from_Σ!(res::Matrix{ComplexF64}, Σ::Vector{ComplexF64}, ϵkGrid::Vector{Float64}, range::AbstractVector{Int}, mP::ModelParameters; μ = mP.μ,  Σloc::Vector{ComplexF64} = ComplexF64[]) 
    first(range) != 0 && error("G_from_Σ only implemented for range == 0:νmax!")
    for (i,ind) in enumerate(range)
        Σi = i <= length(Σ) ? Σ[i] : Σloc[i]
        for (ki, ϵk) in enumerate(ϵkGrid)
            @inbounds res[ki,i] = G_from_Σ(ind, mP.β, μ, ϵk, Σi)
        end
    end
    return nothing
end

function G_from_Σ!(res::Matrix{ComplexF64}, Σ::Matrix{ComplexF64}, ϵkGrid::Vector{Float64}, range::AbstractVector{Int}, mP::ModelParameters; μ = mP.μ,  Σloc::Vector{ComplexF64} = ComplexF64[]) 
    first(range) != 0 && error("G_from_Σ only implemented for range == 0:νmax!")
    for (i,ind) in enumerate(range)
        for (ki, ϵk) in enumerate(ϵkGrid)
            Σi = i < size(Σ,2) ? Σ[ki, i] : Σloc[i]
            @inbounds res[ki,i] = G_from_Σ(ind, mP.β, μ, ϵk, Σi)
        end
    end
    return nothing
end

# =============================================== GLoc ===============================================
#TODO: docs, test, cleanup, consistency with G_from_Σ 
function G_from_Σladder(Σ_ladder::OffsetMatrix{ComplexF64}, Σloc::Vector{ComplexF64}, kG::KGrid, mP::ModelParameters, sP::SimulationParameters;
                        fix_n::Bool=false)
    νRange = 0:last(sP.fft_range)
    Σloc_part = Σloc[1:last(νRange)+1]
    GLoc_new = Matrix{ComplexF64}(undef, size(Σ_ladder,1), length(νRange))

    function fμ(μ::Vector{Float64})
        G_from_Σ!(GLoc_new, Σ_ladder.parent, kG.ϵkGrid, νRange, mP, μ=μ[1], Σloc = Σloc_part)
        filling_pos(GLoc_new, kG, mP.U, μ[1], mP.β) - mP.n
    end
    μ_new_nls = try 
        nlsolve(fμ, [mP.μ])
    catch e
        @warn "μ determination failed with: $e"
        return NaN, nothing, nothing, nothing
    end
    μ = if !μ_new_nls.f_converged 
        @warn "Could not determine μ!"
        @warn μ_new_nls
        μ = NaN
    else
        μ_new_nls.zero[1]
    end
    gLoc_fft = nothing
    gLoc_rfft = nothing
    if fix_n && !isnan(μ)
        gLoc_fft = OffsetArray(Array{ComplexF64,2}(undef, kG.Nk, length(sP.fft_range)), 1:kG.Nk, sP.fft_range)
        gLoc_rfft = OffsetArray(Array{ComplexF64,2}(undef, kG.Nk, length(sP.fft_range)), 1:kG.Nk, sP.fft_range)
        mP.μ = μ
        G_from_Σ!(GLoc_new, Σ_ladder.parent, kG.ϵkGrid, νRange, mP, μ=mP.μ, Σloc = Σloc_part)
        for νi in sP.fft_range
            GLoc_νi  = νi < 0 ? expandKArr(kG, conj(GLoc_new[:,-νi])) : expandKArr(kG, GLoc_new[:,νi+1])
            gLoc_fft[:,νi] .= fft(GLoc_νi)[:]
            gLoc_rfft[:,νi] .= fft(reverse(GLoc_νi))[:]
        end
    end
    return μ, OffsetArray(GLoc_new, 1:size(GLoc_new,1), νRange), gLoc_fft, gLoc_rfft
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
    
function Σ_Dyson!(Σ::AbstractVector{ComplexF64}, GBath::Vector{ComplexF64}, GImp::Vector{ComplexF64})
    Σ[:] =  1 ./ GBath .- 1 ./ GImp
    return nothing
end


# ============================================= Filling ==============================================
"""
    filling(G::Vector{ComplexF64}, [kG::KGrid, ] β::Float64)
    filling(G::Vector, U::Float64, μ::Float64, β::Float64, [shell::Float64])

Computes filling of (non-) local Green's function.

If `U`, `μ` and `β` are provided, asymptotic corrections are used. The shell sum can be precomputed using [`G_shell_sum`](@ref G_shell_sum)
If `G` is defined only over positive Matsubara frequencies [`filling_pos`](@ref filling_pos) can be used.
"""
function filling(G::Vector{ComplexF64}, β::Float64)
    n = 2*sum(G)/β + 1
    imag(n) > 1e-8 && throw("Error: Imaginary part of filling is larger than 10^-8")
    return real(n)
end

function filling(G::Matrix{ComplexF64}, kG::KGrid, β::Float64)
    n = 2*sum(kintegrate(kG, G, 1))/β + 1
    imag(n) > 1e-8 && throw("Error: Imaginary part of filling is larger than 10^-8")
    return real(n)
end

function filling(G::Vector{ComplexF64}, U::Float64, μ::Float64, β::Float64, shell::Float64)::Float64
    2*(real(sum(G))/β + 0.5 + μ * shell) / (1 + U * shell)
end

function filling(G::Vector{ComplexF64}, U::Float64, μ::Float64, β::Float64)
    N = floor(Int, length(G)/2)
    shell = G_shell_sum(N, β)
    filling(G, U, μ, β, shell)
end

function filling(G::Matrix{ComplexF64}, kG::KGrid, U::Float64, μ::Float64, β::Float64)
    filling(kintegrate(kG,G,1)[1,:], U, μ, β)
end

"""
    filling_pos(G::Vector, U::Float64, μ::Float64, β::Float64[, shell::Float64])::Float64

Returns filling from `G` only defined over positive Matsubara frequencies. 
See [`filling`](@ref filling) for further documentation.
"""
function filling_pos(G::Vector{ComplexF64}, U::Float64, μ::Float64, β::Float64, shell::Float64)::Float64
    sG = sum(G)
    2*(real(sG + conj(sG))/β + 0.5 + μ * shell) / (1 + U * shell)
end

function filling_pos(G::Vector{ComplexF64}, U::Float64, μ::Float64, β::Float64)::Float64
    N = floor(Int, length(G)/2)
    shell = G_shell_sum(N, β)
    filling_pos(G, U, μ, β, shell)
end

function filling_pos(G::Matrix{ComplexF64}, kG::KGrid, U::Float64, μ::Float64, β::Float64)
    filling_pos(kintegrate(kG,G,1)[1,:], U, μ, β)
end

"""
    G_shell_sum(N::Int, β::Float64)::Float64

Calculate ``\\frac{1}{\\beta} \\sum_{n \\in \\Omega_\\mathrm{shell}} \\frac{1}{(i \\nu_n)^2}``
`N` should be the index of the largest frequency + 1, NOT the total lenth of the array, i.e. `50` for `indices = 0:49`.
"""
function G_shell_sum(N::Int, β::Float64)::Float64
    (polygamma(1, N + 1/2) - polygamma(1, 1/2 - N)) * β / (4*π^2) + β/4
end


# =============================== Frequency Tail Modification Helpers ================================

"""
    subtract_tail(inp::AbstractArray{T,1}, c::Float64, iω::Array{ComplexF64,1}, power::Int) where T <: Number

subtract the ``\\frac{c}{(i\\omega)^\\text{power}}`` high frequency tail from input array `inp`.
"""
function subtract_tail(inp::AbstractArray{T,1}, c::Float64, iω::Array{ComplexF64,1}, power::Int) where T <: Number
    res = Array{eltype(inp),1}(undef, length(inp))
    subtract_tail!(res, inp, c, iω, power)
    return res
end

"""
    subtract_tail!(outp::AbstractArray{T,1}, inp::AbstractArray{T,1}, c::Float64, iω::Array{ComplexF64,1}, power::Int) where T <: Number

subtract the c/(iω)^power high frequency tail from `inp` and store in `outp`. See also [`subtract_tail`](@ref subtract_tail)
"""
function subtract_tail!(outp::AbstractArray{T,1}, inp::AbstractArray{T,1}, c::Float64, iω::Array{ComplexF64,1}, power::Int) where T <: Number
    for n in 1:length(inp)
        if iω[n] != 0
            outp[n] = inp[n] - (c/(iω[n]^power))
        else
            outp[n] = inp[n]
        end
    end
end

"""
    ω_tail(ωindices::AbstractVector{Int}, coeffs::AbstractVector{Float64}, sP::SimulationParameters) 
    ω_tail(χ_sp::χT, χ_ch::χT, coeffs::AbstractVector{Float64}, sP::SimulationParameters) 


"""
function ω_tail(χ_sp::χT, χ_ch::χT, coeffs::AbstractVector{Float64}, sP::SimulationParameters) 
    ωindices::UnitRange{Int} = (sP.dbg_full_eom_omega) ? (1:size(χ,2)) : intersect(χ_sp.usable_ω, χ_ch.usable_ω)
    ω_tail(ωindices, coeffs, sP) 
end
function ω_tail(ωindices::AbstractArray{Int}, coeffs::AbstractVector{Float64}, sP::SimulationParameters) 
    iωn = (1im .* 2 .* (-sP.n_iω:sP.n_iω)[ωindices] .* π ./ mP.β)
    iωn[findfirst(x->x ≈ 0, iωn)] = Inf
    for (i,ci) in enumerate(coeffs)
        χ_tail::Vector{Float64} = real.(ci ./ (iωn.^i))
    end
end

#TODO: sum_ω functions
# sum_ω(inp::AbstractVector, tail_c::)
