# ==================================================================================================== #
#                                           GFTools.jl                                                 #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
#   Last Edit Date  : 01.09.22                                                                         #
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

iν_array(β::Real, grid::AbstractArray{Int64,1}) = ComplexF64[1.0im*((2.0 *el + 1)* π/β) for el in grid]
iν_array(β::Real, size::Integer)    = ComplexF64[1.0im*((2.0 *i + 1)* π/β) for i in 0:size-1]
iω_array(β::Real, grid::AbstractArray{Int64,1}) = ComplexF64[1.0im*((2.0 *el)* π/β) for el in grid]
iω_array(β::Real, size::Integer)    = ComplexF64[1.0im*((2.0 *i)* π/β) for i in 0:size-1]


# ===================================== Dyson Equations Helpers ======================================

"""
    G_from_Σ(ind::Int64, Σ::Array{ComplexF64,[1,2,3]}, ϵkGrid, β, μ)
    G_from_Σ(freq::[Int64 or ComplexF64], β::Float64, μ::Float64, ϵₖ::Float64, Σ::ComplexF64)

Constructs GF from k-independent self energy, using the Dyson equation
and the dispersion relation of the lattice.
"""
@inline function G_from_Σ(ind::Int64, Σ::Vector{ComplexF64}, 
        ϵkGrid::Union{Array{Float64,1},Base.Generator}, β::Float64, μ::Float64)
    Σν = get_symm_f(Σ,ind)
    return map(ϵk -> G_from_Σ(ind, β, μ, ϵk, Σν), ϵkGrid)
end

@inline function G_from_Σ(ind::Int64, Σ::Array{ComplexF64,2},
                   ϵkGrid, β::Float64, μ::Float64)
    Σνk = get_symm_f_2(Σ,ind)
    return reshape(map(((ϵk, Σνk_i),) -> G_from_Σ(ind, β, μ, ϵk, Σνk_i), zip(ϵkGrid, Σνk)), size(ϵkGrid)...)
end

@inline @fastmath G_from_Σ(ind::Int64, β::Float64, μ::Float64, ϵₖ::Float64, Σ::ComplexF64) =
                    1/((π/β)*(2*ind + 1)*1im + μ - ϵₖ - Σ)
@inline @fastmath G_from_Σ(mf::ComplexF64, μ::Float64, ϵₖ::Float64, Σ::ComplexF64) =
                    1/(mf + μ - ϵₖ - Σ)

function G_from_Σ(Σ::AbstractArray, ϵkGrid::AbstractArray, range::AbstractVector{Int}, mP::ModelParameters; μ = mP.μ ,  Σloc = nothing) 
    res = Array{ComplexF64,2}(undef, length(ϵkGrid), length(range))
    for (i,ind) in enumerate(range)
        res[:,i] = abs(ind) < size(Σ,2) ? G_from_Σ(ind, Σ, ϵkGrid, mP.β, μ) : G_from_Σ(ind, Σloc, ϵkGrid, mP.β, μ)
    end
    return res
end


Σ_Dyson(GBath::Array{ComplexF64,1}, GImp::Array{ComplexF64,1}, eps = 1e-3) =
    Σ::Array{ComplexF64,1} =  1 ./ GBath .- 1 ./ GImp

#TODO: test/documentation (see SC and Sigma Tails notebook)
function filling(G, νn_list, kG::KGrid, β::Float64)
    n = 0.0
    @assert length(νn_list) == size(G,2)
    for (νi, νn) in enumerate(νn_list)
        n += kintegrate(kG, G[:,νi],1)[1] - 1.0/νn
    end
    n = 2*n/β + 0.5*2
end


function GLoc_from_Σladder(Σ_ladder, Σloc, kG::KGrid, mP::ModelParameters, sP::SimulationParameters)
    νRange = sP.fft_range
    #gLoc_red = G_from_Σ(Σ_ladder.parent, kG.ϵkGrid, νRange, mP; Σloc=Σloc);
    νn = iν_array(mP.β, νRange);

    function fix_μ(μ::Vector{Float64})
        gLoc_red = G_from_Σ(Σ_ladder.parent, kG.ϵkGrid, collect(νRange), mP; μ = μ[1], Σloc=Σloc);
        real(filling(gLoc_red, νn, kG, mP.β) - mP.n)
    end
    #res = nlsolve(fix_μ, [mP.μ])
    μ = mP.μ # res.zero[1]
    gLoc_red = G_from_Σ(Σ_ladder.parent, kG.ϵkGrid, collect(νRange), mP; μ = μ, Σloc=Σloc);
    gLoc_red = OffsetArray(gLoc_red, 1:length(kG.ϵkGrid), νRange)
    gLoc_fft = OffsetArray(Array{ComplexF64,2}(undef, kG.Nk, length(sP.fft_range)), 1:kG.Nk, sP.fft_range)
    gLoc_rfft = OffsetArray(Array{ComplexF64,2}(undef, kG.Nk, length(sP.fft_range)), 1:kG.Nk, sP.fft_range)
    ϵk_full = expandKArr(kG, kG.ϵkGrid)[:]
    for νi in sP.fft_range
        GLoc_νi  = expandKArr(kG, gLoc_red[:,νi].parent)
        gLoc_fft[:,νi] .= fft(GLoc_νi)[:]
        gLoc_rfft[:,νi] .= fft(reverse(GLoc_νi))[:]
    end
    return μ, gLoc_red, gLoc_fft, gLoc_rfft
end

# =============================== Frequency Tail Modification Helpers ================================

"""
    subtract_tail(inp::AbstractArray{T,1}, c::Float64, iω::Array{ComplexF64,1}) where T <: Number

subtract the c/(iω)^2 high frequency tail from `inp`.
"""
function subtract_tail(inp::AbstractArray{T,1}, c::Float64, iω::Array{ComplexF64,1}) where T <: Number
    res = Array{eltype(inp),1}(undef, length(inp))
    subtract_tail!(res, inp, c, iω)
    return res
end

"""
    subtract_tail!(outp::AbstractArray{T,1}, inp::AbstractArray{T,1}, c::Float64, iω::Array{ComplexF64,1}) where T <: Number

subtract the c/(iω)^2 high frequency tail from `inp` and store in `outp`. See also [`subtract_tail`](@ref subtract_tail)
"""
function subtract_tail!(outp::AbstractArray{T,1}, inp::AbstractArray{T,1}, c::Float64, iω::Array{ComplexF64,1}) where T <: Number
    for n in 1:length(inp)
        if iω[n] != 0
            outp[n] = inp[n] - (c/(iω[n]^2))
        else
            outp[n] = inp[n]
        end
    end
end

