# ==================================================================================================== #
#                                           helpers.jl                                                 #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#   General purpose helper functions for the ladder DΓA code.                                          #
# -------------------------------------------- TODO -------------------------------------------------- #
#   Add documentation for all functions                                                                #
#   !!!!Cleanup of setup function!!!!                                                                  #
# ==================================================================================================== #



# ========================================== Index Functions =========================================
"""
    νnGrid(ωn::Int, sP::SimulationParameters)

Calculates grid of fermionic Matsubara frequencies for given bosonic frequency `ωn` (including shift, if set through `sP`).
"""
νnGrid_shell(ωn::Int, sP::SimulationParameters) = ((-sP.n_iν-sP.n_iν_shell):(sP.n_iν+sP.n_iν_shell-1)) .- sP.shift * trunc(Int, ωn / 2)
νnGrid_noShell(ωn::Int, sP::SimulationParameters) = ((-sP.n_iν):(sP.n_iν-1)) .- sP.shift * trunc(Int, ωn / 2)

"""
    q0_index(kG::KGrid)   

Index of zero k-vector.
"""
q0_index(kG::KGrid) = findfirst(x -> all(x .≈ zeros(length(gridshape(kG)))), kG.kGrid)

"""
    ω0_index(sP::SimulationParameters)
    ω0_index(χ::[χT or AbstractMatrix])

Index of ω₀ frequency. 
"""
ω0_index(sP::SimulationParameters) = sP.n_iω + 1
ω0_index(χ::χT) = ω0_index(χ.data)
ω0_index(χ::χ₀T) = ω0_index(view(χ.data,1,:,:))
ω0_index(χ::AbstractMatrix) = ceil(Int64, size(χ, 2) / 2)

"""
    OneToIndex_to_Freq(ωi::Int, νi::Int, sP::SimulationParameters)
    OneToIndex_to_Freq(ωi::Int, νi::Int, shift::Int, nBose::Int, nFermi::Int)

Converts `(1:N,1:N)` index tuple for bosonic (`ωi`) and fermionic (`νi`) frequency to
Matsubara frequency number. If the array has a `ν` shell (for example for tail
improvements) this will also be taken into account by providing `Nν_shell`.
This is the inverse function of [`Freq_to_OneToIndex`](@ref Freq_to_OneToIndex).
"""
function OneToIndex_to_Freq(ωi::Int, νi::Int, sP::SimulationParameters)
    OneToIndex_to_Freq(ωi, νi, sP.shift, sP.n_iω, sP.n_iν)
end

function OneToIndex_to_Freq(ωi::Int, νi::Int, shift::Union{Bool,Int}, nBose::Int, nFermi::Int)
    ωn = ωi - nBose - 1
    νn = (νi - nFermi - 1) - shift * trunc(Int, ωn / 2)
    return ωn, νn
end

"""
    OneToIndex_to_Freq(ωi::Int, νpi::Int, νi::Int, sP::SimulationParameters)
    Freq_to_OneToIndex(ωn::Int, νn::Int, νpn::Int, shift::Int, nBose::Int, nFermi::Int)

Converts Matsubara frequency index to array indices, starting at 1.
This is the inverse function of [`OneToIndex_to_Freq`](@ref OneToIndex_to_Freq).
"""
function OneToIndex_to_Freq(ωi::Int, νpi::Int, νi::Int, sP::SimulationParameters)
    Freq_to_OneToIndex(ωi, νpi, νi, sP.shift, sP.n_iω, sP.n_iν)
end

function Freq_to_OneToIndex(ωn::Int, νn::Int, νpn::Int, shift::Union{Bool,Int}, nBose::Int, nFermi::Int)
    ωn + nBose + 1, νn + nFermi + 1 + trunc(Int, shift * ωn / 2), νpn + nFermi + 1 + trunc(Int, shift * ωn / 2)
end

"""
    ν0Index_of_ωIndex(ωi::Int[, sP])::Int

Calculates index of zero fermionic Matsubara frequency (which may depend on the bosonic frequency). 
`ωi` is the index (i.e. starting with 1) of the bosonic Matsubara frequency.
"""
ν0Index_of_ωIndex(ωi::Int, sP::SimulationParameters)::Int = sP.n_iν + sP.shift * (trunc(Int, (ωi - sP.n_iω - 1) / 2)) + 1

"""
    νi_νngrid_pos(ωi::Int, νmax::Int, sP::SimulationParameters)

Indices for positive fermionic Matsubara frequencies, depinding on `ωi`, the index of the bosonic Matsubara frequency.
"""
function νi_νngrid_pos(ωi::Int, νmax::Int, sP::SimulationParameters)
    ν0Index_of_ωIndex(ωi, sP):νmax
end


"""
    get_val_or_zero(arr::Vector{T}, ind::Int)::T where T

Returns value at index, if index inside index range of `arr::Vector{T}`, `zero(T)` otherwise. 
"""
function get_val_or_zero(arr::Vector{T}, ind::Int)::T where T
    return ind ∈ axes(arr,1) ? arr[ind] : zero(T) 
end

# =========================================== Noise Filter ===========================================
"""
    filter_MA(m::Int, X::AbstractArray{T,1}) where T <: Number
    filter_MA!(res::AbstractArray{T,1}, m::Int, X::AbstractArray{T,1}) where T <: Number

Iterated moving average noise filter for inut data. See also [`filter_KZ`](@ref filter_KZ).
"""
function filter_MA(m::Int, X::AbstractArray{T,1}) where {T<:Number}
    res = deepcopy(X)
    offset = trunc(Int, m / 2)
    res[1+offset] = sum(@view X[1:m]) / m
    for (ii, i) in enumerate((2+offset):(length(X)-offset))
        res[i] = res[i-1] + (X[m+ii] - X[ii]) / m
    end
    return res
end

function filter_MA!(res::AbstractArray{T,1}, m::Int, X::AbstractArray{T,1}) where {T<:Number}
    offset = trunc(Int, m / 2)
    res[1+offset] = sum(@view X[1:m]) / m
    for (ii, i) in enumerate((2+offset):(length(X)-offset))
        res[i] = res[i-1] + (X[m+ii] - X[ii]) / m
    end
    return res
end

"""
    filter_KZ(m::Int, k::Int, X::AbstractArray{T,1}) where T <: Number

Iterated moving average noise filter for inut data. See also [`filter_MA`](@ref filter_MA).
"""
function filter_KZ(m::Int, k::Int, X::AbstractArray{T,1}) where {T<:Number}
    res = filter_MA(m, X)
    for ki = 2:k
        res = filter_MA!(res, m, res)
    end
    return res
end

# ======================================== Consistency Checks ========================================
"""
    νi_health(νGrid::AbstractArray{Int}, sP::SimulationParameters)

Returns a list of available bosonic frequencies for each fermionic frequency, given in `νGrid`.
This can be used to estimate the maximum number of usefull frequencies for the equation of motion.
"""
function νi_health(νGrid::AbstractArray{Int}, sP::SimulationParameters)
    t = gen_ν_part(νGrid, sP, 1)[1]
    return [length(filter(x -> x[4] == i, t)) for i in unique(getindex.(t, 4))]
end
# ============================================== Misc. ===============================================
"""
    reduce_range(range::AbstractArray, red_prct::Float64)

Returns indices for 1D array slice, reduced by `red_prct` % (compared to initial `range`).
Range is symmetrically reduced fro mstart and end.
"""
function reduce_range(range::AbstractArray, red_prct::Float64)
    sub = floor(Int, length(range) / 2 * red_prct)
    lst = maximum([last(range) - sub, ceil(Int, length(range) / 2 + iseven(length(range)))])
    fst = minimum([first(range) + sub, ceil(Int, length(range) / 2)])
    return fst:lst
end

