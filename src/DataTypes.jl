# ==================================================================================================== #
#                                           DataTypes.jl                                               #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#   Data types for Matsubara functions.                                                                #
# -------------------------------------------- TODO -------------------------------------------------- #
#   define getproperty to mask internals of MatsubaraFunction                                          #
#      Base.propertynames(F::TYPE, private::Bool=false) = ... see LU type (lu.jl 332)                  #
#   Σ data type                                                                                        #
#   Struct with asymptotics for Green's fnctions                                                       #
#   χ₀: calculate t1,t2 of bubble from GFtails (first: define GF struct)                               #
# ==================================================================================================== #


import Base.copy
const ω_axis = 3;
const ν_axis = 2;
const q_axis = 1;

# =========================================== Static Types ===========================================
const _eltype = ComplexF64
const ΓT = Array{_eltype,3}
const FT = Array{_eltype,3}
const GνqT = OffsetMatrix{ComplexF64, Matrix{ComplexF64}}

abstract type MatsubaraFunction{T,N} <: AbstractArray{T,N} end


# =========================================== Struct Types ===========================================
"""
    χ₀T <: MatsubaraFunction

Struct for the bubble term. The `q`, `ω` dependent asymptotic behavior is computed from the 
`t1` and `t2` input.

Constructor
-------------
    χ₀T(data::Array{_eltype,3}, kG::KGrid, ω_grid::AbstractVector{Int}, n_iν::Int, shift::Bool, mP::ModelParameters; local_tail=false)

Set `local_tail=true` in case of the local bubble constructed fro mthe impurity Green's function. This is necessary in order to construct the correct asymptotics.


Fields
-------------
- **`data`**         : `Array{ComplexF64,3}`, data
- **`asym`**         : `Array{ComplexF64,2}`, `[q, ω]` dependent asymptotic behavior.
- **`axis_types`**   : `Dict{Symbol,Int}`, Dictionary mapping `:q, :ν, :ω` to the axis indices.
"""
struct χ₀T <: MatsubaraFunction{_eltype,3}
    data::Array{_eltype,3}
    asym::Array{_eltype,2}
    axis_types::Dict{Symbol, Int}
    function χ₀T(data::Array{_eltype,3}, kG::KGrid, ω_grid::AbstractVector{Int}, n_iν::Int,
                 shift::Bool, mP::ModelParameters; local_tail=false)
        c1, c2, c3 = χ₀Asym_coeffs(kG, local_tail, mP)
        asym = χ₀Asym(c1, c2, c3, ω_grid, n_iν, shift, mP.β)
        new(data,asym,Dict(:q => 1, :ν => 2, :ω => 3))
    end
end

"""
    χ₀Asym(c1::Float64, c2::Vector{Float64}, c3::Float64, ω_grid::AbstractVector{Int}, n_iν::Int, shift::Int, β::Float64)

Builds asymtotic helper array. See [`calc_bubble`](@ref calc_bubble) implementation for details.

`c1`, `c2` and `c3` are the coefficients for the asymtotic tail expansion and can be obtained through [`χ₀Asym_coeffs`](@ref χ₀Asym_coeffs`).
`n_iν` is the number of positive fermionic Matsubara frequencies, `shift` is either `1` or `0`, depending on the type of frequency grid.
"""
function χ₀Asym(c1::Float64, c2::Vector{Float64}, c3::Float64,
                ω_grid::AbstractVector{Int}, n_iν::Int, shift::Bool, β::Float64)
    if length(ω_grid) == 0 || length(first(ω_grid):last(ω_grid)) == 0
        throw(ArgumentError("Cannot construct χ₀ array with empty frequency mesh!"))
    end
    χ₀_rest = χ₀_shell_sum_core(β, first(ω_grid):last(ω_grid), n_iν, Int(shift))
    asym = Array{_eltype, 2}(undef, length(c2), length(ω_grid))
    for (ωi,ωn) in enumerate(ω_grid)
        for (qi,c2i) in enumerate(c2)
            asym[qi,ωi] = χ₀_shell_sum(χ₀_rest, ωn, β, c1, c2i, c3)
        end
    end
    return asym
end

"""
    χ₀Asym_coeffs(kG::KGrid, local_tail::Bool, mP::ModelParameters)

Builds tail coefficients for the χ₀ asymptotic helper, obtained through [`χ₀Asym`](@ref χ₀Asym).
"""
function χ₀Asym_coeffs(kG::KGrid, local_tail::Bool, mP::ModelParameters)
    t1,  t2 = if local_tail
        convert.(ComplexF64, [mP.U*mP.n/2 - mP.μ]),
        mP.sVk + (mP.U^2)*(mP.n/2)*(1-mP.n/2)
    else
        convert.(ComplexF64, kG.ϵkGrid .+ mP.U*mP.n/2 .- mP.μ),
        (mP.U^2)*(mP.n/2)*(1-mP.n/2)
    end
    c1 = real.(kintegrate(kG, t1))
    c2 = real.(conv_noPlan(kG, t1, t1))
    c3 = real.(kintegrate(kG, t1 .^ 2) .+ t2)
    return c1, c2, c3
end


"""
    χT <: MatsubaraFunction

Struct for the non-local susceptibilities. 

Constructor
-------------
`χT(data::Array{T, 2}; full_range=true, reduce_range_prct=0.1)`: if `full_range` is set to `false`, the usable range 
is determined via [`find_usable_χ_interval`](@ref find_usable_χ_interval).

Fields
-------------
- **`data`**         : `Array{ComplexF64,3}`, data
- **`axis_types`**   : `Dict{Symbol,Int}`, Dictionary mapping `:q, :ω` to the axis indices.
- **`indices_ω`**    : `Vector{Int}`, 
- **`λ`**            : `Float64`, λ correction parameter.
- **`usable_ω`**     : `AbstractArray{Int}`, usable indices for which data is assumed to be correct. See also [`find_usable_interval`](@ref find_usable_interval)
"""
mutable struct χT <: MatsubaraFunction{_eltype, 2}
    data::Matrix
    axis_types::Dict{Symbol, Int}
    indices_ω::AbstractVector{Int}
    tail_c::Vector{Float64}
    λ::Float64
    β::Float64
    usable_ω::AbstractArray{Int}

    function χT(data::Array{T, 2}, β::Float64;
                indices_ω::AbstractVector{Int}=axes(data,2) .- ceil(Int,size(data,2)/2), 
                tail_c::Vector{Float64} = Float64[], full_range=true, reduce_range_prct=0.0) where T <: Union{ComplexF64, Float64}
        range = full_range ? (1:size(data,2)) : find_usable_χ_interval(data, reduce_range_prct=reduce_range_prct)
        new(data, Dict(:q => 1, :ω => 2), indices_ω, tail_c, 0.0, β, range)
    end
end


"""
    γT <: MatsubaraFunction

Struct for the non-local triangular vertex. 

Fields
-------------
- **`data`**         : `Array{ComplexF64,3}`, data
- **`axis_types`**   : `Dict{Symbol,Int}`, Dictionary mapping `:q, :ν, :ω` to the axis indices.
"""
struct γT <: MatsubaraFunction{ComplexF64,3}
    data::Array{ComplexF64,3}
    axis_types::Dict{Symbol, Int}
    function γT(data::Array{ComplexF64, 3})
        new(data, Dict(:q => 1, :ν => 2, :ω => 3))
    end
end

# ============================================= Interface ============================================
# ------------------------------------------- Custom Helpers -----------------------------------------
"""
    ωn_grid(χ::χT)

Computes bosonic frequencies for `χ`: ``2 i \\pi n / \\beta``.
"""
ωn_grid(χ::χT) = 2im .* π .* χ.indices_ω ./ χ.β


"""
    sum_ω(χ::χT)
    sum_ω!(res::Vector{ComplexF64}, work::Vector{ComplexF64}, ωn_arr::Vector{ComplexF64}, χ::χT; force_full_range=false)::Nothing
    sum_ω!(work::Vector{T}, ωn_arr::Vector{T}, χ::AbstractVector{T}, tail_c::Vector{Float64}, β::Float64; force_full_range=false)::T where T <: Union{Float64,ComplexF64}

Sums the physical susceptibility over all usable (if `force_full_range` is not set to `true`) bosonic frequencies, including improvd tail summation, if `χ.tail_c` is set.
"""
function sum_ω(χ::χT; force_full_range=false)
    work = Vector{eltype(χ.data)}(undef, size(χ.data, χ.axis_types[:ω]))
    res  = Vector{eltype(χ.data)}(undef, size(χ.data, χ.axis_types[:q]))
    ωn_arr = ωn_grid(χ)
    sum_ω!(res, work, ωn_arr, χ; force_full_range=force_full_range)
    return res
end

function sum_ω!(res::Vector{ComplexF64}, work::Vector{ComplexF64}, ωn_arr::Vector{ComplexF64}, χ::χT; force_full_range=false)::Nothing
    length(res)  != size(χ,χ.axis_types[:q]) && error("Result array for ω summation must be of :q axis length!")
    ω_slice = 1:size(χ, χ.axis_types[:ω])
    ω_slice = force_full_range ? ω_slice : ω_slice[χ.usable_ω]
    #for (p,ci) in enumerate(χ.tail_c)
    #    if ci != 0
    #    end
    #end
    i = 3
    for qi in eachindex(res)
        subtract_tail!(work, view(χ.data, qi,ω_slice), χ.tail_c[i], ωn_arr, i-1)
        res[qi] = sum_ω!(work, ωn_arr, view(χ,qi,ω_slice), χ.tail_c, χ.β::Float64)
    end
    return nothing
end

function sum_ω!(work::Vector{T}, ωn_arr::Vector{T}, χ::AbstractVector{T}, tail_c::Vector{Float64}, β::Float64)::T where T <: Union{Float64,ComplexF64}
    length(work) != length(χ) && error("Work array for ω summation must be of :ω axis length!")
    !(sum(tail_c) ≈ tail_c[3]) && error("sum_ω only implemented for 1/ω^2, but other tail coefficients are non-zero!")
    limits = [Inf, Inf, -β^2/12] # -i 1.202056903*β^3/(8*π^3), β^4/720
    
    i = 3
    subtract_tail!(work, χ, tail_c[i], ωn_arr, i-1)
    return (sum(work) - limits[i])/β
end

"""
    update_tail!(χ::χT, new_tail_c::Array{Float64}, ωnGrid::Array{ComplexF64})

Updates the ``\\frac{c_i}{\\omega_n^i}`` tail for all coefficients given in `new_tail_c` (index 1 corresponds to ``i=0``).
"""
function update_tail!(χ::χT, new_tail_c::Array{Float64}, ωnGrid::Array{ComplexF64})
    length(new_tail_c) != length(χ.tail_c) && throw(ArgumentError("length of new tail coefficient array ($(length(new_tail_c))) must be the same as the old one ($(length(χ.tail_c)))!"))
    length(ωnGrid) != size(χ.data,2) && throw(ArgumentError("length of ωnGrid ($(length(ωnGrid))) must be the same as for χ ($(size(χ.data,2)))!"))
    !all(isfinite.(new_tail_c)) && throw(ArgumentError("One or more tail coefficients are not finite!"))
    λ_bak = χ.λ
    λ_bak != 0 && reset!(χ)
    update_tail!(χ.data, χ.tail_c, new_tail_c, ωnGrid)

    for ci in 2:length(new_tail_c)
        if !(new_tail_c[ci] == χ.tail_c[ci])
            χ.tail_c[ci] = new_tail_c[ci]
        end
    end
    χ_λ!(χ, λ_bak)
end

function update_tail!(χ::AbstractMatrix, old_tail_c::Vector{Float64}, new_tail_c::Vector{Float64}, ωnGrid::Vector{ComplexF64})
    zero_ind = findall(x->x==0, ωnGrid) 
    for ci in 2:length(new_tail_c)
        if !(new_tail_c[ci] == old_tail_c[ci])
            tmp = (new_tail_c[ci] - old_tail_c[ci]) ./ (ωnGrid .^ (ci-1))  
            tmp[zero_ind] .= 0
            for qi in 1:size(χ,1)
                χ[qi,:] = χ[qi,:] .+ tmp
            end
        end
    end
end


# ---------------------------------------------- Indexing --------------------------------------------
Base.size(arr::T) where T <: MatsubaraFunction = size(arr.data)
Base.getindex(arr::T, i::Int) where T <: MatsubaraFunction = Base.getindex(arr.data, i)
Base.getindex(arr::χ₀T, I::Vararg{Int,3}) = Base.getindex(arr.data, I...)
Base.getindex(arr::χT, I::Vararg{Int,2}) = Base.getindex(arr.data, I...)
Base.getindex(arr::γT, I::Vararg{Int,3}) = Base.getindex(arr.data, I...)
Base.setindex!(arr::T, v, i::Int) where T <: MatsubaraFunction = Base.setindex!(arr.data, v, i)
Base.setindex!(arr::χ₀T, v, I::Vararg{Int,3}) = Base.setindex!(arr.data, v, I...)
Base.setindex!(arr::χT, v, I::Vararg{Int,2}) = Base.setindex!(arr.data, v, I...)
Base.setindex!(arr::γT, v, I::Vararg{Int,3}) = Base.setindex!(arr.data, v, I...)

# --------------------------------------------- Iteration --------------------------------------------
# --------------------------------------------- Broadcast --------------------------------------------
