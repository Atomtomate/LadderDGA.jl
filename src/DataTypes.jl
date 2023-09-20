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

# =========================================== Static Types ===========================================
const _eltype = ComplexF64
const ΓT = Array{_eltype,3}
const FT = Array{_eltype,3}
const GνqT = OffsetArray

const _eltype_RPA = Float64

abstract type MatsubaraFunction{T,N} <: AbstractArray{T,N} end


# =========================================== Struct Types ===========================================
# ------------------------------------------------- χ₀ -----------------------------------------------
"""
    χ₀T <: MatsubaraFunction

Struct for the bubble term. The `q`, `ω` dependent asymptotic behavior is computed from the 
`t1` and `t2` input.  See [`χ₀Asym_coeffs`](@ref χ₀Asym_coeffs) implementation for details.

Constructor
-------------
    χ₀T(type::Symbol, data::Array{_eltype,3}, kG::KGrid, ωnGrid::AbstractVector{Int}, n_iν::Int, shift::Bool, mP::ModelParameters)

Set `local_tail=true` in case of the local bubble constructed fro mthe impurity Green's function. This is necessary in order to construct the correct asymptotics.


Fields
-------------
- **`type`**         : `Symbol`, can be `DMFT`, `local`, `RPA`, `RPA_exact`. TODO: documentation
- **`data`**         : `Array{ComplexF64,3}`, data
- **`asym`**         : `Array{ComplexF64,2}`, `[q, ω]` dependent asymptotic behavior.
- **`axis_types`**   : `Dict{Symbol,Int}`, Dictionary mapping `:q, :ν, :ω` to the axis indices.
- **`indices_νω`**   : `Matrix{Tuple{Int,Int}}`, (n,m) indices of fermionic ``\\nu_n`` and bosonic ``\\omega_m`` Matsubara frequencies.
- **`β`**            : `Float64`, inverse temperature.
"""
struct χ₀T <: MatsubaraFunction{_eltype,3}
    data::Array{_eltype,3}
    asym::Array{_eltype,2}
    axis_types::Dict{Symbol, Int}
    indices_νω::Matrix{Tuple{Int,Int}}
    ν_shell_size::Int
    β::Float64
    # possible inconsistency: ω grid is passed generated, while ν grid is generated in the constructor. The grids must already be known before the calculation of data. From my point of view both grids should be passed to the constructor already generated.
    function χ₀T(type::Symbol, data::Array{_eltype,3}, kG::KGrid, ωnGrid::AbstractVector{Int}, n_iν::Int,
                 shift::Bool, sP::SimulationParameters, mP::ModelParameters; ν_shell_size::Int=0)
        c1, c2, c3 = χ₀Asym_coeffs(type, kG, mP; sVk=sP.sVk)
        asym = χ₀Asym(c1, c2, c3, ωnGrid, n_iν, shift, mP.β)

        νnGrid = -n_iν:n_iν-1
        indices_νω = reshape([(j,i) for i in ωnGrid for j in νnGrid .- trunc(Int64,shift*i/2)],(length(νnGrid), length(ωnGrid)));
        new(data,asym,Dict(:q => 1, :ν => 2, :ω => 3), indices_νω, ν_shell_size, mP.β)
    end
end

    
"""
    core(χ₀::χ₀T)

Select core region (without asymptotic shell) from bubble term.
"""
function core(χ₀::χ₀T)
    view(χ₀.data, :,χ₀.ν_shell_size+1:size(χ₀.data,2)-χ₀.ν_shell_size,:)
end


"""
    χ₀Asym(c1::Float64, c2::Vector{Float64}, c3::Float64, ωnGrid::AbstractVector{Int}, n_iν::Int, shift::Int, β::Float64)

Builds asymtotic helper array. See [`calc_bubble`](@ref calc_bubble) implementation for details.

`c1`, `c2` and `c3` are the coefficients for the asymtotic tail expansion and can be obtained through [`χ₀Asym_coeffs`](@ref χ₀Asym_coeffs).
`n_iν` is the number of positive fermionic Matsubara frequencies, `shift` is either `1` or `0`, depending on the type of frequency grid.
"""
function χ₀Asym(c1::Float64, c2::Vector{Float64}, c3::Float64,
                ωnGrid::AbstractVector{Int}, n_iν::Int, shift::Bool, β::Float64)
    if length(ωnGrid) == 0 || length(first(ωnGrid):last(ωnGrid)) == 0
        throw(ArgumentError("Cannot construct χ₀ array with empty frequency mesh!"))
    end
    χ₀_rest = χ₀_shell_sum_core(β, first(ωnGrid):last(ωnGrid), n_iν, Int(shift))
    asym = Array{_eltype, 2}(undef, length(c2), length(ωnGrid))
    for (ωi,ωn) in enumerate(ωnGrid)
        for (qi,c2i) in enumerate(c2)
            asym[qi,ωi] = χ₀_shell_sum(χ₀_rest, ωn, β, c1, c2i, c3)
        end
    end
    return asym
end

"""
    χ₀Asym_coeffs(type::Symbol, kG::KGrid, mP::ModelParameters; sVk=NaN)

Builds tail coefficients for the χ₀ asymptotic helper, obtained through [`χ₀Asym`](@ref χ₀Asym).

TODO: full documentation
"""
function χ₀Asym_coeffs(type::Symbol, kG::KGrid, mP::ModelParameters; sVk=NaN)
    if type == :local
        c1_tilde = mP.U*mP.n/2 - mP.μ
        c2_tilde = c1_tilde*c1_tilde
        c3_tilde = c1_tilde*c1_tilde + sVk + (mP.U^2)*(mP.n/2)*(1-mP.n/2)
        c1_tilde, [c2_tilde], c3_tilde
    elseif type == :DMFT
        t1 = convert.(ComplexF64, kG.ϵkGrid .+ mP.U*mP.n/2 .- mP.μ)
        t2 = (mP.U^2)*(mP.n/2)*(1-mP.n/2)
        c1_tilde = real.(kintegrate(kG, t1))
        c2_tilde = real.(conv_noPlan(kG, t1, t1))
        c3_tilde = real.(kintegrate(kG, t1 .^ 2) .+ t2)
        c1_tilde, c2_tilde, c3_tilde
    elseif type == :RPA
        t1 = convert.(ComplexF64, kG.ϵkGrid .+ mP.U*mP.n/2 .- mP.μ)
        c1_tilde = real.(kintegrate(kG, t1))
        c2_tilde = real.(conv_noPlan(kG, t1, t1))
        c3_tilde = real.(kintegrate(kG, t1 .^ 2))
        c1_tilde, c2_tilde, c3_tilde
    elseif type == :RPA_exact
        error("Not implemented yet!")
    else 
        throw(ArgumentError("Unkown type for χ₀Asym coefficients!"))
    end
end


"""
    χ₀RPA_T <: MatsubaraFunction

Struct for the RPA bubble term.

Constructor
------------
χ₀RPA_T(data::Array{_eltype,2}, ωnGrid::AbstractVector{Int}, νnGrid::UnitRange{Int64}, β::Float64)

This constructor does not perform any checks for the entered data array in the currently implemented version.
Make sure that the axes match the axis_types field!

Fields
-------------
- **`data`**         : `Array{ComplexF64,3}`, data.
- **`axis_types`**   : `Dict{Symbol,Int}`, Dictionary mapping `:q, :ω` to the axis indices.
- **`indices_ω`**    : `Vector{Int}`, `m` indices m of bosonic ``\\omega_m`` Matsubara frequencies.
- **`β`**            : `Float64`, inverse temperature.
- **`e_kin`**        : `Float64`, kinetic energy.
- **`e_kin_q`**      : `Float64`, so called 'q-dependent kinetic energy'
"""
struct χ₀RPA_T <: MatsubaraFunction{_eltype_RPA,2}
    data::Array{_eltype_RPA,2}
    axis_types::Dict{Symbol, Int}
    indices_ω::Vector{Int}
    β::Float64
    e_kin::Float64
    e_kin_q::Float64        # q-independent part of the so called 'q-dependent RPA kinetic energy' which correpsonds to lim_{ω→∞} (iω)²⋅χ₀(q, ω)
    function χ₀RPA_T(data::Array{_eltype_RPA,2}, ωnGrid::UnitRange{Int}, β::Float64, e_kin::Float64, e_kin_q::Float64)
        indices_ω = [i for i in ωnGrid];
        new(data, Dict(:q => 1, :ω => 2), indices_ω, β, e_kin, e_kin_q)
    end
end
# ------------------------------------------------- χ ------------------------------------------------
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
- **`tail_c`**       : `Vector{Float64}`, tail coefficients of ``1/\\omega^i`` tails. Index `1` corresponds to `i=0`.
- **`λ`**            : `Float64`, λ correction parameter.
- **`β`**            : `Float64`, inverse temperature.
- **`usable_ω`**     : `AbstractArray{Int}`, usable indices for which data is assumed to be correct. See also [`find_usable_χ_interval`](@ref find_usable_χ_interval)
"""
mutable struct χT <: MatsubaraFunction{Float64, 2}
    data::Matrix{Float64}
    axis_types::Dict{Symbol, Int}
    indices_ω::AbstractVector{Int}
    tail_c::Vector{Float64}
    λ::Float64
    β::Float64
    usable_ω::AbstractArray{Int}
    transform!::Function

    function χT(data::Array{Float64, 2}, β::Float64;
                indices_ω::AbstractVector{Int}=axes(data,2) .- ceil(Int,size(data,2)/2), 
                tail_c::Vector{Float64} = Float64[], full_range=true, reduce_range_prct=0.0)
        f!(χ,λ) = nothing
        range = full_range ? (1:size(data,2)) : find_usable_χ_interval(data, reduce_range_prct=reduce_range_prct)
        new(data, Dict(:q => 1, :ω => 2), indices_ω, tail_c, 0.0, β, range, f!)
    end
end

"""
    ωn_grid(χ::χT)

Computes bosonic frequencies for `χ`: ``2 i \\pi n / \\beta``.
"""
ωn_grid(χ::χT) = 2im .* π .* χ.indices_ω ./ χ.β


"""
    sum_kω(kG::kGrid, χ::χT; ωn_arr=ωn_grid(χ), force_full_range=false, [transform::Function])
    sum_kω(kG::kGrid, χ::χT; ωn_arr=ωn_grid(χ), force_full_range=false, [λ::Float64])
    sum_kω(kG::KGrid, χ::AbstractMatrix{Float64}, β::Float64, ωn2_tail::Vector{Float64}; transform=nothing)::Float64

Returns ``\\int_\\mathrm{BZ} dk \\sum_\\omega \\chi^\\omega_k``. The bosonic Matsubara grid can be precomputed and given with `ωn_arr` to increase performance.

TODO: for now this is only implemented for tail correction in the ``1 / \\omega^2_n`` term!
Sums first over k, then over ω (see also [`sum_ω`](@ref sum_ω)), see [`sum_kω`](@ref sum_kω) for the rverse order (results can differ, due to inaccuracies in the asymptotic tail treatment).
The transform function needs to have the signature `f(in::Float64)::Float64` and will be applied before summation. Alternatively, `λ` can be given directly as `Float64`, if the usual [`λ-correction`](@ref χ_λ) should be applied.
"""
function sum_kω(kG::KGrid, χ::χT; ωn_arr=ωn_grid(χ), force_full_range=false, transform=nothing, λ::Float64=NaN)::Float64
    χ.λ != 0 && !isnan(λ) && error("χ already λ-corrected, but external λ provided!")
    !all(χ.tail_c[1:2] .== [0, 0]) && length(χ.tail_c) == 3 && error("sum_kω only implemented for ω^2 tail!") 
    !isnan(λ) && !isnothing(transform) && error("Only transformation OR λ value should be given!")
    !isnan(λ) && (transform = (f(x::Float64)::Float64 = χ_λ(x, λ)))
    ω_slice = 1:size(χ, χ.axis_types[:ω])
    ω_slice = force_full_range ? ω_slice : ω_slice[χ.usable_ω]

    ωn2_tail = real(χ.tail_c[3] ./ ωn_arr .^ 2)
    zero_ind = findfirst(x->!isfinite(x), ωn2_tail)
    ωn2_tail[zero_ind] = 0.0
    res =  sum_kω(kG, view(χ.data,:, ω_slice), χ.β, χ.tail_c[3], ωn2_tail, transform=transform) 
    return res
end

function sum_kω(kG::KGrid, χ::AbstractMatrix{Float64}, β::Float64, c2::Float64, ωn2_tail::Vector{Float64}; transform=nothing)::Float64
    res::Float64  = 0.0
    norm::Float64 = sum(kG.kMult)
    if transform === nothing
        for (i,ωi) in enumerate(ωn2_tail)
            res += kintegrate(kG, view(χ,:,i)) - ωi 
        end
    else
        resi::Float64 = 0.0
        for (i,ωi) in enumerate(ωn2_tail)
            resi = 0.0
            for (ki,km) in enumerate(kG.kMult)
                resi += transform(χ[ki,i]) * km
            end
            res += resi/norm - ωi
        end
    end
    #TODO: add limit
    return (res + (-β^2/12)*c2)/β
end

"""
    sum_ωk(kG::KGrid, χ::χT; force_full_range=false)::Float64

WARNING: This function is a non optimized debugging function! See [`sum_kω`](@ref sum_kω), which should return the same result if the asymptotics are captured correctly.
Optional function `f` transforms `χ` before summation.
"""
function sum_ωk(kG::KGrid, χ::χT; force_full_range=false, λ::Float64=NaN)::Float64
    λ_check = !isnan(λ) && λ != 0
    χ.λ != 0 && λ_check  && error("χ already λ-corrected, but external λ provided!")
    λ_check && χ_λ!(χ, λ)
    res = kintegrate(kG, sum_ω(χ, force_full_range=force_full_range))
    λ_check && reset!(χ)
    return res
end





"""
    sum_ω(χ::χT)
    sum_ω!(res::Vector{ComplexF64}, ωn_arr::Vector{ComplexF64}, χ::χT; force_full_range=false)::Nothing
    sum_ω!(ωn_arr::Vector{T}, χ::AbstractVector{T}, tail_c::Vector{Float64}, β::Float64; force_full_range=false)::T where T <: Union{Float64,ComplexF64}

Sums the physical susceptibility over all usable (if `force_full_range` is not set to `true`) bosonic frequencies, including improvd tail summation, if `χ.tail_c` is set.
"""
function sum_ω(χ::χT; force_full_range=false)
    res  = Vector{eltype(χ.data)}(undef, size(χ.data, χ.axis_types[:q]))
    ωn_arr = ωn_grid(χ)
    sum_ω!(res, ωn_arr, χ; force_full_range=force_full_range)
    return res
end

function sum_ω!(res::Vector{Float64}, ωn_arr::Vector{ComplexF64}, χ::χT; force_full_range=false)::Nothing
    length(res)  != size(χ,χ.axis_types[:q]) && error("Result array for ω summation must be of :q axis length!")
    ω_slice = 1:size(χ, χ.axis_types[:ω])
    ω_slice = force_full_range ? ω_slice : ω_slice[χ.usable_ω]
    for qi in eachindex(res)
        res[qi] = sum_ω(ωn_arr, view(χ,qi,ω_slice), χ.tail_c, χ.β::Float64)
    end
    return nothing
end

function sum_ω(ωn_arr::Vector{ComplexF64}, χ::AbstractVector{Float64}, tail_c::Vector{Float64}, β::Float64)
    !(sum(tail_c) ≈ tail_c[3]) && error("sum_ω only implemented for 1/ω^2, but other tail coefficients are non-zero!")
    limits = [Inf, Inf, -β^2/12] # -i 1.202056903*β^3/(8*π^3), β^4/720
    
    i = 3
    res::Float64 = 0.0
    for n in 1:length(χ)
        if ωn_arr[n] != 0
            res += χ[n] - real(tail_c[i]/(ωn_arr[n]^(i-1)))
        else
            res += χ[n]
        end
    end
    return (res + tail_c[i]*limits[i])/β
end

"""
    update_tail!(χ::χT, new_tail_c::Array{Float64}, ωnGrid::Array{ComplexF64})

Updates the ``\\frac{c_i}{\\omega_n^i}`` tail for all coefficients given in `new_tail_c` (index 1 corresponds to ``i=0``).
"""
function update_tail!(χ::χT, new_tail_c::Array{Float64}, ωnGrid::Array{ComplexF64})
    length(new_tail_c) != length(χ.tail_c) && throw(ArgumentError("length of new tail coefficient array ($(length(new_tail_c))) must be the same as the old one ($(length(χ.tail_c)))!"))
    length(ωnGrid) != size(χ.data,2) && throw(ArgumentError("length of ωnGrid ($(length(ωnGrid))) must be the same as for χ ($(size(χ.data,2)))!"))
    !all(isfinite.(new_tail_c)) && throw(ArgumentError("One or more tail coefficients are not finite!"))
    sum(new_tail_c) != new_tail_c[3] && throw(ArgumentError("update tail only implemented for c_2 coefficient!"))

    λ_bak = χ.λ
    λ_bak != 0 && reset!(χ)
    update_tail!(χ.data, χ.tail_c, new_tail_c, ωnGrid)

    ci = 3
    if !(new_tail_c[ci] == χ.tail_c[ci])
        χ.tail_c[ci] = new_tail_c[ci]
    end
    χ_λ!(χ, λ_bak)
end

function update_tail!(χ::AbstractMatrix{Float64}, old_tail_c::Vector{Float64}, new_tail_c::Vector{Float64}, ωnGrid::Vector{ComplexF64})
    ci::Int = 3

    zero_ind = findfirst(x->abs(x) < 1e-10, ωnGrid) 
    if old_tail_c[ci] != new_tail_c[ci]
        length(new_tail_c) != length(old_tail_c) && throw(ArgumentError("length of new tail coefficient array ($(length(new_tail_c))) must be the same as the old one ($(length(χ.tail_c)))!"))
        !all(isfinite.(new_tail_c)) && throw(ArgumentError("One or more tail coefficients are not finite!"))
        sum(new_tail_c) != new_tail_c[ci] && throw(ArgumentError("update tail only implemented for c_2 coefficient!"))
        for ωi in setdiff(axes(χ,2), zero_ind)
            tmp = real((new_tail_c[ci] - old_tail_c[ci]) / (ωnGrid[ωi] ^ (ci-1)))
            χ[:,ωi] = χ[:,ωi] .+ tmp
        end
    end
end

# ------------------------------------------------- γ ------------------------------------------------
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


# ---------------------------------------------- Indexing --------------------------------------------
Base.size(arr::T) where T <: MatsubaraFunction = size(arr.data)
Base.getindex(arr::T, i::Int) where T <: MatsubaraFunction = Base.getindex(arr.data, i)
Base.getindex(arr::χ₀T, I::Vararg{Int,3}) = Base.getindex(arr.data, I...)
Base.getindex(arr::χ₀RPA_T, I::Vararg{Int,2}) = Base.getindex(arr.data, I...)
Base.getindex(arr::χT, I::Vararg{Int,2}) = Base.getindex(arr.data, I...)
Base.getindex(arr::γT, I::Vararg{Int,3}) = Base.getindex(arr.data, I...)
Base.setindex!(arr::T, v, i::Int) where T <: MatsubaraFunction = Base.setindex!(arr.data, v, i)
Base.setindex!(arr::χ₀T, v, I::Vararg{Int,3}) = Base.setindex!(arr.data, v, I...)
Base.setindex!(arr::χ₀RPA_T, v, I::Vararg{Int,2}) = Base.setindex!(arr.data, v, I...)
Base.setindex!(arr::χT, v, I::Vararg{Int,2}) = Base.setindex!(arr.data, v, I...)
Base.setindex!(arr::γT, v, I::Vararg{Int,3}) = Base.setindex!(arr.data, v, I...)

# --------------------------------------------- Iteration --------------------------------------------
# --------------------------------------------- Broadcast --------------------------------------------
