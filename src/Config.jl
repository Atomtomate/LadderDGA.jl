# ==================================================================================================== #
#                                            Config.jl                                                 #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
#   Last Edit Date  : 03.08.22                                                                         #
# ----------------------------------------- Description ---------------------------------------------- #
#   This file contains legacy functionality for read/write operations of files generated and need      #
#   by a number of auxilliary Fortran codes.                                                           #
# ==================================================================================================== #


import Base.show

abstract type ConfigStruct end

BSum = Union{Symbol, Tuple{Int,Int}}

"""
    ModelParameters <: ConfigStruct

Contains model parameters for the Hubbard model.
This is typically generated from a config.toml file using  the [`readConfig`](@ref readConfig) function.

Fields
-------------
- **`U`**         : `Float64`, Hubbard U
- **`μ`**         : `Float64`, chemical potential
- **`β`**         : `Float64`, inverse temperature
- **`n`**         : `Float64`, filling
- **`sVk`**       : `Float64`, ∑_k Vₖ^2 (TODO: this is used for local tail improvement and will generally not be needed)
- **`Epot_DMFT`** : `Float64`, DMFT potential energy
- **`Ekin_DMFT`** : `Float64`, DMFT kinetic intergy
"""
struct ModelParameters <: ConfigStruct
    U::Float64              # Hubbard U
    μ::Float64              # chemical potential
    β::Float64              # inverse temperature
    n::Float64              # number density
    sVk::Float64            # ∑_k Vₖ^2
    Epot_DMFT::Float64
    Ekin_DMFT::Float64
end
# TODO: save kgrid info here: kGrid::String           # String encoding information about the grid

"""
    SumExtrapolationHelper <: ConfigStruct

Helper for sum extrapolations. Not used right now, since it was replaced by the `BSE_SC.jl` module.
"""
struct SumExtrapolationHelper <: ConfigStruct
    bosonic_tail_coeffs::Array{Int,1}   # tail
    fermionic_tail_coeffs::Array{Int,1}
    ω_smoothing::Symbol                 # nothing, range, full
    sh_f::SumHelper                     # SumHelper for fermionic sums (bosonic sums depend on runtime results)
    fνmax_lo::Int
    fνmax_up::Int
    fνmax_cache_r::Array{Float64,1}
    fνmax_cache_c::Array{ComplexF64,1}
end


"""
    SimulationParameters <: ConfigStruct

Contains simulation parameters for the ladder DGA computations.
This is typically generated from a config.toml file using the [`readConfig`](@ref readConfig) function.

Fields
-------------
- **`n_iω`**                    : `Int`, Number of positive bosonic frequencies (full number will be `2*n_iω+1` 
- **`n_iν`**                    : `Int`, Number of positive fermionic frequencies (full number will be `2*n_iν` 
- **`n_iν_shell`**              : `Int`, Number of fermionic frequencies used for asymptotic sum improvement (`χ_asym_r` arrays with at least these many entries need to be provided)
- **`shift`**                   : `Bool`, Flag specifying if `-n_iν:n_iν-1` is shifted by `-ωₙ/2` at each `ωₙ` slice (centering the main features)
- **`tc_type_f`**               : `Symbol` specifies the type of sum extrapolation for fermionic sums. Implemented for `:nothing` (naive sum) and `:richardson`. See `SeriesAcceleration` package for more details.
- **`tc_type_b`**               : `Symbol`, specifies the type of sum extrapolation for bosonic sums. Implemented for `:nothing` (naive sum), `:coeffs` (subtracted known tail coefficients) and `:richardson`. See `SeriesAcceleration` package for more details.
- **`χ_helper`**                : `struct`, helper struct for asymptotic sum improvements (`nothing` if `n_iν_shell == 0`), see also `BSE_SC.jl`.
- **`ωsum_type`**               : `Symbol`, Either a Symbol `:individual` or `:common`, specifying whether bosonic ranges should be common for all channels or maximal in each channel, or a fixed range (given as Tuple).
- **`λ_rhs`**                   : `Symbol`, specifying the rhs in lambda corrections. Options are `:native`, `:fixed`, `:error_comp`.  
- **`fullChi`**                 : `Bool`, specifying whether values for quantities outside the usable ωrange should be computed anyway.
- **`usable_prct_reduction`**   : `Float64`, percent reduction of usable bosonic frequencies
- **`fft_range`**               : `Int`, Frequencies used for computations of type `f(νₙ + ωₙ)`. 
- **`dbg_full_eom_omega`**      : `Bool`, if true overrides usable ω ranges to `n_iω`.
"""
struct SimulationParameters <: ConfigStruct
    n_iω::Int64             # number of bosonic frequencies
    n_iν::Int64             # number of fermionic frequencies
    n_iν_shell::Int64
    shift::Bool             # shift of center for interval of bosonic frequencies
    tc_type_f::Symbol       # use correction for finite ν sums.
    tc_type_b::Symbol       # use correction for finite ω sums.
    χ_helper # Helper for χ asymptotics improvement
    ωsum_type::BSum
    λ_rhs::Symbol
    fullChi::Bool
    fft_range::AbstractArray
    usable_prct_reduction::Float64      # safety cutoff for usable ranges
    dbg_full_eom_omega::Bool
    sumExtrapolationHelper::Union{SumExtrapolationHelper,Nothing}
end

"""
    EnvironmentVars <: ConfigStruct

Contains various settings, controlling the I/O behaviour of this module.
This is typically generated from a config.toml file using the [`readConfig`](@ref readConfig) function.

Fields
-------------
- **`inputDir`**        : `String`, Directory of input files
- **`inputVars`**       : `String`, File name of .jld2 file containing input.
- **`cast_to_real`**    : `Bool`, 
- **`loglevel`**        : `String`, Options: disabled, error, warn, info, debug
- **`logfile`**         : `String`,    Options: STDOUT, STDERR, filename
- **`progressbar`**     : `Bool`,      Options: true/false enable or disable progress bar
"""
struct EnvironmentVars <: ConfigStruct
    inputDir::String
    inputVars::String
    cast_to_real::Bool
    loglevel::String      # disabled, error, warn, info, debug
    logfile::String       # STDOUT, STDERR, filename
    progressbar::Bool     # true/false enable or disable progress bar
end

"""
	Base.show(io::IO, m::SimulationParameters)

Custom output for SimulationParameters
"""
function Base.show(io::IO, m::SimulationParameters)
    compact = get(io, :compact, false)

    if !compact
        println(io, "B/F range    : $(m.n_iω)/$(m.n_iν) $(m.shift ? "with" : "without") shifted fermionic frequencies")
        println(io, "   ωsum type = $(m.ωsum_type) $(m.fullChi ? "with" : "without") full χ(ω) range computation ($(m.dbg_full_eom_omega ? "with" : "without") full ω range in EoM.")
        println(io, "Asymptotic correction : $(typeof(m.χ_helper))")
        println(io, "B/F sum type : $(m.tc_type_b)/$(m.tc_type_f)")
        println(io, "   $(100*m.usable_prct_reduction) % reduction of usable range and ω smoothing $(m.usable_prct_reduction)")
        println(io, "λ-Correction : $(m.λc_type) with rhs $(m.λ_rhs)")
    else
        print(io, "SimulationParams[nB=$m.n_iω, nF=m.n_iν, shift=m.shift]")
    end
end

function Base.show(io::IO, ::MIME"text/plain", m::SimulationParameters)
    println(io, "LadderDGA.jl SimulationParameters:")
    show(io, m)
end

"""
	Base.show(io::IO, m::ModelParameters)

Custom output for ModelParameters
"""
function Base.show(io::IO, m::ModelParameters)
    compact = get(io, :compact, false)

    if !compact
        println(io, "U=$(m.U), β=$(m.β), n=$(m.n), μ=$(m.μ)")
        println(io, "DMFT Energies: T=$(m.Ekin_DMFT), V=$(m.Epot_DMFT)")
    else
        print(io, "SimulationParams[nB=$m.n_iω, nF=m.n_iν, shift=m.shift]")
    end
end

function Base.show(io::IO, ::MIME"text/plain", m::ModelParameters)
    println(io, "LadderDGA.jl ModelParameters:")
    show(io, m)
end
