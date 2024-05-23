# ==================================================================================================== #
#                                            Config.jl                                                 #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe, Jan Frederik Weißler                                              #
#   Last Edit Date  : 22.09.22                                                                         #
# ----------------------------------------- Description ---------------------------------------------- #
#   This file contains legacy functionality for read/write operations of files generated and need      #
#   by a number of auxilliary Fortran codes.                                                           #
# -------------------------------------------- TODO -------------------------------------------------- #
#   -  Find a better frequency grid representation (instead of n_iω, etc)                              #
#   -  save kgrid info in ModelParams: kGrid::String                                                   #
#   -  remove sVk from ModelParams and save in struct for local GF                                     #
# ==================================================================================================== #


import Base.show

# ============================================= Type Def. ============================================

abstract type ConfigStruct end


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
- **`Epot_DMFT`** : `Float64`, DMFT potential energy
- **`Ekin_DMFT`** : `Float64`, DMFT kinetic energy
"""
mutable struct ModelParameters <: ConfigStruct
    U::Float64              # Hubbard U
    μ::Float64              # chemical potential
    β::Float64              # inverse temperature
    n::Float64              # number density
    Epot_DMFT::Float64
    Ekin_DMFT::Float64
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
- **`χ_helper`**                : `struct`, helper struct for asymptotic sum improvements involving the generalized susceptibility (`nothing` if `n_iν_shell == 0`), see also `BSE_SC.jl`.
- **`sVk`**                     : `Float64`, ∑_k Vₖ^2
- **`fft_range`**               : `Int`, Frequencies used for computations of type `f(νₙ + ωₙ)`. 
- **`usable_prct_reduction`**   : `Float64`, percent reduction of usable bosonic frequencies
- **`dbg_full_eom_omega`**      : `Bool`, if true overrides usable ω ranges to `n_iω`.
"""
struct SimulationParameters <: ConfigStruct
    n_iω::Int64             # number of bosonic frequencies
    n_iν::Int64             # number of fermionic frequencies
    n_iν_shell::Int64
    shift::Bool             # shift of center for interval of bosonic frequencies
    χ_helper::Any # Helper for χ asymptotics improvement
    sVk::Float64            # ∑_k Vₖ^2, TODO: this should be moved somewhere else
    fft_range::AbstractArray
    usable_prct_reduction::Float64      # safety cutoff for usable ranges
    dbg_full_eom_omega::Bool
end

"""
    EnvironmentVars <: ConfigStruct

Contains various settings, controlling the I/O behaviour of this module.
This is typically generated from a config.toml file using the [`readConfig`](@ref readConfig) function.

Fields
-------------
- **`inputDir`**        : `String`, Directory of input files
- **`inputVars`**       : `String`, File name of .jld2 file containing input.
- **`loglevel`**        : `String`, Options: disabled, error, warn, info, debug
- **`logfile`**         : `String`,    Options: STDOUT, STDERR, filename
"""
struct EnvironmentVars <: ConfigStruct
    inputDir::String
    inputVars::String
    loglevel::String      # disabled, error, warn, info, debug
    logfile::String       # STDOUT, STDERR, filename
end

# ============================================= Interface ============================================
