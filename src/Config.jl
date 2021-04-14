abstract type ConfigStruct end

#TODO: this type is probably not needed anymore
BSum = Union{Symbol, Tuple{Int,Int}}

@enum ChiFillType zero_χ_fill lambda_χ_fill χ_fill

"""
    ModelParameters <: ConfigStruct

Contains model parameters for the Hubbard model.
This is typically generated from a config.toml file using 
the [`readConfig`](@ref readConfig) function.
"""
struct ModelParameters <: ConfigStruct
    U::Float64              # Hubbard U
    μ::Float64              # chemical potential
    β::Float64              # inverse temperature
    n::Float64              # number density
    D::Int64
end

"""
    SimulationParameters <: ConfigStruct

Contains simulation parameters for the ladder DGA computations.
This is typically generated from a config.toml file using 
the [`readConfig`](@ref readConfig) function.

Fields
-------------
TODO: describe implications of all fields
"""
struct SimulationParameters <: ConfigStruct
    n_iω::Int64             # number of bosonic frequencies
    n_iν::Int64             # number of fermionic frequencies
    shift::Bool            # shift of center for interval of bosonic frequencies
    Nk::Int64               # Number of k-space integration steps
    tc_type::Symbol  # use correction for finite ν and ω sums.
    λc_type::Symbol  # which type of lambda correction to use (currecntly: nothing, sp, sp_ch, TOOD: sp_ch_q
    ωsum_type::BSum
    λ_rhs::Symbol
    fullChi::Bool
    χFillType::ChiFillType # values to be set outside the usable interval
    bosonic_tail_coeffs::Array{Int,1}   # tail
    fermionic_tail_coeffs::Array{Int,1}
end

"""
    EnvironmentVars <: ConfigStruct

Contains various settings, controlling the I/O behaviour 
of this module.
This is typically generated from a config.toml file using 
the [`readConfig`](@ref readConfig) function.
"""
struct EnvironmentVars <: ConfigStruct
    inputDataType::String
    writeFortran::Bool
    loadAsymptotics::Bool
    inputDir::String
    freqFile::String
    inputVars::String
    asymptVars::String
    cast_to_real::Bool
    loglevel::String      # disabled, error, warn, info, debug
    logfile::String       # STDOUT, STDERR, filename
    progressbar::Bool     # true/false enable or disable progress bar
end
