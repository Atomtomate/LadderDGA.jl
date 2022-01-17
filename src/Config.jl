import Base.show

abstract type ConfigStruct end

#TODO: this type is probably not needed anymore
BSum = Union{Symbol, Tuple{Int,Int}}

@enum ChiFillType zero_χ_fill lambda_χ_fill χ_fill

#TODO: build better constructor, IO, update docstring
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
    sVk::Float64            # ∑_k Vₖ^2
    Epot_DMFT::Float64
    Ekin_DMFT::Float64
end
# TODO: save kgrid info here: kGrid::String           # String encoding information about the grid

"""
    SimulationParameters <: ConfigStruct

Contains simulation parameters for the ladder DGA computations.
This is typically generated from a config.toml file using 
the [`readConfig`](@ref readConfig) function.

Fields
-------------
- **`n_iω`**    : Number of positive bosonic frequencies (full number will be `2*n_iω+1` 
- **`n_iν`**    : Number of positive fermionic frequencies (full number will be `2*n_iν` 
- **`shift`**   : Flag specifying if `-n_iν:n_iν-1` is shifted by `-ωₙ/2` at each `ωₙ` slice (centering the main features)
- **`tc_type_f`** : Symbol specifying the type of sum extrapolation for fermionic sums. Implemented for `:nothing` (naive sum) and `:richardson`. See `SeriesAcceleration` package for more details.
- **`tc_type_b`** : Symbol specifying the type of sum extrapolation for bosonic sums. Implemented for `:nothing` (naive sum), `:coeffs` (subtracted known tail coefficients) and `:richardson`. See `SeriesAcceleration` package for more details.
- **`ωsum_type`**  :
- **`λ_rhs`**   :
- **`fullChi`** :
- **`χFillType`**  :
- **`bosonic_tail_coeffs`**   :
- **`fermionic_tail_coeffs`** :
- **`usable_prct_reduction`** :
"""
struct SimulationParameters <: ConfigStruct
    n_iω::Int64             # number of bosonic frequencies
    n_iν::Int64             # number of fermionic frequencies
    n_iν_shell::Int64
    #νAsymptGrid::AbstractVector{Int}
    shift::Bool            # shift of center for interval of bosonic frequencies
    tc_type_f::Symbol  # use correction for finite ν sums.
    tc_type_b::Symbol  # use correction for finite ω sums.
    χ_helper::Union{BSE_SC_Helper,BSE_Asym_Helper,Nothing} # Helper for χ asymptotics improvement
    ωsum_type::BSum
    λ_rhs::Symbol
    fullChi::Bool
    χFillType::ChiFillType # values to be set outside the usable interval
    #TODO: move sum related stuff tu nu/omega_sum_helper
    bosonic_tail_coeffs::Array{Int,1}   # tail
    fermionic_tail_coeffs::Array{Int,1}
    usable_prct_reduction::Float64      # safety cutoff for usable ranges
    ω_smoothing::Symbol                 # nothing, range, full
    sh_f::SumHelper                     # SumHelper for fermionic sums (bosonic sums depend on runtime results)
    fft_range::AbstractArray
    fft_offset::Int
    dbg_full_eom_omega::Bool
    fνmax_lo::Int
    fνmax_up::Int
    fνmax_cache_r::Array{Float64,1}
    fνmax_cache_c::Array{ComplexF64,1}
end

#TODO: SimulationParameters is becomming too large. introduce additional helper?

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
    inputDir::String
    freqFile::String
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
        println(io, "   ωsum type = $(m.ωsum_type) $(m.fullChi ? "with" : "without") full χ(ω) range computation (filled as $(m.χFillType), $(m.dbg_full_eom_omega ? "with" : "without") full ω range in EoM.")
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
