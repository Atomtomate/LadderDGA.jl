@enum ChiFillType zero_χ_fill lambda_χ_fill χ_fill

BSum = Union{Symbol, Tuple{Int,Int}}

struct ModelParameters
    U::Float64              # Hubbard U
    μ::Float64              # chemical potential
    β::Float64              # inverse temperature
    n::Float64              # number density
    D::Int64
end

struct SimulationParameters
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
    chi_only::Bool          # skip computation of self energy
    bosonic_tail_coeffs::Array{Int,1}   # tail
    fermionic_tail_coeffs::Array{Int,1}
end

struct FreqGrid
    ω::Array{Float64,1}
    ν::Array{Float64,1}
end

struct EnvironmentVars
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
