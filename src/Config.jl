@enum ChiFillType zero_χ_fill lambda_χ_fill χ_fill

struct ModelParameters
    U::Float64              # Hubbard U
    μ::Float64              # chemical potential
    β::Float64              # inverse temperature
    n::Float64              # number density
    D::Int64
end

struct SimulationParameters
    n_iν::Int64             # number of fermionic frequencies
    n_iω::Int64             # number of bosonic frequencies
    shift::Int64            # shift of center for interval of bosonic frequencies
    Nk::Int64               # Number of k-space integration steps
    tail_corrected::Bool    # use correction for finite ν and ω sums.
    fullLocSums::Bool       # full ω sums in computation of local quantities
    fullRange::Bool          # full ω sums in computation of Σ_ladder
    fullChi::Bool
    χFillType::ChiFillType # values to be set outside the usable interval
    chi_only::Bool          # skip computation of self energy
    kInt::String            # Type of k=space integration: naive summation or FFT
end

struct FreqGrid
    ω::Array{Float64,1}
    ν::Array{Float64,1}
end

struct EnvironmentVars
    loadFortran::String
    writeFortran::Bool
    loadAsymptotics::Bool
    inputDir::String
    inputVars::String
    asymptVars::String
    cast_to_real::Bool
    loglevel::String      # disabled, error, warn, info, debug
    logfile::String       # STDOUT, STDERR, filename
    progressbar::Bool     # true/false enable or disable progress bar
end
