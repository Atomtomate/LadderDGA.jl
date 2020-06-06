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
    shift::Int64            # shift of center for intervall of bosonic frequencies
    Nk::Int64               # Number of k-space integration steps
    tail_corrected::Bool    # use correction for finite ν and ω sums.
    chi_only::Bool    # skip computation of self energy
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
end
# TODO: not implemented: LQ, Nint, chi_only, lambdaspin_only, sumallch, sumallsp
