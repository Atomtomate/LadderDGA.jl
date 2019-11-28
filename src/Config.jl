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
    Nint::Int64
    Nq::Int64               # Number of k-space integration steps for Q grid
    inputVars::String       # path to Γ Input
end

# TODO: not implemented: LQ, Nint, chi_only, lambdaspin_only, sumallch, sumallsp
