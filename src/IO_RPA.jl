# ==================================================================================================== #
#                                           IO_RPA.jl                                                  #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Jan Frederik Weißler                                                             #
# ----------------------------------------- Description ---------------------------------------------- #
#   Functions for reading and writing RPA specific data.                                               #
# -------------------------------------------- TODO -------------------------------------------------- #
#   Unit tests                                                                                         #
# ==================================================================================================== #
"""
is_okay(χ₀qω)

check whether the given χ₀qω Array satisfies a set of expected conditions.
"""
function is_okay(χ₀qω)
    leq_zero = all(χ₀qω .≥ 0)
    χΓω0neq0 = χ₀qω[begin, begin] ≠ 0
    χΓωneq0eq0 = all(χ₀qω[begin, begin+1:end] .== 0)
    is_ok = leq_zero & χΓω0neq0 & χΓωneq0eq0
    return is_ok
end


"""
    read_χ₀_RPA(file::String)

TBW
"""
function read_χ₀_RPA(file::String)
    if !HDF5.ishdf5(file)
        throw(ArgumentError("The given file is not a valid hdf5 file!\ngiven path is: '$(file)'"))
    end
    println("--------------------------------------------------------")
    println("Read RPA input from file $(file)\n")
    f = h5open(file, "r")
    chi0_group = f["chi0"]

    # attributes
    attr_dict = attrs(chi0_group)
    β = attr_dict["beta"]               # inverse temperature
    n_bz_k = attr_dict["n_cubes"]            # number of sample points per dimension to sample the first brillouin zone
    n_bz_q = 2 * (attr_dict["n_samples"] - 1)     # number of sample points per dimension to sample the first brillouin zone
    e_kin = attr_dict["e_kin"]
    e_kin_q = attr_dict["tail_coeff_q_indep"]

    # datasets
    ω_integers = read(chi0_group["omega_integers"])
    if ω_integers ≠ collect(0:maximum(ω_integers))
        throw(ArgumentError("ω-integers are distict from (0:$(maximum(ω_integers)))!"))
    end
    χ₀qω = read(chi0_group["values"])

    # consistency checks
    if β ≤ 0.0
        error("β is a positive quantity!")
    end
    if n_bz_q ≤ 8
        error("Number of sample points in reciprocal space per dimension is supposed to be larger than 7!")
    end
    if !is_okay(χ₀qω)
        throw(ArgumentError("The given array for χ₀qω violates at least one of the expected relations."))
    end

    println("χ₀: (1:$(n_bz_q))³ x (0:$(maximum(ω_integers))) → R, indices(q,ω)↦χ₀(q,ω)")
    println("was evaluated ...")
    println("\t... for the inverse temperature $(β)")
    println("\t... using $(n_bz_k) gauss legendre sample points to perform the integration over the first brillouin zone")
    println("\nTail coefficients")
    println("\t... kinetic energy is $(e_kin)")
    println("\t... tail coeff e_kin_q is $(e_kin_q)")
    println("\nIformations about the data array")
    println("\t...Array χ₀qω has size $(size(χ₀qω))")
    close(f)
    println("--------------------------------------------------------")

    data = expand_ω(χ₀qω)
    ωnGrid = (convert(Int64, -maximum(ω_integers)):convert(Int64, maximum(ω_integers)))
    return χ₀RPA_T(data, ωnGrid, β, e_kin, e_kin_q, n_bz_q)
end

"""
    expand_ω(χ₀qω)

    Helper function for reading RPA input. It holds χ₀(q,ω)=χ₀(q,-ω). Take an array for χ₀(q,ω) with ω-integers {0, 1, ..., m} and map onto array with ω-integers {-m, -(m-1), ..., -1, 0, 1, ..., m-1, m}.
"""
function expand_ω(χ₀qω)
    return cat(reverse(χ₀qω, dims=2)[:, begin:end-1], χ₀qω, dims=2)
end


"""
    readConfig_RPA(cfg_in::String)

Reads a config.toml file either as string or from a file and returns 
    - workerpool
    - [`ModelParameter, "../test/test_data/rpa_chi0_1.h5")
    # χ₀ = read_χ₀_RPA(inputfile)s`](@ref)
    - [`SimulationParameters`](@ref)
    - [`EnvironmentVars`](@ref)
    - kGrid (see Dispersions.jl)
"""
function readConfig_RPA(cfg_in::String)
    @info "Reading Inputs..."

    cfg_is_file = true
    try
        cfg_is_file = isfile(cfg_in)
    catch e
        @warn "cfg_file not found as file. Trying to parse as config string."
        cfg_is_file = false
    end
    tml = if cfg_is_file
        TOML.parsefile(cfg_in)
    else
        TOML.parse(cfg_in)
    end

    # sections
    tml_model = tml["Model"]
    tml_simulation = tml["Simulation"]
    tml_enviroment = tml["Environment"]
    tml_debug = tml["Debug"]

    # read section Model
    @warn "The parameter μ should be read from the χ₀-file and not passed via the configuration file!"
    @warn "The parameter n should be read from the χ₀-file and not passed via the configuration file!"
    @warn "The parameter EPot_DMFT should be read from the χ₀-file and not passed via the configuration file!"
    U = tml_model["U"]
    μ = tml_model["mu"]
    n_density = tml_model["n_density"]
    EPot_DMFT = tml_model["EPot_DMFT"]
    kGridStr = tml_model["kGrid"]

    # read section Debug
    full_EoM_omega = tml_debug["full_EoM_omega"]

    # read section Simulation
    Nν = tml_simulation["n_pos_fermi_freqs"]     # Number of positive fermionic matsubara frequencies. The matsubara frequency will be sampled symmetrically around zero. So the space of fermionic matsubara frequencies will be sampled by 2Nν elements in total. Will be used for the triangular vertex as well as the self energy
    chi_asympt_method = tml_simulation["chi_asympt_method"]     # what is this used for? First guess ν-Asymptotics...
    chi_asympt_shell = tml_simulation["chi_asympt_shell"]      # what is this used for? First guess ν-Asymptotics...
    usable_prct_reduction = tml_simulation["usable_prct_reduction"] # what is this used for? First guess ν-Asymptotics...
    omega_smoothing = tml_simulation["omega_smoothing"]       # what is this used for?

    # read section Environment
    inputDir = tml_enviroment["inputDir"]
    inputVars = tml_enviroment["inputVars"]
    logfile = tml_enviroment["logfile"]
    loglevel = tml_enviroment["loglevel"]

    # collect EnvironmentVars
    input_dir = isabspath(inputDir) ? inputDir : abspath(joinpath(dirname(cfg_in), inputDir))
    env = EnvironmentVars(input_dir,
        joinpath(input_dir, inputVars),
        String([(i == 1) ? uppercase(c) : lowercase(c)
                for (i, c) in enumerate(loglevel)]),
        lowercase(logfile))

    # read RPA χ₀ from hdf5 file
    χ₀::χ₀RPA_T = read_χ₀_RPA(env.inputVars)

    # collect ModelParameters
    mP = ModelParameters(U, μ, χ₀.β, n_density, EPot_DMFT, χ₀.e_kin)

    # collect SimulationParameters
    Nω = trunc(Int, (length(χ₀.indices_ω) - 1) / 2) # number of positive bosonic matsubara frequencies. Is there a particular reason for this choice ?
    freq_r = 2 * (Nν + Nω)
    freq_r = -freq_r:freq_r
    sP = SimulationParameters(
        Nω,                            # number of positive bosonic matsubara frequencies
        Nν,                            # number of positive fermionic matsubara frequencies | Set this such that the program crashes whenever this is used...
        -1,                            # number of fermionic frequencies used for asymptotic sum improvement | Set this such that the program crashes whenever this is used...
        false,                         # since there are no fermionic frequencies there is no need for the shift | Is this save?
        undef,                         # χ_helper. When is this guy used?
        0.0,                           # sum(Vk .^ 2). What is this used for?  
        freq_r,                      # fft_range. No idea how this is supposed to be set...
        NaN,                           # usable_prct_reduction. No idea what this is...
        full_EoM_omega                 # A debug flag I guess...
    )

    # build workerpool
    wP = get_workerpool() #TODO setup reasonable pool with clusterManager/Workerconfi # No idea how this is used...
    return χ₀, wP, mP, sP, env, kGridStr
end
