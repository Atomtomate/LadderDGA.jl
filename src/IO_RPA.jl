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

    Nω :: Int, Number of positive bosonic frequencies to be used. Make sure that the given χ₀-file has at least this many bosonic frequencies!  
"""
function read_χ₀_RPA(file::String, Nω::Int)
    if !HDF5.ishdf5(file)
        throw(ArgumentError("The given file is not a valid hdf5 file!\ngiven path is: '$(file)'"))
    end
    println("--------------------------------------------------------")
    println("Read RPA input from file $(file)\n")
    f = h5open(file, "r")
    chi0_group = f["chi0"]

    # attributes
    attr_dict = attrs(chi0_group)
    β      :: Float64 = attr_dict["beta"]           # inverse temperature
    n_bz_k :: Int64   = attr_dict["n_k"]            # number of gauß-legendre sample points that were used to calculate χ₀
    n_bz_q :: Int64   = 2 * (attr_dict["n_q"] - 1)  # number of sample points per dimension to sample the first brillouin zone
    e_kin  :: Float64 = attr_dict["e_kin"]

    # datasets
    ω_integers = read(chi0_group["omega_integers"])
    max_ω_index = indexin(Nω, ω_integers)[1]
    if size(ω_integers,1) > Nω
        ω_integers = ω_integers[begin:max_ω_index]
    elseif size(ω_integers,1) < Nω
        error("Number of bosonic frequencies in the χ₀ file is smaller than n_pos_bose_freqs from the configuration file!") 
    end
    if ω_integers ≠ collect(0:Nω)
        throw(ArgumentError("ω-integers are distict from (0:$(Nω))!"))
    end
    χ₀qω = read(chi0_group["values"])[:,begin:max_ω_index]

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
    println("\nInformations about the data array")
    println("\t...Array χ₀qω has size $(size(χ₀qω))")
    close(f)
    println("--------------------------------------------------------")

    data = expand_ω(χ₀qω)
    ωnGrid = (convert(Int64, -maximum(ω_integers)):convert(Int64, maximum(ω_integers)))
    return χ₀RPA_T(data, ωnGrid, β, e_kin, n_bz_q, n_bz_k)
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
    tml_model      = tml["Model"]
    tml_simulation = tml["Simulation"]
    tml_enviroment = tml["Environment"]
    tml_debug      = tml["Debug"]

    # read section Model
    U        = tml_model["U"]
    kGridStr = tml_model["kGrid"]
    
    @warn "The filling is set to ome!"
    n_density = 1.0 # tml_model["n_density"]
    @warn "The parameter μ is set to Un/2!"
    μ = U * n_density / 2
    @warn "The parameter EPot_DMFT is set to zero!"
    EPot_DMFT = 0.0 # tml_model["EPot_DMFT"] # after inspecting the code: EPot_DMFT is actually never used during the standard-λdm-calculation... 

    # read section Debug
    full_EoM_omega = tml_debug["full_EoM_omega"]

    # read section Simulation
    Nν = tml_simulation["n_pos_fermi_freqs"]       # Number of positive fermionic matsubara frequencies. The matsubara frequency will be sampled symmetrically around zero. So the space of fermionic matsubara frequencies will be sampled by 2Nν elements in total. Will be used for the triangular vertex as well as the self energy
    Nω = tml_simulation["n_pos_bose_freqs"]        # Number of positive bosonic matsubara frequencies. The matsubara frequency will be sampled symmetrically around zero. So the space of fermionic matsubara frequencies will be sampled by 2Nν elements in total. Will be used for the triangular vertex as well as the self energy

    if Nν ≠ Nω
        # Julian: strange things might happen if Nν ≠ Nω... So:
        error("Please use same number of bosonic and fermionic frequencies!")
    end
    # chi_asympt_method     = tml_simulation["chi_asympt_method"]     # what is this used for? First guess ν-Asymptotics...
    # chi_asympt_shell      = tml_simulation["chi_asympt_shell"]      # what is this used for? First guess ν-Asymptotics...
    # usable_prct_reduction = tml_simulation["usable_prct_reduction"] # what is this used for? First guess ν-Asymptotics...
    # omega_smoothing       = tml_simulation["omega_smoothing"]       # what is this used for?

    # read section Environment
    inputDir  = tml_enviroment["inputDir"]
    inputVars = tml_enviroment["inputVars"]
    logfile   = tml_enviroment["logfile"]
    loglevel  = tml_enviroment["loglevel"]

    # collect EnvironmentVars
    input_dir = isabspath(inputDir) ? inputDir : abspath(joinpath(dirname(cfg_in), inputDir))
    env = EnvironmentVars(input_dir,
        joinpath(input_dir, inputVars),
        String([(i == 1) ? uppercase(c) : lowercase(c)
                for (i, c) in enumerate(loglevel)]),
        lowercase(logfile))

    # read RPA χ₀ from hdf5 file
    χ₀::χ₀RPA_T = read_χ₀_RPA(env.inputVars, Nω)

    # collect ModelParameters
    mP = ModelParameters(U, μ, χ₀.β, n_density, EPot_DMFT, χ₀.e_kin)

    # collect SimulationParameters
    # Nω = trunc(Int, (length(χ₀.indices_ω) - 1) / 2) # number of positive bosonic matsubara frequencies. Is there a particular reason for this choice ?
    freq_r = 2 * (Nν + Nω)
    freq_r = -freq_r:freq_r
    sP = SimulationParameters(
        Nω,                            # (n_iω::Int64) number of positive bosonic matsubara frequencies
        Nν,                            # (n_iν::Int64) number of positive fermionic matsubara frequencies
        -1,                            # (n_iν_shell::Int64) number of fermionic frequencies used for asymptotic sum improvement | Set this such that the program crashes whenever this is used...
        true,                          # (shift::Bool) since there are no fermionic frequencies there is no need for the shift | code with option "false" is not tested since 1.5 years => always use the shifted version!
        undef,                         # (χ_helper # Helper) χ_helper. When is this guy used?
        0.0,                           # (sVk::Float64) sum(Vk .^ 2). What is this used for?  
        freq_r,                        # (fft_range::AbstractArray) fft_range. No idea how this is supposed to be set...
        NaN,                           # (usable_prct_reduction::Float64) usable_prct_reduction. No idea what this is...
        full_EoM_omega                 # (dbg_full_eom_omega::Bool) A debug flag I guess...
    )

    # build workerpool
    wP = get_workerpool() #TODO setup reasonable pool with clusterManager/Workerconfi # No idea how this is used...
    return χ₀, wP, mP, sP, env, kGridStr
end
