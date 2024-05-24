# ==================================================================================================== #
#                                           IO_RPA.jl                                                  #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Jan Frederik Weißler                                                             #
# ----------------------------------------- Description ---------------------------------------------- #
#   Functions for reading and writing RPA specific data.                                               #
# -------------------------------------------- TODO -------------------------------------------------- #
#   Unit tests                                                                                         #
#   This only works for local parameters as for now, since I don't have any input files                #
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
    @info "--------------------------------------------------------"
    @info "Read RPA input from file $(file)\n"
    f = h5open(file, "r")
    chi0_group = f["chi0"]

    # attributes
    attr_dict      = attrs(chi0_group)
    β::Float64     = attr_dict["beta"]           # inverse temperature
    n_bz_k::Int64  = attr_dict["n_k"]            # number of gauß-legendre sample points that were used to calculate χ₀
    n_bz_q::Int64  = 2 * (attr_dict["n_q"] - 1)  # number of sample points per dimension to sample the first brillouin zone
    e_kin::Float64 = attr_dict["e_kin"]

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

    @info "χ₀: (1:$(n_bz_q))³ x (0:$(maximum(ω_integers))) → R, indices(q,ω)↦χ₀(q,ω)"
    @info "was evaluated ..."
    @info "\t... for the inverse temperature $(β)"
    @info "\t... using $(n_bz_k) gauss legendre sample points to perform the integration over the first brillouin zone"
    @info "\nTail coefficients"
    @info "\t... kinetic energy is $(e_kin)"
    @info "\nInformations about the data array"
    @info "\t...Array χ₀qω has size $(size(χ₀qω))"
    close(f)
    @info "--------------------------------------------------------"

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
    setupConfig_RPA(KGridStr, Nk::Int)

Sets up RPA calculation directly.
Usually one should use [`readConfig_RPA`](@ref readConfig_RPA) as entry point.
"""
function setupConfig_RPA(kGridStr::String, Nk::Int, U::Float64, β::Float64,
                         n_1Pt::Float64, μ_1Pt::Float64, Epot_1Pt::Float64, Ekin_1Pt::Float64,
                         Nω::Int, Nν::Int, N_iν_shell::Int, shift::Bool;
                         inputDir::String  = "", inputVars::String = "", 
                         loglevel::String = "debug", logfile = "stderr", full_EoM_omega::Bool = true)
    inputVars_full = !isempty(inputDir) && !isempty(inputVars) ? joinpath(inputDir, inputVars) : ""
    freq_r = 2 * (Nν + Nω)
    freq_r = -freq_r:freq_r

    wp = get_workerpool() #TODO setup reasonable pool with clusterManager/Workerconfi # No idea how this is used...
    env = EnvironmentVars(inputDir, inputVars_full, lowercase(loglevel), lowercase(logfile))
    mP = ModelParameters(U, μ_1Pt, β, n_1Pt, Epot_1Pt, Ekin_1Pt)
    sP = SimulationParameters(Nω, Nν, N_iν_shell, shift, undef, NaN, freq_r, 0.0, full_EoM_omega)
    return wp, mP, sP, env, (kGridStr, Nk)
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
    tml_parameters = haskey(tml, "Parameters") ? tml["Parameters"] : nothing

    # read section Model
    kGridStr = tml_model["kGrid"]

    # read section Debug
    full_EoM_omega = tml_debug["full_EoM_omega"]

    # read section Simulation
    Nk = tml_simulation["Nk"]
    Nω = tml_simulation["NBose"]        # Number of positive bosonic matsubara frequencies. The matsubara frequency will be sampled symmetrically around zero. So the space of fermionic matsubara frequencies will be sampled by 2Nν elements in total. Will be used for the triangular vertex as well as the self energy
    Nν = tml_simulation["NFermi"]       # Number of positive fermionic matsubara frequencies. The matsubara frequency will be sampled symmetrically around zero. So the space of fermionic matsubara frequencies will be sampled by 2Nν elements in total. Will be used for the triangular vertex as well as the self energy
    N_iν_shell = 0 
    shift = true

    # Julian: strange things might happen if Nν ≠ Nω... So:
    if Nν ≠ Nω
        error("Please use same number of bosonic and fermionic frequencies!")
    end
    # chi_asympt_method     = tml_simulation["chi_asympt_method"]     # what is this used for? First guess ν-Asymptotics...
    # chi_asympt_shell      = tml_simulation["chi_asympt_shell"]      # what is this used for? First guess ν-Asymptotics...
    # usable_prct_reduction = tml_simulation["usable_prct_reduction"] # what is this used for? First guess ν-Asymptotics...
    # omega_smoothing       = tml_simulation["omega_smoothing"]       # what is this used for?

    # read section Environment
    logfile   = tml_enviroment["logfile"]
    loglevel  = tml_enviroment["loglevel"]

    # collect EnvironmentVars
    inputDir  = haskey(tml_enviroment,"inputDir") ? tml_enviroment["inputDir"] : ""
    inputVars = haskey(tml_enviroment,"input_vars") ? tml["inputVars"] : ""
    has_input_file = !isempty(inputDir) && !isempty(inputVars)
    has_parameters_section = haskey(tml_model, "U") && haskey(tml_model, "beta")
    if has_parameters_section
        @info "Found parameters section in config, starting new computation"
        # TODO: this should be read from the input file, if one is provided
        U        = tml_model["U"]
        β        = tml_model["beta"]
        @warn "The filling is set to ome!"
        n_1Pt = 1.0 # tml_model["n_density"]
        @warn "The parameter μ is set to Un/2!"
        μ_1Pt = U * n_1Pt / 2
        Epot_1Pt = NaN
        Ekin_1Pt = NaN
    else
        if !has_input_file
            error("No input file or explicit parameter section defined!")
        end
    end

    setupConfig_RPA(kGridStr, Nk, U, β,
                    n_1Pt, μ_1Pt, Epot_1Pt, Ekin_1Pt,
                    Nω, Nν, N_iν_shell, shift;
                    inputDir  = inputDir, inputVars = inputVars, 
                    loglevel = loglevel, logfile = logfile, full_EoM_omega = full_EoM_omega)
end
