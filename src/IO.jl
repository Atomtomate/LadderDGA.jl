# ==================================================================================================== #
#                                              IO.jl                                                   #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
#   Last Edit Date  : 01.09.22                                                                         #
# ----------------------------------------- Description ---------------------------------------------- #
#   General input output operations for reading config and data files as well as logging               #
# -------------------------------------------- TODO -------------------------------------------------- #
# TODO: document, file is missleading, can also be a string                                            #
# TODO: many of the fortran functions have the old axis layout                                         #
#       (omega axis first instead of last, update that                                                 #
# ==================================================================================================== #



# ======================================= Legace (Fortran) IO ========================================
include("IO_legacy.jl")

# ============================================ Config IO =============================================
"""
    readConfig(cfg_in::String)

Reads a config.toml file either as string or from a file and returns 
    - workerpool
    - [`ModelParameters`](@ref)
    - [`SimulationParameters`](@ref)
    - [`EnvironmentVars`](@ref)
    - kGrid (see Dispersions.jl)
"""
function readConfig(cfg_in::String)
    @info "Reading Inputs..."

    cfg_is_file = true
    try 
        cfg_is_file = isfile(cfg_in)
    catch e
        cfg_is_file = false
    end
    tml = if cfg_is_file
        TOML.parsefile(cfg_in)
    else
        TOML.parse(cfg_in)
    end
    sim = tml["Simulation"]
    rr = r"^fixed:(?P<start>\N+):(?P<stop>\N+)"
    smoothing = Symbol(lowercase(tml["Simulation"]["omega_smoothing"]))
    if !(smoothing in [:nothing, :range, :full])
        error("Unrecognized smoothing type \"$(smoothing)\"")
    end
    dbg_full_eom_omega = (haskey(tml["Debug"], "full_EoM_omega") && tml["Debug"]["full_EoM_omega"]) ? true : false

    env = EnvironmentVars(   tml["Environment"]["inputDir"],
                             tml["Environment"]["inputVars"],
                             String([(i == 1) ? uppercase(c) : lowercase(c)
                                     for (i, c) in enumerate(tml["Environment"]["loglevel"])]),
                             lowercase(tml["Environment"]["logfile"])
                            )

    mP, χsp_asympt, χch_asympt, χpp_asympt = jldopen(env.inputDir*"/"*env.inputVars, "r") do f
        EPot_DMFT = 0.0
        EKin_DMFT = 0.0
        if haskey(f, "E_kin_DMFT")
            EPot_DMFT = f["E_pot_DMFT"]
            EKin_DMFT = f["E_kin_DMFT"]
        else
            @warn "Could not find E_kin_DMFT, E_pot_DMFT key in input"
        end
        U, μ, β, nden, Vk = if haskey(f, "U")
            @warn "Found Hubbard Parameters in input .jld2, ignoring config.toml"
            f["U"], f["μ"], f["β"], f["nden"], f["Vₖ"]
        else
            @warn "reading Hubbard Parameters from config. These should be supplied through the input jld2!"
            tml["Model"]["U"], tml["Model"]["mu"], tml["Model"]["beta"], tml["Model"]["nden"], f["Vₖ"]
        end
        ModelParameters(U, μ, β, nden, sum(Vk.^2), EPot_DMFT, EKin_DMFT), 
        f["χ_sp_asympt"] ./ f["β"]^2, f["χ_ch_asympt"] ./ f["β"]^2, f["χ_pp_asympt"] ./ f["β"]^2
    end
    freqString =  tml["Environment"]["freqFile"]
    nBose, nFermi, shift = if !isfile(freqString)
        @warn "Frequency file not found, reconstructing grid from config."
        m = match(r"b(?<bf>\d+)f(?<ff>\d+)s(?<s>\d)", freqString)
        parse(Int, m[:bf]), parse(Int, m[:ff]), parse(Int, m[:s])
    else
        load(freqString, "nBose"), load(freqString, "nFermi"), load(freqString, "shift")
    end
    #TODO: BSE inconsistency between direct and SC
    asympt_sc = lowercase(tml["Simulation"]["chi_asympt_method"]) == "asympt" ? 1 : 0
    Nν_shell  = tml["Simulation"]["chi_asympt_shell"]
    Nν_full = nFermi + asympt_sc*Nν_shell
    freq_r = 2*(Nν_full+nBose)#+shift*ceil(Int, nBose)
    fft_range = -freq_r:freq_r

    # chi asymptotics

    χ_helper = if lowercase(tml["Simulation"]["chi_asympt_method"]) == "asympt"
                    BSE_SC_Helper(χsp_asympt, χch_asympt, χpp_asympt, 2*Nν_full, Nν_shell, nBose, Nν_full, shift)
                elseif lowercase(tml["Simulation"]["chi_asympt_method"]) == "direct"
                    BSE_Asym_Helper(χsp_asympt, χch_asympt, χpp_asympt, Nν_shell, mP.U, mP.β, nBose, nFermi, shift)
                elseif lowercase(tml["Simulation"]["chi_asympt_method"]) == "direct_approx2"
                    BSE_Asym_Helper_Approx2(Nν_shell)
                elseif lowercase(tml["Simulation"]["chi_asympt_method"]) == "nothing"
                    nothing
                else
                    @error "could not parse chi_asympt_method $(tml["Simulation"]["chi_asympt_method"]). Options are: asympt/direct/nothing"
                    nothing
                end

    sP = SimulationParameters(nBose,nFermi,Nν_shell,shift,
                               χ_helper,
                               fft_range,
                               tml["Simulation"]["usable_prct_reduction"],
                               dbg_full_eom_omega
    )
    kGrids = Array{Tuple{String,Int}, 1}(undef, length(tml["Simulation"]["Nk"]))
    if typeof(tml["Simulation"]["Nk"]) === String && strip(lowercase(tml["Simulation"]["Nk"])) == "conv"
        kGrids = [(tml["Model"]["kGrid"], 0)]
    else
        for i in 1:length(tml["Simulation"]["Nk"])
            Nk = tml["Simulation"]["Nk"][i]
            kGrids[i] = (tml["Model"]["kGrid"], Nk)
        end
    end

    workerpool = default_worker_pool() #TODO setup reasonable pool with clusterManager/Workerconfi
    return workerpool, mP, sP, env, kGrids
end


# ============================================= Logging ==============================================

function get_log()
    global LOG
    LOG *= String(take!(LOG_BUFFER))
    return LOG
end

function reset_log()
    global LOG
    LOG = ""
end

# ============================================== Misc. ===============================================

"""
    printr_s(x::ComplexF64)
    printr_s(x::Float64)

prints 4 digits of (the real part of) `x`
"""
printr_s(x::ComplexF64) = round(real(x), digits=4)
printr_s(x::Float64) = round(x, digits=4)


# ====================================== Custom Type Custum IO =======================================

"""
	Base.show(io::IO, m::SimulationParameters)

Custom output for SimulationParameters
"""
function Base.show(io::IO, m::SimulationParameters)
    compact = get(io, :compact, false)
    if !compact
        println(io, "B/F range    : $(m.n_iω)/$(m.n_iν) $(m.shift ? "with" : "without") shifted fermionic frequencies")
        println(io, "   ($(m.dbg_full_eom_omega ? "with" : "without") full ω range in EoM.")
        println(io, "Asymptotic correction : $(typeof(m.χ_helper))")
        println(io, "   $(100*m.usable_prct_reduction) % reduction of usable range and ω smoothing $(m.usable_prct_reduction)")
        println(io, "λ-Correction : $(m.λc_type)")
    else
        print(io, "SimulationParams[nB=$(m.n_iω), nF=$(m.n_iν), shift=$(m.shift)]")
    end
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
        print(io, "ModelParams[U=$(m.U), β=$(m.β), μ=$(m.μ), n=$(m.n)]")
    end
end


function Base.show(io::IO, ::MIME"text/plain", m::SimulationParameters)
    println(io, "LadderDGA.jl SimulationParameters:")
    show(io, m)
end

function Base.show(io::IO, ::MIME"text/plain", m::ModelParameters)
    println(io, "LadderDGA.jl ModelParameters:")
    show(io, m)
end
