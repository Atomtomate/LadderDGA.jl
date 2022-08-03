# ==================================================================================================== #
#                                              IO.jl                                                   #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
#   Last Edit Date  : 03.08.22                                                                         #
# ----------------------------------------- Description ---------------------------------------------- #
#   General input output operations for reading config and data files as well as logging               #
# -------------------------------------------- TODO -------------------------------------------------- #
# TODO: document, file is missleading, can also be a string                                            #
# TODO: many of the fortran functions have the old axis layout                                         #
#       (omega axis first instead of last, update that                                                 #
# ==================================================================================================== #



include("IO_legacy.jl")

function readConfig(cfg_in)
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
    χfill = nothing
    rr = r"^fixed:(?P<start>\N+):(?P<stop>\N+)"
    tc_type_f = Symbol(lowercase(tml["Simulation"]["fermionic_tail_correction"]))
    if !(tc_type_f in [:nothing, :richardson, :shanks])
        error("Unrecognized tail correction type \"$(tc_type_f)\"")
    end
    tc_type_b = Symbol(lowercase(tml["Simulation"]["bosonic_tail_correction"]))
    if !(tc_type_b in [:nothing, :richardson, :shanks, :coeffs])
        error("Unrecognized tail correction type \"$(tc_type_b)\"")
    end
    ωsum_inp = lowercase(tml["Simulation"]["bosonic_sum_range"])
    m = match(rr, ωsum_inp)
    ωsum_type = if m !== nothing
        tuple(parse.(Int, m.captures)...)
    elseif ωsum_inp in ["common", "individual", "full"]
        Symbol(ωsum_inp)
    else
        error("Could not parse bosonic_sum. Allowed input is: common, individual, full, fixed:N:M")
    end
    smoothing = Symbol(lowercase(tml["Simulation"]["omega_smoothing"]))
    if !(smoothing in [:nothing, :range, :full])
        error("Unrecognized smoothing type \"$(smoothing)\"")
    end
    λrhs_type = Symbol(lowercase(tml["Simulation"]["rhs"]))
    !(λrhs_type in [:native, :fixed, :error_comp]) && error("Could not parse rhs type for lambda correction. Options are native, fixed, error_comp.")
    dbg_full_eom_omega = (haskey(tml["Debug"], "full_EoM_omega") && tml["Debug"]["full_EoM_omega"]) ? true : false

    env = EnvironmentVars(   tml["Environment"]["inputDir"],
                             tml["Environment"]["inputVars"],
                             tml["Environment"]["cast_to_real"],
                             String([(i == 1) ? uppercase(c) : lowercase(c)
                                     for (i, c) in enumerate(tml["Environment"]["loglevel"])]),
                             lowercase(tml["Environment"]["logfile"]),
                             tml["Environment"]["progressbar"]
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
    sh_f = get_sum_helper(default_fit_range(-Nν_full:Nν_full-1), tml["Simulation"]["fermionic_tail_coeffs"], tc_type_f)
    freq_r = 2*(Nν_full+nBose)#+shift*ceil(Int, nBose)
    fft_range = -freq_r:freq_r
    lo = npartial_sums(sh_f)
    up = 2*Nν_full - lo + 1 

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

    sEH = if ((tc_type_f == :richardson) || (tc_type_b == :richardson)) 
        SumExtrapolationHelper(
                           tml["Simulation"]["bosonic_tail_coeffs"],
                           tml["Simulation"]["fermionic_tail_coeffs"],
                           smoothing,
                           sh_f,
                           lo,
                           up,
                           Array{Float64, 1}(undef, lo),
                           Array{ComplexF64, 1}(undef, lo))
    else
        nothing
    end

    sP = SimulationParameters(nBose,nFermi,Nν_shell,shift,
                               tc_type_f,
                               tc_type_b,
                               χ_helper,
                               ωsum_type,
                               λrhs_type,
                               tml["Simulation"]["force_full_bosonic_chi"],
                               fft_range,
                               tml["Simulation"]["usable_prct_reduction"],
                               dbg_full_eom_omega,
                               sEH
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

function print_chi_bubble(qList, res, simParams)
    for j in 1:size(res,1)
        print(" ========== ω = $(j-(simParams.n_iω + 1)) =============== \n")
        for k in 1:size(res,2)
            print(" ---------- ν = $(k-1) -------------- \n")
            for (qi,q) in enumerate(qList)
                @printf("   q = (%.2f,%.2f): %.2f + %.2fi\n", q[1],q[2], real(res[j, k, qi]), imag(res[j, k, qi]))
            end
        end
    end
end

function get_log()
    global LOG
    LOG *= String(take!(LOG_BUFFER))
    return LOG
end

function reset_log()
    global LOG
    LOG = ""
end
