module LadderDGA

    include("DepsInit.jl")

    # KGrid
    export gen_kGrid, kintegrate

    # Types
    export ModelParameters, SimulationParameters, EnvironmentVars
    export ΓT, FT, χ₀T, χT, γT, GνqT
    
    # Setup and auxilliary functions
    export readConfig, setup_LDGA, calc_bubble, calc_χγ, calc_Σ_ω, calc_Σ, calc_Σ_parts, calc_Σνω, calc_λ0, Σ_loc_correction, filling
    export find_usable_χ_interval, usable_ωindices, subtract_tail, subtract_tail!
    export addprocs

    # LadderDGA main functions
    export calc_bubble_par, calc_χγ_par, calc_Σ_par
    export λ_from_γ, F_from_χ, G_from_Σ, G_from_Σladder

    # Thermodynamics
    export calc_E_ED, calc_E, calc_Epot2

    # LambdaCorrection
    export χ_λ, χ_λ!
    export newton_right
    #TODO: check interface after refactoring
    export λ_correction, λ_correction!, residuals, find_root

end
