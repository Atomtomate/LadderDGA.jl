module LadderDGA

    include("DepsInit.jl")

    # KGrid
    export kintegrate

    # Types
    export ModelParameters, SimulationParameters, EnvironmentVars
    export ΓT, FT, χ₀T, χT, γT, GνqT
    
    # Setup and auxilliary functions
    export readConfig, setup_LDGA, calc_bubble, calc_χγ, calc_Σ_ω, calc_Σ, calc_Σ_parts, calc_Σνω, calc_λ0, Σ_loc_correction, filling
    export find_usable_χ_interval, subtract_tail, subtract_tail!

    # LadderDGA main functions
    export calc_bubble_par, calc_χγ_par, calc_Σ_par
    export λ_from_γ, F_from_χ, G_from_Σ, GLoc_from_Σladder

    # Thermodynamics
    export calc_E, calc_Epot2

    # LambdaCorrection
    export χ_λ, χ_λ!
    export newton_right
    #TODO: check interface after refactoring
    export λ_correction, λ_correction!, calc_λsp_rhs_usable, c2_curve, find_root

end
