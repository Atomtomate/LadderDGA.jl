module LadderDGA

    include("DepsInit.jl")

    export kintegrate
    export ModelParameters, SimulationParameters, EnvironmentVars
    export LocalQuantities, NonLocalQuantities, ΓT, χ₀T, GνqT, FUpDoT
    export readConfig, setup_LDGA, calc_bubble, calc_χγ, calc_Σ_ω, calc_Σ, calc_Σ_parts, calc_Σνω, calc_λ0, Σ_loc_correction
    export calc_bubble_par, calc_χγ_par, calc_Σ_par
    export λsp, λ_correction, λ_correction!, calc_λsp_rhs_usable, calc_λsp_correction!, c2_curve
    export λ_from_γ, F_from_χ, G_from_Σ
    export calc_E, calc_Epot2, flatten_2D
    export χ_λ, χ_λ!, subtract_tail, subtract_tail!

end
