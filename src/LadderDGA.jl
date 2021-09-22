module LadderDGA
#Base.Experimental.@optlevel 3

include("DepsInit.jl")

export kintegrate, sum_freq, get_sum_helper
export ModelParameters, SimulationParameters, EnvironmentVars
export LocalQuantities, NonLocalQuantities, ΓT, BubbleT, GνqT, FUpDoT
export readConfig, setup_LDGA, calc_bubble, calc_χ_trilex, calc_Σ_ω, calc_Σ, Σ_correction, Σ_loc_correction
export λsp, λ_correction, λ_correction!, calc_λsp_rhs_usable, calc_λsp_correction!
export calc_E
export χ_λ!, subtract_tail

end
