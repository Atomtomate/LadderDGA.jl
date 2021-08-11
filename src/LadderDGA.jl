module LadderDGA
#Base.Experimental.@optlevel 3

include("DepsInit.jl")

export kintegrate, sum_freq, get_sum_helper
export ModelParameters, SimulationParameters, FreqGrid, EnvironmentVars
export LocalQuantities, NonLocalQuantities, ΓT, BubbleT, GνqT
export readConfig, setup_LDGA, calc_bubble, calc_χ_trilex, calc_Σ, full_run
export λ_correction, λ_correction!, calc_λsp_rhs_usable, calc_λsp_correction!
export calculate_Σ_ladder, writeFortranΣ

end
