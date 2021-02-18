module LadderDGA
Base.Experimental.@optlevel 3

include("DepsInit.jl")

export gen_kGrid, squareLiattice_ekGrid, reduce_kGrid, kGrid_multiplicity #TODO: this should be in the dispersions package
export ModelParameters, SimulationParameters, FreqGrid, EnvironmentVars
export LocalQuantities, NonLocalQuantities, ΓT, BubbleT, GνqT
export setup_LDGA, calc_bubble, calc_χ_trilex, calc_Σ, full_run
export calc_λsp_rhs_usable, calc_λsp_correction!
export calculate_Σ_ladder, writeFortranΣ

end
