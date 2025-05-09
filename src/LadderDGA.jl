# ==================================================================================================== #
#                                           LadderDGA.jl                                               #
# ---------------------------------------------------------------------------------------------------- #
#   Authors         : Julian Stobbe, Jan Frederik Weissler                                             #
# ----------------------------------------- Description ---------------------------------------------- #
#  Entry point for LadderDGA.jl                                                                        #
# -------------------------------------------- TODO -------------------------------------------------- #
#   Find a consistent way to pass Simulation and ModelParameters (at least consistent order)           #
# ==================================================================================================== #
module LadderDGA

include("DepsInit.jl")

# KGrid
export gen_kGrid, kintegrate

# Types
export ModelParameters, SimulationParameters, EnvironmentVars, lDΓAHelper
export ΓT, FT, χ₀T, χT, γT, GνqT, λ₀T
export RPAHelper

# Setup and auxilliary functions
export filling, filling_pos, G_fft
export find_usable_χ_interval, usable_ωindices, subtract_tail, subtract_tail!
export Freq_to_OneToIndex, OneToIndex_to_Freq
export addprocs

# BSE helper functions
export F_from_χ, calc_local_EoM, check_χ_health, fix_χr, fix_χr!

# GF Helper Functions
export ω2_tail, G_from_Σ, G_from_Σladder, Σ_loc_correction, calc_G_Σ, iν_array, iω_array

# LadderDGA main functions
export ωn_grid, sum_ω, sum_ω!, sum_kω, sum_ωk, core, ω0_index, q0_index
export readConfig, setup_LDGA, calc_bubble, calc_χγ, calc_Σ, calc_Σ_parts, calc_λ0, run_sc
# TODO: parallel version needs refactor and tests
export calc_bubble_par, calc_χγ_par, initialize_EoM, calc_Σ_par, clear_wcache!
export collect_χ₀, collect_χ, collect_γ, collect_Σ
export λ_from_γ, F_from_χ, F_from_χ_gen, G_from_Σ, G_from_Σladder, calc_G_Σ, Σ_from_Gladder, attach_Σloc
export ΣTail, ΣTail_Plain, ΣTail_Full, ΣTail_ExpStep, ΣTail_EoM, ΣTail_λm

# Asymptotic lDGA
export setup_ALDGA
export update_ΓAsym!, run_AlDGA_convergence

# RPA main functions
export setup_RPA!, χ₀RPA_T, read_χ₀_RPA, readConfig_RPA, setupConfig_RPA

# Thermodynamics
export EPot_p1, EPot_p2, PP_p1, PP_p1, EKin_p1, EKin_p2
# TODO: remove old interface in favor of EPot_p1,... etc
export calc_E_ED, calc_E, calc_Epot2

# LambdaCorrection
export χ_λ, χ_λ!, reset!
export λm_correction, λdm_correction, λm_sc_correction, λdm_sc_correction, λm_tsc_correction, λdm_tsc_correction
export λdm_sc_correction, λdm_correction, λm_correction, λ_correction, λ_correction!, λ_result, λm_correction_full_RPA
export Cond_PauliPrinciple, Cond_EPot, Cond_EKin, Cond_Tail, Cond_Causal

export sample_f, EPotCond_λdm_curve

#TODO: check interface after refactoring
export λ_correction, λ_result, converged, sc_converged, PP_diff, EPot_diff

# Postprocessing and Linearized Eliashberg stuff
export calc_λmax_linEliashberg_MatrixFree, calc_λmax_linEliashberg, calc_Γs_ud, calc_gen_χ, F_from_χ_star_gen
export BubbleDiff

# Additional functionality
export estimate_ef, fermi_surface_connected, estimate_connected_ef
end
