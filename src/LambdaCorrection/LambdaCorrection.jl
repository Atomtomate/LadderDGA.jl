module LambdaCorrection

using LinearAlgebra, GenericLinearAlgebra
using OffsetArrays, StaticArrays, DataStructures
using FiniteDiff
using Dispersions
using Roots
using DataFrames
using TimerOutputs
using Printf, Term

import ..χT, ..γT, ..λ₀T, ..GνqT, ..KGrid, ..ModelParameters, ..SimulationParameters, ..eom_ν_cutoff, ..default_Σ_tail_correction
import ..ΣTail, ..ΣTail_EoM, ..ΣTail_Full, ..ΣTail_ExpStep, ..ΣTail_Plain, ..ΣTail_λm
import ..sum_ω, ..sum_ω!, ..sum_kω, ..sum_ωk, ..subtract_tail, ..update_tail!, ..ω0_index, ..usable_ωindices, ..iν_array, ..ω2_tail, ..ωn_grid
import ..G_from_Σ, ..G_fft, ..G_rfft!, ..G_from_Σladder!, ..G_from_Σladder, ..calc_G_Σ, ..calc_G_Σ!, ..calc_Σ, ..calc_Σ!, ..eom, ..eom_rpa
import ..calc_E, ..PP_p1, ..PP_p2, ..EPot_p1, ..EPot_p2, ..EKin_p1, ..EKin_p2, ..EPot_p1_tail, ..EKin_p1_tail
import ..filling, ..filling_pos, ..tail_factor, ..tail_correction_term, ..default_Σ_tail_correction
import ..RPAHelper, ..lDΓAHelper, ..AlDΓAHelper, ..RunHelper
import ..estimate_ef, ..estimate_connected_ef

export χ_λ, χ_λ!, dχ_λ, reset!
export get_λ_min

export λ_correction, λ_result
export newton_right, newton_secular, sample_f
export run_sc

export EPot_diff, PP_diff, n_diff, validate, converged, sc_converged
export λm_correction, λdm_correction, λdm_correction_clean
export λm_sc_correction, λdm_sc_correction, λdm_sc_correction_clean
export λm_tsc_correction, λdm_tsc_correction, λdm_tsc_correction_clean

export Cond_PauliPrinciple, Cond_EPot, Cond_EKin, Cond_Tail, Cond_Causal

include("common.jl")
include("helpers.jl")
include("RootFinding.jl")
include("Types.jl")
include("conditions_singleCore.jl")
include("conditionsRPA_singleCore.jl")
include("ConditionCurves.jl")
# include("conditions_new_mu_test.jl")
# include("conditions_dbg.jl")


end
