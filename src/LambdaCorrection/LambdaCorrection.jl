module LambdaCorrection

using LinearAlgebra, GenericLinearAlgebra
using OffsetArrays, StaticArrays
using FiniteDiff
using Dispersions
using Roots
using DataFrames
using TimerOutputs

import ..χT, ..γT, ..GνqT, ..KGrid, ..ModelParameters, ..SimulationParameters, ..lDΓAHelper
import ..sum_ω, ..sum_ω!, ..sum_kω, ..sum_ωk, ..subtract_tail, ..update_tail!, ..ω0_index, ..usable_ωindices, ..iν_array
import ..G_from_Σ, ..G_fft, ..G_rfft!, ..G_from_Σladder!, ..G_from_Σladder, ..calc_E, ..EPot1, ..calc_Σ, ..calc_Σ_ω!, ..calc_Σ!, ..eom
import ..initialize_EoM, ..calc_Σ_par, ..calc_Σ_par!
import ..filling, ..filling_pos
import ..update_wcaches_G_rfft!

export χ_λ, χ_λ!, dχ_λ, reset!
export get_λ_min

export λ_correction, λm_correction, λdm_correction, λ_result 
export bisect, correct_margins, newton_right
export run_sc

include("helpers.jl")
include("conditions.jl")
include("conditions_new.jl")


end
