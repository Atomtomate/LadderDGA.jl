module LambdaCorrection

using LinearAlgebra, GenericLinearAlgebra
using OffsetArrays
using FiniteDiff
using Dispersions

import ..χT, ..γT, ..GνqT, ..KGrid, ..ModelParameters, ..SimulationParameters 
import ..subtract_tail, ..update_tail!, ..ω0_index, ..usable_ωindices, ..iν_array
import ..G_from_Σ, ..G_fft, ..G_from_Σladder, ..calc_E, ..EPot1, ..calc_Σ, ..calc_Σ_ω!, ..eom, ..initialize_EoM, ..calc_Σ_par
import ..filling, ..filling_pos
import ..update_wcaches_G_rfft!

export χ_λ, χ_λ!, dχ_λ, reset!
export get_λ_min

export λsp_correction
export bisect, correct_margins, newton_right
export run_sc, residuals, find_root

include("helpers.jl")
include("conditions.jl")
include("residualCurves.jl")


end
