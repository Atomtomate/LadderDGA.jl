module LambdaCorrection

using LinearAlgebra, GenericLinearAlgebra
using OffsetArrays
using FiniteDiff
using Dispersions

import ..χT, ..γT, ..GνqT, ..KGrid, ..ModelParameters, ..SimulationParameters 
import ..subtract_tail, ..update_tail!, ..ω0_index, ..usable_ωindices, ..iν_array
import ..G_from_Σ, ..G_from_Σladder, ..calc_E, ..EPot1, ..calc_Σ, ..calc_Σ_ω!, ..eom
import ..filling, ..filling_pos

export χ_λ, χ_λ!, dχ_λ
export get_λ_min

export λsp_correction
export bisect, correct_margins, newton_right
export residuals, find_root

include("helpers.jl")
include("conditions.jl")
include("residualCurves.jl")


end
