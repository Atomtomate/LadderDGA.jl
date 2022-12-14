module LambdaCorrection

using LinearAlgebra, GenericLinearAlgebra
using OffsetArrays
using FiniteDiff

import ..χT, ..γT, ..GνqT, ..KGrid, ..ModelParameters, ..SimulationParameters 

export χ_λ, χ_λ!, dχ_λ
export get_λ_min

export bisect, correct_margins, newton_right
export residuals, find_root

include("helpers.jl")
include("conditions.jl")
include("residualCurves.jl")


end
