using Pkg
using TimerOutputs
path = joinpath(abspath(@__DIR__), "..")
println("activating: ", path)
Pkg.activate(path)
Pkg.instantiate()
using LadderDGA

χ₀, wP, mP, sP, env, kGridStr = readConfig_RPA("/home/coding/LadderDGA.jl/test/test_data/config_rpa_example.toml");

# --------- pull into setup_RPA --------- 
kG = gen_kGrid(kGridStr, χ₀.Nq)

if length(LadderDGA.gridPoints(kG)) ≠ size(χ₀.data)[χ₀.axis_types[:q]]
    error("Number of q points in kGrid does not match number of q-points in χ₀!")
end
# ---------------------------------------

gLoc      = []
gLoc_fft  = []
gLoc_rfft = [] 
helper = RPAHelper(sP, mP, kG, gLoc,gLoc_fft, gLoc_rfft)

χm, γm = calc_χγ(:m, χ₀, kG, mP);
χd, γd = calc_χγ(:d, χ₀, kG, mP);

println("U:$(χd.U)")

λ_result = LadderDGA.λm_correction_full_RPA(χm, χd, helper; verbose=true, validate_threshold=1e-8)
println("done.")
