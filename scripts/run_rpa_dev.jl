using Pkg
using TimerOutputs
path = joinpath(abspath(@__DIR__), "..")
println("activating: ", path)
Pkg.activate(path)
Pkg.instantiate()
using LadderDGA

χ₀, wP, mP, sP, env, kGridStr = readConfig_RPA("/home/coding/LadderDGA.jl/test/test_data/config_rpa_example.toml");

χ₀.e_kin

# inputfile = joinpath(abspath(@__DIR__), "../test/test_data/rpa_chi0_1.h5")
# χ₀ = read_χ₀_RPA(inputfile)

# println(χ₀.e_kin)