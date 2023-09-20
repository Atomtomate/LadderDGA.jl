using Pkg
using TimerOutputs
path = joinpath(abspath(@__DIR__), "..")
println("activating: ", path)
Pkg.activate(path)
Pkg.instantiate()
using LadderDGA


inputfile = joinpath(abspath(@__DIR__), "../test/test_data/rpa_chi0_1.h5")
χ₀ = read_χ₀_RPA(inputfile)

println(χ₀.e_kin)