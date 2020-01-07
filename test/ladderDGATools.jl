include("$(@__DIR__)/../src/Config.jl")
include("$(@__DIR__)/../src/helpers.jl")
include("$(@__DIR__)/../src/IO.jl")
include("$(@__DIR__)/../src/dispersion.jl")
include("$(@__DIR__)/../src/GFTools.jl")
include("$(@__DIR__)/../src/ladderDGATools.jl")
include("$(@__DIR__)/../src/GFFit.jl")
#include("$(@__DIR__)/../test/old_ladderDGATools.jl")
include("old_ladderDGATools.jl")

#= using JLD =#
#= LOAD_FORTRAN = false =#
#= modelParams, simParams = readConfig("./test/test_config.toml") =#
#= if LOAD_FORTRAN =#
#=     g0 = zeros(Complex{Float64}, simParams.n_iν+simParams.n_iω) =#
#=     gi = zeros(Complex{Float64},  simParams.n_iν+simParams.n_iω) =#
#=     readFortranSymmGF!(g0, dir*"g0mand", storedInverse=true) =#
#=     readFortranSymmGF!(gi, dir*"gm_wim", storedInverse=false) =#
#=     ωBox, Γcharge, Γspin = readFortranΓ(dir*"gamma_dir") =#
#=     save("vars.jld", "g0", g0, "gi", gi, "GammaCharge", Γcharge, "GammaSpin", Γspin, "omegaBox", ωBox) =#
#= else =#
#=     vars = load("vars.jld") =# 
#=     g0 = vars["g0"] =#
#=     gi = vars["gi"] =#
#=     Γcharge = vars["GammaCharge"] =#
#=     Γspin   = vars["GammaSpin"] =#
#=     ωBox    = vars["omegaBox"] =#
#= end =#
#= const Σ  = Σ_Dyson(g0, gi) =#

#= const kGrid  = gen_kGrid(simParams.Nint, modelParams.D; min = 0, max = 2π, include_min = false) =# 
#= const qGrid  = reduce_kGrid(gen_kGrid(simParams.Nint, modelParams.D; min = 0, max = π, include_min = true)) =#
#= const ϵkList = squareLattice_ekGrid(kGrid) =#
#= const ϵkqList = gen_squareLattice_ekq_grid(kGrid, qGrid) =#
#= const kList = collect(kGrid) =#
#= const qList = collect(qGrid) =#

#= @testset "ladderDGATools.jl" begin =#

#=     @testset "bubble" begin =#
#=         bubble_check = load("test_vars_large.jld", "chi_bubble") =#
#=         @time bubble = calc_bubble_naive(Σ, modelParams, simParams) =#
#=         @time bubble2 = calc_bubble_naive_macro(Σ, kList, modelParams, simParams) =#
#=         @time bubble3 = calc_bubble_parallel(Σ, kList, qList, ϵkList, ϵkqList, modelParams, simParams) =#
#=         @time bubble3 = calc_bubble_parallel2(Σ, kList, qList, ϵkList, ϵkqList, modelParams, simParams) =#
#=         @test all(bubble2 .≈ bubble) =#
#=         @test all(bubble3 .≈ bubble2) =#
#=     end =#
#=     @testset "trilex" begin =#
#=         bubble = calc_bubble_naive(Σ, modelParams, simParams) =#
#=         @time trilexCharge = calc_χ_trilex_v1(Γcharge, bubble, modelParams.β, modelParams.U, kList) =#
#=         @time trilexCharge2 = calc_χ_trilex_parallel_2(Γcharge, bubble, modelParams.β, modelParams.U, kList) =#
#=         @time trilexCharge3 = calc_χ_trilex_parallel(Γcharge, bubble, modelParams.β, modelParams.U, kList) =#
#=         @test all(trilexCharge .≈ trilexCharge2) =#
#=         @test all(trilexCharge2 .≈ trilexCharge3) =#
#=     end =#
#= end =#
