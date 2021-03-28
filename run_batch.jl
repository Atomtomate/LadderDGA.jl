using JLD
using DelimitedFiles

path = ARGS[1]
config = ARGS[2]
println(ARGS)
println(path)
println(config)
flush(stdout)
flush(stderr)
include("/scratch/usr/hhpstobb/lDGA/LadderDGA.jl/run.jl")



bubbleLoc, locQ_sp, locQ_ch, bubble, nlQ_ch, nlQ_sp, nlQ_ch_λ, nlQ_sp_λ, Σ_bare, Σ_ladder, Σ_ladderLoc = run_sim(config)

println(dbg1)

save(path*"/lDGA_vars.jld", "bubbleLoc", bubbleLoc, "locQ_sp", locQ_sp, "locQ_ch", locQ_ch, 
                           "bubble", bubble, "nlQ_sp", nlQ_sp, "nlQ_ch", nlQ_ch, "nlQ_sp_l", nlQ_sp_λ, 
                           "Sigma_bare", Σ_bare, "SigmaLadder", Σ_ladder, "SigmaLadderLoc", Σ_ladderLoc, 
                           "usable_w", usable_ω, "usable_w_l" usable_ω_λc, "usable_w_S", usable_ω_Σ, 
                           "tmp", tmp, "SigmaLadder_w", Σ_ladder_ω)
writedlm("chisp_pi", [real(usable_ω) real.(nlQ_sp_λ.χ[usable_ω,end])], '\t')
