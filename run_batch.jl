using JLD
using DelimitedFiles
using Distributed

config = ARGS[1]
out_file = ARGS[2]
println("config path: ", config)
println("output file: ", out_file)
flush(stdout)
flush(stderr)
include("./run.jl")

#bubbleLoc, locQ_sp, locQ_ch, bubble, nlQ_ch, nlQ_sp, nlQ_ch_λ, nlQ_sp_λ, Σ_bare, Σ_ladder, Σ_ladderLoc = run_sim(config)
nlQ_ch, nlQ_sp, Σ_ladder = run2(config)

println("computation complete. saving.")


#save(path*"/lDGA_vars.jld", "bubbleLoc", bubbleLoc, "locQ_sp", locQ_sp, "locQ_ch", locQ_ch, 
#                           "bubble", bubble, "nlQ_sp", nlQ_sp, "nlQ_ch", nlQ_ch, "nlQ_sp_l", nlQ_sp_λ, 
#                           "Sigma_bare", Σ_bare, "SigmaLadder", Σ_ladder, "SigmaLadderLoc", Σ_ladderLoc, 
#                           "usable_w", usable_ω, "usable_w_l" usable_ω_λc, "usable_w_S", usable_ω_Σ, 
#                           "tmp", tmp, "SigmaLadder_w", Σ_ladder_ω)
#writedlm("chisp_pi", [real(usable_ω) real.(nlQ_sp_λ.χ[usable_ω,end])], '\t')
save(out_file*".jld", "nlQ_ch", nlQ_ch, "nlQ_sp", nlQ_sp, "SigmaLadder", Σ_ladder)
