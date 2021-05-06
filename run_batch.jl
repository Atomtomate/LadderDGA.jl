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
run_sim(cfg_file=config)
