using DelimitedFiles
using Distributed

config = ARGS[1]
out_path = ARGS[2]
println("config path: ", config)
println("output path: ", out_path)
flush(stdout)
flush(stderr)
include("./run.jl")

#bubbleLoc, locQ_sp, locQ_ch, bubble, nlQ_ch, nlQ_sp, nlQ_ch_λ, nlQ_sp_λ, Σ_bare, Σ_ladder, Σ_ladderLoc = run_sim(config)
run_sim(cfg_file=config, res_prefix=out_path*"/", save_results=true)
