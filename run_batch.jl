using LinearAlgebra
using TimerOutputs
using Logging
using Pkg
Pkg.activate(@__DIR__)
using LadderDGA


cfg_file = ARGS[1]
out_path = ARGS[2]
NProcs = parse(Int,ARGS[3])
BLAS.set_num_threads(NProcs)

#TODO: read the log file name from config
#TODO: also use this name for output file in run.jl
mP, sP, env, kGridsStr = readConfig(cfg_file);
tc_s = (sP.tc_type_f != :nothing) ? "rtc" : "ntc"
logfile_path = out_path*"/lDGA_"*tc_s*"_$(kGridsStr[1][2])to$(kGridsStr[end][2]).log"
i = 1
while isfile(logfile_path)
    global i
    global logfile_path
    postf = "_$i"
    logfile_path = i > 1 ? logfile_path[1:end-(length("_$(i-1)"))]*postf : logfile_path*postf
    i += 1
end

println("config path: $cfg_file\noutput path: $out_path\nlogging to $(logfile_path)")
flush(stdout)
flush(stderr)
include("./run.jl")
include("/scratch/projects/hhp00048/codes/LadderDGA_utils/new_lambda_analysis.jl")
description = "exploration of PT region"

open(logfile_path,"w") do io
    redirect_stdout(io) do
        run_sim(descr=description, cfg_file=cfg_file, res_postfix="", res_prefix=out_path*"/", save_results=true, log_io=io)
    end
end
