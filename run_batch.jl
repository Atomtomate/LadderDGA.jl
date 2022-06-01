using TimerOutputs
using Logging
using Pkg
Pkg.activate(@__DIR__)
using LadderDGA

using Distributed
addprocs(parse(Int,ARGS[3]))
@everywhere using Pkg
@everywhere Pkg.activate(@__DIR__)
using LadderDGA
@everywhere using LadderDGA

cfg_file = ARGS[1]
out_path = ARGS[2]

#TODO: read the log file name from config
#TODO: also use this name for output file in run.jl
@timeit LadderDGA.to "input" wp, mP, sP, env, kGridsStr = readConfig(cfg_file);
println("using workerpool: ", wp)
tc_s = (sP.tc_type_f != :nothing) ? "rtc" : "ntc"
#BSE_Asym_Helper_Approx1
asym_str = if typeof(sP.χ_helper) === LadderDGA.BSE_Asym_Helper
    "direct_asym"
elseif typeof(sP.χ_helper) === LadderDGA.BSE_Asym_Helper_Approx1
    "direct_asym_approx1"
elseif typeof(sP.χ_helper) === LadderDGA.BSE_Asym_Helper_Approx2
    "direct_asym_approx2"
elseif typeof(sP.χ_helper) === LadderDGA.BSE_Asym_Helper
    "sc_asym"
else
    "no_asym"
end

name = "lDGA_" * tc_s * "_" * asym_str * "_$(kGridsStr[1][2])to$(kGridsStr[end][2])"
logfile_path = out_path*"/"*name*".log"
i = 1
while isfile(logfile_path)
    global i
    global logfile_path
    postf = "_$i.log"
    logfile_path = i > 1 ? logfile_path[1:end-4-(length("_$(i-1)"))]*postf : logfile_path*postf
    i += 1
end

println("config path: $cfg_file\noutput path: $out_path\nlogging to $(logfile_path)")
flush(stdout)
flush(stderr)
include("./run.jl")
include("/scratch/projects/hhp00048/codes/scripts/LadderDGA_utils/new_lambda_analysis.jl")
description = "lDGA at U=$(mP.U), β=$(mP.β) with direct asymptotic improvement."

open(logfile_path,"w") do io
    redirect_stdout(io) do
        run_sim(fname = out_path *"/"*name,descr=description, cfg_file=cfg_file, res_postfix="", res_prefix=out_path*"/", save_results=true, log_io=io)
    end
end
