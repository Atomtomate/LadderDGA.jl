using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using LadderDGA
using JLD2
using LinearAlgebra

LinearAlgebra.BLAS.set_num_threads(parse(Int, ARGS[3]))


function run(ARGS; use_cache::Bool = true, cache_name::String = "lDGA_cache.jld2")
    cfg_file = ARGS[1]
    println("starting for $cfg_file")
    cfg_dir,_ = splitdir(cfg_file)
    wp, mP, sP, env, kGridsStr = readConfig(cfg_file);
    lDGAhelper = setup_LDGA(kGridsStr[1], mP, sP, env, silent=false);

    χm = nothing
    χd = nothing
    γm = nothing
    γd = nothing
    restored_run = false

    cache_file = joinpath(cfg_dir, cache_name)
    run_id = hash(string(readlines(cfg_file)...))
    run_nk = lDGAhelper.kG.Ns
    if isfile(cache_file) 
        jldopen(cache_file, "r") do cache_f
            if haskey(cache_f, "$run_id")
                if cache_f["$run_id/Nk"] != run_nk
                    println("run id $run_id does match, but Nk does not!!!")
                else
                    restored_run = true
                end
            end
        end
    end
    if !restored_run
        bubble = calc_bubble(:DMFT, lDGAhelper);
        @time χm, γm = calc_χγ(:m, lDGAhelper, bubble);
        @time χd, γd = calc_χγ(:d, lDGAhelper, bubble);
    end
    if !restored_run
        cache_file = joinpath(cfg_dir, cache_name)
        cfg_content = readlines(cfg_file)
        jldopen(cache_file, "a+") do cache_f
            cache_f["$run_id/Nk"] = run_nk
            cache_f["$run_id/config"] = cfg_content
            cache_f["$run_id/chi_m"] = χm
            cache_f["$run_id/chi_d"] = χd
            cache_f["$run_id/gamma_m"] = γm
            cache_f["$run_id/gamma_d"] = γd
        end
    end
    println("completed $cfg_file")
end

run(ARGS)
