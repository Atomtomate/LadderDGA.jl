using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using LadderDGA
using JLD2
using Logging


function gen_run_id(mP, sP, kgrid_str)
    run_id_str = "$(round(mP.U,digits=8))_$(round(mP.n,digits=8))_$(round(mP.β,digits=8))_$(round(mP.μ,digits=8))"
    run_id_str = run_id_str*"$(sP.n_iν)_$(sP.n_iω)_$(sP.n_iν_shell)_$(sP.shift)_$(round(sP.usable_prct_reduction,digits=8))_$(sP.dbg_full_eom_omega)_$(sP.dbg_full_chi_omega)"
    run_id_str = run_id_str*"$(kgrid_str)"
    return hash(run_id_str)
end


function run(ARGS; use_cache::Bool = true, cache_name::String = "lDGA_cache.jld2")
    cfg_file = ARGS[1]
    fname = ARGS[2]
    cfg_dir,_ = splitdir(cfg_file)
    cfg_content = readlines(cfg_file)

    wp, mP, sP, env, kGridsStr = readConfig(cfg_file);

    χm = nothing
    χd = nothing
    γm = nothing
    γd = nothing
    restored_run = false


    run_id = gen_run_id(mP, sP, kGridsStr)

    println("[$run_id]: ======================================"); 
    println("[$run_id]: = beta = $(round(mP.β,digits=4)), U = $(round(mP.U,digits=4)), n = $(round(mP.n,digits=4)), Nk = $(kGridsStr[1][2])  ");
    println("[$run_id]: ======================================"); 

    lDGAhelper = setup_LDGA(kGridsStr[1], mP, sP, env, silent=false);
    if use_cache
        cache_file = joinpath(cfg_dir, cache_name)
        println("[$run_id]: Trying to open cache file $cache_file."); flush(stdout)
        run_nk = lDGAhelper.kG.Ns
        if isfile(cache_file) 
            try
                jldopen(cache_file, "r") do cache_f
                    if haskey(cache_f, "$run_id")
                        χm = cache_f["$run_id/chi_m"]
                        χd = cache_f["$run_id/chi_d"]
                        γm = cache_f["$run_id/gamma_m"]
                        γd = cache_f["$run_id/gamma_d"]
                        println("[$run_id]: restored χ_r and γ_r."); flush(stdout)
                        restored_run =  (!isnothing(χm) && !isnothing(χd)) ? true : false
                    end
                end
            catch e
                println("[$run_id]: WARNING! opening $cache_file resulted in $e"); 
                #println("[$run_id]: WARNING! deleting $cache_file"); flush(stdout)
                #rm(cache_file)
            end
        end
    end

    bubble = calc_bubble(:DMFT, lDGAhelper);

    if isnothing(χm)
        println("[$run_id]: Inverting BSE, no cached values found."); flush(stdout)
        χm, γm = calc_χγ(:m, lDGAhelper, bubble; ω_symmetric=true, use_threads=true);
        χd, γd = calc_χγ(:d, lDGAhelper, bubble; ω_symmetric=true, use_threads=true);
    end
    if use_cache && !restored_run
        cache_file = joinpath(cfg_dir, cache_name)
        run_nk = lDGAhelper.kG.Ns
        println("[$run_id]: Trying to store χ_r and γ_r in $cache_file.");
        faulty_cache = false
        if isfile(cache_file) 
            jldopen(cache_file, "r") do cache_f
                if !haskey(cache_f, "$run_id")
                    println("[$run_id]: WARNING! run_id found in $cache_file but restored run = $restored_run. This indicates a faulty previous run.!"); flush(stdout)
                    faulty_cache = true
                end
                if haskey(cache_f, "$run_id") && cache_f["$run_id/Nk"] != lDGAhelper.kG.Ns
                    println("[$run_id]: WARNING! run_id found in $cache_file but number of k-points do not match $(cache_f["$run_id/Nk"]) != $(lDGAhelper.kG.Ns)!"); flush(stdout)
                    faulty_cache = true
                end
            end
            # Removing groupds is not supported by JLD2, creating new fiile
            if faulty_cache
                bak_file = joinpath(cfg_dir, "bak_$run_id"*cache_name)
                mv(cache_file, bak_file, force=true)
                if isfile(bak_file)
                    jldopen(cache_file, "w") do cache_f
                        jldopen(bak_file, "r") do bak_cache_f
                            for ki in keys(bak_cache_f)
                                if ki != run_id
                                    for kj in keys(bak_cache_f["$ki"])
                                        cache_f["$ki/$kj"] = bak_cache_f["$ki/$kj"]
                                    end
                                end
                            end
                        end
                    end
                else
                    error("[$run_id]: ERROR! Could not create backup file for cache rebuild!")
                end
                rm(bak_file)
            end
        end

        jldopen(cache_file, "a+") do cache_f
            cache_f["$run_id/Nk"] = run_nk
            cache_f["$run_id/config"] = cfg_content
            cache_f["$run_id/chi_m"] = χm
            cache_f["$run_id/chi_d"] = χd
            cache_f["$run_id/gamma_m"] = γm
            cache_f["$run_id/gamma_d"] = γd
            println("[$run_id]: Saved data."); flush(stdout)
        end
    end

end

# Set to Logging.Warn or Info for more verbose output
logger = ConsoleLogger(stdout, Logging.Error)
with_logger(logger) do
    run(ARGS)
end
