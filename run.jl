using Distributed
using JLD2, FileIO
using Logging
@everywhere using Pkg
@everywhere Pkg.activate(@__DIR__)
Pkg.instantiate()
@everywhere using LadderDGA
#using LadderDGA
println("Modules loaded")
flush(stdout)
flush(stderr)

if myid() == 1
    io = open("lDGA.log","w+")
    #io = stdout
    logger = ConsoleLogger(io, Logging.Info, 
                      meta_formatter=Logging.default_metafmt,
                      show_limited=true, right_justify=0)
    #logger = SimpleLogger(io)
    global_logger(logger)
end

function run_sim(; cfg_file=nothing, res_prefix="", res_postfix="", save_results=true)
    @warn "assuming linear, continuous nu grid for chi/trilex"
    if cfg_file === nothing
        print("specify location of config file: ")
        cfg_file = readline()
    end
    mP, sP, env, kGrids, qGridLoc,  freqList = readConfig(cfg_file)#

    for kIteration in 1:length(kGrids)
        kG = kGrids[kIteration]

        @info " ===== Iteration $(kIteration)/$(length(kGrids)) with $(kG.Nk) k points. ===== "
        νGrid, sumHelper_f, impQ_sp, impQ_ch, GImp_fft, GLoc, GLoc_fft, Σ_loc, FUpDo  = setup_LDGA(kG, freqList, mP, sP, env);

        @info "Calculating local quantities: "
        flush(io)
        bubbleLoc = calc_bubble(νGrid, GImp_fft, qGridLoc, mP, sP)
        locQ_sp = calc_χ_trilex(impQ_sp.Γ, bubbleLoc, qGridLoc, νGrid, sumHelper_f, mP.U, mP, sP);
        locQ_ch = calc_χ_trilex(impQ_ch.Γ, bubbleLoc, qGridLoc, νGrid, sumHelper_f, -mP.U, mP, sP);

        Σ_ladderLoc = calc_Σ(locQ_sp, locQ_ch, bubbleLoc, GImp_fft, FUpDo, qGridLoc, sumHelper_f, mP, sP)[:,1] .+ mP.n * mP.U/2.0;

        @info "Calculating bubble: "
        flush(io)
        bubble = calc_bubble(νGrid, GLoc_fft, kG, mP, sP);

        @info "Calculating χ and γ: "
        flush(io)
        nlQ_sp = calc_χ_trilex(impQ_sp.Γ, bubble, kG, νGrid, sumHelper_f, mP.U, mP, sP);
        nlQ_ch = calc_χ_trilex(impQ_ch.Γ, bubble, kG, νGrid, sumHelper_f, -mP.U, mP, sP);

        
        @info "Calculating λsp correction: "
        flush(io)
        @time λ_correction!(:sp, impQ_sp, impQ_ch, FUpDo, Σ_loc, Σ_ladderLoc, nlQ_sp, nlQ_ch, bubble, GLoc_fft, kG, mP, sP)
        @info "Calculating Σ ladder: "
        Σ_ladder = calc_Σ(nlQ_sp, nlQ_ch, bubble, GLoc_fft, FUpDo, kG, sumHelper_f, mP, sP)
        Σ_ladder_corrected = Σ_ladder .- Σ_ladderLoc .+ Σ_loc[1:size(Σ_ladder,1)];
        @info "Done."
        flush(io)

        if save_results
        fname = res_prefix*"lDGA_k$(kG.Ns)_sp_"*res_postfix*".jld2"
        @info "Writing to $(fname)"
        save(fname, "sP", sP, "mP", mP, "kG", kG ,"bubbleLoc", bubbleLoc, "locQ_sp", locQ_sp, "locQ_ch", locQ_ch, "Σ_ladderLoc", Σ_ladderLoc, "bubble", bubble, "nlQ_ch", nlQ_ch, "nlQ_sp", nlQ_sp, "Σ_ladder", Σ_ladder)
        end
    end
end

function run2(cfg_file)
    run_sim(cfg_file=cfg_file)
end
