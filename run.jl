using Distributed
using JLD2, FileIO
using Logging
#if nprocs() == 1
#    addprocs(7)
#end
@everywhere using Pkg
@everywhere Pkg.activate(@__DIR__)
#Pkg.instantiate()
@everywhere using LadderDGA


#io = open("lDGA.log","w+")
io = stdout
#logger = ConsoleLogger(io, Logging.Info, 
#                  meta_formatter=Logging.default_metafmt,
#                  show_limited=true, right_justify=0)
logger = SimpleLogger(io)
global_logger(logger)

function run_sim(; cfg_file=nothing, res_prefix="", res_postfix="", save_results=true)
    @warn "assuming linear, continuous nu grid for chi/trilex"
    if cfg_file === nothing
        print("specify location of config file: ")
        cfg_file = readline()
    end
    mP, sP, env, kGrids, qGrids, qGridLoc, freqRed_map, freqList, freqList_min, parents, ops, nFermi, nBose, shift, base, offset = readConfig(cfg_file)#

    last_λsp = nothing
    last_λspch = nothing
    for kIteration in 1:length(kGrids)
        kG = kGrids[kIteration]
        qG = qGrids[kIteration]
        @info " ===== Iteration $(kIteration)/$(length(kGrids)) with $(kG.Nk) k points. ===== "
        νGrid, sumHelper_f, impQ_sp, impQ_ch, GImp_fft, GLoc_fft, Σ_loc, FUpDo, gImp, gLoc = setup_LDGA(kG, freqList, mP, sP, env);

        @info "Calculating local quantities: "
        flush(io)
        bubbleLoc = calc_bubble(νGrid, GImp_fft, qGridLoc, mP, sP)
        locQ_sp = calc_χ_trilex(impQ_sp.Γ, bubbleLoc, qGridLoc, νGrid, sumHelper_f, mP.U, mP, sP);
        locQ_ch = calc_χ_trilex(impQ_ch.Γ, bubbleLoc, qGridLoc, νGrid, sumHelper_f, -mP.U, mP, sP);

        Σ_ladderLoc = calc_Σ(locQ_sp, locQ_ch, bubbleLoc, GImp_fft, FUpDo,
                             qGridLoc, sumHelper_f, mP, sP)
        Σ_ladderLoc = Σ_ladderLoc .+ mP.n * mP.U/2.0;


        @info "Calculating bubble: "
        flush(io)
        bubble = calc_bubble(νGrid, GLoc_fft, qG, mP, sP);

        @info "Calculating χ and γ: "
        flush(io)
        nlQ_sp = calc_χ_trilex(impQ_sp.Γ, bubble, qG, νGrid, sumHelper_f, mP.U, mP, sP);
        nlQ_ch = calc_χ_trilex(impQ_ch.Γ, bubble, qG, νGrid, sumHelper_f, -mP.U, mP, sP);

        
        @info "Calculating λ correction: "
        flush(io)
        λ_sp, λ_spch  = λ_correction!(impQ_sp, impQ_ch, FUpDo, Σ_loc, Σ_ladderLoc, nlQ_sp, nlQ_ch, bubble, GLoc_fft, qG, mP, sP, init_sp=last_λsp, init_spch=last_λspch)
        last_λsp = λ_sp
        last_λspch = λ_spch

        @info "Calculating Σ ladder: "
        flush(io)
        Σ_ladder = calc_Σ(nlQ_sp, nlQ_ch, bubble, GLoc_fft, FUpDo, qG, sumHelper_f, mP, sP)
        Σ_ladder_corrected = Σ_ladder .- Σ_ladderLoc .+ Σ_loc[1:size(Σ_ladder,1)]
        @info "Done."
        flush(io)

        if save_results
        fname = res_prefix*"lDGA_b$(mP.β)_U$(mP.U)_k$(kG.Ns)_"*String(sP.tc_type)*"_lambda"*String(sP.λc_type)*res_postfix*".jld2"
        @info "Writing to $(fname)"
        save(fname, "λ_sp", λ_sp, "λ_spch", λ_spch, "bubbleLoc", bubbleLoc, "locQ_sp", locQ_sp, "locQ_ch", locQ_ch, "bubble", bubble, "nlQ_ch", nlQ_ch, "nlQ_sp", nlQ_sp, "Σ_ladder", Σ_ladder, "Σ_ladderLoc", Σ_ladderLoc)
        end
    end
end

function run2(cfg_file)
     λ_sp, λ_new_sp, λ_new_ch, _, _, _, _, nlQ_ch, nlQ_sp, Σ_ladder, _ = run_sim(cfg_file=cfg_file)
    return λ_sp, λ_new_sp, λ_new_ch, nlQ_ch, nlQ_sp, Σ_ladder
end
