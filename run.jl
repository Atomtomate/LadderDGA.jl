using Distributed
#if nprocs() == 1
#    addprocs(7)
#end
@everywhere using Pkg
@everywhere Pkg.activate(@__DIR__)
Pkg.instantiate()
@everywhere using LadderDGA
#TODO: this could be a macro modifying the 3 main functions
# ======================== Setup ==========================

function run_sim(; cfg_file=nothing)
    @warn "assuming linear, continuous nu grid for chi/trilex"
    if cfg_file === nothing
        print("specify location of config file: ")
        cfg_file = readline()
    end
    modelParams, simParams, env, qGrid_loc, kGrid, qGrid, νGrid, sumHelper_f, impQ_sp, impQ_ch, GImp_fft, GLoc_fft, Σ_loc_pos, FUpDo, gImp, GLoc = setup_LDGA(cfg_file, false);
        
    @info "Calculating local quantities: "
    bubbleLoc = calc_bubble(νGrid, GImp_fft, qGrid_loc, modelParams, simParams)
    locQ_sp = calc_χ_trilex(impQ_sp.Γ, bubbleLoc, qGrid_loc, νGrid, sumHelper_f, modelParams.U, modelParams, simParams);
    locQ_ch = calc_χ_trilex(impQ_ch.Γ, bubbleLoc, qGrid_loc, νGrid, sumHelper_f, -modelParams.U, modelParams, simParams);

    Σ_ladder = nothing
    Σ_ladderLoc = nothing
    #Σ_ladderLoc = calc_Σ(locQ_sp, locQ_ch, bubbleLoc, GImp_fft, FUpDo,
    #                     qGrid_loc, sumHelper_f, modelParams, simParams)
    #Σ_ladderLoc = Σ_ladderLoc .+ modelParams.n * modelParams.U/2.0;


    @info "Calculating bubble: "
    bubble = calc_bubble(νGrid, GLoc_fft, qGrid, modelParams, simParams);

    @info "Calculating χ and γ: "
    nlQ_sp = calc_χ_trilex(impQ_sp.Γ, bubble, qGrid, νGrid, sumHelper_f, modelParams.U, modelParams, simParams);
    nlQ_ch = calc_χ_trilex(impQ_ch.Γ, bubble, qGrid, νGrid, sumHelper_f, -modelParams.U, modelParams, simParams);

    
    @info "Calculating λ correction: "
    λ_sp, λ_new_sp, λ_new_ch  = λ_correction!(impQ_sp, impQ_ch, FUpDo, Σ_loc_pos, Σ_ladderLoc, nlQ_sp, nlQ_ch, bubble, GLoc_fft, qGrid, modelParams, simParams)

    @info "Calculating Σ ladder: "
    #Σ_ladder = calc_Σ(nlQ_sp, nlQ_ch, bubble, GLoc_fft, FUpDo, qGrid, sumHelper_f, modelParams, simParams)
    #Σ_ladder_corrected = Σ_ladder .- Σ_ladderLoc .+ Σ_loc_pos[1:size(Σ_ladder,1)]
    #Σ_ladder, Σ_ladder_corrected, Σ_ladderLoc
    @info "Done."
    return λ_sp, λ_new_sp, λ_new_ch, bubbleLoc, locQ_sp, locQ_ch, bubble, nlQ_ch, nlQ_sp, Σ_ladder, Σ_ladderLoc
end

function run2(cfg_file)
     λ_sp, λ_new_sp, λ_new_ch, _, _, _, _, nlQ_ch, nlQ_sp, Σ_ladder, _ = run_sim(cfg_file=cfg_file)
    return λ_sp, λ_new_sp, λ_new_ch, nlQ_ch, nlQ_sp, Σ_ladder
end

flush(LadderDGA.io)
