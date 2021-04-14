using Distributed
#addprocs(7, enable_threaded_blas=true)
@everywhere using Pkg
@everywhere Pkg.activate("/home/julian/Hamburg/LadderDGA.jl")
@everywhere using LadderDGA
#TODO: this could be a macro modifying the 3 main functions
# ======================== Setup ==========================

function run_sim(; cfg_file=nothing)
    @warn "assuming linear, continuous nu grid for chi/trilex"
    if cfg_file === nothing
        print("specify location of config file: ")
        cfg_file = readline()
    end
    modelParams, simParams, env, kGrid, qGrid, νGrid, sumHelper_f, impQ_sp, impQ_ch, GImp_fft, GLoc_fft, Σ_loc_pos, FUpDo, gImp = setup_LDGA(cfg_file, false);
        
    @info "Calculating local quantities: "
    bubbleLoc = calc_bubble(νGrid, GImp_fft, 1, modelParams, simParams)
    locQ_sp = calc_χ_trilex(impQ_sp.Γ, bubbleLoc, [1.0], νGrid, sumHelper_f, modelParams.U, modelParams, simParams);
    locQ_ch = calc_χ_trilex(impQ_ch.Γ, bubbleLoc, [1.0], νGrid, sumHelper_f, -modelParams.U, modelParams, simParams);

    Σ_ladderLoc = calc_Σ(locQ_sp, locQ_ch, bubbleLoc, GImp_fft, FUpDo,
                         [(1,1,1)], 1, sumHelper_f, modelParams, simParams)
    Σ_ladderLoc = Σ_ladderLoc .+ modelParams.n * modelParams.U/2.0;


    @info "Calculating bubble: "
    bubble = calc_bubble(νGrid, GLoc_fft, length(qGrid.multiplicity), modelParams, simParams);

    @info "Calculating χ and γ: "
    nlQ_sp = calc_χ_trilex(impQ_sp.Γ, bubble, qGrid.multiplicity, νGrid, sumHelper_f, modelParams.U, modelParams, simParams);
    nlQ_ch = calc_χ_trilex(impQ_ch.Γ, bubble, qGrid.multiplicity, νGrid, sumHelper_f, -modelParams.U, modelParams, simParams);

    
    @info "Calculating λ correction: "
    λ_correction!(impQ_sp, impQ_ch, FUpDo, Σ_loc_pos, Σ_ladderLoc, nlQ_sp, nlQ_ch, bubble, GLoc_fft, qGrid, modelParams, simParams)

    @info "Calculating Σ ladder: "
    Σ_ladder = calc_Σ(nlQ_sp, nlQ_ch, bubble, GLoc_fft, FUpDo,
                        qGrid.indices, simParams.Nk, sumHelper_f, modelParams, simParams)
    Σ_ladder_corrected = Σ_ladder .- Σ_ladderLoc .+ Σ_loc_pos[1:size(Σ_ladder,1)]
    Σ_ladder, Σ_ladder_corrected, Σ_ladderLoc
    @info "Done."
    return bubbleLoc, locQ_sp, locQ_ch, bubble, nlQ_ch, nlQ_sp, Σ_bare, Σ_ladder, Σ_ladderLoc
end

function run2(cfg_file)
     _, _, _, _, nlQ_ch, nlQ_sp, _, Σ_ladder, _ = run_sim(cfg_file=cfg_file)
    return nlQ_ch, nlQ_sp, Σ_ladder
end

flush(LadderDGA.io)
