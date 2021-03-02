using Pkg
Pkg.activate("/home/julian/Hamburg/LadderDGA")
using LadderDGA
#TODO: this could be a macro modifying the 3 main functions
# ======================== Setup ==========================

function run_sim()
    print("specify location of config file: ")
    cfg_file = readline()
    modelParams, simParams, env, kGrid, qGrid, νGrid, impQ_sp, impQ_ch, GImp_fft, GLoc_fft, Σ_loc_pos, FUpDo, χDMFTsp, χDMFTch, gImp = setup_LDGA(cfg_file, false);
        
    @info "Calculating local quantities: "
    bubbleLoc = calc_bubble(νGrid, GImp_fft, 1, modelParams, simParams)
    locQ_sp = calc_χ_trilex(impQ_sp.Γ, bubbleLoc, [1.0], modelParams.U, modelParams, simParams);
    locQ_ch = calc_χ_trilex(impQ_ch.Γ, bubbleLoc, [1.0], -modelParams.U, modelParams, simParams);

    @info "Calculating bubble: "
    bubble = calc_bubble(νGrid, GLoc_fft, length(qGrid.multiplicity), modelParams, simParams);

    @info "Calculating χ and γ: "
    nlQ_sp = calc_χ_trilex(impQ_sp.Γ, bubble, qGrid.multiplicity, modelParams.U, modelParams, simParams);
    nlQ_ch = calc_χ_trilex(impQ_ch.Γ, bubble, qGrid.multiplicity, -modelParams.U, modelParams, simParams);

    @info "Calculating λ correction in the spin channel: "
    rhs, usable_ω = calc_λsp_rhs_usable(impQ_sp, impQ_ch, nlQ_sp, nlQ_ch, qGrid.multiplicity, simParams, modelParams)
    @info "Computing λ corrected χsp, using " simParams.χFillType " as fill value outside usable ω range."

# function calc_Σ(nlQ_sp::NonLocalQuantities, nlQ_ch::NonLocalQuantities, bubble::BubbleT, 
#                         Gνω::GνqT, FUpDo::Array{Complex{Float64},3}, qIndices::AbstractArray, 
#                         ωindices::AbstractArray{Int64},νGrid::Vector{AbstractArray}, Nk::Int64,
#                         mP::ModelParameters, sP::SimulationParameters)
    usable_ω_λc = simParams.maxRange ? nlQ_sp.usable_ω : intersect(nlQ_sp.usable_ω, nlQ_ch.usable_ω)
    nlQ_sp_λ = calc_λsp_correction!(nlQ_sp, usable_ω_λc, rhs, qGrid, modelParams, simParams)
    νZero = simParams.n_iν
    dbg1, dbg2, tmp, Σ_ladder_ω, Σ_bare, Σ_ladder, Σ_ladderLoc = if !simParams.chi_only
        println("param: ", simParams.fullωRange_Σ)
        usable_ω_Σ = simParams.fullωRange_Σ ? (1:2*simParams.n_iω+1) : intersect(nlQ_sp.usable_ω, nlQ_ch.usable_ω)
        println("range: ", usable_ω_Σ)
        @info "Calculating Σ ladder: "
        dbg1,dbg2,c,d,Σ_ladderLoc = calc_Σ(locQ_sp, locQ_ch, bubbleLoc, GImp_fft, FUpDo,
                             [(1,1,1)], usable_ω_Σ, νGrid, νZero, 1, modelParams, simParams,false)
        Σ_ladderLoc = Σ_ladderLoc .+ modelParams.n * modelParams.U/2.0;
        a,b,tmp, Σ_ladder_ω,Σ_ladder = calc_Σ(nlQ_sp_λ, nlQ_ch, bubble, GLoc_fft, FUpDo,
                                qGrid.indices, usable_ω_Σ, νGrid, νZero, simParams.Nk, modelParams, simParams, false)
        Σ_ladder_corrected = Σ_ladder[1:νZero] .- Σ_ladderLoc[1:νZero] .+ Σ_loc_pos[eachindex(Σ_ladderLoc[1:νZero])]
        dbg1, dbg2, tmp, Σ_ladder_ω, Σ_ladder, Σ_ladder_corrected, Σ_ladderLoc
    end
    @info "Done."
    return bubbleLoc, locQ_sp, locQ_ch, bubble, nlQ_sp, nlQ_ch, nlQ_sp_λ, Σ_bare, Σ_ladder, Σ_ladderLoc, usable_ω, usable_ω_λc, usable_ω_Σ, tmp, Σ_ladder_ω, dbg1, dbg2
end

flush(LadderDGA.io)
