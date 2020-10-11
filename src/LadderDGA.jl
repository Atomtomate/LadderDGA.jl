module LadderDGA
Base.Experimental.@optlevel 3
include("$(@__DIR__)/DepsInit.jl")

export calculate_Σ_ladder, writeFortranΣ


#TODO: implement generic indexing, especially dynamic freq_grids
function calc_Σ()
    @info "Calculating local quantities: "
    bubbleLoc = calc_bubble(GImp, 1);
    locQ_sp = calc_χ_trilex(impQ_sp.Γ, bubbleLoc, [1.0], modelParams.U);
    locQ_ch = calc_χ_trilex(impQ_ch.Γ, bubbleLoc, [1.0], -modelParams.U);

    @info "Calculating bubble: "
    bubble = calc_bubble(GLoc_fft, length(qMultiplicity));

    @info "Calculating χ and γ: "
    nlQ_sp = calc_χ_trilex(impQ_sp.Γ, bubble, qMultiplicity, modelParams.U);
    nlQ_ch = calc_χ_trilex(impQ_ch.Γ, bubble, qMultiplicity, -modelParams.U);

    @info "Calculating λ correction in the spin channel: "
    rhs, usable_ω = calc_λsp_rhs_usable(impQ_sp, impQ_ch, nlQ_sp, nlQ_ch, qMultiplicity, simParams, modelParams)
    @info "Computing λ corrected χsp, using " simParams.χFillType " as fill value outside usable ω range."

    usable_ω_λc = simParams.maxRange ? nlQ_sp.usable_ω : intersect(nlQ_sp.usable_ω, nlQ_ch.usable_ω)
    nlQ_sp_λ = calc_λsp_correction!(nlQ_sp, usable_ω_λc, rhs, qMultiplicity, 
                                    modelParams.β, simParams.tail_corrected, simParams.χFillType)

    Σ, Σ_ladderLoc = if !simParams.chi_only
        usable_ω_Σ = simParams.fullωRange_Σ ? (1:simParams.n_iω) : intersect(nlQ_sp.usable_ω, nlQ_ch.usable_ω)
        @info "Calculating Σ ladder: "
        Σ_ladderLoc = calc_Σ(locQ_sp, locQ_ch, bubbleLoc, GImp,
                             [(1,1,1)], usable_ω_Σ, 1)
        Σ_ladderLoc = Σ_ladderLoc .+ modelParams.n * modelParams.U/2.0;
        Σ_ladder = calc_Σ(nlQ_sp_λ, nlQ_ch, bubble, GLoc_fft, 
                                qIndices, usable_ω_Σ, simParams.Nk)
        #simParams.n_iν
        Σ_ladder_corrected = Σ_ladder .- Σ_ladderLoc .+ Σ_loc_pos[eachindex(Σ_ladderLoc)]
        Σ_ladder_corrected, Σ_ladderLoc
    end
    #env.progressbar && put!(channel, false)
    @info "Done."
    flush(LadderDGA.io)
    return bubble, nlQ_sp, nlQ_sp_λ, nlQ_ch, sdata(Σ), sdata(Σ_ladderLoc)
end;

end
