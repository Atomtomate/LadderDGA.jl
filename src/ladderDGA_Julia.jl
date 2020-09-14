module LadderDGA

Base.Experimental.@optlevel 3
include("$(@__DIR__)/Dependencies.jl")

export calculate_Σ_ladder




#TODO: implement generic indexing, especially dynamic freq_grids
if myid() == 1
    args = parse_args(ARGS, s)
    @info "Reading Inputs..."
    const modelParams, simParams, env, impQ_sp, impQ_ch, GImp_tmp, Σ_loc, FUpDo  = setup_LDGA(args["config"], false);
    const kIndices, kGrid = gen_kGrid(simParams.Nk, modelParams.D);
    const ϵkGrid          = squareLattice_ekGrid(kGrid);
    const qIndices, qGrid = reduce_kGrid.(cut_mirror.((kIndices, kGrid)));
    const qMultiplicity   = kGrid_multiplicity(qIndices);

    const fft_range = -simParams.n_iω:(simParams.n_iν+simParams.n_iω-1)
    const GImp_sym = store_symm_f(GImp_tmp, fft_range)
    const GImp = convert(SharedArray,reshape(GImp_sym, (length(GImp_sym),1)));
    const GLoc_fft_tmp = Gfft_from_Σ(Σ_loc, ϵkGrid, fft_range, modelParams);
    const GLoc_fft = convert(SharedArray, GLoc_fft_tmp);

    @info """Inputs Read. Starting Computation.
    Found usable intervals for local susceptibility of length 
          sp: $(length(impQ_sp.usable_ω))
          ch: $(length(impQ_ch.usable_ω)) 
          χLoc_sp = $(printr_s(impQ_sp.χ_loc)), χLoc_ch = $(printr_s(impQ_ch.χ_loc))"""

    const calc_bubble(G::SharedArray{Complex{Float64},2}, len::Int64) = calc_bubble_int(G, len, modelParams, simParams);
    const calc_χ_trilex(bubble::SharedArray{Complex{Float64},3}, qMult::Array{Float64,1}, U::Float64) = 
            calc_χ_trilex_int(impQ_sp.Γ, bubble, qMult, U, modelParams.β,
                              tc=simParams.tail_corrected, fullRange=simParams.fullChi);

    const calc_Σ(Qsp, Qch, bubble, G, usable_ω, νindices, Nk) = calc_DΓA_Σ_int(Qsp, Qch, bubble, G, FUpDo, 
                                                   qIndices, usable_ω, usable_ν, Nk,
                                                   modelParams, simParams, simParams.tail_corrected)
    #usable_ν = 1:(last(usable_ν) - sP.n_iν)
end

function calc_Σ(G::SharedArray{Complex{Float64},2}=GLoc_fft)
    @info "Calculating local quantities: "
    bubbleLoc = calc_bubble(GImp, 1);
    locQ_sp = calc_χ_trilex(bubbleLoc, [1.0], modelParams.U);
    locQ_ch = calc_χ_trilex(bubbleLoc, [1.0], -modelParams.U);

    @info "Calculating bubble: "
    @time bubble = calc_bubble(GLoc_fft, length(qMultiplicity));

    @info "Calculating χ and γ: "
    @time nlQ_sp = calc_χ_trilex(bubble, qMultiplicity, modelParams.U);
    @time nlQ_ch = calc_χ_trilex(bubble, qMultiplicity, -modelParams.U);


    @info "Calculating λ correction in the spin channel: "
    rhs, usable_ω = calc_λsp_rhs_usable(impQ_sp, impQ_ch, nlQ_sp, nlQ_ch, simParams, modelParams)
    usable_ω = simParams.fullRange ? (1:(2*simParams.n_iω+1)) : usable_ω
    @info "Computing λ corrected χsp, using " simParams.χFillType " as fill value outside usable ω range."
    @time λsp,χsp_λ = calc_λsp_correction(nlQ_sp.χ_general, usable_ω, rhs, qMultiplicity, 
                                    modelParams.β, simParams.tail_corrected, simParams.χFillType)
    nlQ_sp_λ = NonLocalQuantities(χsp_λ, χ_ω, nlQ_sp.γ, nlQ_sp.usable_ω)
    nlQ_ch_λ = nlQ_ch


    Σ_ladder = nothing
    Σ_ladderLoc = nothing
    if !simParams.chi_only
        @info "Calculating Σ ladder: "
        println("DEBUG: computing 2 versions of Sigma")
        @time Σ_ladderLoc_cut = calc_DΓA_Σ_fft(impQ_sp, impQ_ch, locQ_sp, locQ_ch, bubbleLoc, GImp,
                                               [1], usable_ω, 1:simParams.n_iν, 1)
        Σ_ladderLoc_cut = Σ_ladderLoc_cut .+ modelParams.n * modelParams.U/2.0;
        @time Σ_ladder_cut = calc_DΓA_Σ_fft(χsp_λ, χch, trilexsp, trilexch, bubble, GLoc_fft, 
                                            qIndices, usable_ω, 1:simParams.n_iν, simParams.Nk)
        Σ_ladder_cut_corrected = Σ_ladder_cut .- Σ_ladderLoc_cut .+ Σ_loc[eachindex(Σ_ladderLoc_cut)]
        #save("sigma.jld","Sigma", Σ_ladder, 
        #      compress=true, compatible=true)
    end
    return bubble, χsp, χsp_λ, χch, usable_sp, usable_ch, trilexsp, trilexch, 
           Σ_ladder_cut, Σ_ladder_cut_corrected, Σ_ladderLoc_cut
end;

end
