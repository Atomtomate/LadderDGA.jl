function E_Pot(Σ_ladder, ϵkGrid, mP::ModelParameters, sP::SimulationParameters)
    println("TODO: this function has to be tested")
    νGrid = 0:simParams.n_iν-1
    Σ_hartree = mP.n * mP.U/2
    ϵkGrid_red = reduce_kGrid(cut_mirror(collect(ϵkGrid)))
    tail_corr_0 = 0.0
    tail_corr_inv_0 = mP.β * Σ_hartree/2
    tail_corr_1 = (mP.U^2 * 0.5 * mP.n * (1-0.5*mP.n) .+ Σ_hartree .* (ϵkGrid_red .+ Σ_hartree .- mP.μ))' ./ (iν_array(mP.β, 0:(sP.n_iν-1)) .^ 2)
    tail_corr_inv_1 = 0.5 * mP.β * (mP.U^2 * 0.5 * mP.n * (1-0.5*mP.n) .+ Σ_hartree .* (ϵkGrid_red .+ Σ_hartree .- mP.μ))

    Σ_ladder_corrected = Σ_ladder.+ Σ_hartree
    G0_full = G_from_Σ(zeros(Complex{Float64}, sP.n_iν), ϵkGrid, νGrid, mP);
    G0 = flatten_2D(reduce_kGrid.(cut_mirror.(G0_full)))
    G_new = flatten_2D(G_from_Σ(Σ_ladder_corrected, ϵkGrid_red, νGrid, mP));

    norm = (mP.β * sP.Nk^mP.D)
    tmp = real.(G_new .* (Σ_ladder_corrected) .+ tail_corr_0 .- tail_corr_1);
    res = [sum( (2 .* sum(tmp[1:i,:], dims=[1])[1,:] .+ tail_corr_inv_0 .- tail_corr_inv_1 .* 0.5 .* mP.β) .* qMultiplicity) / norm for i in 1:sP.n_iν]
    return res[end]
end
