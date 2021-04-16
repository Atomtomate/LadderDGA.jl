function calc_E_ED(iνₙ, ϵₖ, Vₖ, GImp, mP; full=false)
    full && @error "Not implemented for full GF."
    E_kin = 0.0
    E_pot = 0.0
    vk = sum(Vₖ .^ 2)
    Σ_hartree = mP.n * mP.U/2
    E_pot_tail = (mP.U^2)/2 * mP.n * (1-mP.n/2) - Σ_hartree*(Σ_hartree-mP.μ)
    E_kin_tail = vk

    for n in 1:length(GImp)
        Δ_n = sum((Vₖ .^ 2) ./ (iνₙ[n] .- ϵₖ))
        Σ_n = iνₙ[n] .- Δ_n .- 1.0 ./ GImp[n] .+ mP.μ
        E_kin += 2*real(GImp[n] * Δ_n - E_kin_tail/(iνₙ[n]^2))
        E_pot += 2*real(GImp[n] * Σ_n - E_pot_tail/(iνₙ[n]^2))
    end
    E_kin = E_kin .* (2/mP.β) - (mP.β/2) .* E_kin_tail
    E_pot = E_pot .* (1/mP.β) .+ 0.5*Σ_hartree .- (mP.β/4) .* E_pot_tail
    return E_kin, E_pot
end

function calc_E(Σ, ϵqGrid, qM, Nk, mP, sP)
    #println("TODO: E_pot function has to be tested")
    #println("TODO: use GNew/GLoc/GImp instead of Sigma")
    #println("TODO: make frequency summation with sum_freq an optional")
    νGrid = 0:sP.n_iν-1
    iν_n = iν_array(mP.β, νGrid)
    Σ_hartree = mP.n * mP.U/2
    Σ_corr = Σ .+ Σ_hartree

    norm = (mP.β * Nk^mP.D)
    E_kin_tail_c = [zeros(size(ϵqGrid)), (ϵqGrid .+ Σ_hartree .- mP.μ)]
    E_pot_tail_c = [zeros(size(ϵqGrid)),
                    (mP.U^2 * 0.5 * mP.n * (1-0.5*mP.n) .+ Σ_hartree .* (ϵqGrid .+ Σ_hartree .- mP.μ))]
    tail = [1 ./ (iν_n .^ n) for n in 1:length(E_kin_tail_c)]
    E_pot_tail = sum(E_pot_tail_c[i]' .* tail[i] for i in 1:length(tail))
    E_kin_tail = sum(E_kin_tail_c[i]' .* tail[i] for i in 1:length(tail))
    E_pot_tail_inv = sum((mP.β/2)  .* [Σ_hartree .* ones(size(ϵqGrid)), (-mP.β/2) .* E_pot_tail_c[2]])
    E_kin_tail_inv = sum(map(x->x .* (mP.β/2) .* ϵqGrid , [1, -(mP.β) .* E_kin_tail_c[2]]))

    G_corr = flatten_2D(G_from_Σ(Σ_corr, ϵqGrid, νGrid, mP));
    E_pot = real.(G_corr .* Σ_corr .- E_pot_tail);
    E_kin = ϵqGrid' .* real.(G_corr .- E_kin_tail);

    E_pot = [sum( (2 .* sum(E_pot[1:i,:], dims=[1])[1,:] .+ E_pot_tail_inv) .* qM) / norm for i in 1:sP.n_iν]
    E_kin = [sum( (4 .* sum(E_kin[1:i,:], dims=[1])[1,:] .+ E_kin_tail_inv) .* qM) / norm for i in 1:sP.n_iν]
    return E_kin, E_pot
end

"""

Specialized function for DGA potential energy. Better performance than calc_E.
"""
function calc_E_pot(G, Σ, tail, tail_inv, qM, norm)
    E_pot = real.(G .* Σ .- tail);
    return sum((2 .* sum(E_pot, dims=[1])[1,:] .+ tail_inv) .* qM) / norm
end

function calc_E_pot_νn(G, Σ, tail, tail_inv, qM, norm)
    E_pot = real.(G .* Σ .- tail);
    return [sum((2 .* sum(E_pot[1:i,:], dims=[1])[1,:] .+ tail_inv) .* qM) / norm for i in 1:size(E_pot,1)]
end


function calc_E_kin(G, ϵqGrid, tail, tail_inv, qM, norm)
    E_kin = ϵqGrid' .* real.(G .- tail)
    return sum((4 .* sum(E_kin, dims=[1])[1,:] .+ tail_inv) .* qM) / norm
end