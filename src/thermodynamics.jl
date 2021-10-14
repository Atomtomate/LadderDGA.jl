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

function calc_E(Σ, kG, mP::ModelParameters, sP::SimulationParameters; trace=false)
    #println("TODO: make frequency summation with sum_freq optional")
    νmax = ndims(Σ) == 2 ? size(Σ,2) : length(Σ)
    νGrid = 0:(νmax-1)
    iν_n = iν_array(mP.β, νGrid)
    Σ_hartree = mP.n * mP.U/2

	E_kin_tail_c = [zeros(size(kG.ϵkGrid)), (kG.ϵkGrid .+ Σ_hartree .- mP.μ)]
	E_pot_tail_c = [zeros(size(kG.ϵkGrid)),
					(mP.U^2 * 0.5 * mP.n * (1-0.5*mP.n) .+ Σ_hartree .* (kG.ϵkGrid .+ Σ_hartree .- mP.μ))]
	tail = [1 ./ (iν_n .^ n) for n in 1:length(E_kin_tail_c)]
	E_pot_tail = sum(E_pot_tail_c[i] .* transpose(tail[i]) for i in 1:length(tail))
	E_kin_tail = sum(E_kin_tail_c[i] .* transpose(tail[i]) for i in 1:length(tail))
	E_pot_tail_inv = sum((mP.β/2)  .* [Σ_hartree .* ones(size(kG.ϵkGrid)), (-mP.β/2) .* E_pot_tail_c[2]])
	E_kin_tail_inv = sum(map(x->x .* (mP.β/2) .* kG.ϵkGrid , [1, -(mP.β) .* E_kin_tail_c[2]]))

	G_corr = transpose(flatten_2D(G_from_Σ(Σ, kG.ϵkGrid, νGrid, mP)));
	E_pot_full = real.(G_corr .* Σ.- E_pot_tail);
	E_kin_full = kG.ϵkGrid .* real.(G_corr .- E_kin_tail);
    E_kin, E_pot = if trace
        [kintegrate(kG, 4 .* sum(view(E_kin_full,:,1:i), dims=[2])[:,1] .+ E_kin_tail_inv) for i in 1:νmax] ./ mP.β,
        [kintegrate(kG, 2 .* sum(view(E_pot_full,:,1:i), dims=[2])[:,1] .+ E_pot_tail_inv) for i in 1:νmax] ./ mP.β
    else
        kintegrate(kG, 4 .* sum(view(E_kin_full,:,1:νmax), dims=[2])[:,1] .+ E_kin_tail_inv) ./ mP.β,
        kintegrate(kG, 2 .* sum(view(E_pot_full,:,1:νmax), dims=[2])[:,1] .+ E_pot_tail_inv) ./ mP.β
    end
    return E_kin, E_pot
end

"""

Specialized function for DGA potential energy. Better performance than calc_E.
"""
function calc_E_pot(kG::ReducedKGrid, G::Array{ComplexF64, 2}, Σ::Array{ComplexF64, 2}, 
                    tail::Array{ComplexF64, 2}, tail_inv::Array{Float64, 1}, β::Float64)::Float64
    E_pot = real.(G .* Σ .- tail);
    return kintegrate(kG, 2 .* sum(view(E_pot,:,1:size(Σ,2)), dims=[2])[:,1] .+ tail_inv) / β
end

function calc_E_pot_νn(kG, G, Σ, tail, tail_inv)
    E_pot = real.(G .* Σ .- tail);
    @warn "possibly / β missing in this version! Test this first against calc_E_pot"
    return [kintegrate(kG, 2 .* sum(E_pot[:,1:i], dims=[2])[:,1] .+ tail_inv) for i in 1:size(E_pot,1)]
end


function calc_E_kin(kG::ReducedKGrid, G::Array{ComplexF64, 1}, ϵqGrid, tail::Array{ComplexF64, 1}, tail_inv::Vector{Float64}, β::Float64)
    E_kin = ϵqGrid' .* real.(G .- tail)
    return kintegrate(kG, 4 .* E_kin .+ tail_inv) / β
end

function calc_E_kin(kG::ReducedKGrid, G::Array{ComplexF64, 2}, ϵqGrid, tail::Array{ComplexF64, 2}, tail_inv::Vector{Float64}, β::Float64)
    E_kin = ϵqGrid' .* real.(G .- tail)
    return kintegrate(kG, 4 .* sum(E_kin, dims=[2])[:,1] .+ tail_inv) / β
end
