# ==================================================================================================== #
#                                        thermodynamics.jl                                             #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#   Thermodynamic quantities from impurity and lDΓA GFs.                                               #
# -------------------------------------------- TODO -------------------------------------------------- #
#   Finish cleanup and documentation                                                                   #
# ==================================================================================================== #

# ================================================ ED ================================================

"""
    calc_E_ED(ϵₖ::Vector{Float64}, Vₖ::Vector{Float64}, GImp::Vector{ComplexF64}, U, n, μ, β)
    calc_E_ED(ϵₖ::Vector{Float64}, Vₖ::Vector{Float64}, GImp::Vector{ComplexF64}, mP::ModelParameters)
    calc_E_ED(fname::String)

Computes kinetic and potential energies from Anderson parameters.


Returns: 
-------------
(EKin, EPot): `Tuple{Float64,Float64}`, kinetic and potential energy.

Arguments:
-------------
- **`fname`** : `jld2`-file containing the fields: `[gImp, β, ϵₖ, Vₖ, U, nden, μ]` (see below)
- **`ϵₖ`**    : bath levels
- **`Vₖ`**    : hoppend amplitudes
- **`GImp`**  : impurity Green's function. WARNING: the arguments are assumed to by fermionic Matsuabra indices 0:length(GImp)-1!
- **`U`**     : Coulomb interaction strength
- **`n`**     : number density
- **`μ`**     : chemical potential
- **`β`**     : inverse temperature
- **`mP`**    : Alternative call with model parameters as `Float64`. See also [`ModelParameters`](ModelParameters).
"""
function calc_E_ED(fname::String)
    E_kin, E_pot = jldopen(fname,"r") do f
        calc_E_ED(f["ϵₖ"], f["Vₖ"], f["gImp"], f["U"], f["nden"], f["μ"], f["β"])
    end
    return E_kin, E_pot 
end

calc_E_ED(ϵₖ::Vector{Float64}, Vₖ::Vector{Float64}, GImp::Vector{ComplexF64}, mP) = calc_E_ED(ϵₖ, Vₖ, GImp, mP.U, mP.n, mP.μ, mP.β)

function calc_E_ED(ϵₖ::Vector{Float64}, Vₖ::Vector{Float64}, GImp::Vector{ComplexF64}, U::Float64, n::Float64, μ::Float64, β::Float64)
    E_kin = 0.0
    E_pot = 0.0
    vk = sum(Vₖ .^ 2)
    Σ_hartree = n * U/2
    E_pot_tail = (U^2)/2 * n * (1-n/2) - Σ_hartree*(Σ_hartree-μ)
    E_kin_tail = vk
    iνₙ = iν_array(β, length(GImp))

    for n in 1:length(GImp)
        Δ_n = sum((Vₖ .* conj.(Vₖ)) ./ (iνₙ[n] .- ϵₖ))
        Σ_n = iνₙ[n] .- Δ_n .- 1.0 ./ GImp[n] .+ μ
        E_kin += 2*real(GImp[n] * Δ_n - E_kin_tail/(iνₙ[n]^2))
        E_pot += 2*real(GImp[n] * Σ_n - E_pot_tail/(iνₙ[n]^2))
    end
    E_kin = E_kin * (2/β) - (β/2) * E_kin_tail
    E_pot = E_pot * (1/β) + 0.5*Σ_hartree - (β/4) * E_pot_tail
    return E_kin, E_pot
end


# ========================================== lDΓA Energies ===========================================
# ----------------------------------------------- EPot -----------------------------------------------
function calc_EPot2(χ_sp::χT, γ_sp::γT, χ_ch::χT, γ_ch::γT, kG::KGrid, 
                sP::SimulationParameters, mP::ModelParameters)
    ωindices::UnitRange{Int} = (sP.dbg_full_eom_omega) ? (1:size(χ,2)) : intersect(χ_sp.usable_ω, χ_ch.usable_ω)
    iωn = (1im .* 2 .* (-sP.n_iω:sP.n_iω)[ωindices] .* π ./ mP.β)
    iωn[findfirst(x->x ≈ 0, iωn)] = Inf
    Epot2 = mP.U * calc_EPot2_int(χ_sp, χ_ch, kG.kMult, Nk(kG), mP.β)
end

function calc_EPot2(χ_sp::χT, χ_ch::χT,
                      kMult::Vector{Float64}, k_norm::Int, β::Float64)
    lhs_c2 = 0.0
    for ωi in 1:size(χ_sp, 2)
        tmp2 = 0.0
        for (qi,km) in enumerate(kMult)
            χsp_i_λ = real(χ_sp[qi,ωi])
            χch_i_λ = real(χ_ch[qi,ωi])
            tmp2 += (χch_i_λ - χsp_i_λ) * km
        end
        lhs_c2 += 0.5*tmp2/k_norm
    end
    lhs_c2 = lhs_c2/β
    return lhs_c2 
end

"""

Specialized function for DGA potential energy. Better performance than calc_E.
"""
function EPot1(kG::KGrid, G::AbstractArray{ComplexF64, 2}, Σ::Array{ComplexF64, 2}, 
                    tail::Array{ComplexF64, 2}, tail_inv::Array{Float64, 1}, β::Float64)::Float64
    E_pot = real.(G .* Σ .- tail);
    return kintegrate(kG, 2 .* sum(view(E_pot,:,1:size(Σ,2)), dims=[2])[:,1] .+ tail_inv) / β
end

function calc_E_pot_νn(kG, G, Σ, tail, tail_inv)
    E_pot = real.(G .* Σ .- tail);
    @warn "possibly / β missing in this version! Test this first against calc_E_pot"
    return [kintegrate(kG, 2 .* sum(E_pot[:,1:i], dims=[2])[:,1] .+ tail_inv) for i in 1:size(E_pot,1)]
end


# ----------------------------------------------- EKin -----------------------------------------------
function calc_E_kin(kG::KGrid, G::Array{ComplexF64, 1}, ϵqGrid, tail::Array{ComplexF64, 1}, tail_inv::Vector{Float64}, β::Float64)
    E_kin = ϵqGrid' .* real.(G .- tail)
    return kintegrate(kG, 4 .* E_kin .+ tail_inv) / β
end

function calc_E_kin(kG::KGrid, G::Array{ComplexF64, 2}, ϵqGrid, tail::Array{ComplexF64, 2}, tail_inv::Vector{Float64}, β::Float64)
    E_kin = ϵqGrid' .* real.(G .- tail)
    return kintegrate(kG, 4 .* sum(E_kin, dims=[2])[:,1] .+ tail_inv) / β
end

# ---------------------------------------------- Common ----------------------------------------------

function calc_E(χ_sp::χT, γ_sp::γT, χ_ch::χT, γ_ch::γT, λ₀, gLoc_rfft, kG::KGrid, mP::ModelParameters, sP::SimulationParameters; νmax=sP.n_iν)
    Σ_ladder = LadderDGA.calc_Σ(χ_sp::χT, γ_sp::γT, χ_ch::χT, γ_ch::γT, λ₀, gLoc_rfft, kG, mP, sP);
    E_kin, E_pot = calc_E(Σ_ladder.parent, kG, mP, νmax = νmax)
    return E_kin, E_pot
end


function calc_E(Σ::AbstractArray{ComplexF64,2}, kG, mP; νmax::Int = floor(Int,3*size(Σ,2)/8),  trace::Bool=false)
    νGrid = 0:(νmax-1)
    G = G_from_Σ(Σ[:,1:νmax], kG.ϵkGrid, νGrid, mP);
    return calc_E(G, Σ, kG, mP; νmax = νmax,  trace=trace)
end

function calc_E(G::AbstractArray{ComplexF64,2}, Σ::AbstractArray{ComplexF64,2}, kG, mP; νmax::Int = floor(Int,3*size(Σ,2)/8),  trace::Bool=false)
    #println("TODO: make frequency summation with sum_freq optional")
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

	E_pot_full = real.(G .* Σ[:,1:νmax].- E_pot_tail);
	E_kin_full = kG.ϵkGrid .* real.(G .- E_kin_tail);
    E_kin, E_pot = if trace
        [kintegrate(kG, 4 .* sum(view(E_kin_full,:,1:i), dims=[2])[:,1] .+ E_kin_tail_inv) for i in 1:νmax] ./ mP.β,
        [kintegrate(kG, 2 .* sum(view(E_pot_full,:,1:i), dims=[2])[:,1] .+ E_pot_tail_inv) for i in 1:νmax] ./ mP.β
    else
        kintegrate(kG, 4 .* sum(view(E_kin_full,:,1:νmax), dims=[2])[:,1] .+ E_kin_tail_inv) ./ mP.β,
        kintegrate(kG, 2 .* sum(view(E_pot_full,:,1:νmax), dims=[2])[:,1] .+ E_pot_tail_inv) ./ mP.β
    end
    return E_kin, E_pot
end
