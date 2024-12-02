# ==================================================================================================== #
#                                        thermodynamics.jl                                             #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe, Jan Frederik Weissler                                             #
# ----------------------------------------- Description ---------------------------------------------- #
#   Thermodynamic quantities from impurity and lDΓA GFs.                                               #
# -------------------------------------------- TODO -------------------------------------------------- #
#   Enery calculation does a lot of copying, provide subroutines                                       #
#   Some of these functions are performance cirtical, benchmark those                                  #
# ==================================================================================================== #

# ================================================ ED ================================================

"""
    PP_p1(χm::χT, χd::χT, λm::Float64, λd::Float64, n::Float64, U::Float64)::Float64

Pauli-Principle on 1-particle level: ``n/2 (1-n/2)``.
"""
function PP_p1(χm::χT, χd::χT, λm::Float64, λd::Float64, n::Float64, U::Float64)::Float64
    return n/2 * 1-n/2
end

"""
    PP_p2(χm::χT, χd::χT, λm::Float64, λd::Float64, n::Float64, U::Float64)::Float64

Pauli-Principle on 2-particle level: ``(\\sum_{k,\\omega} \\chi^{\\lambda,\\omega}_{m,k} + \\sum_{k,\\omega} \\chi^{\\lambda,\\omega}_{d,k})/2``.
"""
function PP_p2(χm::χT, χd::χT, λm::Float64, λd::Float64, n::Float64, U::Float64, kG::KGrid)::Float64
    return real(sum_kω(kG, χd, λ = λd) + sum_kω(kG, χm, λ = λm))/2
end

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
- **`mP`**    : Alternative call with model parameters as `Float64`. See also [`ModelParameters`](@ref ModelParameters).
"""
function calc_E_ED(fname::String)
    E_kin, E_pot = jldopen(fname, "r") do f
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
    E_pot_tail = (U^2) * (n/2) * (1 - n/2) - Σ_hartree * (Σ_hartree - μ)
    E_kin_tail = vk
    iνₙ = iν_array(β, length(GImp))

    for n = 1:length(GImp)
        Δ_n = sum((Vₖ .* conj.(Vₖ)) ./ (iνₙ[n] .- ϵₖ))
        Σ_n = iνₙ[n] .- Δ_n .- 1.0 ./ GImp[n] .+ μ
        E_kin += 2 * real(GImp[n] * Δ_n - E_kin_tail / (iνₙ[n]^2))
        E_pot += 2 * real(GImp[n] * Σ_n - E_pot_tail / (iνₙ[n]^2))
    end
    E_kin = E_kin * (2 / β) - (β / 2) * E_kin_tail
    E_pot = E_pot * (1 / β) + 0.5 * Σ_hartree - (β / 4) * E_pot_tail
    return E_kin, E_pot
end

# ========================================== lDΓA Energies ===========================================
# ----------------------------------------------- EPot -----------------------------------------------
"""
    EPot_p1_tail(νGrid::Vector{ComplexF64}, μ::Float64, h)
    EPot_p1_tail(iν_n::Vector{ComplexF64}, μ::Float64, U::Float64, β::Float64, n::Float64, kG::KGrid)

Tail cofficients for the one-particle potential energy [`EPot_p1`](@ref EPot_p1)
"""
EPot_p1_tail(νGrid::Vector{ComplexF64}, μ::Float64, h) = EPot_p1_tail(νGrid, μ, h.mP.U, h.mP.β, h.mP.n, h.kG)

function EPot_p1_tail(iν_n::Vector{ComplexF64}, μ::Float64, U::Float64, β::Float64, n::Float64, kG::KGrid)
    Σ_hartree = n * U / 2
    EPot_tail_c = (U^2 * (n/2) * (1 - n/2) .+ Σ_hartree .* (kG.ϵkGrid .+ Σ_hartree .- μ))
    tail = 1 ./ (iν_n .^ 2)
    EPot_tail = EPot_tail_c .* transpose(tail)
    EPot_tail_inv = (β / 2) .* Σ_hartree .+ (β / 2) * (-β / 2) .* EPot_tail_c
    return EPot_tail, EPot_tail_inv
end

"""
    EPot_p1(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::λ₀T, h; λm::Float64 = 0.0, λd::Float64 = 0.0, tc::Type{<: ΣTail}=default_Σ_tail_correction())
    EPot_p1(G_ladder::AbstractMatrix, Σ_ladder::AbstractMatrix, μ::Float64, h; tc::Type{<: ΣTail}=default_Σ_tail_correction())

Potential energy an the one-particle level.
"""
function EPot_p1(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::λ₀T, h; λm::Float64 = 0.0, λd::Float64 = 0.0, tc::Type{<: ΣTail}=default_Σ_tail_correction())::Float64
    μ, G_ladder, Σ_ladder = calc_G_Σ(χm, γm, χd, γd, λ₀, λm, λd, h; tc=tc, fix_n = true);
    νGrid = 0:last(axes(Σ_ladder,2))
    iν_n = iν_array(h.mP.β, νGrid)
    EPot_tail, EPot_tail_inv = EPot_p1_tail(iν_n, μ, h)
    return EPot_p1(view(G_ladder,:, νGrid), view(Σ_ladder,:, νGrid), EPot_tail, EPot_tail_inv, h.mP.β, h.kG)
end

function EPot_p1(G_ladder::AbstractMatrix, Σ_ladder::AbstractMatrix, EPot_tail::AbstractMatrix, EPot_tail_inv, β::Float64, kG::KGrid)::Float64
    EPot_k = 2 .* sum(real.((G_ladder .* Σ_ladder) .- EPot_tail), dims = [2])[:, 1] .+ EPot_tail_inv
    return kintegrate(kG, EPot_k) / β
end

"""
    EPot_p2(χm::χT, χd::χT, λm::Float64, λd::Float64, n::Float64, U::Float64)::Float64

Potential energy on the two-particle level.
"""
function EPot_p2(χm::χT, χd::χT, λm::Float64, λd::Float64, n::Float64, U::Float64, kG::KGrid)::Float64
    return (U/2)*real(sum_kω(kG, χd, λ = λd) - sum_kω(kG, χm, λ = λm)) + U * (n/2)^2
    #=lhs_c2 = 0.0
    for ωi = 1:size(χm, 2)
        tmp2 = 0.0
        for (qi, km) in enumerate(kG.kMult)
            χm_i_λ = 1 / (1/real(χm[qi, ωi]) + λm)
            χd_i_λ = 1 / (1/real(χd[qi, ωi]) + λd)
            tmp2 += (χd_i_λ - χm_i_λ) * km
        end
        lhs_c2 += 0.5 * tmp2 / Nk(kG)
    end
    lhs_c2 = lhs_c2 / χm.β
    return lhs_c2=#
end

# ----------------------------------------------- EKin -----------------------------------------------
"""
    EKin_p1_tail(νGrid::Vector{ComplexF64}, μ::Float64, h)
    EKin_p1_tail(iν_n::Vector{ComplexF64}, μ::Float64, U::Float64, β::Float64, n::Float64, kG::KGrid)

Tail cofficients for the one-particle kinetic energy [`EKin_p1`](@ref EKin_p1)
"""
EKin_p1_tail(νGrid::Vector{ComplexF64}, μ::Float64, h) = EKin_p1_tail(νGrid, μ, h.mP.U, h.mP.β, h.mP.n, h.kG)

function EKin_p1_tail(iν_n::Vector{ComplexF64}, μ::Float64, U::Float64, β::Float64, n::Float64, kG::KGrid)
    Σ_hartree = n * U / 2
    EKin_tail_c = (kG.ϵkGrid .+ Σ_hartree .- μ)
    tail = 1 ./ (iν_n .^ 2)
    EKin_tail = EKin_tail_c .* transpose(tail)
    EKin_tail_inv = (β / 2) .* kG.ϵkGrid .* (1 .+ -(β) .* EKin_tail_c)
    return EKin_tail, EKin_tail_inv
end

"""
    EKin_p1(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::λ₀T, h; λm::Float64 = 0.0, λd::Float64 = 0.0, tc::Type{<: ΣTail}=default_Σ_tail_correction())
    EKin_p1(G_ladder::AbstractMatrix, EKin_tail, EKin_tail_inv, β::Float64, kG::KGrid)

Kinetic energy an the one-particle level.
"""
function EKin_p1(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::λ₀T, h; λm::Float64 = 0.0, λd::Float64 = 0.0, tc::Type{<: ΣTail}=default_Σ_tail_correction())::Float64
    μ, G_ladder, Σ_ladder = calc_G_Σ(χm, γm, χd, γd, λ₀, λm, λd, h; tc=tc, fix_n = true);
    νGrid = 0:last(axes(Σ_ladder,2))
    iν_n = iν_array(h.mP.β, νGrid)
    EKin_tail, EKin_tail_inv = EKin_p1_tail(iν_n, μ, h)
    return EKin_p1(view(G_ladder,:, νGrid), EKin_tail, EKin_tail_inv, h.mP.β, h.kG)
end

function EKin_p1(G_ladder::AbstractMatrix, EKin_tail::AbstractMatrix, EKin_tail_inv, β::Float64, kG::KGrid)::Float64
    EKin_k = 4 .* sum(kG.ϵkGrid .* real.(G_ladder .- EKin_tail), dims=2)[:, 1] .+ EKin_tail_inv
    return kintegrate(kG, EKin_k) / β
end

"""
    EKin_p2(χm::χT, χd::χT, λm::Float64, λd::Float64, n::Float64, U::Float64)::Float64

Kinetic energy on the two-particle level.
"""
function EKin_p2(χm::χT, χd::χT)::Float64
    !(χm.tail_c[3] ≈ χd.tail_c[3]) && error("Kinetic energies on the two particle level in magnetic and density channel do not match!")
    return χm.tail_c[3]
end
# ---------------------------------------------- Common ----------------------------------------------
"""
calc_E(χ_sp::χT, γ_sp::γT, χ_ch::χT, γ_ch::γT, λ₀, Σ_loc, gLoc_rfft, kG::KGrid, mP::ModelParameters, sP::SimulationParameters; 
        νmax::Int = eom_ν_cutoff(sP), tc::Bool=true)
calc_E([G::Array{ComplexF64,2},] Σ::AbstractArray{ComplexF64,2}, kG::KGrid, mP::ModelParameters; trace::Bool=false)

Returns kinetic and potential energies from given 
    * self-energy `Σ` or
    * the ingredients of the equation of motion: the physical susceptibilies as well as the triangular vertices in spin and charge channel, the correction term and the greensfunction to be used.
"""
function calc_E(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::λ₀T, h; λm::Float64 = 0.0, λd::Float64 = 0.0, tc::Type{<: ΣTail}=default_Σ_tail_correction())::Float64
    μ, G_ladder, Σ_ladder = calc_G_Σ(χm, γm, χd, γd, λ₀, λm, λd, h; tc=tc);
    E_kin, E_pot = calc_E(G_ladder, Σ_ladder, μ, h.kG, h.mP)
    return E_kin, E_pot
end

function calc_E(Σ::OffsetMatrix{ComplexF64}, h; trace::Bool = false, fix_n = fix_n)
    νGrid = 0:last(axes(Σ,2))
    μ_new, G_ladder = G_from_Σladder(Σ, h.Σ_loc, h.kG, h.mP, h.sP; fix_n = fix_n)
    return calc_E(G_ladder, Σ, μ_new, h.kG, h.mP; trace = trace)
end

function calc_E(G::OffsetMatrix{ComplexF64}, Σ::OffsetMatrix{ComplexF64}, μ::Float64, kG::KGrid, mP::ModelParameters; trace::Bool = false)
    first(axes(Σ, 2)) != 0 && error("Calc_E assumes a ν grid starting at 0! check G and Σ axes.")
    νGrid = 0:last(axes(Σ,2))
    iν_n = iν_array(mP.β, νGrid)
    EKin_tail, EKin_tail_inv = EKin_p1_tail(iν_n, μ, mP.U, mP.β, mP.n, kG)
    EKin = EKin_p1(view(G,:, νGrid), EKin_tail, EKin_tail_inv, mP.β, kG)
    EPot_tail, EPot_tail_inv = EPot_p1_tail(iν_n, μ, mP.U, mP.β, mP.n, kG)
    EPot = EPot_p1(view(G,:, νGrid), view(Σ,:, νGrid), EPot_tail, EPot_tail_inv, mP.β, kG)
    return EKin, EPot
end

function calc_E_trace(G::OffsetMatrix{ComplexF64}, Σ::OffsetMatrix{ComplexF64}, μ::Float64, kG::KGrid, mP::ModelParameters; trace::Bool = false)
    first(axes(Σ, 2)) != 0 && error("Calc_E assumes a ν grid starting at 0! check G and Σ axes.")
    νGrid = 0:last(axes(Σ,2))
    iν_n = iν_array(mP.β, νGrid)
    Σ_hartree = mP.n * mP.U / 2

    E_kin_tail_c = (kG.ϵkGrid .+ Σ_hartree .- μ)
    E_pot_tail_c = (mP.U^2 *(mP.n/2) * (1 - (mP.n/2) ) .+ Σ_hartree .* (kG.ϵkGrid .+ Σ_hartree .- μ))
    tail = 1 ./ (iν_n .^ 2)
    tail_1 = 1 ./ (iν_n .^ 1)
    E_pot_tail = E_pot_tail_c .* transpose(tail)
    E_kin_tail = E_kin_tail_c .* transpose(tail)
    E_pot_tail_inv = (mP.β / 2) .* Σ_hartree .+ (mP.β / 2) * (-mP.β / 2) .* E_pot_tail_c
    E_kin_tail_inv = (mP.β / 2) .* kG.ϵkGrid .* (1 .+ -(mP.β) .* E_kin_tail_c)
    E_pot_full = real.((G[:, νGrid] .* Σ[:, νGrid]) .- E_pot_tail)
    E_kin_full = kG.ϵkGrid .* real.(G[:, νGrid] .- E_kin_tail)
    E_kin, E_pot = if trace
        [kintegrate(kG, 4 .* sum(view(E_kin_full, :, 1:i), dims = [2])[:, 1] .- tail_1 .+ E_kin_tail_inv) for i = 1:last(νGrid)] ./ mP.β,
        [kintegrate(kG, 2 .* sum(view(E_pot_full, :, 1:i), dims = [2])[:, 1] .+ E_pot_tail_inv) for i = 1:last(νGrid)] ./ mP.β
    else
        kintegrate(kG, 4 .* sum(E_kin_full, dims = [2])[:, 1] .+ E_kin_tail_inv) ./ mP.β, kintegrate(kG, 2 .* sum(E_pot_full, dims = [2])[:, 1] .+ E_pot_tail_inv) ./ mP.β
    end
    return E_kin, E_pot
end
