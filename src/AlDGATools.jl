# ==================================================================================================== #
#                                         AlDGATools.jl                                                #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#   Asymptotic ladder DΓA related functions                                                            #
# -------------------------------------------- TODO -------------------------------------------------- #
#   Cleanup, complete refactor??                                                                       #
# ==================================================================================================== #



# ========================================== Correction Term =========================================
"""
    correction_term(mP::ModelParameters, kG::KGrid, χm::χT, grid::AbstractArray{Int64,1})

Calculates the so called tail correcion term of the ladder self energy. The purpose of this term is to enforce the limit

`\\lim_{n\\rightarrow\\infty}i\\nu_n\\Sigma_{\\mathbf{q}}^{\\nu_n}=U^2\\frac{n}{2}\\left(1-\\frac{n}{2} \\right )`.

This can be archived by adding the term
    * RPA: `-\\frac{U^2}{i\\nu}\\sum_{\\omega,\\mathbf{q}}\\left( \\chi_{m,\\mathbf{q}}^{\\omega}-\\chi_{0,\\mathbf{q}}^{\\omega}\\right )`
    * ladder-DGA: `-\\frac{U^2}{i\\nu}\\left(\\sum_{\\omega,\\mathbf{q}}\\chi_{m,\\mathbf{q}}^{\\omega}-\\chi_{m,loc} \\right )`
    * Asymptotic-ladder-DGA: see RPA

TODO: derive asymptotic corretion properly
from the ladder self energy.
Arguments
-------------
- **`mP`**         : ModelParameters
- **`kG`**         : KGrid
- **`χm`**         : χT
- **`χ_m_sum`**    : Union{Float64,ComplexF64}. RPA: `\\sum_{\\omega,\\mathbf{q}}\\chi_{0,\\mathbf{q}}^{\\omega}`, lDGA: 'χ_m_sum'.
- **`grid`**       : AbstractArray{Int64,1}
"""
function correction_term(mP::ModelParameters, kG::KGrid, χm::χT, grid::AbstractArray{Int64,1})
    return - (mP.U .^2) .* (sum_kω(kG, χm) - 0.25) ./ iν_array(mP.β, collect(grid))
end

"""
    calc_λ0(χ₀::χ₀T, h::lDΓAHelper)
    calc_λ0(χ₀::χ₀T, Fr::FT, h::lDΓAHelper)
    calc_λ0(χ₀::χ₀T, Fr::FT, χ::χT, γ::γT, mP::ModelParameters, sP::SimulationParameters)

Correction term, TODO: documentation
"""
function calc_λ0(χ₀::χ₀T, h::AlDΓAHelper)
    F_m = F_from_χ(:m, h)
    @error("Not Implemented Yet")
    calc_λ0_asym(χ₀, F_m, h)
end

function calc_λ0(χ₀::χ₀T, Fr::FT, h::AlDΓAHelper)::λ₀T
    @error("Not Implemented Yet")
    calc_λ0_asym(χ₀, Fr, h.χ_m_loc, h.γ_m_loc, h.mP, h.sP)
end

function calc_λ0_asym(χ₀::χ₀T, Fr::FT, χ::χT, γ::γT, mP::ModelParameters, sP::SimulationParameters; improved_sums::Bool = true)::λ₀T
    @error("Not Implemented Yet")
    #TODO: store nu grid in sP?
    Niν = size(Fr, 1)
    ω_range = 1:size(χ₀.data, χ₀.axis_types[:ω])
    λ0 = λ₀T(undef, size(χ₀.data, χ₀.axis_types[:q]), Niν, length(ω_range))
    λ0[:] = calc_λ0_impr(:m, -sP.n_iω:sP.n_iω, Fr, χ₀.data, χ₀.asym, view(γ.data, 1, :, :), view(χ.data, 1, :), mP.U, mP.β, sP.χ_helper)
end

function calc_Σ(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::λ₀T, h::AlDΓAHelper; νmax::Int=eom_ν_cutoff(h), λm::Float64 = 0.0, λd::Float64 = 0.0, tc::Bool = false)
    calc_Σ(χm, γm, χd, γd, 0.25, λ₀, Σ_loc, h.gLoc_rfft, h.kG, h.mP, h.sP, νmax=νmax, λm = λm, λd = λd, tc = tc)
end

function calc_Σ(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::λ₀T, gLoc_rfft, h::AlDΓAHelper; νmax::Int=eom_ν_cutoff(h), λm::Float64 = 0.0, λd::Float64 = 0.0, tc::Bool = false)
    Σ_loc = OffsetArray(zeros(ComplexF64, νmax), 0:νmax-1)
    calc_Σ(χm, γm, χd, γd, 0.25, λ₀, Σ_loc, gLoc_rfft, h.kG, h.mP, h.sP, νmax=νmax, λm = λm, λd = λd, tc = tc)
end

function update_ΓAsym!(χm::χT, χd::χT, χpp::χT, sP::SimulationParameters, mP::ModelParameters, h::AlDΓAHelper)
    U = mP.U
    h.Γ_d .=  U
    h.Γ_m .= -U 
    χd_loc = kintegrate(h.kG, χd, 1)[1,:] 
    χm_loc = kintegrate(h.kG, χm, 1)[1,:] 
    χpp_loc = kintegrate(h.kG, χpp, 1)[1,:] 
    skip = 0
    tot = 0
    for (ωi, ωn) in enumerate(-sP.n_iω:sP.n_iω)
        nGrid = νnGrid_noShell(ωn, sP)
        for (νi, νn) in enumerate(nGrid)
            for (νpi, νpn) in enumerate(nGrid)
                i1_pre = νpn - νn
                i2_pre = νn + νn - 1 + ωn
                i1 = (sP.n_iω + 1) + (i1_pre)  
                i2 = (sP.n_iω + 1) + (i2_pre)  
                h.Γ_d[νi, νpi, ωi] += (U^2/2) * get_val_or_zero(χd_loc,i1) + (3*U^2/2) * get_val_or_zero(χm_loc,i1) - (U^2) * get_val_or_zero(χpp_loc,i2) 
                h.Γ_m[νi, νpi, ωi] += (U^2/2) * get_val_or_zero(χd_loc,i1) -   (U^2/2) * get_val_or_zero(χm_loc,i1) + (U^2) * get_val_or_zero(χpp_loc,i2) 
            end
        end
    end
end
