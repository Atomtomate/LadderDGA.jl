# ==================================================================================================== #
#                                        ladderDGATools.jl                                             #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe, Jan Frederik Weißler                                              #
# ----------------------------------------- Description ---------------------------------------------- #
#   ladder DΓA related functions                                                                       #
# -------------------------------------------- TODO -------------------------------------------------- #
#   Cleanup, complete refactor??                                                                       #
# ==================================================================================================== #



# ========================================== Correction Term =========================================

"""
    Σ_hartree(mP::ModelParameters)

Calculates the hartree term of the self energy

`\\Sigma_{hartree}=\\frac{U\\cdot n}{2}`.

Arguments
-------------
- **`mP`** : ModelParameters

Returns
-------
Float64 : Hartree term 
"""
function Σ_hartree(mP::ModelParameters)
    return mP.n * mP.U/2.0
end

"""
    correction_term(mP::ModelParameters, kG::KGrid, χm::χT, χ_m_sum::Union{Float64,ComplexF64}, grid::AbstractArray{Int64,1})

Calculates the so called tail correcion term of the ladder self energy. The purpose of this term is to enforce the limit

`\\lim_{n\\rightarrow\\infty}i\\nu_n\\Sigma_{\\mathbf{q}}^{\\nu_n}=U^2\\frac{n}{2}\\left(1-\\frac{n}{2} \\right )`.

This can be archived by adding the term
    * RPA: `-\\frac{U^2}{i\\nu}\\sum_{\\omega,\\mathbf{q}}\\left( \\chi_{m,\\mathbf{q}}^{\\omega}-\\chi_{0,\\mathbf{q}}^{\\omega}\\right )`
    * ladder-DGA: `-\\frac{U^2}{i\\nu}\\left(\\sum_{\\omega,\\mathbf{q}}\\chi_{m,\\mathbf{q}}^{\\omega}-\\chi_{m,loc} \\right )`
from the ladder self energy.

Arguments
-------------
- **`mP`**         : ModelParameters
- **`kG`**         : KGrid
- **`χm`**         : χT
- **`χ_m_sum`**    : Union{Float64,ComplexF64}. RPA: `\\sum_{\\omega,\\mathbf{q}}\\chi_{0,\\mathbf{q}}^{\\omega}`, lDGA: 'χ_m_sum'.
- **`grid`**       : AbstractArray{Int64,1}
"""
#function correction_term(mP::ModelParameters, kG::KGrid, χm::χT, χ_m_sum::Union{Float64,ComplexF64}, grid::AbstractArray{Int64,1})
#    return - (mP.U) .* (sum_kω(kG, χm) - χ_m_sum) ./ iν_array(mP.β, collect(grid))
#end

"""
    calc_λ0(χ₀::χ₀T, h::lDΓAHelper)
    calc_λ0(χ₀::χ₀T, Fr::FT, h::lDΓAHelper)
    calc_λ0(χ₀::χ₀T, Fr::FT, χ::χT, γ::γT, mP::ModelParameters, sP::SimulationParameters)

Correction term, TODO: documentation
"""
function calc_λ0(χ₀::χ₀T, h::lDΓAHelper; diag_zero::Bool=true, dbg=true)
    F_m = F_from_χ(:m, h)
    calc_λ0(χ₀, F_m, h; diag_zero=diag_zero, dbg=dbg)
end

function calc_λ0(χ₀::χ₀T, Fr::FT, h::lDΓAHelper; diag_zero::Bool=true, dbg=true)::λ₀T
    calc_λ0(χ₀, Fr, h.χ_m_loc, h.γ_m_loc, h.mP, h.sP; diag_zero=diag_zero)
end

function calc_λ0(χ₀::χ₀T, Fr::FT, χ::χT, γ::γT, mP::ModelParameters, sP::SimulationParameters;
        improved_sums::Bool = true, diag_zero::Bool=true, use_threads::Bool=true)::λ₀T
    #TODO: store nu grid in sP?
    Niν = size(Fr, 1)
    Nq = size(χ₀.data, χ₀.axis_types[:q])
    ω_range = 1:size(χ₀.data, χ₀.axis_types[:ω])
    λ0 = λ₀T(undef, size(χ₀.data, χ₀.axis_types[:q]), Niν, length(ω_range))

    if improved_sums && typeof(sP.χ_helper) <: BSE_Asym_Helpers
        #TODO: decide what to do about the warning for ignoring the diagonal part
        λ0[:] = calc_λ0_impr(:m, -sP.n_iω:sP.n_iω, Fr, χ₀.data, χ₀.asym, view(γ.data, 1, :, :), view(χ.data, 1, :), mP.U, mP.β, sP.χ_helper; diag_zero=diag_zero)
    else
        #TODO: this is not well optimized, but also not often executed
        @warn "Using plain summation for λ₀, check Σ_ladder tails!"
        fill!(λ0, 0.0)
        for ωi in ω_range
            for νi = 1:Niν
                #TODO: export realview functions?
                v1 = view(Fr, νi, :, ωi)
                for qi = 1:Nq
                    v2 = view(χ₀.data, qi, (sP.n_iν_shell+1):(size(χ₀.data, 2)-sP.n_iν_shell), ωi)
                    λ0[qi, :, ωi] = λ0[qi, :, ωi] .+ v1 .* v2 ./ mP.β^2
                end
            end
        end
    end
    return λ0
end


#=
function calc_λ0_impr(type::Symbol, ωgrid::AbstractVector{Int},
                 F::AbstractArray{ComplexF64,3}, χ₀::AbstractArray{ComplexF64,3}, 
                 χ₀_asym::Array{ComplexF64,2}, γ::AbstractArray{ComplexF64,2}, 
                 χ::AbstractArray{T,1},
                 U::Float64, β::Float64, h; diag_zero::Bool=true) where T <: Union{ComplexF64,Float64}
    s = (type == :d) ? -1 : +1
    ind_core = (h.Nν_shell+1):(size(χ₀,2)-h.Nν_shell)
    Nq = size(χ₀,1)
    Nν = length(ind_core)
    Nω = size(χ₀,3)
    λasym = Array{ComplexF64,1}(undef, Nν)
    λcore = Array{ComplexF64,1}(undef, Nν)
    res = Array{ComplexF64,3}(undef, Nq, Nν, Nω)

    F_diag!(diag_asym_buffer::Vector{ComplexF64}, qi::Int, ωi::Int,  ωn::Int, χ₀::Array{ComplexF64,3},
                        buffer::OffsetMatrix{ComplexF64}, h::BSE_Asym_Helper)

    for (ωi,ωn) in enumerate(ωgrid)
        λasym = -(view(γ,:,ωi) .* (1 .+ s*U .* χ[ωi]) ) .+ 1
        for qi in 1:Nq
            λcore[:] = [s*dot(view(χ₀,qi,ind_core,ωi), view(F,νi,:,ωi))/(β^2) for νi in 1:size(F,1)]
            diag_term = if !diag_zero && hasfield(typeof(h), :diag_asym_buffer)
                F_diag!(diag_asym_buffer, qi, ωi, ωn, χ₀, h.buffer_m, h)
                view(diag_asym_buffer, ind_core)
            else
                0.0
            end
            res[qi,:,ωi] = λcore + χ₀_asym[qi,ωi].*U.*(λasym .- 1) .+ diag_term
        end
    end
    return res
end
=#