# ==================================================================================================== #
#                                          BSE_Tools.jl                                                #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#   Functions related to the solution of the Bethe Salpeter Equation                                   #                                                                       #
# -------------------------------------------- TODO -------------------------------------------------- #
# ==================================================================================================== #


# ========================================== Transformations =========================================
"""
    λ_from_γ(type::Symbol, γ::γT, χ::χT, U::Float64)

TODO: documentation
"""
function λ_from_γ(type::Symbol, γ::γT, χ::χT, U::Float64)
    s = (type == :d) ? -1 : 1
    res = similar(γ.data)
    for ωi = 1:size(γ, 3)
        for qi = 1:size(γ, 1)
            res[qi, :, ωi] = s .* view(γ, qi, :, ωi) .* (1 .+ s * U .* χ.data[qi, ωi]) .- 1
        end
    end
    return res
end


"""
    F_from_χ(type::Symbol, h::lDΓAHelper; diag_term=true)
    F_from_χ(χ::AbstractArray{ComplexF64,3}, G::AbstractArray{ComplexF64,1}, sP::SimulationParameters, β::Float64; diag_term=true)

TODO: documentation
"""
function F_from_χ(type::Symbol, h::lDΓAHelper; diag_term = true)
    χDMFT = if type == :m
        h.χDMFT_m
    elseif type == :d
        h.χDMFT_d
    else
        error("Unkown channel `$type`!")
    end
    F_from_χ(χDMFT, h.gImp[1, :], h.sP, h.mP.β; diag_term = diag_term)
end

function F_from_χ(
    χ::AbstractArray{ComplexF64,3},
    G::AbstractVector{ComplexF64},
    sP::SimulationParameters,
    β::Float64;
    diag_term = true,
)
    F = similar(χ)
    for ωi = 1:size(F, 3)
        for νpi = 1:size(F, 2)
            ωn, νpn = OneToIndex_to_Freq(ωi, νpi, sP)
            for νi = 1:size(F, 1)
                _, νn = OneToIndex_to_Freq(ωi, νi, sP)
                F[νi, νpi, ωi] =
                    -(χ[νi, νpi, ωi] + (νn == νpn && diag_term) * β * G[νn] * G[ωn+νn]) /
                    (G[νn] * G[ωn+νn] * G[νpn] * G[ωn+νpn])
            end
        end
    end
    return F
end

"""
    F_from_χ_gen(χ₀::χ₀T, χr::Array{ComplexF64,4})::Array{ComplexF64,4}

Calculates the full vertex from the generalized susceptibility ``\\chi^{\\nu\\nu'\\omega}_r`` and the bubble term ``\\chi_0`` via
``F^{\\nu\\nu'\\omega}_{r,\\mathbf{q}} 
     =
     \\beta^2 \\left( \\chi^{\\nu\\nu'\\omega}_{0,\\mathbf{q}} \\right)^{-1} 
     - \\left( \\chi^{\\nu\\omega}_{0,\\mathbf{q}} \\right)^{-1}  \\chi^{\\nu\\nu'\\omega}_{r,\\mathbf{q}} \\left( \\chi^{\\nu'\\omega}_{0,\\mathbf{q}} \\right)^{-1}``

For a version using the physical susceptibilities see [`F_from_χ_gen`](@ref F_from_χ_gen).
"""
function F_from_χ_gen(χ₀::χ₀T, χr::Array{ComplexF64,4})::Array{ComplexF64,4}
    F = similar(χr)
    for ωi = 1:size(χr, 4)
        for qi = 1:size(χr, 3)
            F[:, :, qi, ωi] =
                Diagonal(χ₀.β^2 ./ core(χ₀)[qi, :, ωi]) .-
                χ₀.β^2 .* χr[:, :, qi, ωi] ./ (core(χ₀)[qi, :, ωi] .* transpose(core(χ₀)[qi, :, ωi]))
        end
    end
    return F
end


"""
    F_from_χ_star_gen(χ₀::χ₀T, χstar_r::Array{ComplexF64,4}, χr::χT, γr::γT, Ur::Float64)

Calculates the full vertex from the generalized susceptibility ``\\chi^{\\nu\\nu'\\omega}_r``, the physical susceptibility ``\\chi^{\\omega}_r`` and the triangular vertex ``\\gamma^{\\nu\\omega}_r``.
This is usefull to calculate a ``\\lambda``-corrected full vertex. 

``F^{\\nu\\nu'\\omega}_{r,\\mathbf{q}} 
     =
     \\beta^2 \\left( \\chi^{\\nu\\nu'\\omega}_{0,\\mathbf{q}} \\right)^{-1} 
     - \\beta^2 (\\chi^{\\nu\\omega}_{0,\\mathbf{q}})^{-1} \\chi^{*,\\nu\\nu'\\omega}_{r,\\mathbf{q}} (\\chi^{\\nu'\\omega}_{0,\\mathbf{q}})^{-1} 
    + U_r (1 - U_r \\chi^{\\omega}_{r,\\mathbf{q}}) \\gamma^{\\nu\\omega}_{r,\\mathbf{q}} \\gamma^{\\nu'\\omega}_{r,\\mathbf{q}}``
For a version using the physical susceptibilities see [`F_from_χ_gen`](@ref F_from_χ_gen).
"""
function F_from_χ_star_gen(χ₀::χ₀T, χstar_r::Array{ComplexF64,4}, χr::χT, γr::γT, Ur::Float64)
    F = similar(χstar_r)
    for ωi in axes(χstar_r, 4)
        for qi in axes(χstar_r, 3)
            pre_factor = Ur * (1 - Ur * χr[qi, ωi])
            for νpi in axes(χstar_r, 2)
                @simd for νi in axes(χstar_r, 1)
                    @inbounds F[νi, νpi, qi, ωi] =
                        -χ₀.β^2 * χstar_r[νi, νpi, qi, ωi] / (core(χ₀)[qi, νi, ωi] * core(χ₀)[qi, νpi, ωi])
                    @inbounds F[νi, νpi, qi, ωi] += pre_factor * γr[qi, νi, ωi] * γr[qi, νpi, ωi]
                end
                F[νpi, νpi, qi, ωi] += χ₀.β^2 / core(χ₀)[qi, νpi, ωi]
            end
        end
    end
    return F
end

# ========================================= Bubble Calculation =======================================
"""
    calc_bubble(type::Symbol, h <: RunHelper; mode=:ph)
    calc_bubble(type::Symbol, Gνω::GνqT, Gνω_r::GνqT, kG::KGrid, mP::ModelParameters, sP::SimulationParameters; mode=:ph)

Calculates bubble term.

``\\chi^{\\omega}_{0,\\mathbf{q}} = -\\Sigma_{\\mathbf{k}} \\Sigma_{\\nu} G^{\\nu}_{\\mathbf{k}} \\cdot G^{\\nu+\\omega}_{\\mathbf{k}+\\mathbf{q}}``

where
    ``\\nu``     : Fermionic Matsubara frequencies
    ``\\omega``  : Bosonic Matsubara frequencies
    ``\\mathbf{k}, \\mathbf{q}``: Element of the first Brillouin zone

    This is a real-valued quantity.

Returns
-------------
Bubble, `χ₀::χ₀T`

Arguments
-------------
- **`type`**      : `Symbol`, can be `DMFT`, `local`, `RPA`, `RPA_exact`. TODO: documentation
- **`RPAHelper`** :  Helper struct generated by [`setup_RPA`](@ref setup_RPA).
- **`β`**         : `Float64`, Inverse temperature in natural units
- **`kG`**        : `KGrid`,   The k-grid on which to perform the calculation
- **`sP`**        : `SimulationParameters`, (to construct a frequency range)
- **`mode`**      : selects particle-hole (`:ph`, default) or particle-particle (`:pp`) notation 
"""
function calc_bubble(type::Symbol, h::RunHelper; mode = :ph)
    calc_bubble(type, h.gLoc_fft, h.gLoc_rfft, h.kG, h.mP, h.sP; mode = mode)
end

function calc_bubble(
    type::Symbol,
    Gνω::GνqT,
    Gνω_r::GνqT,
    kG::KGrid,
    mP::ModelParameters,
    sP::SimulationParameters;
    mode = :ph,
)
    #TODO: fix the size (BSE_SC inconsistency)
    data = Array{ComplexF64,3}(undef, length(kG.kMult), 2 * (sP.n_iν + sP.n_iν_shell), 2 * sP.n_iω + 1)
    νdim = ndims(Gνω) > 2 ? length(gridshape(kG)) + 1 : 2 # TODO: this is a fallback for gImp
    function conv_function!(νi::Int, νn::Int, ωi::Int, ωn::Int)
        if mode == :ph
            conv_fft!(kG, view(data, :, νi, ωi), selectdim(Gνω, νdim, νn), selectdim(Gνω_r, νdim, ωn + νn))
        elseif mode == :pp
            conv_fft!(
                kG,
                view(data, :, νi, ωi),
                selectdim(Gνω, νdim, νn),
                selectdim(Gνω, νdim, ωn - νn);
                crosscorrelation = false,
            )
        else
            error("Unkown mode $mode for bubble calculation. Options are :ph and :pp.")
        end
    end
    #     conv_fft!(kG, view(data,:,νi,ωi), selectdim(Gνω,νdim,νn), selectdim(Gνω_r,νdim, ωn+νn))
    # elseif mode == :pp
    #     conv_fft!(kG, view(data,:,νi,ωi), selectdim(Gνω,νdim,νn), selectdim(Gνω  ,νdim, ωn-νn); crosscorrelation=false)


    for (ωi, ωn) in enumerate(-sP.n_iω:sP.n_iω)
        νrange = ((-(sP.n_iν + sP.n_iν_shell)):(sP.n_iν+sP.n_iν_shell-1)) .- trunc(Int, sP.shift * ωn / 2)
        #TODO: fix the offset (BSE_SC inconsistency)
        for (νi, νn) in enumerate(νrange)
            conv_function!(νi, νn, ωi, ωn)
            data[:, νi, ωi] .*= -mP.β
        end
    end
    #TODO: not necessary after real fft
    data = _eltype === Float64 ? real.(data) : data
    return χ₀T(type, data, kG, -sP.n_iω:sP.n_iω, sP.n_iν, sP.shift, sP, mP, ν_shell_size = sP.n_iν_shell)
end

# ========================================== Correction Term =========================================

"""
    calc_χγ(type::Symbol, h::lDΓAHelper, χ₀::χ₀T)
    calc_χγ(type::Symbol, Γr::ΓT, χ₀::χ₀T, kG::KGrid, mP::ModelParameters, sP::SimulationParameters)

Calculates susceptibility and triangular vertex in `type` channel. See [`calc_χγ_par`](@ref calc_χγ_par) for parallel calculation.

This method solves the following equation:
``
\\chi_r = \\chi_0 - \\frac{1}{\\beta^2} \\chi_0 \\Gamma_r \\chi_r \\\\
\\Leftrightarrow (1 + \\frac{1}{\\beta^2} \\chi_0 \\Gamma_r) = \\chi_0 \\\\
\\Leftrightarrow (\\chi^{-1}_r - \\chi^{-1}_0) = \\frac{1}{\\beta^2} \\Gamma_r
``
"""
function calc_χγ(type::Symbol, h::Union{lDΓAHelper,AlDΓAHelper}, χ₀::χ₀T; verbose=true)
    calc_χγ(type, getfield(h, Symbol("Γ_$(type)")), χ₀, h.kG, h.mP, h.sP, verbose=verbose)
end

function calc_χγ(type::Symbol, Γr::ΓT, χ₀::χ₀T, kG::KGrid, mP::ModelParameters, sP::SimulationParameters; verbose=true)
    #TODO: find a way to reduce initialization clutter: move lo,up to sum_helper
    #TODO: χ₀ should know about its tail c2, c3
    s = if type == :d
        -1
    elseif type == :m
        1
    else
        error("Unkown type")
    end

    Nν = 2 * sP.n_iν
    Nq = length(kG.kMult)
    Nω = size(χ₀.data, χ₀.axis_types[:ω])
    #TODO: use predifened ranks for Nq,... cleanup definitions
    γ = Array{ComplexF64,3}(undef, Nq, Nν, Nω)
    χ = Array{Float64,2}(undef, Nq, Nω)

    νi_range = 1:Nν
    qi_range = 1:Nq

    χ_ω = Vector{_eltype}(undef, Nω)
    χννpω = Matrix{_eltype}(undef, Nν, Nν)
    ipiv = Vector{Int}(undef, Nν)
    work = _gen_inv_work_arr(χννpω, ipiv)
    λ_cache = Array{eltype(χννpω),1}(undef, Nν)

    for (ωi, ωn) in enumerate(-sP.n_iω:sP.n_iω)
        for qi in qi_range
            χννpω[:, :] = deepcopy(Γr[:, :, ωi])
            for l in νi_range
                χννpω[l, l] += 1.0 / χ₀.data[qi, χ₀.ν_shell_size+l, ωi]
            end
            inv!(χννpω, ipiv, work)
            if typeof(sP.χ_helper) <: BSE_Asym_Helpers
                χ[qi, ωi] = real(
                    calc_χλ_impr!(λ_cache, type, ωn, χννpω, 
                                  view(χ₀.data, qi, :, ωi), mP.U, mP.β,
                                  χ₀.asym[qi, ωi], sP.χ_helper,
                    ),
                )
                γ[qi, :, ωi] = (1 .- s * λ_cache) ./ (1 .+ s * mP.U .* χ[qi, ωi])
            else
                if typeof(sP.χ_helper) === BSE_SC_Helper
                    improve_χ!(type, ωi, view(χννpω, :, :, ωi), view(χ₀, qi, :, ωi), mP.U, mP.β, sP.χ_helper)
                end
                χ[qi, ωi] = real(sum(χννpω)) / mP.β^2
                for νk in νi_range
                    γ[qi, νk, ωi] = sum(view(χννpω, :, νk)) / (χ₀.data[qi, νk, ωi] * (1.0 + s * mP.U * χ[qi, ωi]))
                end
            end
        end
        #TODO: write macro/function for ths "real view" beware of performance hits
        #v = _eltype === Float64 ? view(χ,:,ωi) : @view reinterpret(Float64,view(χ,:,ωi))[1:2:end]
        v = view(χ, :, ωi)
        χ_ω[ωi] = kintegrate(kG, v)
    end
    log_q0_χ_check(kG, sP, χ, type; verbose=verbose)
    usable_ω = find_usable_χ_interval(real.(χ_ω), reduce_range_prct=sP.usable_prct_reduction)

    return χT(χ, mP.β, tail_c = [0, 0, mP.Ekin_1Pt], full_range=sP.dbg_full_chi_omega, usable_ω = usable_ω), γT(γ)
end


"""
    calc_gen_χ(Γr::ΓT, χ₀::χ₀T, kG::KGrid)

Calculates generalized susceptibility from `Γr` by solving the Bethe Salpeter Equation. 
See [`calc_χγ`](@ref calc_χγ) for direct (and more efficient) calculation of physical susceptibility and triangular vertex.

Returns: ``\\chi^{\\nu\\nu'\\omega}_q`` as 4-dim array with axis: `νi, νpi, qi, ωi`.
"""
function calc_gen_χ(Γr::ΓT, χ₀::χ₀T, kG::KGrid)

    Nν = size(Γr, 1)
    χννpω = similar(Γr, Nν, Nν, size(kG.kMult, 1), size(Γr, 3))
    χννpω_work = Matrix{ComplexF64}(undef, Nν, Nν)
    ipiv = Vector{Int}(undef, Nν)
    work = _gen_inv_work_arr(χννpω_work, ipiv)
    νi_range = 1:Nν

    for ωi = 1:size(Γr, 3)
        for qi = 1:length(kG.kMult)

            χννpω_work[:, :] = deepcopy(Γr[:, :, ωi])
            for l = 1:size(Γr, 1)
                χννpω_work[l, l] += 1.0 / χ₀.data[qi, χ₀.ν_shell_size+l, ωi]
            end
            inv!(χννpω_work, ipiv, work)
            χννpω[:, :, qi, ωi] = χννpω_work
        end
    end

    return χννpω
end


"""
    calc_local_EoM(lDGAhelper)
    calc_local_EoM(Fm, Fd, gImp::OffsetVector, mP::ModelParameters, sP::SimulationParameters)

Calculates local equation of motion from the full DMFT vertices in the magnetic and density channel `Fm` and `Fd`
as well as the impurity Green's function `gImp`.
The result should be equal to the impuirty self-energy and can also be used to determine a usable ν-range 
for the non-local equation of motion.
"""
function calc_local_EoM(lDGAhelper)
    F_m = F_from_χ(lDGAhelper.χDMFT_m, lDGAhelper.gImp[1, :], lDGAhelper.sP, lDGAhelper.mP.β)
    F_d = F_from_χ(lDGAhelper.χDMFT_d, lDGAhelper.gImp[1, :], lDGAhelper.sP, lDGAhelper.mP.β)
    ΣLoc_m, ΣLoc_d = calc_local_EoM(F_m, F_d, lDGAhelper.gImp[1, :], lDGAhelper.mP, lDGAhelper.sP)
    return 0.5 .* (ΣLoc_m .+ ΣLoc_d)
end

function calc_local_EoM(Fm, Fd, gImp::OffsetVector, mP::ModelParameters, sP::SimulationParameters)
    νmax = floor(Int, size(Fm,1)/2)
    ωGrid = -sP.n_iω:sP.n_iω
    ΣLoc_m = OffsetVector(zeros(ComplexF64, νmax), 0:νmax-1) 
    ΣLoc_d = OffsetVector(zeros(ComplexF64, νmax), 0:νmax-1)
    for (ωi,ωm) in enumerate(ωGrid)
        νnGrid = LadderDGA.νnGrid_noShell(ωm, sP)
        for (νi,νn) in enumerate(νnGrid)
            for (νpi,νpn) in enumerate(νnGrid)
                if νn >= 0 && νn < νmax   
                    ΣLoc_m[νn] += gImp[νpn] * gImp[νpn + ωm] * gImp[νn + ωm] * Fm[νi,νpi,ωi]
                    ΣLoc_d[νn] -= gImp[νpn] * gImp[νpn + ωm] * gImp[νn + ωm] * Fd[νi,νpi,ωi]
                end
            end
        end
    end
    return  mP.U .* ΣLoc_m/mP.β^2 .+ mP.U*mP.n/2,  mP.U .* ΣLoc_d/mP.β^2 .+ mP.U*mP.n/2
end
