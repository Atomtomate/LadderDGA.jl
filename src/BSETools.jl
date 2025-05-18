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
            ωm, νpn = OneToIndex_to_Freq(ωi, νpi, sP)
            for νi = 1:size(F, 1)
                _, νn = OneToIndex_to_Freq(ωi, νi, sP)
                F[νi, νpi, ωi] =
                    -(χ[νi, νpi, ωi] + (νn == νpn && diag_term) * β * G[νn] * G[ωm+νn]) /
                    (G[νn] * G[ωm+νn] * G[νpn] * G[ωm+νpn])
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
function calc_bubble(type::Symbol, h::RunHelper; mode::Symbol = :ph, use_threads::Bool=true)
    calc_bubble(type, h.gLoc_fft, h.gLoc_rfft, h.kG, h.mP, h.sP; mode = mode, use_threads=use_threads)
end

function calc_bubble(
    type::Symbol,
    Gνω::GνqT,
    Gνω_r::GνqT,
    kG::KGrid,
    mP::ModelParameters,
    sP::SimulationParameters;
    ωrange::AbstractVector = -sP.n_iω:sP.n_iω,
    mode::Symbol = :ph,
    use_threads::Bool=true
)
    mode in (:ph, :pp) || error("Unknown mode `$mode`!")

    data = Array{ComplexF64,3}(undef, length(kG.kMult), 2 * (sP.n_iν + sP.n_iν_shell), 2 * sP.n_iω + 1)
    νdim = ndims(Gνω) > 2 ? length(gridshape(kG)) + 1 : 2 # TODO: this is a fallback for gImp

    if use_threads
        bthreads = BLAS.get_num_threads()
        BLAS.set_num_threads(1)
        NT = Threads.nthreads()
        caches = [similar(kG.cache1) for ti in 1:NT]
        Threads.@threads for ωi in eachindex(ωrange)
            ωm = ωrange[ωi]
            νrange = ((-(sP.n_iν + sP.n_iν_shell)):(sP.n_iν+sP.n_iν_shell-1)) .- trunc(Int, sP.shift * ωm / 2)
            for (νi, νn) in enumerate(νrange)
                conv_fft_inlined!(
                    kG, 
                    view(data, :, νi, ωi), 
                    selectdim(Gνω, νdim, νn), 
                    selectdim(Gνω_r, νdim, ωm + νn), 
                    crosscorrelation = mode == :ph,
                    cache=caches[Threads.threadid()]
                )
                data[:, νi, ωi] .*= -mP.β
            end
        end
        BLAS.set_num_threads(bthreads)
    else
        for (ωi, ωm) in enumerate(-sP.n_iω:sP.n_iω)
            νrange = ((-(sP.n_iν + sP.n_iν_shell)):(sP.n_iν+sP.n_iν_shell-1)) .- trunc(Int, sP.shift * ωm / 2)
            for (νi, νn) in enumerate(νrange)
                conv_fft!(
                    kG, 
                    view(data, :, νi, ωi), 
                    selectdim(Gνω, νdim, νn), 
                    selectdim(Gνω_r, νdim, ωm + νn), 
                    crosscorrelation = mode == :ph,
                    cache=kG.cache1
                )
                data[:, νi, ωi] .*= -mP.β
            end
        end
    end

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
function calc_χγ(type::Symbol, h::Union{lDΓAHelper,AlDΓAHelper}, χ₀::χ₀T; verbose=true, ω_symmetric::Bool=false, use_threads=true)
    calc_χγ(type, getfield(h, Symbol("Γ_$(type)")), χ₀, h.kG, h.mP, h.sP; verbose=verbose, ω_symmetric=ω_symmetric, use_threads=use_threads)
end

function calc_χγ(type::Symbol, Γr::ΓT, χ₀::χ₀T, kG::KGrid, mP::ModelParameters, sP::SimulationParameters;
         verbose=true, ω_symmetric::Bool=false, use_threads=true)
    s = type == :d ? -1 : 1
    !(type in (:d, :m)) && error("Unkown type $type")
    
    Nν = 2 * sP.n_iν
    Nq = length(kG.kMult)
    Nω = size(χ₀.data, χ₀.axis_types[:ω])
    ωm_range = ω_symmetric ? (0:(sP.n_iω)) : ((-sP.n_iω):(sP.n_iω))
    γ = Array{ComplexF64,3}(undef, Nq, Nν, Nω)
    χ = Array{Float64,2}(undef, Nq, Nω)

    qi_range = 1:Nq
    χ_ω = Vector{Float64}(undef, Nω)

    if use_threads
        bthreads = BLAS.get_num_threads()
        BLAS.set_num_threads(1)
        NT = Threads.nthreads()
        χννpω = [Matrix{ComplexF64}(undef, Nν, Nν) for ti in 1:NT]
        ipiv = [Vector{Int}(undef, Nν) for ti in 1:NT]
        work = [_gen_inv_work_arr(χννpω[Threads.threadid()], ipiv[Threads.threadid()]) for ti in 1:NT]
        λ_cache = [Vector{ComplexF64}(undef, Nν) for ti in 1:NT]
        diag_cache = [similar(sP.χ_helper.diag_asym_buffer) for ti in 1:NT]
        Threads.@threads for qi in qi_range
            for ωm in ωm_range
                ωi =  ωm + sP.n_iω + 1
                invert_BSE!(χ, γ, χννpω[Threads.threadid()], ipiv[Threads.threadid()], work[Threads.threadid()], 
                        sP.χ_helper, λ_cache[Threads.threadid()], diag_cache[Threads.threadid()], type, ω_symmetric, s, Γr, χ₀.data, χ₀.asym, 
                        χ₀.ν_shell_size, qi, sP.n_iω, ωm, ωi, mP.U, mP.β)
            end
        end
        BLAS.set_num_threads(bthreads)
    else
        χννpω = Matrix{ComplexF64}(undef, Nν, Nν)
        ipiv = Vector{Int}(undef, Nν)
        work = _gen_inv_work_arr(χννpω, ipiv)
        λ_cache = Vector{ComplexF64}(undef, Nν)

        for ωm in ωm_range
            ωi =  ωm + sP.n_iω + 1
            for qi in qi_range
                invert_BSE!(χ, γ, χννpω, ipiv, work, 
                    sP.χ_helper, λ_cache, sP.χ_helper.diag_asym_buffer, type, ω_symmetric, s, Γr, χ₀.data, χ₀.asym, 
                    χ₀.ν_shell_size, qi, sP.n_iω, ωm, ωi, mP.U, mP.β)
            end
        end
    end

    for (ωi, ωm) in enumerate(-sP.n_iω:sP.n_iω)
        v = view(χ, :, ωi)
        χ_ω[ωi] = kintegrate(kG, v)
    end
    log_q0_χ_check(kG, sP, χ, type; verbose=verbose)
    usable_ω = find_usable_χ_interval(real.(χ_ω), reduce_range_prct=sP.usable_prct_reduction)

    return χT(χ, mP.β, tail_c = [0, 0, mP.Ekin_1Pt], full_range=sP.dbg_full_chi_omega, usable_ω = usable_ω), γT(γ)
end

function invert_BSE!(χ::AbstractArray{Float64,2}, γ::AbstractArray{ComplexF64,3}, χννpω::AbstractArray{ComplexF64,2}, 
                    ipiv::Vector{Int}, work::Vector{ComplexF64}, χ_helper::BSE_Asym_Helper, λ_cache::Vector{ComplexF64}, diag_cache::Vector{ComplexF64}, 
                    type::Symbol, ω_symmetric::Bool, s::Int, 
                    Γr::AbstractArray{ComplexF64,3}, χ₀_data::AbstractArray{ComplexF64,3}, χ₀_asym::AbstractArray{ComplexF64,2},
                    ν_shell_size::Int, qi::Int, ωm_max::Int, ωm::Int, ωi::Int, U::Float64, β::Float64)
    
    copy!(χννpω, view(Γr,:,:, ωi))
    for l in axes(χννpω, 1)
        @inbounds χννpω[l, l] += 1.0 / χ₀_data[qi, ν_shell_size+l, ωi]
    end
    inv!(χννpω, ipiv, work)
    if typeof(χ_helper) <: BSE_Asym_Helpers
        χ[qi, ωi] = real(
            calc_χλ_impr!(λ_cache, diag_cache, type, qi, ωi, ωm, χννpω, 
                          χ₀_data, U, β,
                          χ₀_asym[qi, ωi], χ_helper,
            ))
        for νi in axes(γ,2)
            @inbounds γ[qi, νi, ωi] = (1 - s * λ_cache[νi]) / (1 + s * U * χ[qi, ωi])
        end
    else
        if typeof(χ_helper) === BSE_SC_Helper
            improve_χ!(type, ωi, view(χννpω, :, :, ωi), view(χ₀_data, qi, :, ωi), U, β, χ_helper)
        end
        χ[qi, ωi] = real(sum(χννpω)) / β^2
        for νk in axes(γ, 2)
            γ[qi, νk, ωi] = sum(view(χννpω, :, νk)) / (χ₀_data[qi, ν_shell_size+νk, ωi] * (1.0 + s * U * χ[qi, ωi]))
        end
    end
    if ω_symmetric && ωm > 0
        ωi_mirror =  ωm_max + 1 - ωm
        χ[qi, ωi_mirror] = χ[qi, ωi]
        γ[qi, :, ωi_mirror] = conj(reverse(γ[qi, :, ωi]))
    end
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

    for ωi = axes(Γr, 3)
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


# ======================================== Consistency Checks ========================================
"""
    log_q0_χ_check(kG::KGrid, sP::SimulationParameters, χ::AbstractArray{_eltype,2}, type::Symbol)

TODO: documentation
"""
function log_q0_χ_check(kG::KGrid, sP::SimulationParameters, χ::AbstractArray{Float64,2}, type::Symbol; verbose=true)::Float64
    q0_ind = q0_index(kG)
    res = NaN
    if !isnothing(q0_ind)
        #TODO: adapt for arbitrary ω indices
        ω_ind = setdiff(1:size(χ, 2), sP.n_iω + 1)
        res = sum(abs.(view(χ,q0_ind,ω_ind)))
        if verbose
            @info "$type channel: |∑χ(q=0,ω≠0)| = $(round(res,digits=12)) ≟ 0"
        end
    end
    return res
end





# ============================================= Fixes ================================================
"""
    check_χ_health(χr, channel::Symbol, h::lDΓAHelper; q0_check_eps = 0.1, λmmin_check_eps = 1000)

Checks properties of the physical susceptibility `χr` in the given `channel`.
Returns a list of two boolean values indicating the health of the susceptibility.
(1) `q0_check_res`: `true` if the sum of the susceptibility at q=0 is close to a delta distribution.
(2) `λmmin_check_res`: `true` if the minimal λ-value is small.
"""
function check_χ_health(χr, channel::Symbol, h::lDΓAHelper; q0_check_eps = 0.1, λmin_check_eps = 1000)
    q0_r = log_q0_χ_check(h.kG, h.sP, χr, channel)
    λmin = get_λ_min(χr)
    magnitute = sum_kω(h.kG, χr)
    @info "Channel $channel: λm_min = $(λmin)"
    @info "∑_kω = $(magnitute)"
    #@info "Channel $#channel: q0_m = $(q0_r)"

    q0_check_res = q0_r < q0_check_eps
    λmin_check_res = λmin < λmin_check_eps

    if !q0_check_res
        @warn "Channel $channel:  |∑χ(q=0,ω≠0)| is not close to a delta distribution!"
    end

    
    if !λmin_check_res
        @warn "Channel $channel: λ_min very large an positive! This often indicates a channel with small weight."
        @warn "Consider calling fix_small_channel()"
    end
    return q0_check_res, λmin_check_res
end

"""
    fix_χr(χr, negative_eps = 1e-2)

(1) For ω ≠ 0: Sets negative values from the susceptibility `χr` to zero.
(2) For ω = 0: Inverts the sign of all negative values in `χr` WHICH ARE TOO CLOSE TO ZERO (Otherwise this would artificially prevent phase transitions).
"""
function fix_χr(χr; negative_eps = 1e-2)
    χr_copy = deepcopy(χr)
    fix_χr!(χr_copy; negative_eps = negative_eps)
    return χr_copy
end

function fix_χr!(χr; negative_eps = 1e-2)
    @warn "This will artificially invert the sign of all χ[q, ω₀] which are negative and close to zero AND set all χ[q, ω ≠ 0] to zero"
    @warn "Alternatively, you may consider setting λmin explicitly to a small value."

    ω0ind = ω0_index(χr)
    ii_inv = findall(x-> x < 0 && abs(x) < negative_eps, χr[:,ω0ind])
    χr[ii_inv,ω0ind] .= -χr[ii_inv,ω0ind]

    non_zero_ind = union(first(axes(χr,2)):(ω0_index(χr)-1),(ω0ind+1):last(axes(χr,2)))
    ii_zero = findall(x-> x < 0, χr)
    filter!(x-> x[2] in non_zero_ind,ii_zero)
    χr[ii_zero] .= 0.0

    return χr
end