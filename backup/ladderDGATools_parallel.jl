# ==================================================================================================== #
#                                        ladderDGATools.jl                                             #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe, Jan Frederik Weißler                                              #
# ----------------------------------------- Description ---------------------------------------------- #
#   ladder DΓA related functions                                                                       #
# -------------------------------------------- TODO -------------------------------------------------- #
#   Cleanup, combine *singlecore.jl                                                                    #
# ==================================================================================================== #


# ======================================== LadderDGA Functions =======================================
# ------------------------------------------- Bubble Term --------------------------------------------
function χ₀_conv(ωi_range::Vector{NTuple{2,Int}})
    kG = wcache[].kG
    mP = wcache[].mP
    sP = wcache[].sP
    n_iν = 2 * (sP.n_iν + sP.n_iν_shell)
    νdim = length(gridshape(wcache[].kG)) + 1
    data::Array{ComplexF64,3} = Array{ComplexF64,3}(undef, length(kG.kMult), n_iν, length(ωi_range))
    for (iω, ωnR) in enumerate(ωi_range)
        _, ωn = ωnR
        νGrid = νnGrid_shell(ωn, sP)
        for (iν, νn) in enumerate(νGrid)
            conv_fft_noPlan!(kG, view(data, :, iν, iω), selectdim(wcache[].G_fft, νdim, νn), selectdim(wcache[].G_fft_reverse, νdim, νn + ωn))
        end
    end
    c1, c2, c3 = χ₀Asym_coeffs(:DMFT, kG, mP)
    asym = χ₀Asym(c1, c2, c3, map(x -> x[2], ωi_range), sP.n_iν, sP.shift, mP.β)
    update_wcache!(:χ₀, -mP.β .* data)
    update_wcache!(:χ₀Indices, ωi_range)
    update_wcache!(:χ₀Asym, asym)
    return nothing
end

"""
    calc_bubble_par(h::lDΓAHelper; collect_data=true)
    calc_bubble_par(kG::KGrid, mP::ModelParameters, sP::SimulationParameters; collect_data=true)

Calculates the bubble, based on two fourier-transformed Greens functions where the second one has to be reversed.
"""
calc_bubble_par(h::lDΓAHelper; collect_data = true) = calc_bubble_par(h.kG, h.mP, h.sP; collect_data = collect_data)

function calc_bubble_par(kG::KGrid, mP::ModelParameters, sP::SimulationParameters; collect_data = true)
    worker_list = workers()
    !(length(worker_list) > 1) && throw(ErrorException("Add workers and run lDGA_setup before calling parallel functions!"))
    ωi_range, ωi_part = gen_ω_part(sP, length(worker_list))

    @sync begin
        for (i, ind) in enumerate(ωi_part)
            @async remotecall_fetch(χ₀_conv, worker_list[i], ωi_range[ind])
        end
    end

    return collect_data ? collect_χ₀(kG, mP, sP) : nothing
end

"""
    collect_χ₀(kG::KGrid, mP::ModelParameters, sP::SimulationParameters)

Collect non-local bubble ``\\chi_0^{\\omega}(q)`` from workers. Values first need to be calculated using [`calc_bubble_par`](@ref calc_bubble_par).
"""
collect_χ₀(h::lDΓAHelper) = collect_χ₀(h.kG, h.mP, h.sP)

function collect_χ₀(kG::KGrid, mP::ModelParameters, sP::SimulationParameters)
    !(length(workers()) > 1) && throw(ErrorException("Add workers and run lDGA_setup before calling parallel functions!"))
    data::Array{ComplexF64,3} = Array{ComplexF64,3}(undef, length(kG.kMult), 2 * (sP.n_iν + sP.n_iν_shell), 2 * sP.n_iω + 1)

    @sync begin
        for w in workers()
            @async begin
                indices = @fetchfrom w LadderDGA.wcache[].χ₀Indices
                χdata = @fetchfrom w LadderDGA.wcache[].χ₀
                for i = 1:length(indices)
                    ωi, _ = indices[i]
                    data[:, :, ωi] = χdata[:, :, i]
                end
            end
        end
    end
    χ₀T(:DMFT, data, kG, -sP.n_iω:sP.n_iω, sP.n_iν, sP.shift, sP, mP)
end

# --------------------------------------------- χ and γ ----------------------------------------------
"""
    bse_inv(type::Symbol, Γr::Array{ComplexF64,3}) 

Kernel for calculation of susceptibility and triangular vertex. Used by [`calc_χγ_par`](@ref calc_χγ_par).
"""
function bse_inv(type::Symbol, Γr::Array{ComplexF64,3})
    !(typeof(wcache[].sP.χ_helper) <: BSE_Asym_Helpers) && throw("Current version ONLY supports BSE_Asym_Helper!")
    s = type === :d ? -1 : 1
    Nν = size(Γr, 1)
    Nq = size(wcache[].χ₀, 1)
    Nω = length(wcache[].χ₀Indices)
    χ_data = Array{Float64,2}(undef, Nq, Nω)
    γ_data = Array{eltype(Γr),3}(undef, Nq, Nν, Nω)

    χννpω = Matrix{eltype(Γr)}(undef, Nν, Nν)
    ipiv = Vector{Int}(undef, Nν)
    work = _gen_inv_work_arr(χννpω, ipiv)
    λ_cache = Array{eltype(χννpω),1}(undef, Nν)

    for (i, ω_el) in enumerate(wcache[].χ₀Indices)
        ωi, ωn = ω_el
        for qi = 1:Nq
            copy!(χννpω, view(Γr, :, :, ωi))
            for l = 1:size(χννpω, 1)
                χννpω[l, l] += 1.0 / wcache[].χ₀[qi, wcache[].sP.n_iν_shell+l, i]
            end
            inv!(χννpω, ipiv, work)
            χ_data[qi, i] = real(calc_χλ_impr!(λ_cache, type, ωn, χννpω, view(wcache[].χ₀, qi, :, i), wcache[].mP.U, wcache[].mP.β, wcache[].χ₀Asym[qi, i], wcache[].sP.χ_helper))
            γ_data[qi, :, i] = (1 .- s * λ_cache) ./ (1 .+ s * wcache[].mP.U .* χ_data[qi, i])
        end
    end
    # TODO: unify naming ch/_d
    update_wcache!(Symbol(string("χ", type, "_part")), χ_data)
    update_wcache!(Symbol(string("γ", type)), γ_data)
end

"""
    calc_χγ_par(type::Symbol, h::lDΓAHelper)
    calc_χγ_par(type::Symbol, Γr::ΓT, kG::KGrid, mP::ModelParameters, sP::SimulationParameters; collect_data=true)

Calculate susceptibility and triangular vertex parallel on workerpool. 

Set `collect_data` to return both quantities, or call [`collect_χ`](@ref collect_χ) and [`collect_γ`](@ref collect_γ) at a later point.
[`calc_χγ`](@ref calc_χγ) can be used for single core computations.

"""
calc_χγ_par(type::Symbol, h::lDΓAHelper; collect_data = true) = calc_χγ_par(type, getfield(h, Symbol("Γ_$(type)")), h.kG, h.mP, h.sP; collect_data = collect_data)

function calc_χγ_par(type::Symbol, Γr::ΓT, kG::KGrid, mP::ModelParameters, sP::SimulationParameters; collect_data = true)
    @sync begin
        for w in workers()
            @async remotecall_fetch(bse_inv, w, type, Γr)
        end
    end

    return collect_data ? (collect_χ(type, kG, mP, sP), collect_γ(type, kG, mP, sP)) : nothing
end


"""
    calc_Σ_eom_par(νmax::Int) 

Equation of motion for self energy. See [`calc_Σ_par`](@ref calc_Σ_par).
"""
function calc_Σ_eom_par(λm::Float64, λd::Float64; tc::Bool = true)
    if !wcache[].EoMVars_initialized
        error("Call initialize_EoM before calc_Σ_par in order to initialize worker caches.")
    end
    U = wcache[].mP.U
    kG = wcache[].kG
    #TODO: this is inefficient and should only be done on one processor
    λm != 0 && χ_λ!(wcache[].χm, λm)
    λd != 0 && χ_λ!(wcache[].χd, λd)
    νdim = length(gridshape(kG)) + 1
    tail_correction = (tc ? -wcache[].mP.U * (sum_kω(kG, wcache[].χm) - wcache[].χloc_m_sum) : 0)
    Nq::Int = size(wcache[].χm, 1)
    fill!(wcache[].Σ_ladder, 0)
    for (νi, νn) in enumerate(wcache[].νn_indices)
        for (ωi, ωn) in enumerate(wcache[].ωn_ranges[νi])
            for qi = 1:Nq
                wcache[].Kνωq_pre[qi] = eom(U, wcache[].γm[qi, ωi, νi], wcache[].γd[qi, ωi, νi], wcache[].χm[qi, ωi], wcache[].χd[qi, ωi], wcache[].λ₀[qi, ωi, νi])
            end
            conv_tmp_add!(view(wcache[].Σ_ladder, :, νi), kG, wcache[].Kνωq_pre, selectdim(wcache[].G_fft_reverse, νdim, νn + ωn))
        end
        tc && (wcache[].Σ_ladder[:, νi] .= wcache[].Σ_ladder[:, νi] ./ wcache[].mP.β .+ tail_correction / (1im * (2 * νn + 1) * π / wcache[].mP.β))

    end
    λm != 0.0 && reset!(wcache[].χm)
    λd != 0.0 && reset!(wcache[].χd)
end

"""
    collect_Σ!(Σ_ladder::OffsetMatrix{ComplexF64, Matrix{ComplexF64}}, mP::ModelParameters; λm=0.0)

Collects self-energy from workers.
"""
function collect_Σ!(Σ_ladder::OffsetMatrix{ComplexF64,Matrix{ComplexF64}})
    !(length(workers()) > 1) && throw(ErrorException("Add workers and run lDGA_setup before calling parallel functions!"))
    Σ_hartree = wcache[].mP.n * wcache[].mP.U / 2.0

    @sync begin
        for w in workers()
            @async begin
                indices = @fetchfrom w getfield(LadderDGA.wcache[], :νn_indices)
                Σ_ladder[:, indices] = @fetchfrom w getfield(LadderDGA.wcache[], :Σ_ladder)
            end
        end
    end
    Σ_ladder.parent[:, :] = Σ_ladder.parent[:, :] .+ Σ_hartree
    return nothing
end

"""
    calc_Σ_par(; λm::Float64=0.0, λd::Float64=0.0, collect_data=true, tc::Bool=true)

Calculates self-energy on worker pool. Workers must first be initialized using [`initialize_EoM`](@ref initialize_EoM).
#TODO: νrange must be equal to the one used during initialization. remove one.
"""
function calc_Σ_par(; λm::Float64 = 0.0, λd::Float64 = 0.0, collect_data = true, tc::Bool = true)
    Nk::Int = collect_data ? length(wcache[].kG.kMult) : 1
    νrange = wcache[].EoM_νGrid
    νrange = collect_data ? νrange : UnitRange(0, 0)
    Σ_ladder = collect_data ? OffsetArray(Matrix{ComplexF64}(undef, Nk, length(νrange)), 1:Nk, νrange) : OffsetArray(Matrix{ComplexF64}(undef, 0, 0), 1:1, 0:0)
    return calc_Σ_par!(Σ_ladder; λm = λm, λd = λd, collect_data = collect_data, tc = tc)
end

function calc_Σ_par!(Σ_ladder::OffsetMatrix{ComplexF64,Matrix{ComplexF64}}; λm::Float64 = 0.0, λd::Float64 = 0.0, collect_data = true, tc::Bool = true)
    !(length(workers()) > 1) && throw(ErrorException("Add workers and run lDGA_setup before calling parallel functions!"))

    @sync begin
        for wi in workers()
            @async remotecall_fetch(calc_Σ_eom_par, wi, λm, λd, tc = tc)
        end
    end
    if collect_data
        collect_Σ!(Σ_ladder)
    end

    return collect_data ? Σ_ladder : nothing
end
