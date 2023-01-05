# ==================================================================================================== #
#                                        ladderDGATools.jl                                             #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
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
    n_iν = 2*(sP.n_iν + sP.n_iν_shell)
    data::Array{ComplexF64,3} = Array{ComplexF64,3}(undef, length(kG.kMult), n_iν, length(ωi_range))
    for (iω,ωnR) in enumerate(ωi_range)
        _,ωn = ωnR
        νGrid = νnGrid(ωn, sP)
        for (iν, νn) in enumerate(νGrid)
            conv_fft_noPlan!(kG, view(data,:,iν,iω), 
                     reshape(wcache[].G_fft[:,νn].parent,gridshape(kG)),
                     reshape(wcache[].G_fft_reverse[:,νn+ωn].parent,gridshape(kG)))
        end
    end
    c1, c2, c3 = χ₀Asym_coeffs(kG, false, mP)
    asym = χ₀Asym(c1, c2, c3, map(x->x[2],ωi_range), sP.n_iν, sP.shift, mP.β)
    update_wcache!(:χ₀, -mP.β .* data)
    update_wcache!(:χ₀Indices, ωi_range)
    update_wcache!(:χ₀Asym, asym)
    return nothing
end

"""
    calc_bubble_par(kG::KGrid, mP::ModelParameters, sP::SimulationParameters; collect_data=true)

Calculates the bubble, based on two fourier-transformed Greens functions where the second one has to be reversed.
"""
function calc_bubble_par(kG::KGrid, mP::ModelParameters, sP::SimulationParameters; collect_data=true)
    workerpool = get_workerpool()
    !(length(workerpool) > 0) && throw(ErrorException("Add workers and rund lDGA_setup before calling parallel functions!"))
    ωi_range, ωi_part = gen_ω_part(sP, workerpool)

    @sync begin
        for ind in ωi_part
            @async remotecall_fetch(χ₀_conv, workerpool, ωi_range[ind])
        end
    end

    return collect_data ? collect_χ₀(kG, mP, sP) : nothing
end

"""
    collect_χ₀(kG::KGrid, mP::ModelParameters, sP::SimulationParameters)

Collect non-local bubble ``\\chi_0^{\\omega}(q)`` from workers. Values first need to be calculated using [`calc_bubble_par`](@ref `calc_bubble_par`).
"""
function collect_χ₀(kG::KGrid, mP::ModelParameters, sP::SimulationParameters)
    workerpool = get_workerpool()
    !(length(workerpool) > 0) && throw(ErrorException("Add workers and rund lDGA_setup before calling parallel functions!"))
    data::Array{ComplexF64,3} = Array{ComplexF64,3}(undef, length(kG.kMult), 2*(sP.n_iν+sP.n_iν_shell), 2*sP.n_iω+1)

    @sync begin
    for w in workers(workerpool)
        @async begin
            indices = @fetchfrom w LadderDGA.wcache[].χ₀Indices
            χdata   = @fetchfrom w LadderDGA.wcache[].χ₀
            for i in 1:length(indices)
                ωi,_ = indices[i]
                data[:,:,ωi] = χdata[:,:,i]
            end
        end
    end
    end
    χ₀T(data, kG, -sP.n_iω:sP.n_iω, sP.n_iν, sP.shift, mP, local_tail=false) 
end

# --------------------------------------------- χ and γ ----------------------------------------------
"""
    bse_inv(type::Symbol, Γr::Array{ComplexF64,3}) 

Kernel for calculation of susceptibility and triangular vertex. Used by [`calc_χγ_par`](@ref calc_χγ_par).
"""
function bse_inv(type::Symbol, Γr::Array{ComplexF64,3}) 
    !(typeof(wcache[].sP.χ_helper) <: BSE_Asym_Helpers) && throw("Current version ONLY supports BSE_Asym_Helper!")
    s = type === :ch ? -1 : 1
    Nν = size(Γr,1)
    Nq = size(wcache[].χ₀, 1)
    Nω = length(wcache[].χ₀Indices) 
    χ_data = Array{eltype(Γr),2}(undef, Nq, Nω)
    γ_data = Array{eltype(Γr),3}(undef, Nq, Nν, Nω)

    χννpω = Matrix{eltype(Γr)}(undef, Nν, Nν)
    ipiv = Vector{Int}(undef, Nν)
    work = _gen_inv_work_arr(χννpω, ipiv)
    λ_cache = Array{eltype(χννpω),1}(undef, Nν)

    for (i,ω_el) in enumerate(wcache[].χ₀Indices)
        ωi,ωn = ω_el
        for qi in 1:Nq
            copy!(χννpω, view(Γr,:,:,ωi))
            for l in 1:size(χννpω,1)
                χννpω[l,l] += 1.0/wcache[].χ₀[qi, wcache[].sP.n_iν_shell+l, i]
            end
            inv!(χννpω, ipiv, work)
            χ_data[qi, i] = calc_χλ_impr!(λ_cache, type, ωn, χννpω, view(wcache[].χ₀,qi,:,i), 
                    wcache[].mP.U, wcache[].mP.β, wcache[].χ₀Asym[qi,i], wcache[].sP.χ_helper);
            γ_data[qi, :, i] = (1 .- s*λ_cache) ./ (1 .+ s* wcache[].mP.U .* χ_data[qi,i])
        end
    end
    update_wcache!(Symbol(string("χ", type)), χ_data)
    update_wcache!(Symbol(string("γ", type)), γ_data)
end

"""
    calc_χγ_par(type::Symbol, Γr::ΓT, kG::KGrid, mP::ModelParameters, sP::SimulationParameters; collect_data=true)

Calculate susceptibility and triangular vertex parallel on workerpool. 

Set `collect_data` to return both quantities, or call [`collect_χγ`](@ref collect_χγ) at a later point.
[`calc_χγ`](@ref calc_χγ) can be used for single core computations.

"""
function calc_χγ_par(type::Symbol, Γr::ΓT, kG::KGrid, mP::ModelParameters, sP::SimulationParameters; collect_data=true)
    workerpool = get_workerpool()

    @sync begin
    for w in workers(workerpool)
        @async remotecall_fetch(bse_inv, w, type, Γr)
    end
    end

    return collect_data ? (collect_χ(type, kG, mP, sP), collect_γ(type, kG, mP, sP)) : nothing
end

# -------------------------------------------- EoM: Main ---------------------------------------------
"""
    calc_Σ_eom_par(νmax::Int) 

Equation of motion for self energy. See [`calc_Σ_par`](@ref calc_Σ_par).
"""
function calc_Σ_eom_par(λsp::Float64, λch::Float64) 
    if !wcache[].Σ_initialized 
        error("Call initialize_EoM before calc_Σ_par in order to initialize worker caches.")
    end
    λsp != 0 && χ_λ!(wcache[].χsp, λsp) 
    λch != 0 && χ_λ!(wcache[].χch, λch) 

    Nq::Int = size(wcache[].χsp,1)
    U = wcache[].mP.U
    fill!(wcache[].Σ_ladder, 0)
    for (νi,νn) in enumerate(wcache[].νn_indices)
        for (ωi,ωn) in enumerate(wcache[].ωn_ranges[νi])
            v = reshape(view(wcache[].G_fft_reverse,:,νn + ωn), gridshape(wcache[].kG)...)
            for qi in 1:Nq
                wcache[].Kνωq_pre[qi] = eom(U, wcache[].γsp[qi,ωi,νi], wcache[].γch[qi,ωi,νi], wcache[].χsp[qi,ωi], 
                        wcache[].χch[qi,ωi], wcache[].λ₀[qi,ωi,νi])
            end
            conv_fft1_noPlan!(wcache[].kG, wcache[].Kνωq_post, wcache[].Kνωq_pre, v)
            wcache[].Σ_ladder[:,νi] += wcache[].Kνωq_post
        end
    end
    λsp != 0 && χ_λ!(wcache[].χsp, -λsp) 
    λch != 0 && χ_λ!(wcache[].χch, -λch) 
end

"""
collect_Σ(Nk::Int, νrange::AbstractVector{Int}, mP::ModelParameters, sP::SimulationParameters)
"""
function collect_Σ(Nk::Int, νrange::AbstractVector{Int}, mP::ModelParameters, sP::SimulationParameters)
    workerpool = get_workerpool()
    !(length(workerpool) > 0) && throw(ErrorException("Add workers and rund lDGA_setup before calling parallel functions!"))
    Σ_hartree = mP.n * mP.U/2.0;
    Σ_ladder = OffsetArray(zeros(ComplexF64, Nk, length(νrange)), 1:Nk, νrange)

    @sync begin
    for w in workers(workerpool)
        @async begin
            indices = @fetchfrom w getfield(LadderDGA.wcache[], :νn_indices)
            Σ_ladder[:,indices] = @fetchfrom w getfield(LadderDGA.wcache[], :Σ_ladder)
        end
    end
    end
    Σ_ladder = Σ_ladder ./ mP.β .+ Σ_hartree
    return Σ_ladder
end

"""
    calc_Σ_par(mP::ModelParameters, sP::SimulationParameters; νmax=sP.n_iν; collect_data=true)

Calculates self-energy on worker pool. Workers must first be initialized using [`initialize_EoM`](@ref initialize_EoM).
#TODO: νrange must be equal to the one used during initialization. remove one.
"""
function calc_Σ_par(kG::KGrid, mP::ModelParameters, sP::SimulationParameters; 
                    λsp::Float64=0.0, λch::Float64=0.0, νrange=0:sP.n_iν-1, collect_data=true)
    workerpool = get_workerpool()
    !(length(workerpool) > 0) && throw(ErrorException("Add workers and rund lDGA_setup before calling parallel functions!"))
    Nk::Int = length(kG.kMult)
    @sync begin
    for wi in workers(workerpool)
        @async remotecall_fetch(calc_Σ_eom_par, wi, λsp, λch)
    end
    end

    return collect_data ? collect_Σ(Nk, νrange, mP, sP) : nothing
end


# ---------------------------------------------- Misc. -----------------------------------------------
function Σ_loc_correction(Σ_ladder::AbstractArray{T1, 2}, Σ_ladderLoc::AbstractArray{T2, 2}, Σ_loc::AbstractArray{T3, 1}) where {T1 <: Number, T2 <: Number, T3 <: Number}
    res = similar(Σ_ladder)
    for qi in axes(Σ_ladder,1)
        for νi in axes(Σ_ladder,2)
            res[qi,νi] = Σ_ladder[qi,νi] .- Σ_ladderLoc[νi] .+ Σ_loc[νi]
        end
    end
    return res
end

# -------------------------------------------- EoM: Defs ---------------------------------------------
@inline eom(U::Float64, γsp::ComplexF64, γch::ComplexF64, χsp::Float64, χch::Float64, λ₀::ComplexF64)::ComplexF64 = U*(γsp * 1.5 * (1 + U * χsp) - γch * 0.5 * (1 - U * χch) - 1.5 + 0.5 + λ₀)
@inline eom(U::Float64, γsp::ComplexF64, γch::ComplexF64, χsp::ComplexF64, χch::ComplexF64, λ₀::ComplexF64)::ComplexF64 = U*(γsp * 1.5 * (1 + U * χsp) - γch * 0.5 * (1 - U * χch) - 1.5 + 0.5 + λ₀)

@inline eom_χsp(U::Float64, γsp::ComplexF64, γch::ComplexF64, χsp::ComplexF64, χch::ComplexF64, λ₀::ComplexF64)::ComplexF64 = U*(γsp * 1.5 * (U * χsp) )
@inline eom_χch(U::Float64, γsp::ComplexF64, γch::ComplexF64, χsp::ComplexF64, χch::ComplexF64, λ₀::ComplexF64)::ComplexF64 = -U*(γch * 0.5 * ( - U * χch))
@inline eom_γsp(U::Float64, γsp::ComplexF64, γch::ComplexF64, χsp::ComplexF64, χch::ComplexF64, λ₀::ComplexF64)::ComplexF64 = U*(γsp * 1.5)
@inline eom_γch(U::Float64, γsp::ComplexF64, γch::ComplexF64, χsp::ComplexF64, χch::ComplexF64, λ₀::ComplexF64)::ComplexF64 = -U*(γch * 0.5)
@inline eom_rest_01(U::Float64, γsp::ComplexF64, γch::ComplexF64, χsp::ComplexF64, χch::ComplexF64, λ₀::ComplexF64)::ComplexF64 = -U*1.0 + 0.0im

@inline eom_sp_01(U::Float64, γsp::ComplexF64, γch::ComplexF64, χsp::ComplexF64, χch::ComplexF64, λ₀::ComplexF64)::ComplexF64 = U*(γsp * 0.5 * (1 + U * χsp) - 0.5)
@inline eom_sp_02(U::Float64, γsp::ComplexF64, γch::ComplexF64, χsp::ComplexF64, χch::ComplexF64, λ₀::ComplexF64)::ComplexF64 = U*(γsp * 1.0 * (1 + U * χsp) - 1.0)
@inline eom_sp(U::Float64, γsp::ComplexF64, γch::ComplexF64, χsp::ComplexF64, χch::ComplexF64, λ₀::ComplexF64)::ComplexF64 = U*(γsp * 1.5 * (1 + U * χsp) - 1.5)
@inline eom_ch(U::Float64, γsp::ComplexF64, γch::ComplexF64, χsp::ComplexF64, χch::ComplexF64, λ₀::ComplexF64)::ComplexF64 = -U*(γch * 0.5 * (1 - U * χch) - 0.5)
@inline eom_rest(U::Float64, γsp::ComplexF64, γch::ComplexF64, χsp::ComplexF64, χch::ComplexF64, λ₀::ComplexF64)::ComplexF64 = U*λ₀
