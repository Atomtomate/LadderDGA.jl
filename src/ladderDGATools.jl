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
    νdim = length(gridshape(wcache[].kG))+1 
    data::Array{ComplexF64,3} = Array{ComplexF64,3}(undef, length(kG.kMult), n_iν, length(ωi_range))
    for (iω,ωnR) in enumerate(ωi_range)
        _,ωn = ωnR
        νGrid = νnGrid(ωn, sP)
        for (iν, νn) in enumerate(νGrid)
            conv_fft_noPlan!(kG, view(data,:,iν,iω), 
                selectdim(wcache[].G_fft,νdim,νn),
                selectdim(wcache[].G_fft_reverse,νdim,νn+ωn))
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
    worker_list = workers()
    !(length(worker_list) > 1) && throw(ErrorException("Add workers and run lDGA_setup before calling parallel functions!"))
    ωi_range, ωi_part = gen_ω_part(sP, length(worker_list))

    @sync begin
        for (i,ind) in enumerate(ωi_part)
            @async remotecall_fetch(χ₀_conv, worker_list[i], ωi_range[ind])
        end
    end

    return collect_data ? collect_χ₀(kG, mP, sP) : nothing
end

"""
    collect_χ₀(kG::KGrid, mP::ModelParameters, sP::SimulationParameters)

Collect non-local bubble ``\\chi_0^{\\omega}(q)`` from workers. Values first need to be calculated using [`calc_bubble_par`](@ref `calc_bubble_par`).
"""
function collect_χ₀(kG::KGrid, mP::ModelParameters, sP::SimulationParameters)
    !(length(workers()) > 1) && throw(ErrorException("Add workers and run lDGA_setup before calling parallel functions!"))
    data::Array{ComplexF64,3} = Array{ComplexF64,3}(undef, length(kG.kMult), 2*(sP.n_iν+sP.n_iν_shell), 2*sP.n_iω+1)

    @sync begin
    for w in workers()
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
    s = type === :d ? -1 : 1
    Nν = size(Γr,1)
    Nq = size(wcache[].χ₀, 1)
    Nω = length(wcache[].χ₀Indices) 
    χ_data = Array{Float64,2}(undef, Nq, Nω)
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
            χ_data[qi, i] = real(calc_χλ_impr!(λ_cache, type, ωn, χννpω, view(wcache[].χ₀,qi,:,i), 
                                 wcache[].mP.U, wcache[].mP.β, wcache[].χ₀Asym[qi,i], wcache[].sP.χ_helper));
            γ_data[qi, :, i] = (1 .- s*λ_cache) ./ (1 .+ s* wcache[].mP.U .* χ_data[qi,i])
        end
    end
    # TODO: unify naming ch/_d
    update_wcache!(Symbol(string("χ", type,"_part")), χ_data)
    update_wcache!(Symbol(string("γ", type)), γ_data)
end

"""
    calc_χγ_par(type::Symbol, Γr::ΓT, kG::KGrid, mP::ModelParameters, sP::SimulationParameters; collect_data=true)

Calculate susceptibility and triangular vertex parallel on workerpool. 

Set `collect_data` to return both quantities, or call [`collect_χγ`](@ref collect_χγ) at a later point.
[`calc_χγ`](@ref calc_χγ) can be used for single core computations.

"""
function calc_χγ_par(type::Symbol, Γr::ΓT, kG::KGrid, mP::ModelParameters, sP::SimulationParameters; collect_data=true)
    @sync begin
    for w in workers()
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
function calc_Σ_eom_par(λm::Float64, λd::Float64) 
    if !wcache[].EoMVars_initialized 
        error("Call initialize_EoM before calc_Σ_par in order to initialize worker caches.")
    end

    U = wcache[].mP.U
    kG = wcache[].kG

    #TODO: this is inefficient and should only be done on one processor
    λm != 0 && χ_λ!(wcache[].χm, λm) 
    λd != 0 && χ_λ!(wcache[].χd, λd) 
    νdim = length(gridshape(kG))+1 
    err = wcache[].mP.U^2 * wcache[].mP.β * sum_kω(kG, wcache[].χm) - wcache[].χloc_m_sum

    Nq::Int = size(wcache[].χm,1)
    fill!(wcache[].Σ_ladder, 0)
    for (νi,νn) in enumerate(wcache[].νn_indices)
        for (ωi,ωn) in enumerate(wcache[].ωn_ranges[νi])
            for qi in 1:Nq
                wcache[].Kνωq_pre[qi] = eom(U, wcache[].γm[qi,ωi,νi], wcache[].γd[qi,ωi,νi], wcache[].χm[qi,ωi], 
                        wcache[].χd[qi,ωi], wcache[].λ₀[qi,ωi,νi])
            end

        conv_tmp!(view(wcache[].Σ_ladder,:,νi), kG, wcache[].Kνωq_pre, wcache[].Kνωq_post, selectdim(wcache[].G_fft_reverse,νdim, νn + ωn))
            #TODO: find a way to not unroll this!
        end
    end
    reset!(wcache[].χm) 
    reset!(wcache[].χd) 
end
function conv_tmp!(res::AbstractVector{ComplexF64}, kG::KGrid, arr1::Vector{ComplexF64}, arr2::Vector{ComplexF64}, GView::AbstractArray{ComplexF64,N})::Nothing where N
            expandKArr!(kG, kG.cache1, arr1)
            mul!(kG.cache1, kG.fftw_plan, kG.cache1)
            for i in eachindex(kG.cache1)
                kG.cache1[i] *= GView[i]
            end
            kG.fftw_plan \ kG.cache1
            Dispersions.conv_post!(kG, arr2, kG.cache1)
            res[:] += arr2[:] #.- err / νn
    return nothing
end

"""
    collect_Σ!(Σ_ladder::OffsetMatrix{ComplexF64, Matrix{ComplexF64}}, mP::ModelParameters; λm=0.0)

Collects self-energy from workers.
"""
function collect_Σ!(Σ_ladder::OffsetMatrix{ComplexF64, Matrix{ComplexF64}}, mP::ModelParameters)
    !(length(workers()) > 1) && throw(ErrorException("Add workers and run lDGA_setup before calling parallel functions!"))
    Σ_hartree = mP.n * mP.U/2.0;

    @sync begin
        for w in workers()
            @async begin
                indices = @fetchfrom w getfield(LadderDGA.wcache[], :νn_indices)
                Σ_ladder[:,indices] = @fetchfrom w getfield(LadderDGA.wcache[], :Σ_ladder)
            end
        end
    end
    Σ_ladder[:,:] = Σ_ladder[:,:] ./ mP.β .+ Σ_hartree
    return nothing
end

"""
    calc_Σ_par(mP::ModelParameters, sP::SimulationParameters; νmax=sP.n_iν; collect_data=true)

Calculates self-energy on worker pool. Workers must first be initialized using [`initialize_EoM`](@ref initialize_EoM).
#TODO: νrange must be equal to the one used during initialization. remove one.
"""
function calc_Σ_par(kG::KGrid, mP::ModelParameters; 
                    λm::Float64=0.0, λd::Float64=0.0, collect_data=true)
    !(length(workers()) > 1) && throw(ErrorException("Add workers and run lDGA_setup before calling parallel functions!"))
    Nk::Int = collect_data ? length(kG.kMult) : 1
    νrange = wcache[].EoM_νGrid
    νrange = collect_data ?  νrange : UnitRange(0,0)

    Σ_ladder = collect_data ? OffsetArray(Matrix{ComplexF64}(undef, Nk, length(νrange)), 1:Nk, νrange) : OffsetArray(Matrix{ComplexF64}(undef, 0, 0), 1:1, 0:0)
    return calc_Σ_par!(Σ_ladder, mP; λm=λm, λd=λd, collect_data=collect_data)
end

function calc_Σ_par!(Σ_ladder::OffsetMatrix{ComplexF64, Matrix{ComplexF64}}, 
                    mP::ModelParameters; 
                    λm::Float64=0.0, λd::Float64=0.0, collect_data=true)
    !(length(workers()) > 1) && throw(ErrorException("Add workers and run lDGA_setup before calling parallel functions!"))
    @sync begin
    for wi in workers()
        @async remotecall_fetch(calc_Σ_eom_par, wi, λm, λd)
    end
    end
    collect_data && collect_Σ!(Σ_ladder, mP) 

    return collect_data ? Σ_ladder : nothing
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
@inline eom(U::Float64, γ_m::ComplexF64, γ_d::ComplexF64, χ_m::Float64, χ_d::Float64, λ₀::ComplexF64)::ComplexF64 = U*(γ_m * 1.5 * (1 + U * χ_m) - γ_d * 0.5 * (1 - U * χ_d) - 1.5 + 0.5 + λ₀)
@inline eom_χ_m(U::Float64, γ_m::ComplexF64, γ_d::ComplexF64, χ_m::Float64, χ_d::Float64, λ₀::ComplexF64)::ComplexF64 = U*(γ_m * 1.5 * (U * χ_m) )
@inline eom_χ_d(U::Float64, γ_m::ComplexF64, γ_d::ComplexF64, χ_m::Float64, χ_d::Float64, λ₀::ComplexF64)::ComplexF64 = -U*(γ_d * 0.5 * ( - U * χ_d))
@inline eom_γ_m(U::Float64, γ_m::ComplexF64, γ_d::ComplexF64, χ_m::Float64, χ_d::Float64, λ₀::ComplexF64)::ComplexF64 = U*(γ_m * 1.5)
@inline eom_γ_d(U::Float64, γ_m::ComplexF64, γ_d::ComplexF64, χ_m::Float64, χ_d::Float64, λ₀::ComplexF64)::ComplexF64 = -U*(γ_d * 0.5)
@inline eom_rest_01(U::Float64, γ_m::ComplexF64, γ_d::ComplexF64, χ_m::Float64, χ_d::Float64, λ₀::ComplexF64)::ComplexF64 = -U*1.0 + 0.0im

@inline eom_sp_01(U::Float64, γ_m::ComplexF64, γ_d::ComplexF64, χ_m::Float64, χ_d::Float64, λ₀::ComplexF64)::ComplexF64 = U*(γ_m * 0.5 * (1 + U * χ_m) - 0.5)
@inline eom_sp_02(U::Float64, γ_m::ComplexF64, γ_d::ComplexF64, χ_m::Float64, χ_d::Float64, λ₀::ComplexF64)::ComplexF64 = U*(γ_m * 1.0 * (1 + U * χ_m) - 1.0)
@inline eom_sp(U::Float64, γ_m::ComplexF64, γ_d::ComplexF64, χ_m::Float64, χ_d::Float64, λ₀::ComplexF64)::ComplexF64 = U*(γ_m * 1.5 * (1 + U * χ_m) - 1.5)
@inline eom_ch(U::Float64, γ_m::ComplexF64, γ_d::ComplexF64, χ_m::Float64, χ_d::Float64, λ₀::ComplexF64)::ComplexF64 = -U*(γ_d * 0.5 * (1 - U * χ_d) - 0.5)
@inline eom_rest(U::Float64, γ_m::ComplexF64, γ_d::ComplexF64, χ_m::Float64, χ_d::Float64, λ₀::ComplexF64)::ComplexF64 = U*λ₀

@inline eom(U::Float64, γ_m::ComplexF64, γ_d::ComplexF64, χ_m::ComplexF64, χ_d::ComplexF64, λ₀::ComplexF64)::ComplexF64 = U*(γ_m * 1.5 * (1 + U * χ_m) - γ_d * 0.5 * (1 - U * χ_d) - 1.5 + 0.5 + λ₀)
@inline eom_χ_m(U::Float64, γ_m::ComplexF64, γ_d::ComplexF64, χ_m::ComplexF64, χ_d::ComplexF64, λ₀::ComplexF64)::ComplexF64 = U*(γ_m * 1.5 * (U * χ_m) )
@inline eom_χ_d(U::Float64, γ_m::ComplexF64, γ_d::ComplexF64, χ_m::ComplexF64, χ_d::ComplexF64, λ₀::ComplexF64)::ComplexF64 = -U*(γ_d * 0.5 * ( - U * χ_d))
@inline eom_γ_m(U::Float64, γ_m::ComplexF64, γ_d::ComplexF64, χ_m::ComplexF64, χ_d::ComplexF64, λ₀::ComplexF64)::ComplexF64 = U*(γ_m * 1.5)
@inline eom_γ_d(U::Float64, γ_m::ComplexF64, γ_d::ComplexF64, χ_m::ComplexF64, χ_d::ComplexF64, λ₀::ComplexF64)::ComplexF64 = -U*(γ_d * 0.5)
@inline eom_rest_01(U::Float64, γ_m::ComplexF64, γ_d::ComplexF64, χ_m::ComplexF64, χ_d::ComplexF64, λ₀::ComplexF64)::ComplexF64 = -U*1.0 + 0.0im

@inline eom_sp_01(U::Float64, γ_m::ComplexF64, γ_d::ComplexF64, χ_m::ComplexF64, χ_d::ComplexF64, λ₀::ComplexF64)::ComplexF64 = U*(γ_m * 0.5 * (1 + U * χ_m) - 0.5)
@inline eom_sp_02(U::Float64, γ_m::ComplexF64, γ_d::ComplexF64, χ_m::ComplexF64, χ_d::ComplexF64, λ₀::ComplexF64)::ComplexF64 = U*(γ_m * 1.0 * (1 + U * χ_m) - 1.0)
@inline eom_sp(U::Float64, γ_m::ComplexF64, γ_d::ComplexF64, χ_m::ComplexF64, χ_d::ComplexF64, λ₀::ComplexF64)::ComplexF64 = U*(γ_m * 1.5 * (1 + U * χ_m) - 1.5)
@inline eom_ch(U::Float64, γ_m::ComplexF64, γ_d::ComplexF64, χ_m::ComplexF64, χ_d::ComplexF64, λ₀::ComplexF64)::ComplexF64 = -U*(γ_d * 0.5 * (1 - U * χ_d) - 0.5)
@inline eom_rest(U::Float64, γ_m::ComplexF64, γ_d::ComplexF64, χ_m::ComplexF64, χ_d::ComplexF64, λ₀::ComplexF64)::ComplexF64 = U*λ₀
