# ==================================================================================================== #
#                                        ladderDGATools.jl                                             #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#   ladder DΓA related functions                                                                       #
# -------------------------------------------- TODO -------------------------------------------------- #
#   Cleanup, combine *singlecore.jl                                                                    #
#   Reasonable parallelization for EoM                                                                 #
# ==================================================================================================== #


# ========================================== Transformations =========================================

"""
    λ_from_γ(type::Symbol, γ::γT, χ::χT, U::Float64)

TODO: documentation
"""
function λ_from_γ(type::Symbol, γ::γT, χ::χT, U::Float64)
    s = (type == :ch) ? -1 : 1
    res = similar(γ.data)
    for ωi in 1:size(γ,3)
        for qi in 1:size(γ,1)
            res[qi,:,ωi] = s .* view(γ,qi,:,ωi) .* (1 .+ s*U .* χ.data[qi, ωi]) .- 1
        end
    end
    return res
end


"""
    F_from_χ(χ::AbstractArray{ComplexF64,3}, G::AbstractArray{ComplexF64,1}, sP::SimulationParameters, β::Float64[; diag_term=true])

TODO: documentation
"""
function F_from_χ(χ::AbstractArray{ComplexF64,3}, G::AbstractArray{ComplexF64,1}, sP::SimulationParameters, β::Float64; diag_term=true)
    F = similar(χ)
    for ωi in 1:size(F,3)
    for νpi in 1:size(F,2)
        ωn, νpn = OneToIndex_to_Freq(ωi, νpi, sP) #, sP.n_iν_shell)
        for νi in 1:size(F,1)
            _, νn = OneToIndex_to_Freq(ωi, νi, sP) #, sP.n_iν_shell)
            F[νi,νpi,ωi] = -(χ[νi,νpi,ωi] + (νn == νpn && diag_term) * β * G[νn] * G[ωn+νn])/(
                                          G[νn] * G[ωn+νn] * G[νpn] * G[ωn+νpn])
        end
        end
    end
    return F
end


# ========================================== Correction Term =========================================
"""
    calc_λ0(χ₀::χ₀T, Fr::FT, χ::χT, γ::γT, mP::ModelParameters, sP::SimulationParameters)

Correction term, TODO: documentation
"""
function calc_λ0(χ₀::χ₀T, Fr::FT, χ::χT, γ::γT, mP::ModelParameters, sP::SimulationParameters)
    #TODO: store nu grid in sP?
    Niν = size(Fr,ν_axis)
    Nq  = size(χ₀.data, χ₀.axes[:q])
    ω_range = 1:size(χ₀.data,ω_axis)
    λ0 = Array{ComplexF64,3}(undef,size(χ₀.data,q_axis),Niν,length(ω_range))

    @warn "Forcing naiive computation of λ₀"
    if typeof(sP.χ_helper) <: BSE_Asym_Helpers
       λ0[:] = calc_λ0_impr(:sp, -sP.n_iω:sP.n_iω, Fr, χ₀.data, χ₀.asym, view(γ.data,1,:,:), view(χ.data,1,:),
                            mP.U, mP.β, sP.χ_helper)
    else
        #TODO: this is not well optimized, but also not often executed
        @warn "Using plain summation for λ₀, check Σ_ladder tails!"
        fill!(λ0, 0.0)
        for ωi in ω_range
            for νi in 1:Niν
                #TODO: export realview functions?
                v1 = view(Fr,νi,:,ωi)
                for qi in 1:Nq
                    v2 = view(χ₀.data,qi,(sP.n_iν_shell+1):(size(χ₀.data,2)-sP.n_iν_shell),ωi)
                    λ0[qi,:,ωi] = λ0[qi,:,ωi] .+ v1 .* v2 ./ mP.β^2
                end
            end
        end
    end
    return λ0
end

# ======================================== LadderDGA Functions =======================================
# ------------------------------------------- Bubble Term --------------------------------------------
function χ₀_conv(ωi_range::Vector{NTuple{2,Int}})
    kG = wcache[].kG
    mP = wcache[].mP
    sP = wcache[].sP
    n_iν = 2*(sP.n_iν + sP.n_iν_shell)
    data::Array{ComplexF64,3} = Array{ComplexF64,3}(undef, length(kG.kMult), n_iν, length(ωi_range))
    for (iω,ωnR) in enumerate(ωi_range)
        ωn,_ = ωnR
        νGrid = νnGrid(ωn, sP)
        for (iν, νn) in enumerate(νGrid)
            conv_fft_noPlan!(kG, view(data,:,iν,iω), 
                     reshape(wcache[].GLoc_fft[:,νn].parent,gridshape(kG)),
                     reshape(wcache[].GLoc_fft_reverse[:,νn+ωn].parent,gridshape(kG)))
        end
    end
    c1, c2, c3 = χ₀Asym_coeffs(kG, false, mP)
    asym = χ₀Asym(c1, c2, c3, map(x->x[1],ωi_range), sP.n_iν, sP.shift, mP.β)
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
            @async r = remotecall_fetch(χ₀_conv, workerpool, ωi_range[ind])
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
                _,ωi = indices[i]
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
        ωn,ωi = ω_el
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
    update_wcache!(Symbol(string("χ_", type)), χ_data)
    update_wcache!(Symbol(string("γ_", type)), γ_data)
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
        @async r = remotecall_fetch(bse_inv, w, type, Γr)
    end
    end

    return collect_data ? collect_χγ(type, kG, mP, sP) : nothing
end

"""
    collect_χγ(type::Symbol, kG::KGrid, mP::ModelParameters, sP::SimulationParameters)

Collects susceptibility and triangular vertex from workers, after parallel computation (see [`calc_χγ_par`](@ref calc_χγ_par)).
"""
function collect_χγ(type::Symbol, kG::KGrid, mP::ModelParameters, sP::SimulationParameters)
    workerpool = get_workerpool()
    !(length(workerpool) > 0) && throw(ErrorException("Add workers and rund lDGA_setup before calling parallel functions!"))
    χ_data::Array{ComplexF64,2} = Array{ComplexF64,2}(undef, length(kG.kMult), 2*sP.n_iω+1)
    γ_data::Array{ComplexF64,3} = Array{ComplexF64,3}(undef, length(kG.kMult), 2*sP.n_iν, 2*sP.n_iω+1)
    χfield = Symbol(string("χ_",type))
    γfield = Symbol(string("γ_",type))

    @sync begin
    for w in workers(workerpool)
        @async begin
            indices = @fetchfrom w LadderDGA.wcache[].χ₀Indices
            χdata   = @fetchfrom w getfield(LadderDGA.wcache[], χfield)
            γdata   = @fetchfrom w getfield(LadderDGA.wcache[], γfield)
            for i in 1:length(indices)
                _,ωi = indices[i]
                χ_data[:,ωi] = χdata[:,i]
                γ_data[:,:,ωi] = γdata[:,:,i]
            end
        end
    end
    end
    log_q0_χ_check(kG, sP, χ_data, type)
    χT(χ_data, tail_c=[0,0,mP.Ekin_DMFT]), γT(γ_data)
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

# -------------------------------------------- EoM: Main ---------------------------------------------
function calc_Σ_eom_par() 
    Nq::Int = size(wcache[].χsp,1)
    Kνωq_pre::Vector{ComplexF64} = Vector{ComplexF64}(undef, Nq)
    Σ_tmp::Vector{ComplexF64} = Vector{ComplexF64}(undef, Nq)
    Σ_res::Array{ComplexF64,2} = zeros(ComplexF64, Nq, νmax)
    for (ωi, ωn, νi, νii) in wcache.νω_map
        ωii = wcache.ωind_map[ωi]
        v = reshape(view(wcache.G,:,(νii-1) + ωn), gridshape(wcache.kG)...)
        for qi in 1:Nq
            Kνωq_pre[qi] = eom(U, wcache.γsp[qi,νi,ωii], wcache.γch[qi,νi,ωii], wcache.χsp[qi,ωii], 
                            wcache.χch[qi,ωii], wcache.λ₀[qi,νi,ωii])
        end
        conv_fft1_noPlan!(wcache.kG, Σ_tmp, Kνωq_pre, v)
        Σ_res[:,νii] += Σ_tmp
    end
    return Σ_res
end

function calc_Σ_eom(νωindices::Vector{NTuple{4,Int}}, ωind_map::Dict{Int,Int}, νmax::Int, 
    χsp::Array{ComplexF64,2}, χch::Array{ComplexF64,2},γsp::Array{ComplexF64,3}, γch::Array{ComplexF64,3},
                    Gνω::GνqT, λ₀::Array{ComplexF64,3}, U::Float64, kG::KGrid) 
            
    Kνωq_pre::Vector{ComplexF64} = Vector{ComplexF64}(undef, size(χsp,1))
    Σ_tmp::Vector{ComplexF64} = Vector{ComplexF64}(undef, size(χsp,1))
    Σ_res::Array{ComplexF64,2} = zeros(ComplexF64, size(γsp,1), νmax)
    for (ωi, ωn, νi, νii) in νωindices
        ωii = ωind_map[ωi]
        v = reshape(view(Gνω,:,(νii-1) + ωn), gridshape(kG)...)
        for qi in 1:size(χsp,1)
            Kνωq_pre[qi] = eom(U, γsp[qi,νi,ωii], γch[qi,νi,ωii], χsp[qi,ωii], 
                            χch[qi,ωii], λ₀[qi,νi,ωii])
        end
        conv_fft1_noPlan!(kG, Σ_tmp, Kνωq_pre, v)
        Σ_res[:,νii] += Σ_tmp
    end
    return Σ_res
end

function calc_Σ_par(χ_sp::χT, γ_sp::γT, χ_ch::χT, γ_ch::γT,
                λ₀::AbstractArray{_eltype,3},
                Gνω::GνqT, kG::KGrid,
                mP::ModelParameters, sP::SimulationParameters; νmax=sP.n_iν, pre_expand=true)
    workerpool = get_workerpool()
    # initialize
    Σ_hartree = mP.n * mP.U/2.0;
    Nk::Int = length(kG.kMult)
    Nω::Int = size(χ_sp.data,χ_sp.axes[:ω])
    ωindices::UnitRange{Int} = (sP.dbg_full_eom_omega) ? (1:Nω) : intersect(χ_sp.usable_ω, χ_ch.usable_ω)
    νrange = 0:(νmax-1)# 0:sP.n_iν-1
    Σ_ladder::Matrix{ComplexF64} = zeros(ComplexF64, Nk, length(νrange))
    νω_range::Array{NTuple{4,Int}} = Array{NTuple{4,Int}}[]

    # generate distribution
    for (ωi,ωn) in enumerate(-sP.n_iω:sP.n_iω)
        νZero = ν0Index_of_ωIndex(ωi, sP)
        maxn = min(size(γ_ch,ν_axis), νZero + νmax - 1)
        for (νii,νi) in enumerate(νZero:maxn)
            push!(νω_range, (ωi, ωn, νi, νii))
        end
    end
    νωi_part = par_partition(νω_range, length(workerpool))
    remote_results = Vector{Future}(undef, length(νωi_part))

    for (i,ind) in enumerate(νωi_part)
        ωi = sort(unique(map(x->x[1],νω_range[ind])))
        ωind_map::Dict{Int,Int} = Dict(zip(ωi, 1:length(ωi)))
        remote_results[i] = remotecall(calc_Σ_eom, νω_range[ind], ωind_map, νmax, χ_sp[:,ωi],
                                       χ_ch[:,ωi], γ_sp[:,:,ωi], γ_ch[:,:,ωi], Gνω, λ₀[:,:,ωi], mP.U, kG)
    end

    for (i,ind) in enumerate(νωi_part)
        data_i = fetch(remote_results[i])
        Σ_ladder[:,:] += data_i
    end
    Σ_ladder = Σ_ladder ./ mP.β .+ Σ_hartree
    return  OffsetArray(Σ_ladder, 1:Nk, νrange)
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
