# ==================================================================================================== #
#                                        ladderDGATools.jl                                             #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
#   Last Edit Date  : 01.09.22                                                                         #
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


# ======================================== LadderDGA Functions =======================================
# ------------------------------------------- Bubble Term --------------------------------------------

function χ₀_conv(kG::KGrid, Gνω::GνqT, Gνω_r::GνqT, νωi_range::Vector{NTuple{4,Int}})::Array{ComplexF64,2}
    data::Array{ComplexF64,2} = Array{ComplexF64,2}(undef, length(kG.kMult), length(νωi_range))
    for (i,ωνn) in enumerate(νωi_range)
        ωn,νn,_,_ = ωνn
        conv_fft_noPlan!(kG, view(data,:,i), reshape(Gνω[:,νn].parent,gridshape(kG)), reshape(Gνω_r[:,νn+ωn].parent,gridshape(kG)))
    end
    return data
end

"""
    calc_bubble(Gνω::GνqT, Gνω_r::GνqT, kG::KGrid, mP::ModelParameters, sP::SimulationParameters; local_tail=false)

Calculates the bubble, based on two fourier-transformed Greens functions where the second one has to be reversed.
"""
function calc_bubble_par(Gνω::GνqT, Gνω_r::GνqT, kG::KGrid, mP::ModelParameters, sP::SimulationParameters; local_tail=false, workerpool::AbstractWorkerPool=default_worker_pool())
    data::Array{ComplexF64,3} = Array{ComplexF64,3}(undef, length(kG.kMult), 2*(sP.n_iν+sP.n_iν_shell), 2*sP.n_iω+1)

    νωi_range, νωi_part = gen_νω_part(sP, workerpool)
    remote_results = Vector{Future}(undef, length(νωi_part))
    
    for (i,ind) in enumerate(νωi_part)
        remote_results[i] = remotecall(χ₀_conv, workerpool, kG, Gνω, Gνω_r, νωi_range[ind])
    end
    for (i,ind) in enumerate(νωi_part)
        data_i = fetch(remote_results[i])
        for (j,ωνind) in enumerate(νωi_range[ind])
            _,_,ωi,νi = ωνind
            data[:,νi,ωi] = data_i[:,j] .* -mP.β
        end
    end

    #TODO: move tail calculation to definition of GF (GF should know about its tail)
    t1, t2 = if local_tail
        convert.(ComplexF64, [mP.U*mP.n/2 - mP.μ]),
        mP.sVk + (mP.U^2)*(mP.n/2)*(1-mP.n/2)
    else
        convert.(ComplexF64, kG.ϵkGrid .+ mP.U*mP.n/2 .- mP.μ),
        (mP.U^2)*(mP.n/2)*(1-mP.n/2)
    end
    return χ₀T(data, kG, t1, t2, mP.β, -sP.n_iω:sP.n_iω, sP.n_iν, Int(sP.shift)) 
end

# --------------------------------------------- χ and γ ----------------------------------------------
function bse_inv(type::Symbol, qωi_range::Vector{Tuple{Int,Int}}, ωind_map::Dict{Int,Int}, ω_offset::Int,
        Γr::Array{ComplexF64,3}, χ₀Data::Array{ComplexF64,3}, χ₀Asym::Array{ComplexF64,2}, 
        n_iν_shell::Int, χ_helper, U::Float64, β::Float64)
    s = type === :ch ? -1 : 1
    Nν = size(Γr,1)
    data = Array{eltype(Γr),2}(undef, Nν+1, length(qωi_range))

    χννpω = Matrix{eltype(Γr)}(undef, Nν, Nν)
    ipiv = Vector{Int}(undef, Nν)
    work = _gen_inv_work_arr(χννpω, ipiv)
    λ_cache = Array{eltype(χννpω),1}(undef, Nν)

    for i in 1:length(qωi_range)
        qi,ωi = qωi_range[i]
        ωii = ωind_map[ωi]
        ωn = ωii + ω_offset
        copy!(χννpω, view(Γr,:,:,ωii))
        for l in 1:size(χννpω,1)
            χννpω[l,l] += 1.0/χ₀Data[qi, n_iν_shell+l, ωii]
        end
        inv!(χννpω, ipiv, work)
        data[1,i] = calc_χλ_impr!(λ_cache, type, ωn, χννpω, view(χ₀Data,qi,:,ωii), 
                                   U, β, χ₀Asym[qi,ωii], χ_helper);
        data[2:end, i] = (1 .- s*λ_cache) ./ (1 .+ s* U .* data[1,i])
    end
    return data
end

"""
    calc_χ_trilex(Γr::ΓT, χ₀, kG::KGrid, U::Float64, mP, sP)

Solve χ = χ₀ - 1/β² χ₀ Γ χ
⇔ (1 + 1/β² χ₀ Γ) χ = χ₀
⇔      (χ⁻¹ - χ₀⁻¹) = 1/β² Γ

with indices: χ[ω, q] = χ₀[]
"""
function calc_χγ_par(type::Symbol, Γr::ΓT, χ₀::χ₀T, kG::KGrid, mP::ModelParameters, sP::SimulationParameters; workerpool::AbstractWorkerPool=default_worker_pool())
    !(typeof(sP.χ_helper) <: BSE_Asym_Helpers) && throw("Current version ONLY supports BSE_Asym_Helper!")
    #TODO: reactivate integration to find usable frequencies: χ_ω = Array{_eltype, 1}(undef, Nω)
    Nν::Int = 2*sP.n_iν
    Nq::Int = size(χ₀.data,χ₀.axes[:q])
    Nω::Int = size(χ₀.data,χ₀.axes[:ω])

    γ = Array{ComplexF64,3}(undef, Nq, Nν, Nω)
    χ = Array{ComplexF64,2}(undef, Nq, Nω)
    qωi_range = collect(Base.product(1:Nq, 1:Nω))[:]
    qωi_part = par_partition(qωi_range, length(workerpool))
    remote_results = Vector{Future}(undef, length(qωi_part))

    for (i,ind) in enumerate(qωi_part)
        ωi = sort(unique(map(x->x[2],qωi_range[ind])))
        ωind_map::Dict{Int,Int} = Dict(zip(ωi, 1:length(ωi)))
        remote_results[i] = remotecall(bse_inv, workerpool, type, qωi_range[ind], ωind_map, -sP.n_iω-1,
                                       Γr[:,:,ωi], χ₀.data[:,:,ωi], χ₀.asym[:,ωi], sP.n_iν_shell, 
                                       sP.χ_helper, mP.U, mP.β)
    end
    for (i,ind) in enumerate(qωi_part)
        data_i = fetch(remote_results[i])
        for (j,qωind) in enumerate(qωi_range[ind])
            χ[qωind...] = data_i[1,j]
            γ[qωind[1],:,qωind[2]] .= data_i[2:end,j]
        end
    end

    log_q0_χ_check(kG, sP, χ, type)

    return χT(χ), γT(γ)
end

function calc_λ0(χ₀::χ₀T, Fr::FT, χ::χT, γ::γT, mP::ModelParameters, sP::SimulationParameters)
    #TODO: store nu grid in sP?
    Niν = size(Fr,ν_axis)
    ω_range = 1:size(χ₀.data,ω_axis)
    λ0 = Array{ComplexF64,3}(undef,size(χ₀.data,q_axis),Niν,length(ω_range))

    if typeof(sP.χ_helper) <: BSE_Asym_Helpers
        λ0[:] = calc_λ0_impr(:sp, -sP.n_iω:sP.n_iω, Fr, χ₀.data, χ₀.asym, view(γ.data,1,:,:), view(χ.data,1,:),
                             mP.U, mP.β, sP.χ_helper)
    else
        #TODO: this is not well optimized, but also not often executed
        fill!(λ0, 0.0)
        for ωi in ω_range
            for νi in 1:Niν
                #TODO: export realview functions?
                v1 = view(Fr,νi,:,ωi)
                for qi in 1:Nq
                    v2 = χ₀.data[qi,(sP.n_iν_shell+1):(size(χ₀.data,2)-sP.n_iν_shell),ωi]
                    for νpi in 1:Niν 
                        λ0[qi,νi,ωi] += v1[νpi] * v2[νpi] / mP.β^2
                    end
                end
            end
        end
    end
    return λ0
end

# ----------------------------------------------- EoM ------------------------------------------------
@inline eom(U::Float64, γsp::ComplexF64, γch::ComplexF64, χsp::Float64, χch::Float64, λ₀::ComplexF64)::ComplexF64 = U*(γsp * 1.5 * (1 + U * χsp) - γch * 0.5 * (1 - U * χch) - 1.5 + 0.5 + λ₀)
@inline eom(U::Float64, γsp::ComplexF64, γch::ComplexF64, χsp::ComplexF64, χch::ComplexF64, λ₀::ComplexF64)::ComplexF64 = U*(γsp * 1.5 * (1 + U * χsp) - γch * 0.5 * (1 - U * χch) - 1.5 + 0.5 + λ₀)

@inline eom_χsp(U::Float64, γsp::ComplexF64, γch::ComplexF64, χsp::ComplexF64, χch::ComplexF64, λ₀::ComplexF64)::ComplexF64 = U*(γsp * 1.5 * (U * χsp) )
@inline eom_χch(U::Float64, γsp::ComplexF64, γch::ComplexF64, χsp::ComplexF64, χch::ComplexF64, λ₀::ComplexF64)::ComplexF64 = -U*(γch * 0.5 * ( - U * χch))
@inline eom_γsp(U::Float64, γsp::ComplexF64, γch::ComplexF64, χsp::ComplexF64, χch::ComplexF64, λ₀::ComplexF64)::ComplexF64 = U*(γsp * 1.5)
@inline eom_γch(U::Float64, γsp::ComplexF64, γch::ComplexF64, χsp::ComplexF64, χch::ComplexF64, λ₀::ComplexF64)::ComplexF64 = -U*(γch * 0.5)
@inline eom_rest_01(U::Float64, γsp::ComplexF64, γch::ComplexF64, χsp::ComplexF64, χch::ComplexF64, λ₀::ComplexF64)::ComplexF64 = U*1.0 + 0.0im

@inline eom_sp_01(U::Float64, γsp::ComplexF64, γch::ComplexF64, χsp::ComplexF64, χch::ComplexF64, λ₀::ComplexF64)::ComplexF64 = U*(γsp * 0.5 * (1 + U * χsp) - 0.5)
@inline eom_sp_02(U::Float64, γsp::ComplexF64, γch::ComplexF64, χsp::ComplexF64, χch::ComplexF64, λ₀::ComplexF64)::ComplexF64 = U*(γsp * 1.0 * (1 + U * χsp) - 1.0)
@inline eom_sp(U::Float64, γsp::ComplexF64, γch::ComplexF64, χsp::ComplexF64, χch::ComplexF64, λ₀::ComplexF64)::ComplexF64 = U*(γsp * 1.5 * (1 + U * χsp) - 1.5)
@inline eom_ch(U::Float64, γsp::ComplexF64, γch::ComplexF64, χsp::ComplexF64, χch::ComplexF64, λ₀::ComplexF64)::ComplexF64 = -U*(γch * 0.5 * (1 - U * χch) - 0.5)
@inline eom_rest(U::Float64, γsp::ComplexF64, γch::ComplexF64, χsp::ComplexF64, χch::ComplexF64, λ₀::ComplexF64)::ComplexF64 = U*λ₀

#@inline ν0Index_of_ωIndex(ωi::Int, sP)::Int = sP.n_iν + sP.shift*(trunc(Int, (ωi - sP.n_iω - 1)/2)) + 1
#  2*sP.n_iν + sP.shift*(trunc(Int, (ωi - sP.n_iω - 1)/2))
function calc_Σ_eom_par(νmax::Int, U::Float64) 
    
    Nq::Int = size(wcache.χsp,1)
    Kνωq_pre::Vector{ComplexF64} = Vector{ComplexF64}(undef, Nq)
    Σ_tmp::Vector{ComplexF64} = Vector{ComplexF64}(undef, Nq)
    Σ_res::Array{ComplexF64,2} = zeros(ComplexF64, Nq, νmax)
    # wcache.νω_map = νω_map
    # wcache.ωind_map = ωind_map
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
                mP::ModelParameters, sP::SimulationParameters; νmax=sP.n_iν, pre_expand=true, 
                workerpool::AbstractWorkerPool=default_worker_pool())
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
        remote_results[i] = remotecall(calc_Σ_eom, workerpool, νω_range[ind], ωind_map, νmax, χ_sp[:,ωi],
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
