#TODO: define GF type that knows about which dimension stores which variable
using Base.Iterators
using FFTW


function λ_from_γ(type::Symbol, γ::AbstractArray{ComplexF64,3}, χ::AbstractArray{_eltype,2}, U::Float64)
    s = (type == :ch) ? -1 : 1
    res = similar(γ)
    for ωi in 1:size(γ,3)
        for qi in 1:size(γ,1)
            res[qi,:,ωi] = s .* view(γ,qi,:,ωi) .* (1 .+ s*U .* χ[qi, ωi]) .- 1
        end
    end
    return res
end

function F_from_χ(χ::AbstractArray{ComplexF64,3}, G::AbstractArray{ComplexF64,1}, sP::SimulationParameters, β::Float64)
    F = similar(χ)
    for ωi in 1:size(F,3)
    for νpi in 1:size(F,2)
        ωn, νpn = OneToIndex_to_Freq(ωi, νpi, sP) #, sP.n_iν_shell)
        for νi in 1:size(F,1)
        _, νn = OneToIndex_to_Freq(ωi, νi, sP) #, sP.n_iν_shell)
        @inbounds F[νi,νpi,ωi] = -(χ[νi,νpi,ωi] + (νn == νpn) * β * G[νn] * G[ωn+νn])/(
                                                  G[νn] * G[ωn+νn] * G[νpn] * G[ωn+νpn])
        end
        end
    end
    return F
end

#TODO: implement complex to real fftw
#TODO: gImp should know about its tail instead of χ₀
function calc_bubble(Gνω::GνqT, Gνω_r::GνqT, kG::KGrid, mP::ModelParameters, sP::SimulationParameters; local_tail=false)
    #TODO: fix the size (BSE_SC inconsistency)
    data = Array{ComplexF64,3}(undef, length(kG.kMult), 2*(sP.n_iν+sP.n_iν_shell), 2*sP.n_iω+1)
    for (ωi,ωn) in enumerate(-sP.n_iω:sP.n_iω)
        νrange = ((-(sP.n_iν+sP.n_iν_shell)):(sP.n_iν+sP.n_iν_shell-1)) .- trunc(Int,sP.shift*ωn/2)
        #TODO: fix the offset (BSE_SC inconsistency)
        for (νi,νn) in enumerate(νrange)
            conv_fft!(kG, view(data,:,νi,ωi), reshape(Gνω[:,νn].parent,gridshape(kG)), reshape(Gνω_r[:,νn+ωn].parent,gridshape(kG)))
            data[:,νi,ωi] .*= -mP.β
        end
    end
    #TODO: not necessary after real fft
    data = _eltype === Float64 ? real.(data) : data

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


#imports: BSE_SC, LapackWrapper, LinearAlgebra, bse_inv on all workers
function bse_inv(type::Symbol, qωi_range::Vector{Tuple{Int,Int}}, ω_offset::Int, Γr::Array{ComplexF64,3},
        χ₀Data::Array{ComplexF64,3}, χ₀Asym::Array{ComplexF64,2}, n_iν_shell::Int, χ_helper, U, β)
    s = type === :ch ? -1 : 1
    Nν = size(Γr,1)
    χ = Array{eltype(Γr),1}(undef, length(qωi_range))
    γ = Array{eltype(Γr),2}(undef, Nν, length(qωi_range))

    χννpω = Matrix{eltype(χ)}(undef, Nν, Nν)
    ipiv = Vector{Int}(undef, Nν)
    work = _gen_inv_work_arr(χννpω, ipiv)
    λ_cache = Array{eltype(χννpω),1}(undef, Nν)

    for i in 1:length(qωi_range)
        qi,ωi = qωi_range[i]
        ωn = ωi + ω_offset
        copy!(χννpω, view(Γr,:,:,ωi))
        for l in 1:size(χννpω,1)
            χννpω[l,l] += 1.0/χ₀Data[qi, n_iν_shell+l, ωi]
        end
        inv!(χννpω, ipiv, work)
        χ[i] = calc_χλ_impr!(λ_cache, type, ωn, χννpω, view(χ₀Data,qi,:,ωi), 
                                   U, β, χ₀Asym[qi,ωi], χ_helper);
        γ[:, i] = (1 .- s*λ_cache) ./ (1 .+ s* U .* χ[i])
        (qi == 1 && ωi == 1) && println("dbg...: ", χ[i])
    end
    return χ, γ
end

"""
    calc_χ_trilex(Γr::ΓT, χ₀, kG::KGrid, U::Float64, mP, sP)

Solve χ = χ₀ - 1/β² χ₀ Γ χ
⇔ (1 + 1/β² χ₀ Γ) χ = χ₀
⇔      (χ⁻¹ - χ₀⁻¹) = 1/β² Γ

with indices: χ[ω, q] = χ₀[]
"""
function calc_χγ_par(type::Symbol, Γr::ΓT, χ₀::χ₀T, kG::KGrid, mP::ModelParameters, sP::SimulationParameters)
    !(typeof(sP.χ_helper) <: BSE_Asym_Helpers) && throw("Current version ONLY supports BSE_Asym_Helper!")
    #TODO: reactivate integration to find usable frequencies: χ_ω = Array{_eltype, 1}(undef, Nω)
    Nν = 2*sP.n_iν
    Nq  = size(χ₀.data,χ₀.axes[:q])
    Nω  = size(χ₀.data,χ₀.axes[:ω])
    γ = γT(undef, Nq, Nν, Nω)
    χ = χT(undef, Nq, Nω)
    qωi_range = collect(Base.product(1:Nq, 1:Nω))[:]
    qωi_part = par_partition(qωi_range, length(workerpool))
    remote_results = Vector{Future}(undef, length(qωi_part))

    v1, v2 = 0,0;
    for (i,ind) in enumerate(qωi_part)
        #println("part: ", qωi_range[ind])
        remote_results[i] = remotecall(bse_inv, workerpool, type, qωi_range[ind], -sP.n_iω-1, Γr, 
                                       χ₀.data, χ₀.asym, sP.n_iν_shell, sP.χ_helper, mP.U, mP.β)
        #v1, v2 = bse_inv(type, qωi_range[ind], -sP.n_iω-1, Γr, 
        #                               χ₀.data, χ₀.asym, sP.n_iν_shell, sP.χ_helper, mP.U, mP.β)
    end
    for (i,ind) in enumerate(qωi_part)
        χv, γv = fetch(remote_results[i])
        for (j,qωind) in enumerate(qωi_range[ind])
            χ[qωind...] = χv[j]
            γ[qωind[1],:,qωind[2]] .= γv[j]
        end
    end


    log_q0_χ_check(kG, sP, χ, type)
    @warn "DBG: currently forcing omega FULL range!!"
    usable = 1:Nω

    return qωi_part, qωi_range, remote_results, NonLocalQuantities(χ, γ, usable, 0.0)
end

function calc_χγ(type::Symbol, Γr::ΓT, χ₀::χ₀T, kG::KGrid, mP::ModelParameters, sP::SimulationParameters)
    #TODO: find a way to reduce initialization clutter: move lo,up to sum_helper
    #TODO: χ₀ should know about its tail c2, c3
    s = type === :ch ? -1 : 1
    Nν = 2*sP.n_iν
    Nq  = length(kG.kMult)
    Nω  = size(χ₀.data,ω_axis)
    γ = γT(undef, Nq, Nν, Nω)
    χ = χT(undef, Nq, Nω)
    ωi_range = 1:Nω
    νi_range = 1:Nν
    qi_range = 1:Nq

    χ_ω = Array{_eltype, 1}(undef, Nω)
    χννpω = Matrix{_eltype}(undef, Nν, Nν)
    ipiv = Vector{Int}(undef, Nν)
    work = _gen_inv_work_arr(χννpω, ipiv)
    λ_cache = Array{eltype(χννpω),1}(undef, Nν)

    for ωi in ωi_range
        ωn = (ωi - sP.n_iω) - 1
        for qi in qi_range
            χννpω[:,:] = deepcopy(Γr[:,:,ωi])
            for l in νi_range
                χννpω[l,l] += 1.0/χ₀.data[qi,sP.n_iν_shell+l,ωi]
            end
            @timeit to "inv" inv!(χννpω, ipiv, work)
            @timeit to "χ Impr." if typeof(sP.χ_helper) <: BSE_Asym_Helpers
                χ[qi, ωi] = calc_χλ_impr!(λ_cache, type, ωn, χννpω, view(χ₀.data,qi,:,ωi), 
                                           mP.U, mP.β, χ₀.asym[qi,ωi], sP.χ_helper);
                γ[qi, :, ωi] = (1 .- s*λ_cache) ./ (1 .+ s*mP.U .* χ[qi, ωi])
            else
                if typeof(sP.χ_helper) === BSE_SC_Helper
                    improve_χ!(type, ωi, view(χννpω,:,:,ωi), view(χ₀,qi,:,ωi), mP.U, mP.β, sP.χ_helper);
                end
                #TODO: this is not necessary, sum_freq defaults to sum!
                if sP.tc_type_f == :nothing
                    χ[qi,ωi] = sum(χννpω)/mP.β^2
                    for νk in νi_range
                        γ[qi,νk,ωi] = sum(view(χννpω,:,νk))/(χ₀.data[qi,νk,ωi] * (1.0 + s*mP.U * χ[qi,ωi]))
                    end
                else
                    sEH = sP.sumExtrapolationHelper
                    χ[qi, ωi] = sum_freq_full!(χννpω, sEH.sh_f, mP.β, sEH.fνmax_cache_c, sEH.lo, sEH.up)
                    for νk in νi_range
                        γ[qi,νk,ωi] = sum_freq_full!(view(χννpω,:,νk),sEH.sh_f,1.0,
                                                     sEH.fνmax_cache_c,sEH.lo,sEH.up)  / (χ₀.data[qi, νk, ωi] * (1.0 + s*mP.U * χ[qi, ωi]))
                    end
                    extend_γ!(view(γ,qi,:, ωi), 2*π/mP.β)
                end
            end
        end
        #TODO: write macro/function for ths "real view" beware of performance hits
        v = _eltype === Float64 ? view(χ,:,ωi) : @view reinterpret(Float64,view(χ,:,ωi))[1:2:end]
        χ_ω[ωi] = kintegrate(kG, v)
    end
    log_q0_χ_check(kG, sP, χ, type)
    usable = find_usable_interval(real.(collect(χ_ω)), sum_type=sP.ωsum_type, reduce_range_prct=sP.usable_prct_reduction)
    sP.χ_helper != nothing && @warn "DBG: currently forcing omega FULL range!!"
    sP.χ_helper != nothing && (usable = 1:length(χ_ω))
    return NonLocalQuantities(χ, γ, usable, 0.0)
end


function calc_λ0(χ₀::χ₀T, Fr::FT, Qr::NonLocalQuantities, mP::ModelParameters, sP::SimulationParameters)
    #TODO: store nu grid in sP?
    Niν = size(Fr,ν_axis)
    ω_range = 1:size(χ₀.data,ω_axis)
    λ0 = Array{ComplexF64,3}(undef,size(χ₀.data,q_axis),Niν,length(ω_range))

    if typeof(sP.χ_helper) <: BSE_Asym_Helpers
        λ0[:] = calc_λ0_impr(:sp, -sP.n_iω:sP.n_iω, Fr, χ₀.data, χ₀.asym, view(Qr.γ,1,:,:), view(Qr.χ,1,:),
                             mP.U, mP.β, sP.χ_helper)
    else
        #TODO: this is not well optimized, but also not often executed
        tmp = Array{ComplexF64, 1}(undef, Niν)
        for ωi in ω_range
            for νi in 1:Niν
                #TODO: export realview functions?
                v1 = view(Fr,νi,:,ωi)
                for qi in axes(χ₀.data,q_axis)
                    v2 = view(χ₀.data,qi,:,ωi) 
                    @simd for νpi in 1:Niν 
                        @inbounds tmp[νpi] = v1[νpi] * v2[νpi]
                    end
                    #TODO: update sum_freq_full to accept sP und figure out sum type by itself
                    sEH = sP.sumExtrapolationHelper
                    λ0[qi,νi,ωi] = if sEH !== nothing
                        sum_freq_full!(tmp, sP.sh_f, 1.0, sEH.fνmax_cache_c, sEH.lo, sEH.up)/(mP.β^2)
                    else
                        sum(tmp)/mP.β^2
                    end
                end
            end
        end
        (sP.tc_type_f != :nothing) && extend_corr!(λ0)
    end
    return λ0
end

@fastmath @inline eom(U::Float64, γsp::ComplexF64, γch::ComplexF64, χsp::ComplexF64, χch::ComplexF64, λ₀::ComplexF64) = U*(γsp * 1.5 * (1 + U * χsp) - γch * 0.5 * (1 - U * χch) - 1.5 + 0.5 + λ₀)

function calc_Σ_ω!(Σ::AbstractArray{ComplexF64,3}, Kνωq::Array{ComplexF64}, Kνωq_pre::Array{ComplexF64, 1},
            ωindices::AbstractArray{Int,1},
            Q_sp::NonLocalQuantities, Q_ch::NonLocalQuantities, 
            Gνω::GνqT, λ₀::AbstractArray{ComplexF64,3}, U::Float64, kG::KGrid, 
            sP::SimulationParameters)
    fill!(Σ, zero(ComplexF64))
    for ωii in 1:length(ωindices)
        ωi = ωindices[ωii]
        ωn = (ωi - sP.n_iω) - 1
        νZero = ν0Index_of_ωIndex(ωi, sP)
        maxn = minimum([νZero + sP.n_iν - 1, size(Q_ch.γ,ν_axis), νZero + size(Σ, ν_axis) - 1])
        for (νii,νi) in enumerate(νZero:maxn)
            v = view(Gνω,:,(νii-1) + ωn)
            if kG.Nk == 1
                Σ[1,νii-1,ωn] = eom(U, Q_sp.γ[1,νi,ωi], Q_ch.γ[1,νi,ωi], Q_sp.χ[1,ωi], 
                                    Q_ch.χ[1,ωi], λ₀[1,νi,ωi]) * v[1]
            else
                @simd for qi in 1:size(Σ,q_axis)
                @inbounds Kνωq_pre[qi] = eom(U, Q_sp.γ[qi,νi,ωi], Q_ch.γ[qi,νi,ωi], Q_sp.χ[qi,ωi], 
                                    Q_ch.χ[qi,ωi], λ₀[qi,νi,ωi])
                end
                conv_fft1!(kG, view(Σ,:,νii-1,ωn), Kνωq_pre, v)
            end
        end
    end
end

function calc_Σ(Q_sp::NonLocalQuantities, Q_ch::NonLocalQuantities, λ₀::AbstractArray{_eltype,3},
                Gνω::GνqT, kG::KGrid,
                mP::ModelParameters, sP::SimulationParameters; pre_expand=true)
    if (size(Q_sp.χ,1) != size(Q_ch.χ,1)) || (size(Q_sp.χ,1) != length(kG.kMult))
        @error "q Grids not matching"
    end
    Σ_hartree = mP.n * mP.U/2.0;
    Nq = size(Q_sp.χ,1)
    Nω = size(Q_sp.χ,2)
    ωrange = -sP.n_iω:sP.n_iω
    ωindices = (sP.dbg_full_eom_omega) ? (1:Nω) : intersect(Q_sp.usable_ω, Q_ch.usable_ω)
    νmax = floor(Int,length(ωindices)/3)

    Kνωq = Array{ComplexF64, length(gridshape(kG))}(undef, gridshape(kG)...)
    Kνωq_pre = Array{ComplexF64, 1}(undef, length(kG.kMult))
    #TODO: implement real fft and make _pre real
    Σ_ladder_ω = OffsetArray( Array{Complex{Float64},3}(undef,Nq, sP.n_iν, length(ωrange)),
                              1:Nq, 0:sP.n_iν-1, ωrange)
    @timeit to "Σ_ω" calc_Σ_ω!(Σ_ladder_ω, Kνωq, Kνωq_pre, ωindices, Q_sp, Q_ch, Gνω, λ₀, mP.U, kG, sP)
    @timeit to "sum Σ_ω" res = dropdims(sum(Σ_ladder_ω, dims=[3]),dims=3) ./ mP.β .+ Σ_hartree
    return  res
end

function Σ_loc_correction(Σ_ladder::AbstractArray{T1, 2}, Σ_ladderLoc::AbstractArray{T2, 2}, Σ_loc::AbstractArray{T3, 1}) where {T1 <: Number, T2 <: Number, T3 <: Number}
    res = similar(Σ_ladder)
    for qi in axes(Σ_ladder,1)
        for νi in axes(Σ_ladder,2)
            res[qi,νi] = Σ_ladder[qi,νi] .- Σ_ladderLoc[νi] .+ Σ_loc[νi]
        end
    end
    return res
end
