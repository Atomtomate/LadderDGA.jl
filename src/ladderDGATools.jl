#TODO: define GF type that knows about which dimension stores which variable
using Base.Iterators


function calc_bubble(Gνω::GνqT, kG::ReducedKGrid, mP::ModelParameters, sP::SimulationParameters)
    bubble = create_wkn(sP, length(kG.kMult))
    nd = length(gridshape(kG))
    for ωi in axes(bubble,ω_axis), νi in axes(bubble,ν_axis)
        ωn, νn = OneToIndex_to_Freq(ωi, νi, sP)
        conv_fft!(kG, view(bubble,:,νi,ωi), selectdim(Gνω,nd+1,νn+sP.fft_offset), selectdim(Gνω,nd+1,νn+ωn+sP.fft_offset))
        bubble[:,νi,ωi] .*= -mP.β
    end
    return bubble
end


"""
Solve χ = χ₀ - 1/β² χ₀ Γ χ
    ⇔ (1 + 1/β² χ₀ Γ) χ = χ₀
    ⇔      (χ⁻¹ - χ₀⁻¹) = 1/β² Γ
    with indices: χ[ω, q] = χ₀[]
"""
function calc_χ_trilex(Γr::Array{Complex{Float64},3}, bubble::Array{Complex{Float64},3}, 
                       kG::ReducedKGrid, U::Float64, mP::ModelParameters, sP::SimulationParameters)
    #TODO: find a way to reduce initialization clutter
    Nk = size(bubble,q_axis)
    Niν = size(bubble,ν_axis)
    γ = create_wkn(sP, Nk)
    χ = create_wk(sP, Nk)
    χ_ω = Array{Float64, 1}(undef, size(bubble,ω_axis))
    ωindices = ωindex_range(sP)
    lo = npartial_sums(sP.sh_f)
    up = Niν - lo + 1 
    fνmax_cache  = Array{eltype(bubble), 1}(undef, lo)
    χ_full = Matrix{eltype(bubble)}(undef, Niν, Niν)
    _one = one(eltype(Γr))
    ipiv = Vector{Int}(undef, Niν)
    work = _gen_inv_work_arr(χ_full, ipiv)

    for ωi in axes(bubble,ω_axis)
        Γview = view(Γr,:,:,ωi)
        for qi in axes(bubble,q_axis)
            @inbounds χ_full[:,:] = view(Γr,:,:,ωi)
            for l in 1:Niν 
                @inbounds @views χ_full[l,l] += _one/bubble[qi,l,ωi]
            end
            @timeit to "inv" inv!(χ_full, ipiv, work)
            @inbounds χ[qi, ωi] = sum_freq_full!(χ_full, sP.sh_f, mP.β, sP.fνmax_cache_c, lo, up)
            #TODO: absor this loop into sum_freq, partial sum is carried out twice
            @timeit to "γ" for νk in axes(bubble,ν_axis)
                @inbounds γ[qi, νk, ωi] = sum_freq_full!(view(χ_full,:,νk), sP.sh_f, 1.0, sP.fνmax_cache_c, lo, up) / (bubble[qi, νk, ωi] * (1.0 + U * χ[qi, ωi]))
            end
            (sP.tc_type_f != :nothing) && extend_γ!(view(γ,qi,:, ωi), 2*π/mP.β)
        end
        #TODO: write macro/function for ths "reak view" beware of performance hits
        v = @view reinterpret(Float64,view(χ,:,ωi))[1:2:end]
        χ_ω[ωi] = kintegrate(kG, v)
    end

    usable = find_usable_interval(collect(χ_ω), sum_type=sP.ωsum_type, reduce_range_prct=sP.usable_prct_reduction)
    return NonLocalQuantities(χ, γ, usable, 0.0)
end


function Σ_correction(ωindices::AbstractArray{Int,1}, bubble::BubbleT, FUpDo::AbstractArray{ComplexF64,3}, sP::SimulationParameters)

    Niν = size(bubble,ν_axis)
    tmp = Array{Float64, 1}(undef, Niν)
    corr = Array{Float64,3}(undef,size(bubble,q_axis),Niν,length(ωindices))

    #TODO: this is not well optimized, but also not often executed
    for (ωi,ωii) in enumerate(ωindices)
        for νi in 1:Niν
            #TODO: export realview functions?
            v1 = @view reinterpret(Float64,FUpDo[νi,:,ωii])[1:2:end]
            for qi in axes(bubble,q_axis)
                v2 = @view reinterpret(Float64,bubble[qi,:,ωii])[1:2:end]
                @simd for νpi in 1:Niν 
                    @inbounds tmp[νpi] = v1[νpi] * v2[νpi]
                end
                #@inbounds @views corr[qi,νi,ωii] = sum_freq_full!(tmp, sP.sh_f, 1.0, sP.fνmax_cache_r, sP.fνmax_lo, sP.fνmax_up)
                @inbounds @views corr[qi,νi,ωi] = sum(tmp)
                #TODO: reactivate impr sum!!!!!
                #sum_freq_full!(tmp, sP.sh_b, 1.0, sP.fνmax_cache_r, sP.fνmax_lo, sP.fνmax_up)
            end
        end
    end
    return corr
end

function calc_Σ_ω_old!(Σ::AbstractArray{Complex{Float64},3}, ωindices::AbstractArray{Int,1},
            Q_sp::TQ, Q_ch::TQ,Gνω::GνqT, corr::AbstractArray{Float64,3}, U::Float64, kG::ReducedKGrid, 
            sP::SimulationParameters; lopWarn=false) where TQ <: Union{NonLocalQuantities, ImpurityQuantities}
    #Kνωq = Array{ComplexF64, length(gridshape(kG))}(undef, gridshape(kG)...)
    fill!(Σ, zero(eltype(Σ)))
    nd = length(gridshape(kG))
    for ωi in axes(Σ,ω_axis)
        ωn = (ωi - sP.n_iω) - 1
        fsp = 1.5 .* (1 .+ U .* view(Q_sp.χ,:,ωi))
        fch = 0.5 .* (1 .- U .* view(Q_ch.χ,:,ωi))
        νZero = 1#ν0Index_of_ωIndex(ωi, sP)
        for νi in axes(Q_sp.γ, ν_axis)
        ωni, νni = OneToIndex_to_Freq(ωi, νi, sP)
        s =  -trunc(Int64,sP.shift*ωni/2) + trunc(Int64,sP.shift*sP.n_iω/2)
        Kνωq = view(Q_sp.γ,:,νi,ωi) .* fsp .- view(Q_ch.γ,:,νi,ωi) .* fch .- 1.5 .+ 0.5 .+ view(corr,:,νi,ωi)
        conv_fft1!(kG, view(Σ,:,νi+s,ωi), Kνωq, selectdim(Gνω,nd+1,ωni + νni + sP.fft_offset))
        end
    end
end


function calc_Σ_ω!(Σ::AbstractArray{Complex{Float64},3}, ωindices::AbstractArray{Int,1},
            Q_sp::TQ, Q_ch::TQ,Gνω::GνqT, corr::AbstractArray{Float64,3}, U::Float64, kG::ReducedKGrid, 
            sP::SimulationParameters; lopWarn=false) where TQ <: Union{NonLocalQuantities, ImpurityQuantities}
    Kνωq = Array{ComplexF64, length(gridshape(kG))}(undef, gridshape(kG)...)
    Kνωq_pre = Array{ComplexF64, 1}(undef, size(corr,q_axis))
    fill!(Σ, zero(eltype(Σ)))

    nd = length(gridshape(kG))
    for ωii in 1:length(ωindices)
        ωi = ωindices[ωii]
        ωn = (ωi - sP.n_iω) - 1
        @inbounds fsp = 1.5 .* (1 .+ U .* view(Q_sp.χ,:,ωi))
        @inbounds fch = 0.5 .* (1 .- U .* view(Q_ch.χ,:,ωi))
        νZero = ν0Index_of_ωIndex(ωi, sP)
        maxn = minimum([νZero + sP.n_iν - 1, size(Q_ch.γ,ν_axis)])
        for (νi,νn) in enumerate(νZero:maxn)
            #TODO: manual unroll of conv_fft1
            if kG.Nk == 1
                Σ[1,νi,ωii] = (Q_sp.γ[1,νn,ωi] * fsp[1] - Q_ch.γ[1,νn,ωi] * fch[1] - 1.5 + 0.5 + corr[1,νn,ωii]) * selectdim(Gνω,nd+1,(νi-1) + ωn + sP.fft_offset)[1]
            else

                @simd for qi in 1:size(corr,q_axis)
                    @inbounds Kνωq_pre[qi] = Q_sp.γ[qi,νn,ωi] * fsp[qi] - Q_ch.γ[qi,νn,ωi] * fch[qi] - 1.5 + 0.5 + corr[qi,νn,ωii]
                end
                expandKArr!(kG,Kνωq,Kνωq_pre)
                Dispersions.mul!(Kνωq, kG.fftw_plan, Kνωq)
                v = selectdim(Gνω,nd+1,(νi-1) + ωn + sP.fft_offset)
                @simd for ki in 1:length(Kνωq)
                    @inbounds Kνωq[ki] *= v[ki]
                end
                Dispersions.ldiv!(Kνωq, kG.fftw_plan, Kνωq)
                Dispersions.ifft_post!(typeof(kG), Kνωq)
                reduceKArr!(kG,  view(Σ,:,νi,ωii), Kνωq) 
                @simd for i in 1:size(corr,q_axis)
                    @inbounds Σ[i,νi,ωii] /= (kG.Nk)
                end
            end
            #TODO: end manual unroll of conv_fft1
            #@inbounds conv_fft!(kG, view(Σ,:,νn,ωii), Gνω[(νn-1) + ωn + sP.fft_offset], Kνωq)
        end
    end
end

function calc_Σ_dbg(Q_sp::NonLocalQuantities, Q_ch::NonLocalQuantities, bubble::BubbleT,
                Gνω::GνqT, FUpDo::AbstractArray{Complex{Float64},3}, kGrid::ReducedKGrid,
                mP::ModelParameters, sP::SimulationParameters; pre_expand=true)
    ωindices = (sP.dbg_full_eom_omega) ? (1:size(bubble,ω_axis)) : intersect(Q_sp.usable_ω, Q_ch.usable_ω)

    Σ_ladder_ω = Array{Complex{Float64},3}(undef,size(bubble,1),size(bubble,2)+sP.n_iω,size(bubble,3))
    @timeit to "corr" corr = Σ_correction(ωindices, bubble, FUpDo, sP)
    (sP.tc_type_f != :nothing) && extend_corr!(corr)
    @timeit to "Σ_ω old" calc_Σ_ω_old!(Σ_ladder_ω, ωindices, Q_sp, Q_ch, Gνω, corr, mP.U, kGrid, sP)
    res = (mP.U/mP.β) .* sum(Σ_ladder_ω, dims=[3])[:,:,1]
    return  Σ_ladder_ω,res,corr
end

function calc_Σ(Q_sp::NonLocalQuantities, Q_ch::NonLocalQuantities, bubble::BubbleT,
                Gνω::GνqT, FUpDo::AbstractArray{Complex{Float64},3}, kGrid::ReducedKGrid,
                mP::ModelParameters, sP::SimulationParameters; pre_expand=true)
    ωindices = (sP.dbg_full_eom_omega) ? (1:size(bubble,ω_axis)) : intersect(Q_sp.usable_ω, Q_ch.usable_ω)
    Σ_ladder_ω = Array{Complex{Float64},3}(undef,size(bubble,q_axis), sP.n_iν, size(bubble,ω_axis))
    @timeit to "corr" corr = Σ_correction(ωindices, bubble, FUpDo, sP)
    (sP.tc_type_f != :nothing) && extend_corr!(corr)
    @timeit to "Σ_ω" calc_Σ_ω!(Σ_ladder_ω, ωindices, Q_sp, Q_ch, Gνω, corr, mP.U, kGrid, sP)
    @timeit to "sum Σ_ω" res = (mP.U/mP.β) .* sum(Σ_ladder_ω, dims=[3])[:,:,1]
    return  res
end
