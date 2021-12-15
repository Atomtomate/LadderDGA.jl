#TODO: define GF type that knows about which dimension stores which variable
using Base.Iterators
using FFTW


#TODO: implement complex to real fftw
#TODO: get rid of selectdim
function calc_bubble(Gνω::GνqT, kG::ReducedKGrid, mP::ModelParameters, sP::SimulationParameters)
    #TODO: fix the size (BSE_SC inconsistency)
    bubble = Array{ComplexF64,3}(undef, length(kG.kMult), 2*(sP.n_iν+sP.χ_helper.Nν_shell), 2*sP.n_iω+1)
    nd = length(gridshape(kG))
    for ωi in axes(bubble,ω_axis)
        #TODO: fix the offset (BSE_SC inconsistency)
        for νi in axes(bubble,ν_axis)
            ωn, νn = OneToIndex_to_Freq(ωi, νi, sP, sP.χ_helper.Nν_shell)
            conv_fft!(kG, view(bubble,:,νi,ωi), selectdim(Gνω,nd+1,νn+sP.fft_offset), selectdim(Gνω,nd+1,νn+ωn+sP.fft_offset))
            bubble[:,νi,ωi] .*= -mP.β
        end
    end
    #TODO: not necessary after real fft
    return _eltype === Float64 ? real.(bubble) : bubble
end

"""
    calc_χ_trilex(Γr::ΓT, bubble, kG::ReducedKGrid, U::Float64, mP, sP)

Solve χ = χ₀ - 1/β² χ₀ Γ χ
⇔ (1 + 1/β² χ₀ Γ) χ = χ₀
⇔      (χ⁻¹ - χ₀⁻¹) = 1/β² Γ

with indices: χ[ω, q] = χ₀[]
"""
function calc_χ_γ(type::Symbol, Γr::ΓT, χ₀::BubbleT, kG::ReducedKGrid, 
        mP::ModelParameters, sP::SimulationParameters)
    #TODO: find a way to reduce initialization clutter: move lo,up to sum_helper
    Nk = size(χ₀,q_axis)
    Niν = 2*sP.n_iν
    γ = γT(undef, Nk, Niν, 2*sP.n_iω+1)
    χ = χT(undef, Nk, 2*sP.n_iω+1)
    χ_ω = Array{Float64, 1}(undef, size(χ₀, ω_axis))
    ωindices = ωindex_range(sP)
    lo = npartial_sums(sP.sh_f)
    up = Niν - lo + 1 
    fνmax_cache  = Array{_eltype, 1}(undef, lo)
    χννpω = Matrix{_eltype}(undef, Niν, Niν)
    ipiv = Vector{Int}(undef, Niν)
    work = _gen_inv_work_arr(χννpω, ipiv)
    fνmax_cache = _eltype === Float64 ? sP.fνmax_cache_r : sP.fνmax_cache_c

    U = type === :ch ? -mP.U : mP.U
    c_μ_U = mP.μ - mP.U*mP.n/2
    c_n   = mP.U^2 * mP.n/2 * (1 - mP.n/2)
    for ωi in axes(χ₀,ω_axis)
        ωn = (ωi - sP.n_iω) - 1
        for qi in axes(χ₀,q_axis)
            c2 = mP.μ - mP.U + c_μ_U + kG.ϵkGrid[qi]
            c3 = c_μ_U^2 + c_n + kG.ϵkGrid[qi]^2 - 2*kG.ϵkGrid[qi]*c_μ_U
            χννpω[:,:] = deepcopy(Γr[:,:,ωi])
            for l in 1:Niν 
                #TODO: fix the offset (BSE_SC inconsistency)
                χννpω[l,l] = Γr[l,l,ωi] + 1.0/χ₀[qi,sP.χ_helper.Nν_shell+l,ωi]
            end
            #TODO: only testing
            @timeit to "inv" inv!(χννpω, ipiv, work)
            @timeit to "χ Impr." if typeof(sP.χ_helper) === BSE_SC_Helper
                #TODO: test SC version improve_χ!(type, ωi, χννpω, view(χ₀,qi,:,ωi), mP.U, mP.β, sP.χ_helper);
                bs = -sum(χ₀[qi,:,ωi])/mP.β^2
                xsp_s,xch_s,lsp,lch = improve_χλ_direct(ωi, χννpω, χννpω, view(χ₀,qi,:,ωi), mP.U, mP.β, bs, sP.χ_helper)
                γ[qi, :, ωi] = type == :sp ? lsp .-1 : lch .+ 1
                χ[qi, ωi] = type == :sp ? xsp_s : xch_s
            elseif typeof(sP.χ_helper) === BSE_Asym_Helper
                χ[qi, ωi], γ[qi, :, ωi], calc_χλ(type, ωn, χννpω, view(χ₀,qi,:,ωi), 
                                                 mP.U, mP.β, c2, c3, sP.χ_helper);
            else
                χ[qi, ωi] = sum_freq_full!(χννpω, sP.sh_f, mP.β, fνmax_cache, lo, up)
                for νk in axes(χ₀,ν_axis)
                    γ[qi, νk, ωi] = sum_freq_full!(view(χννpω,:,νk), sP.sh_f, 1.0, fνmax_cache, lo, 
                                                   up) / (χ₀[qi, νk, ωi] * (1.0 + U * χ[qi, ωi]))
                end
                (sP.tc_type_f != :nothing) && extend_γ!(view(γ,qi,:, ωi), 2*π/mP.β)
            end
        end
        #TODO: write macro/function for ths "real view" beware of performance hits
        v = _eltype === Float64 ? view(χ,:,ωi) : @view reinterpret(Float64,view(χ,:,ωi))[1:2:end]
        χ_ω[ωi] = kintegrate(kG, v)
    end

    usable = find_usable_interval(collect(χ_ω), sum_type=sP.ωsum_type, reduce_range_prct=sP.usable_prct_reduction)
    return NonLocalQuantities(χ, γ, usable, 0.0)
end


#TODO: replace by F_m and calculate F_m in improved form
function Σ_correction(ωindices::AbstractArray{Int,1}, bubble::BubbleT, FUpDo::FUpDoT, sP::SimulationParameters)

    Niν = size(bubble,ν_axis)
    tmp = Array{Float64, 1}(undef, Niν)
    corr = Array{Float64,3}(undef,size(bubble,q_axis),Niν,length(ωindices))
    lo = npartial_sums(sP.sh_f)
    up = Niν - lo + 1 

    #TODO: this is not well optimized, but also not often executed
    for (ωi,ωii) in enumerate(ωindices)
        for νi in 1:Niν
            #TODO: export realview functions?
            v1 = _eltype === Float64 ? view(FUpDo,νi,:,ωii) : @view reinterpret(Float64,view(FUpDo,νi,:,ωii))[1:2:end]
            for qi in axes(bubble,q_axis)
                v2 = _eltype === Float64 ? view(bubble,qi,:,ωii) : @view reinterpret(Float64,view(bubble,qi,:,ωii))[1:2:end]
                @simd for νpi in 1:Niν 
                    @inbounds tmp[νpi] = v1[νpi] * v2[νpi]
                end
                corr[qi,νi,ωi] = sum_freq_full!(tmp, sP.sh_f, 1.0, sP.fνmax_cache_r, lo, up)
                #@inbounds @views corr[qi,νi,ωi] = sum(tmp)
                #TODO: reactivate impr sum!!!!!
                #sum_freq_full!(tmp, sP.sh_b, 1.0, sP.fνmax_cache_r, sP.fνmax_lo, sP.fνmax_up)
            end
        end
    end
    (sP.tc_type_f != :nothing) && extend_corr!(corr)
    return corr
end

function calc_Σ_ω!(Σ::AbstractArray{ComplexF64,3}, Kνωq::Array{ComplexF64}, Kνωq_pre::Array{ComplexF64, 1},
            ωindices::AbstractArray{Int,1},
            χsp::AbstractArray{ComplexF64,2}, γsp::AbstractArray{ComplexF64,3},
            χch::AbstractArray{ComplexF64,2}, γch::AbstractArray{ComplexF64,3},
            Gνω::GνqT, corr::AbstractArray{Float64,3}, U::Float64, kG::ReducedKGrid, 
            sP::SimulationParameters; lopWarn=false) where TQ <: Union{NonLocalQuantities, ImpurityQuantities}
    fill!(Σ, zero(ComplexF64))
    nd = length(gridshape(kG))
    for ωii in 1:length(ωindices)
        ωi = ωindices[ωii]
        ωn = (ωi - sP.n_iω) - 1
        fsp = 1.5 .* (1 .+ U .* view(χsp,:,ωi))
        fch = 0.5 .* (1 .- U .* view(χch,:,ωi))
        νZero = ν0Index_of_ωIndex(ωi, sP)
        maxn = minimum([νZero + sP.n_iν - 1, size(γsp,ν_axis), νZero + size(Σ, ν_axis) - 1])
        for (νi,νn) in enumerate(νZero:maxn)
            #TODO: remove manual unroll of conv_fft1
            v = selectdim(Gνω,nd+1,(νi-1) + ωn + sP.fft_offset)
            if kG.Nk == 1
                Σ[1,νi,ωii] = (γsp[1,νn,ωi] * fsp[1] - γch[1,νn,ωi] * fch[1] - 1.5 + 0.5 + corr[1,νn,ωii]) * v[1]
            else
                @simd for qi in 1:size(Σ,q_axis)
                    @inbounds Kνωq_pre[qi] = γsp[qi,νn,ωi] * fsp[qi] - γch[qi,νn,ωi] * fch[qi] - 1.5 + 0.5 + corr[qi,νn,ωii]
                end
                expandKArr!(kG,Kνωq,Kνωq_pre)
                Dispersions.mul!(Kνωq, kG.fftw_plan, Kνωq)
                @simd for ki in 1:length(Kνωq)
                    @inbounds Kνωq[ki] *= v[ki]
                end
                Dispersions.ldiv!(Kνωq, kG.fftw_plan, Kνωq)
                reduceKArr!(kG,  view(Σ,:,νi,ωii), Dispersions.ifft_post(kG, Kνωq)) 
                @simd for qi in 1:size(Σ,q_axis)
                    @inbounds Σ[qi,νi,ωii] /= (kG.Nk)
                end
            end
            #TODO: end manual unroll of conv_fft1
            #@inbounds conv_fft!(kG, view(Σ,:,νn,ωii), Gνω[(νn-1) + ωn + sP.fft_offset], Kνωq)
        end
    end
end


function calc_Σ_ω!(Σ::AbstractArray{Complex{Float64},3}, Kνωq::Array{ComplexF64}, Kνωq_pre::Array{ComplexF64, 1},
            ωindices::AbstractArray{Int,1},
            Q_sp::TQ, Q_ch::TQ,Gνω::GνqT, corr::AbstractArray{Float64,3}, U::Float64, kG::ReducedKGrid, 
            sP::SimulationParameters; lopWarn=false) where TQ <: Union{NonLocalQuantities, ImpurityQuantities}
    fill!(Σ, zero(ComplexF64))
    nd = length(gridshape(kG))
    for ωii in 1:length(ωindices)
        ωi = ωindices[ωii]
        ωn = (ωi - sP.n_iω) - 1
        fsp = 1.5 .* (1 .+ U .* view(Q_sp.χ,:,ωi))
        fch = 0.5 .* (1 .- U .* view(Q_ch.χ,:,ωi))
        νZero = ν0Index_of_ωIndex(ωi, sP)
        maxn = minimum([νZero + sP.n_iν - 1, size(Q_ch.γ,ν_axis), νZero + size(Σ, ν_axis) - 1])
        #TODO: rewrite to plain sum using offset array
        for (νi,νn) in enumerate(νZero:maxn)
            #TODO: remove manual unroll of conv_fft1
            v = selectdim(Gνω,nd+1,(νi-1) + ωn + sP.fft_offset)
            if kG.Nk == 1
                Σ[1,νi,ωii] = (Q_sp.γ[1,νn,ωi] * fsp[1] - Q_ch.γ[1,νn,ωi] * fch[1] - 1.5 + 0.5 + corr[1,νn,ωii]) * v[1]
            else
                @simd for qi in 1:size(Σ,q_axis)
                    @inbounds Kνωq_pre[qi] = Q_sp.γ[qi,νn,ωi] * fsp[qi] - Q_ch.γ[qi,νn,ωi] * fch[qi] - 1.5 + 0.5 + corr[qi,νn,ωii]
                end
                expandKArr!(kG,Kνωq,Kνωq_pre)
                Dispersions.mul!(Kνωq, kG.fftw_plan, Kνωq)
                @simd for ki in 1:length(Kνωq)
                    @inbounds Kνωq[ki] *= v[ki]
                end
                Dispersions.ldiv!(Kνωq, kG.fftw_plan, Kνωq)
                reduceKArr!(kG,  view(Σ,:,νi,ωii), Dispersions.ifft_post(kG, Kνωq)) 
                @simd for qi in 1:size(Σ,q_axis)
                    @inbounds Σ[qi,νi,ωii] /= (kG.Nk)
                end
            end
            #TODO: end manual unroll of conv_fft1
            #@inbounds conv_fft!(kG, view(Σ,:,νn,ωii), Gνω[(νn-1) + ωn + sP.fft_offset], Kνωq)
        end
    end
end

function calc_Σνω(Q_sp::NonLocalQuantities, Q_ch::NonLocalQuantities, bubble::BubbleT,
                Gνω::GνqT, FUpDo::FUpDoT, kG::ReducedKGrid,
                mP::ModelParameters, sP::SimulationParameters; pre_expand=true)
    if (size(Q_sp.χ,1) != size(Q_ch.χ,1)) || (size(Q_sp.χ,1) != size(bubble,1)) || (size(Q_sp.χ,1) != length(kG.kMult))
        @error "q Grids not matching"
    end
    @warn "Selfenergie now contains Hartree term!"
    Σ_hartree = mP.n * mP.U/2.0;
    ωindices = (sP.dbg_full_eom_omega) ? (1:size(bubble,ω_axis)) : intersect(Q_sp.usable_ω, Q_ch.usable_ω)
    Kνωq = Array{ComplexF64, length(gridshape(kG))}(undef, gridshape(kG)...)
    Kνωq_pre = Array{ComplexF64, 1}(undef, length(kG.kMult))
    Σ_ladder_ω = Array{Complex{Float64},3}(undef,size(bubble,q_axis), sP.n_iν, size(bubble,ω_axis))
    @timeit to "corr" corr = Σ_correction(ωindices, bubble, FUpDo, sP)
    (sP.tc_type_f != :nothing) && extend_corr!(corr)
    @timeit to "Σ_ω" calc_Σ_ω!(Σ_ladder_ω, Kνωq, Kνωq_pre, ωindices, Q_sp, Q_ch, Gνω, corr, mP.U, kG, sP)
    return  Σ_ladder_ω
end

function calc_Σ(Q_sp::NonLocalQuantities, Q_ch::NonLocalQuantities, bubble::BubbleT,
                Gνω::GνqT, FUpDo::FUpDoT, kG::ReducedKGrid,
                mP::ModelParameters, sP::SimulationParameters; pre_expand=true)
    if (size(Q_sp.χ,1) != size(Q_ch.χ,1)) || (size(Q_sp.χ,1) != size(bubble,1)) || (size(Q_sp.χ,1) != length(kG.kMult))
        @error "q Grids not matching"
    end
    @warn "Selfenergie now contains Hartree term and is cut to νmax = length(usable_ω)/3!"
    Σ_hartree = mP.n * mP.U/2.0;
    ωindices = (sP.dbg_full_eom_omega) ? (1:size(bubble,ω_axis)) : intersect(Q_sp.usable_ω, Q_ch.usable_ω)
    νmax = floor(Int,length(ωindices)/3)

    Kνωq = Array{ComplexF64, length(gridshape(kG))}(undef, gridshape(kG)...)
    Kνωq_pre = Array{ComplexF64, 1}(undef, length(kG.kMult))
    #TODO: implement real fft and make _pre real
    Σ_ladder_ω = Array{Complex{Float64},3}(undef,size(bubble,q_axis), νmax, size(bubble,ω_axis))
    @timeit to "corr" corr = Σ_correction(ωindices, bubble, FUpDo, sP)
    (sP.tc_type_f != :nothing) && extend_corr!(corr)
    @timeit to "Σ_ω" calc_Σ_ω!(Σ_ladder_ω, Kνωq, Kνωq_pre, ωindices, Q_sp, Q_ch, Gνω, corr, mP.U, kG, sP)
    #TODO: *U should be in calc+Sigma_w
    @timeit to "sum Σ_ω" res = (mP.U/mP.β) .* sum(Σ_ladder_ω, dims=[3])[:,:,1] .+ Σ_hartree
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
