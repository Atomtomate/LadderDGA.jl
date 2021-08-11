#TODO: define GF type that knows about which dimension stores which variable
using Base.Iterators


function calc_bubble(Gνω::GνqT, kG::ReducedKGrid, mP::ModelParameters, sP::SimulationParameters)
    bubble = create_wkn(sP, length(kG.kMult))
    for ωi in axes(bubble,ω_axis), νi in axes(bubble,ν_axis)
        ωn, νn = OneToIndex_to_Freq(ωi, νi, sP)
        conv_fft!(kG, view(bubble,:,νi,ωi), Gνω[νn+sP.fft_offset], Gνω[νn+ωn+sP.fft_offset])
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
        χ_ω[ωi] = kintegrate(kG, view(reshape(reinterpret(Float64,view(χ,:,ωi)),1:Nk,2),:,1))
    end

    usable = find_usable_interval(collect(χ_ω), sum_type=sP.ωsum_type, reduce_range_prct=sP.usable_prct_reduction)
    return NonLocalQuantities(χ, γ, usable, 0.0)
end


function Σ_correction!(corr::AbstractArray{Float64,3}, ωindices::AbstractArray{Int,1}, bubble::BubbleT, 
                     FUpDo::AbstractArray{Complex{Float64},3}, sP::SimulationParameters) where T <: SumHelper

    Niν = size(bubble,ν_axis)
    tmp = Array{Float64, 1}(undef, Niν)

    #TODO: this is not well optimized, but also not often executed
    for (ωi,ωii) in enumerate(ωindices)
        for νi in 1:Niν
            for qi in axes(bubble,q_axis)
                #TODO: export realview functions?
                v1 = view(reshape(reinterpret(Float64,view(FUpDo,:,νi,ωi)),1:Niν,2),:,1)
                v2 = view(reshape(reinterpret(Float64,view(bubble,qi,:,ωi)),1:Niν,2),:,1)
                @simd for νpi in 1:Niν 
                    @inbounds tmp[νpi] = v1[νpi] * v2[νpi]
                end
                #TODO: use normal summation for correction term?
                @inbounds @views corr[qi,νi,ωii] = sum_freq_full!(tmp, sP.sh_f, 1.0, sP.fνmax_cache_r, sP.fνmax_lo, sP.fνmax_up)
            end
        end
    end
end

"""
    calc_Σ_ω!

TODO: docstring
- provide `Σ` with `ν` range (third rank) half size in order to only compute positive fermionic indices.
"""
function calc_Σ_ω!(Σ::AbstractArray{Complex{Float64},3}, ωindices::AbstractArray{Int,1},
            Q_sp::TQ, Q_ch::TQ,Gνω::GνqT, corr::AbstractArray{Float64,3}, U::Float64, kG::ReducedKGrid, 
            sP::SimulationParameters; lopWarn=false) where TQ <: Union{NonLocalQuantities, ImpurityQuantities}
    Kνωq = Array{ComplexF64, length(gridshape(kG))}(undef, gridshape(kG)...)
    for ωii in 1:length(ωindices)
        ωi = ωindices[ωii]
        ωn = (ωi - sP.n_iω) - 1
        fsp = 1.5 .* (1 .+ U*Q_sp.χ[:,ωi])
        fch = 0.5 .* (1 .- U*view(Q_ch.χ,:,ωi))
        νZero = ν0Index_of_ωIndex(ωi, sP)
        maxn = minimum([νZero + sP.n_iν-1, size(corr,3)])
        if lopWarn && (νZero + sP.n_iν-1 > size(corr,3)) 
            ωn1, νn1 = OneToIndex_to_Freq(ωi, size(corr,3) + 1, sP)
            ωn2, νn2 = OneToIndex_to_Freq(ωi, νZero + sP.n_iν - 1, sP)
            @warn "running out of data for νn = $(νn1) to $(νn2) at ωn = $ωn1, $ωn"
        end
        for (νn,νi) in enumerate(νZero:maxn)
            #TODO: : : : not general enough for arbirtrary grids
            #TODO: preexpand!! 
            expandKArr!(kG,view(Kνωq,:,:,:), view(Q_sp.γ,:,νi,ωi) .* fsp .- view(Q_ch.γ,:,νi,ωi) .* fch .- 1.5 .+ 0.5 .+ view(corr,:,νi,ωii))
            conv_fft1!(kG, view(Σ,:,νn,ωii), Gνω[(νn-1) + ωn + sP.fft_offset], Kνωq)
        end
    end
end

function calc_Σ(Q_sp::NonLocalQuantities, Q_ch::NonLocalQuantities, bubble::BubbleT,
                Gνω::GνqT, FUpDo::AbstractArray{Complex{Float64},3}, kGrid::ReducedKGrid,
                mP::ModelParameters, sP::SimulationParameters)
    ωindices = (sP.dbg_full_eom_omega) ? (1:size(bubble,ω_axis)) : intersect(Q_sp.usable_ω, Q_ch.usable_ω)
    sh_b = DirectSum()#get_sum_helper(ωindices, sP, :b)

    #TODO: parallelization
    corr = Array{Float64,3}(undef,size(bubble,q_axis),size(bubble,ν_axis),length(ωindices))
    Σ_ladder_ω = Array{Complex{Float64},3}(undef,size(bubble,q_axis),sP.n_iν,size(bubble,ω_axis))

    res = nothing
    @timeit to "corr" Σ_correction!(corr, ωindices, bubble, FUpDo, sP)
    @timeit to "corr extend" (sP.tc_type_f != :nothing) && extend_tmp!(corr)
    @timeit to "Σ_ω" calc_Σ_ω!(Σ_ladder_ω, ωindices, Q_sp, Q_ch, Gνω, corr, mP.U, kGrid, sP)
    @timeit to "sum Σ_ω" res = mP.U .* sum_freq(Σ_ladder_ω, [ω_axis], sh_b, mP.β)[:,:,1]
    return  res
end
