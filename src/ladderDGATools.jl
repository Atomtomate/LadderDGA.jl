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

function F_from_χ(χ::AbstractArray{ComplexF64,3}, G::AbstractArray{ComplexF64,1}, n_iω::Int, n_iν::Int, shift, β::Float64)
    F = similar(χ)
    for (i,ωn) in enumerate(-n_iω:n_iω)
        s = -shift*trunc(Int, ωn/2)
        for (j,νn) in enumerate((-n_iν:n_iν-1) .+ s)
        for (k,νpn) in enumerate((-n_iν:n_iν-1) .+ s)
            @inbounds F[j,k,i] = -(χ[j,k,i] + (νn == νpn) * β * get_symm_f(G,νn) * get_symm_f(G,ωn+νn))/(
                         get_symm_f(G,νn) * get_symm_f(G,ωn+νn)
                       * get_symm_f(G,νpn) * get_symm_f(G,ωn+νpn))
        end
        end
    end
    return F
end



#TODO: implement complex to real fftw
#TODO: get rid of selectdim
#TODO: gImp should know about its tail instead of χ₀
function calc_bubble(Gνω::GνqT, kG::ReducedKGrid, mP::ModelParameters, sP::SimulationParameters; local_tail=false)
    #TODO: fix the size (BSE_SC inconsistency)
    data = Array{ComplexF64,3}(undef, length(kG.kMult), 2*(sP.n_iν+sP.n_iν_shell), 2*sP.n_iω+1)
    nd = length(gridshape(kG))
    for ωi in axes(data,ω_axis)
        #TODO: fix the offset (BSE_SC inconsistency)
        for νi in axes(data,ν_axis)
            ωn, νn = OneToIndex_to_Freq(ωi, νi, sP, sP.n_iν_shell)
            conv_fft!(kG, view(data,:,νi,ωi), selectdim(Gνω,nd+1,νn+sP.fft_offset), selectdim(Gνω,nd+1,νn+ωn+sP.fft_offset))
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

"""
    calc_χ_trilex(Γr::ΓT, χ₀, kG::ReducedKGrid, U::Float64, mP, sP)

Solve χ = χ₀ - 1/β² χ₀ Γ χ
⇔ (1 + 1/β² χ₀ Γ) χ = χ₀
⇔      (χ⁻¹ - χ₀⁻¹) = 1/β² Γ

with indices: χ[ω, q] = χ₀[]
"""
function calc_χγ(type::Symbol, Γr::ΓT, χ₀::χ₀T, kG::ReducedKGrid, mP::ModelParameters, sP::SimulationParameters)
    #TODO: find a way to reduce initialization clutter: move lo,up to sum_helper
    #TODO: χ₀ should know about its tail c2, c3
    s = type === :ch ? -1 : 1
    Niν = 2*sP.n_iν
    γ = γT(undef, size(χ₀.data,q_axis), Niν, size(χ₀.data,ω_axis))
    χ = χT(undef, size(χ₀.data,q_axis), size(χ₀.data,ω_axis))
    ωi_range = axes(χ₀.data,ω_axis)
    νi_range = 1:Niν
    qi_range = axes(χ₀.data,q_axis)
    χ_ω = Array{Float64, 1}(undef, size(χ₀.data,ω_axis))
    lo = npartial_sums(sP.sh_f)
    up = Niν - lo + 1 
    fνmax_cache  = Array{_eltype, 1}(undef, lo)
    χννpω = Matrix{_eltype}(undef, Niν, Niν)
    ipiv = Vector{Int}(undef, Niν)
    work = _gen_inv_work_arr(χννpω, ipiv)
    fνmax_cache = _eltype === Float64 ? sP.fνmax_cache_r : sP.fνmax_cache_c

    #TODO: clean up loop definition
    for ωi in ωi_range
        ωn = (ωi - sP.n_iω) - 1
        for qi in qi_range
            χννpω[:,:] = deepcopy(Γr[:,:,ωi])
            for l in νi_range
                #TODO: fix the offset (BSE_SC inconsistency)
                χννpω[l,l] = Γr[l,l,ωi] + 1.0/χ₀.data[qi,sP.n_iν_shell+l,ωi]
            end
            @timeit to "inv" inv!(χννpω, ipiv, work)
            @timeit to "χ Impr." if typeof(sP.χ_helper) === BSE_Asym_Helper
                χ[qi, ωi], λ_out = calc_χλ_impr(type, ωn, χννpω, view(χ₀.data,qi,:,ωi), 
                                           mP.U, mP.β, χ₀.asym[qi,ωi], sP.χ_helper);
                γ[qi, :, ωi] = (1 .- s*λ_out) ./ (1 .+ s*mP.U .* χ[qi, ωi])
            else
                if typeof(sP.χ_helper) === BSE_SC_Helper
                    improve_χ!(type, ωi, view(χννpω,:,:,ωi), view(χ₀,qi,:,ωi), mP.U, mP.β, sP.χ_helper);
                end
                χ[qi, ωi] = sum_freq_full!(χννpω, sP.sh_f, mP.β, fνmax_cache, lo, up)
                for νk in axes(χ₀.data,ν_axis)
                    γ[qi, νk, ωi] = sum_freq_full!(view(χννpω,:,νk), sP.sh_f, 1.0, fνmax_cache, lo, 
                                                   up) / (χ₀.data[qi, νk, ωi] * (1.0 + s*mP.U * χ[qi, ωi]))
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


function calc_λ0(χ₀::χ₀T, F::FT, locQ::NonLocalQuantities, mP::ModelParameters, sP::SimulationParameters)

    #TODO: store nu grid in sP?
    Niν = size(F,ν_axis)
    ω_range = 1:size(χ₀.data,ω_axis)
    λ0 = Array{ComplexF64,3}(undef,size(χ₀.data,q_axis),Niν,length(ω_range))

    if typeof(sP.χ_helper) === BSE_Asym_Helper
        λ0[:] = calc_λ0_impr(:sp, -sP.n_iω:sP.n_iω, F, χ₀.data, χ₀.asym, view(locQ.γ,1,:,:), view(locQ.χ,1,:),
                             mP.U, mP.β, sP.χ_helper)
    else
        #TODO: this is not well optimized, but also not often executed
        tmp = Array{ComplexF64, 1}(undef, Niν)
        lo = npartial_sums(sP.sh_f)
        up = Niν - lo + 1 
        for ωi in ω_range
            for νi in 1:Niν
                #TODO: export realview functions?
                v1 = view(F,νi,:,ωi)
                for qi in axes(χ₀.data,q_axis)
                    v2 = view(χ₀.data,qi,:,ωi) 
                    @simd for νpi in 1:Niν 
                        @inbounds tmp[νpi] = v1[νpi] * v2[νpi]
                    end
                    λ0[qi,νi,ωi] = sum_freq_full!(tmp, sP.sh_f, 1.0, sP.fνmax_cache_c, lo, up)/(mP.β^2)
                end
            end
        end
        (sP.tc_type_f != :nothing) && extend_corr!(λ0)
    end
    return λ0
end

function calc_Σ_ω!(Σ::AbstractArray{ComplexF64,3}, Kνωq::Array{ComplexF64}, Kνωq_pre::Array{ComplexF64, 1},
            ωindices::AbstractArray{Int,1},
            Q_sp::NonLocalQuantities, Q_ch::NonLocalQuantities, 
            Gνω::GνqT, λ₀::AbstractArray{ComplexF64,3}, U::Float64, kG::ReducedKGrid, 
            sP::SimulationParameters)
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
        #TODO: manual unrolling is slightly faster. use snoopcompiler and cthulhu.jl to investigate
        for (νi,νn) in enumerate(νZero:maxn)
            v = selectdim(Gνω,nd+1,(νi-1) + ωn + sP.fft_offset)
            if kG.Nk == 1
                Σ[1,νi,ωii] = U*(Q_sp.γ[1,νn,ωi] * fsp[1] - Q_ch.γ[1,νn,ωi] * fch[1] - 1.5 + 0.5 + λ₀[1,νn,ωi]) * v[1]
            else
                Kνωq_pre[:] = U*(Q_sp.γ[qi,νn,ωi] * fsp[qi] - Q_ch.γ[qi,νn,ωi] * fch[qi] - 1.5 + 0.5 + λ₀[1,νn,ωi])
                conv_fft1!(kG, view(Σ,:,νi,ωii), Kνωq_pre, v)
            end
        end
    end
end

function calc_Σ(Q_sp::NonLocalQuantities, Q_ch::NonLocalQuantities, λ₀::AbstractArray{_eltype,3},
                Gνω::GνqT, Fsp::FT, kG::ReducedKGrid,
                mP::ModelParameters, sP::SimulationParameters; pre_expand=true)
    if (size(Q_sp.χ,1) != size(Q_ch.χ,1)) || (size(Q_sp.χ,1) != length(kG.kMult))
        @error "q Grids not matching"
    end
    @warn "Selfenergie now contains Hartree term and is cut to νmax = length(usable_ω)/3!"
    Σ_hartree = mP.n * mP.U/2.0;
    Nq = size(Q_sp.χ,1)
    Nω = size(Q_sp.χ,2)
    ωrange = -sP.n_iω:sP.n_iω
    ωindices = (sP.dbg_full_eom_omega) ? (1:Nω) : intersect(Q_sp.usable_ω, Q_ch.usable_ω)
    νmax = floor(Int,length(ωindices)/3)

    Kνωq = Array{ComplexF64, length(gridshape(kG))}(undef, gridshape(kG)...)
    Kνωq_pre = Array{ComplexF64, 1}(undef, length(kG.kMult))
    #TODO: implement real fft and make _pre real
    Σ_ladder_ω = Array{Complex{Float64},3}(undef,Nq, sP.n_iν, length(ωrange))#OffsetArray( 
                             #Array{Complex{Float64},3}(undef,Nq, sP.n_iν, length(ωrange)),
                             #1:Nq, 0:n_iν, ωrange)
    @timeit to "Σ_ω" calc_Σ_ω!(Σ_ladder_ω, Kνωq, Kνωq_pre, ωindices, Q_sp, Q_ch, Gνω, λ₀, mP.U, kG, sP)
    #TODO: *U should be in calc_Sigma_w
    @timeit to "sum Σ_ω" res = sum(Σ_ladder_ω, dims=[3])[:,:,1] ./ mP.β .+ Σ_hartree
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
