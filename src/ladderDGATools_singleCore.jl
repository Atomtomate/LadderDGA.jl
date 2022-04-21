#TODO: implement complex to real fftw
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
            inv!(χννpω, ipiv, work)
            if typeof(sP.χ_helper) <: BSE_Asym_Helpers
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


