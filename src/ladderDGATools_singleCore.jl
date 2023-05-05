# ==================================================================================================== #
#                                        ladderDGATools.jl                                             #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#   ladder DΓA related functions                                                                       #
# -------------------------------------------- TODO -------------------------------------------------- #
#   Cleanup                                                                                            #
# ==================================================================================================== #

# ========================================== Transformations =========================================
"""
    λ_from_γ(type::Symbol, γ::γT, χ::χT, U::Float64)

TODO: documentation
"""
function λ_from_γ(type::Symbol, γ::γT, χ::χT, U::Float64)
    s = (type == :d) ? -1 : 1
    res = similar(γ.data)
    for ωi in 1:size(γ,3)
        for qi in 1:size(γ,1)
            res[qi,:,ωi] = s .* view(γ,qi,:,ωi) .* (1 .+ s*U .* χ.data[qi, ωi]) .- 1
        end
    end
    return res
end


"""
    F_from_χ(type::Symbol, h::lDΓAHelper; diag_term=true)
    F_from_χ(χ::AbstractArray{ComplexF64,3}, G::AbstractArray{ComplexF64,1}, sP::SimulationParameters, β::Float64; diag_term=true)

TODO: documentation
"""
function F_from_χ(type::Symbol, h::lDΓAHelper; diag_term=true)
    type != :m && error("only F_m tested!")
    F_from_χ(h.χDMFT_m, h.gImp[1,:], h.sP, h.mP.β; diag_term=diag_term)
end
function F_from_χ(χ::AbstractArray{ComplexF64,3}, G::AbstractVector{ComplexF64}, sP::SimulationParameters, β::Float64; diag_term=true)
    F = similar(χ)
    for ωi in 1:size(F,3)
    for νpi in 1:size(F,2)
        ωn, νpn = OneToIndex_to_Freq(ωi, νpi, sP)
        for νi in 1:size(F,1)
            _, νn = OneToIndex_to_Freq(ωi, νi, sP)
            F[νi,νpi,ωi] = -(χ[νi,νpi,ωi] + (νn == νpn && diag_term) * β * G[νn] * G[ωn+νn])/(
                                          G[νn] * G[ωn+νn] * G[νpn] * G[ωn+νpn])
        end
        end
    end
    return F
end
# ========================================== Correction Term =========================================
"""
    calc_λ0(χ₀::χ₀T, h::lDΓAHelper)
    calc_λ0(χ₀::χ₀T, Fr::FT, h::lDΓAHelper)
    calc_λ0(χ₀::χ₀T, Fr::FT, χ::χT, γ::γT, mP::ModelParameters, sP::SimulationParameters)

Correction term, TODO: documentation
"""
function calc_λ0(χ₀::χ₀T, h::lDΓAHelper)
    F_m   = F_from_χ(:m, h);
    calc_λ0(χ₀, F_m, h)
end

function calc_λ0(χ₀::χ₀T, Fr::FT, h::lDΓAHelper)
    calc_λ0(χ₀, Fr, h.χ_m_loc, h.γ_m_loc, h.mP, h.sP)
end

function calc_λ0(χ₀::χ₀T, Fr::FT, χ::χT, γ::γT, mP::ModelParameters, sP::SimulationParameters)
    #TODO: store nu grid in sP?
    Niν = size(Fr,ν_axis)
    Nq  = size(χ₀.data, χ₀.axis_types[:q])
    ω_range = 1:size(χ₀.data,ω_axis)
    λ0 = Array{ComplexF64,3}(undef,size(χ₀.data,q_axis),Niν,length(ω_range))

    if typeof(sP.χ_helper) <: BSE_Asym_Helpers
       λ0[:] = calc_λ0_impr(:m, -sP.n_iω:sP.n_iω, Fr, χ₀.data, χ₀.asym, view(γ.data,1,:,:), view(χ.data,1,:),
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
"""
    calc_bubble(Gνω::GνqT, Gνω_r::GνqT, kG::KGrid, mP::ModelParameters, sP::SimulationParameters; local_tail=false)
    calc_bubble(h::lDΓAHelper, kG::KGrid, mP::ModelParameters, sP::SimulationParameters; local_tail=false)

TODO: documentation
"""
function calc_bubble(h::lDΓAHelper; local_tail=false)
    calc_bubble(h.gLoc_fft, h.gLoc_rfft, h.kG, h.mP, h.sP, local_tail=local_tail)
end

function calc_bubble(Gνω::GνqT, Gνω_r::GνqT, kG::KGrid, mP::ModelParameters, sP::SimulationParameters; local_tail=false)
    #TODO: fix the size (BSE_SC inconsistency)
    data = Array{ComplexF64,3}(undef, length(kG.kMult), 2*(sP.n_iν+sP.n_iν_shell), 2*sP.n_iω+1)
    νdim = ndims(Gνω) > 2 ? length(gridshape(kG))+1 : 2 # TODO: this is a fallback for gImp
    for (ωi,ωn) in enumerate(-sP.n_iω:sP.n_iω)
        νrange = ((-(sP.n_iν+sP.n_iν_shell)):(sP.n_iν+sP.n_iν_shell-1)) .- trunc(Int,sP.shift*ωn/2)
        #TODO: fix the offset (BSE_SC inconsistency)
        for (νi,νn) in enumerate(νrange)
            conv_fft!(kG, view(data,:,νi,ωi), selectdim(Gνω,νdim,νn), selectdim(Gνω_r,νdim,νn+ωn))
            data[:,νi,ωi] .*= -mP.β
        end
    end
    #TODO: not necessary after real fft
    data = _eltype === Float64 ? real.(data) : data
    return χ₀T(data, kG, -sP.n_iω:sP.n_iω, sP.n_iν, sP.shift, mP, local_tail=local_tail) 
end

"""
    calc_χγ(type::Symbol, h::lDΓAHelper, χ₀::χ₀T)
    calc_χγ(type::Symbol, Γr::ΓT, χ₀::χ₀T, kG::KGrid, mP::ModelParameters, sP::SimulationParameters)

Calculates susceptibility and triangular vertex in `type` channel. See [`calc_χγ_par`](@ref calc_χγ_par) for parallel calculation.

This method solves the following equation:
``
\\chi_r = \\chi_0 - \\frac{1}{\\beta^2} \\chi_0 \\Gamma_r \\chi_r \\\\
\\Leftrightarrow (1 + \\frac{1}{\\beta^2} \\chi_0 \\Gamma_r) = \\chi_0 \\\\
\\Leftrightarrow (\\chi^{-1}_r - \\chi^{-1}_0) = \\frac{1}{\\beta^2} \\Gamma_r
``
"""
function calc_χγ(type::Symbol, h::lDΓAHelper, χ₀::χ₀T)
    calc_χγ(type, getfield(h,Symbol("Γ_$(type)")), χ₀, h.kG, h.mP, h.sP)
end

function calc_χγ(type::Symbol, Γr::ΓT, χ₀::χ₀T, kG::KGrid, mP::ModelParameters, sP::SimulationParameters)
    #TODO: find a way to reduce initialization clutter: move lo,up to sum_helper
    #TODO: χ₀ should know about its tail c2, c3
    s = if type == :d 
        -1 
    elseif type == :m
        1
    else
        error("Unkown type")
    end

    Nν = 2*sP.n_iν
    Nq  = length(kG.kMult)
    Nω  = size(χ₀.data,ω_axis)
    #TODO: use predifened ranks for Nq,... cleanup definitions
    γ = Array{ComplexF64,3}(undef, Nq, Nν, Nω)
    χ = Array{Float64,2}(undef, Nq, Nω)
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
                χ[qi, ωi] = real(calc_χλ_impr!(λ_cache, type, ωn, χννpω, view(χ₀.data,qi,:,ωi), 
                                               mP.U, mP.β, χ₀.asym[qi,ωi], sP.χ_helper));
                γ[qi, :, ωi] = (1 .- s*λ_cache) ./ (1 .+ s*mP.U .* χ[qi, ωi])
            else
                if typeof(sP.χ_helper) === BSE_SC_Helper
                    improve_χ!(type, ωi, view(χννpω,:,:,ωi), view(χ₀,qi,:,ωi), mP.U, mP.β, sP.χ_helper);
                end
                χ[qi,ωi] = sum(χννpω)/mP.β^2
                for νk in νi_range
                    γ[qi,νk,ωi] = sum(view(χννpω,:,νk))/(χ₀.data[qi,νk,ωi] * (1.0 + s*mP.U * χ[qi,ωi]))
                end
            end
        end
        #TODO: write macro/function for ths "real view" beware of performance hits
        #v = _eltype === Float64 ? view(χ,:,ωi) : @view reinterpret(Float64,view(χ,:,ωi))[1:2:end]
        v = view(χ,:,ωi)
        χ_ω[ωi] = kintegrate(kG, v)
    end
    log_q0_χ_check(kG, sP, χ, type)

    return χT(χ, mP.β, tail_c=[0,0,mP.Ekin_DMFT]), γT(γ)
end

#TODO: THIS NEEDS CLEANUP!
function conv_tmp!(res::AbstractVector{ComplexF64}, kG::KGrid, arr1::Vector{ComplexF64}, GView::AbstractArray{ComplexF64,N})::Nothing where N
    if Nk(kG) == 1 
        res[:] = arr1 .* GView
    else
        expandKArr!(kG, kG.cache1, arr1)
        mul!(kG.cache1, kG.fftw_plan, kG.cache1)
        for i in eachindex(kG.cache1)
            kG.cache1[i] *= GView[i]
        end
        kG.fftw_plan \ kG.cache1
        Dispersions.conv_post!(kG, res, kG.cache1)
    end
    return nothing
end


function calc_Σ_ω!(eomf::Function, Σ_ω::OffsetArray{ComplexF64,3}, Kνωq_pre::Vector{ComplexF64},
            χ_m::χT, γ_m::γT, χ_d::χT, γ_d::γT,
            Gνω::GνqT, λ₀::AbstractArray{ComplexF64,3}, U::Float64, kG::KGrid, 
            sP::SimulationParameters)

    νdim = ndims(Gνω) > 2 ? length(gridshape(kG))+1 : 2 # TODO: this is a fallback for gIm
    fill!(Σ_ω, zero(ComplexF64))
    for (ωi,ωn) in enumerate(axes(Σ_ω,3))
        νZero = ν0Index_of_ωIndex(ωi, sP)
        maxn = minimum([size(γ_d,ν_axis), νZero + size(Σ_ω, 2) - 1])
        νlist = νZero:maxn
        length(νlist) > size(Σ_ω,2) && (νlist = νlist[1:size(Σ_ω,2)])
        for (νii,νi) in enumerate(νlist)
            for qi in 1:size(Σ_ω,q_axis)
                Kνωq_pre[qi] = eomf(U, γ_m[qi,νi,ωi], γ_d[qi,νi,ωi], χ_m[qi,ωi], χ_d[qi,ωi], λ₀[qi,νi,ωi])
            end
            #TODO: find a way to not unroll this!
            conv_tmp!(view(Σ_ω,:,νii-1,ωn), kG, Kνωq_pre, selectdim(Gνω,νdim,(νii-1) + ωn))
        end
    end
end

function calc_Σ_ω!(eomf::Function, Σ_ladder::OffsetMatrix{ComplexF64}, Kνωq_pre::Vector{ComplexF64},
            χ_m::χT, γ_m::γT, χ_d::χT, γ_d::γT,
            Gνω::GνqT, λ₀::AbstractArray{ComplexF64,3}, U::Float64, kG::KGrid, 
            sP::SimulationParameters)

    νdim = ndims(Gνω) > 2 ? length(gridshape(kG))+1 : 2 # TODO: this is a fallback for gIm
    fill!(Σ_ladder, zero(ComplexF64))
    ω_axis = χ_m.indices_ω 
    for (ωi,ωn) in enumerate(ω_axis)
        νZero = ν0Index_of_ωIndex(ωi, sP)
        maxn = minimum([size(γ_d,ν_axis), νZero + size(Σ_ladder, 2) - 1])
        νlist = νZero:maxn
        length(νlist) > size(Σ_ladder,2) && (νlist = νlist[1:size(Σ_ladder,2)])
        for (νii,νi) in enumerate(νlist)
            for qi in 1:size(Σ_ladder,1)
                Kνωq_pre[qi] = eomf(U, γ_m[qi,νi,ωi], γ_d[qi,νi,ωi], χ_m[qi,ωi], χ_d[qi,ωi], λ₀[qi,νi,ωi])
            end
            #TODO: find a way to not unroll this!
            conv_tmp_add!(view(Σ_ladder,:,νii-1), kG, Kνωq_pre, selectdim(Gνω,νdim,(νii-1) + ωn))
        end
    end
end

function calc_Σ!(Σ_ladder::OffsetMatrix{ComplexF64}, Kνωq_pre::Vector{ComplexF64},
                χ_m::χT, γ_m::γT, χ_d::χT, γ_d::γT, 
                χ_m_sum::Union{Float64,ComplexF64}, λ₀::AbstractArray{_eltype,3},
                Gνω::GνqT, kG::KGrid,
                mP::ModelParameters, sP::SimulationParameters; tc::Bool=true)::Nothing
    Σ_hartree = mP.n * mP.U/2.0;
    calc_Σ_ω!(eom, Σ_ladder, Kνωq_pre, χ_m, γ_m, χ_d, γ_d, Gνω, λ₀, mP.U, kG, sP)
    tail_correction = (tc ? mP.U^2 * (sum_kω(kG, χ_m) - χ_m_sum) : 0) ./ iν_array(mP.β, collect(axes(Σ_ladder)[2]))
    Σ_ladder.parent[:,:] = Σ_ladder.parent[:,:] ./ mP.β .+ reshape(tail_correction, 1, length(tail_correction)) .+ Σ_hartree
    return nothing
end

function calc_Σ!(Σ_ladder::OffsetMatrix{ComplexF64}, Σ_ladder_ω::OffsetArray{ComplexF64,3}, Kνωq_pre::Vector{ComplexF64},
                χ_m::χT, γ_m::γT, χ_d::χT, γ_d::γT, 
                χ_m_sum::Union{Float64,ComplexF64}, λ₀::AbstractArray{_eltype,3},
                Gνω::GνqT, kG::KGrid,
                mP::ModelParameters, sP::SimulationParameters; tc::Bool=true)::Nothing

    Σ_hartree = mP.n * mP.U/2.0;
    calc_Σ_ω!(eom, Σ_ladder_ω, Kνωq_pre, χ_m, γ_m, χ_d, γ_d, Gνω, λ₀, mP.U, kG, sP)
    sum!(Σ_ladder, Σ_ladder_ω)
    tail_correction = (tc ? mP.U^2 * (sum_kω(kG, χ_m) - χ_m_sum) : 0) ./ iν_array(mP.β, collect(axes(Σ_ladder)[2]))
    Σ_ladder.parent[:,:] = Σ_ladder.parent[:,:] ./ mP.β .+ reshape(tail_correction, 1, length(tail_correction)) .+ Σ_hartree
    return nothing
end

function calc_Σ(χ_m::χT, γ_m::γT, χ_d::χT, γ_d::γT, 
                λ₀::AbstractArray{_eltype,3}, h::lDΓAHelper;
                νmax::Int = h.sP.n_iν, λm::Float64=0.0, λd::Float64=0.0, tc::Bool=true)
    calc_Σ(χ_m, γ_m, χ_d, γ_d, h.χloc_m_sum, λ₀, h.gLoc_rfft, h.kG, h.mP, h.sP, νmax=νmax, λm=λm, λd=λd, tc=tc)
end

function calc_Σ(χ_m::χT, γ_m::γT, χ_d::χT, γ_d::γT, 
                χ_m_sum::Union{Float64,ComplexF64}, λ₀::AbstractArray{_eltype,3},
                Gνω::GνqT, kG::KGrid,
                mP::ModelParameters, sP::SimulationParameters; 
                νmax::Int = sP.n_iν,
                λm::Float64=0.0, λd::Float64=0.0, tc::Bool=true)
    χ_m.λ != 0 && λm != 0 && error("Stopping self energy calculation: λm = $λm AND χ_m.λ = $(χ_m.λ)")
    χ_d.λ != 0 && λd != 0 && error("Stopping self energy calculation: λd = $λd AND χ_d.λ = $(χ_d.λ)")
    Nq, Nω = size(χ_m)
    ωrange::UnitRange{Int}   = -sP.n_iω:sP.n_iω

    Kνωq_pre::Vector{ComplexF64} = Vector{ComplexF64}(undef, length(kG.kMult))
    Σ_ladder   = OffsetArray(Array{Complex{Float64},2}(undef,Nq, νmax), 1:Nq, 0:νmax-1)
    
    λm != 0.0 && χ_λ!(χ_m, λm)
    λd != 0.0 && χ_λ!(χ_d, λd)

    calc_Σ!(Σ_ladder, Kνωq_pre, χ_m, γ_m, χ_d, γ_d, χ_m_sum, λ₀, Gνω, kG, mP, sP, tc=tc)

    λm != 0.0 && reset!(χ_m)
    λd != 0.0 && reset!(χ_d)
    return Σ_ladder
end

function calc_Σ_parts(χ_m::χT, γ_m::γT, χ_d::χT, γ_d::γT,
                      χ_m_sum::Union{Float64,ComplexF64}, λ₀::AbstractArray{_eltype,3},
                      Gνω::GνqT, kG::KGrid,
                      mP::ModelParameters, sP::SimulationParameters;
                      λm::Float64=0.0, λd::Float64=0.0)
    Σ_hartree = mP.n * mP.U/2.0;
    Nq, Nω = size(χ_m)
    ωrange::UnitRange{Int} = -sP.n_iω:sP.n_iω
    ωindices::UnitRange{Int} = (sP.dbg_full_eom_omega) ? (1:Nω) : intersect(χ_m.usable_ω, χ_d.usable_ω)

    Kνωq_pre::Vector{ComplexF64} = Vector{ComplexF64}(undef, length(kG.kMult))
    #TODO: implement real fft and make _pre real
    Σ_ladder_ω = OffsetArray(Array{Complex{Float64},3}(undef,Nq, sP.n_iν, length(ωrange)),
                              1:Nq, 0:sP.n_iν-1, ωrange)
    Σ_ladder = Array{Complex{Float64},3}(undef,Nq, sP.n_iν, 7)

    λm != 0.0 && χ_λ!(χ_m, λm)
    λd != 0.0 && χ_λ!(χ_d, λd)

    tail_correction = mP.U^2 .* (sum_kω(kG, χ_m) - χ_m_sum) ./ iν_array(mP.β, 0:sP.n_iν-1)
    calc_Σ_ω!(eom_χ_m, Σ_ladder_ω, Kνωq_pre, χ_m, γ_m, χ_d, γ_d, Gνω, λ₀, mP.U, kG, sP)
    Σ_ladder[:,:,1] = dropdims(sum(Σ_ladder_ω, dims=[3]),dims=3) ./ mP.β
    calc_Σ_ω!(eom_γ_m, Σ_ladder_ω, Kνωq_pre, χ_m, γ_m, χ_d, γ_d, Gνω, λ₀, mP.U, kG, sP)
    Σ_ladder[:,:,2] = dropdims(sum(Σ_ladder_ω, dims=[3]),dims=3) ./ mP.β
    calc_Σ_ω!(eom_χ_d, Σ_ladder_ω, Kνωq_pre, χ_m, γ_m, χ_d, γ_d, Gνω, λ₀, mP.U, kG, sP)
    Σ_ladder[:,:,3] = dropdims(sum(Σ_ladder_ω, dims=[3]),dims=3) ./ mP.β 
    calc_Σ_ω!(eom_γ_d, Σ_ladder_ω, Kνωq_pre, χ_m, γ_m, χ_d, γ_d, Gνω, λ₀, mP.U, kG, sP)
    Σ_ladder[:,:,4] = dropdims(sum(Σ_ladder_ω, dims=[3]),dims=3) ./ mP.β 
    calc_Σ_ω!(eom_rest_01, Σ_ladder_ω, Kνωq_pre, χ_m, γ_m, χ_d, γ_d, Gνω, λ₀, mP.U, kG, sP)
    Σ_ladder[:,:,5] = dropdims(sum(Σ_ladder_ω, dims=[3]),dims=3) ./ mP.β
    calc_Σ_ω!(eom_rest, Σ_ladder_ω, Kνωq_pre, χ_m, γ_m, χ_d, γ_d, Gνω, λ₀, mP.U, kG, sP)
    Σ_ladder[:,:,6] = dropdims(sum(Σ_ladder_ω, dims=[3]),dims=3) ./ mP.β .+ Σ_hartree
    for qi in 1:size(Σ_ladder,1)
        Σ_ladder[qi,:,7] .= tail_correction 
    end
    λm != 0.0 && reset!(χ_m)
    λd != 0.0 && reset!(χ_d)

    return  OffsetArray(Σ_ladder, 1:Nq, 0:sP.n_iν-1, 1:7)
end
