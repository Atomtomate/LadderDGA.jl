# ==================================================================================================== #
#                                           helpers.jl                                                 #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#   General purpose helper functions for the ladder DΓA code.                                          #
# -------------------------------------------- TODO -------------------------------------------------- #
#   Add documentation for all functions                                                                #
#   Cleanup of setup function                                                                          #
# ==================================================================================================== #


# =============================================== Setup ==============================================

"""
    setup_LDGA(kGridStr::Tuple{String,Int}, mP::ModelParameters, sP::SimulationParameters, env::EnvironmentVars [; local_correction=true])

Computes all needed objects for DΓA calculations.

Returns: 
-------------
NTuple{16, Any}[
    Σ_ladderLoc, 
    Σ_loc, 
    imp_density, 
    kGrid,
    gLoc,
    gLoc_fft,
    gLoc_rfft, 
    Γ_m,
    Γ_d, 
    χDMFTm,
    χDMFTd,
    χ_m_loc,
    γ_m_loc,
    χ_d_loc,
    γ_d_loc,
    χ₀Loc,
    gImp
]
"""
function setup_LDGA(kGridStr::Tuple{String,Int}, mP::ModelParameters, sP::SimulationParameters, env::EnvironmentVars; local_correction=true)

    @info "Setting up calculation for kGrid $(kGridStr[1]) of size $(kGridStr[2])"
    @timeit to "gen kGrid" kG    = gen_kGrid(kGridStr[1], kGridStr[2])
    @timeit to "load f" χDMFT_m, χDMFT_d, Γ_m, Γ_d, gImp_in, Σ_loc = jldopen(env.inputVars, "r") do f 
        #TODO: permute dims creates inconsistency between user input and LadderDGA.jl data!!
        Ns = typeof(sP.χ_helper) === BSE_SC_Helper ? sP.χ_helper.Nν_shell : 0
        χDMFT_m = zeros(_eltype, 2*sP.n_iν, 2*sP.n_iν, 2*sP.n_iω+1)
        χDMFT_m[(Ns+1):(end-Ns),(Ns+1):(end-Ns),:] = permutedims(_eltype === Float64 ? real.(f["χDMFTsp"]) : f["χDMFTsp"], (2,3,1))
        χDMFT_d = zeros(_eltype, 2*sP.n_iν, 2*sP.n_iν, 2*sP.n_iω+1)
        χDMFT_d[(Ns+1):(end-Ns),(Ns+1):(end-Ns),:] = permutedims(_eltype === Float64 ? real.(f["χDMFTch"]) : f["χDMFTch"], (2,3,1))
        Γ_m = zeros(_eltype, 2*sP.n_iν, 2*sP.n_iν, 2*sP.n_iω+1)
        Γ_m[(Ns+1):(end-Ns),(Ns+1):(end-Ns),:] = permutedims(_eltype === Float64 ? real.(-f["Γsp"]) : -f["Γsp"], (2,3,1))
        Γ_d = zeros(_eltype, 2*sP.n_iν, 2*sP.n_iν, 2*sP.n_iω+1)
        Γ_d[(Ns+1):(end-Ns),(Ns+1):(end-Ns),:] = permutedims(_eltype === Float64 ? real.(-f["Γch"]) : -f["Γch"], (2,3,1))
        gImp, Σ_loc = if haskey(f, "g0")
            gImp = f["gImp"]
            g0 = f["g0"]
            Σ_loc = Σ_Dyson(g0, gImp)
            gImp, Σ_loc
        else
            gImp = f["gImp"]
            Σ_loc = f["SigmaLoc"]
            gImp, Σ_loc
        end
        χDMFT_m, χDMFT_d, Γ_m, Γ_d, gImp, Σ_loc
    end

    @timeit to "Compute GLoc" begin
        rm = maximum(abs.(sP.fft_range))
        t = cat(conj(reverse(gImp_in[1:rm])),gImp_in[1:rm], dims=1)
        gImp = OffsetArray(reshape(t,1,length(t)),1:1,-length(gImp_in[1:rm]):length(gImp_in[1:rm])-1)
        gLoc = OffsetArray(Array{ComplexF64,2}(undef, length(kG.kMult), length(sP.fft_range)), 1:length(kG.kMult), sP.fft_range)
        gLoc_i = Array{ComplexF64, length(gridshape(kG))}(undef, gridshape(kG)...)
        gLoc_fft = OffsetArray(Array{ComplexF64,2}(undef, kG.Nk, length(sP.fft_range)), 1:kG.Nk, sP.fft_range)
        gLoc_rfft = OffsetArray(Array{ComplexF64,2}(undef, kG.Nk, length(sP.fft_range)), 1:kG.Nk, sP.fft_range)
        ϵk_full = expandKArr(kG, kG.ϵkGrid)[:]
        for νi in sP.fft_range
            Σ_loc_i = (νi < 0) ? conj(Σ_loc[-νi]) : Σ_loc[νi + 1]
            gLoc_i  = reshape(map(ϵk -> G_from_Σ(νi, mP.β, mP.μ, ϵk, Σ_loc_i), ϵk_full), gridshape(kG))
            gLoc[:,νi] = reduceKArr(kG, gLoc_i)
            gLoc_fft[:,νi] .= fft(gLoc_i)[:]
            gLoc_rfft[:,νi] .= fft(reverse(gLoc_i))[:]
        end
    end

    @timeit to "local correction" begin
        #TODO: unify checks
        kGridLoc = gen_kGrid(kGridStr[1], 1)
        F_m   = F_from_χ(χDMFT_m, gImp[1,:], sP, mP.β);
        χ₀Loc = calc_bubble(gImp, gImp, kGridLoc, mP, sP, local_tail=true);
        χ_m_loc, γ_m_loc = calc_χγ(:m, Γ_m, χ₀Loc, kGridLoc, mP, sP);
        χ_d_loc, γ_d_loc = calc_χγ(:d, Γ_d, χ₀Loc, kGridLoc, mP, sP);
        λ₀Loc = calc_λ0(χ₀Loc, F_m, χ_m_loc, γ_m_loc, mP, sP)
        Σ_ladderLoc = calc_Σ(χ_m_loc, γ_m_loc, χ_d_loc, γ_d_loc, 0.0, λ₀Loc, gImp, kGridLoc, mP, sP)
        any(isnan.(Σ_ladderLoc)) && @error "Σ_ladderLoc contains NaN"

        χLoc_m_ω = similar(χDMFT_m, size(χDMFT_m,3))
        χLoc_d_ω = similar(χDMFT_d, size(χDMFT_d,3))
        for ωi in axes(χDMFT_m,ω_axis)
            if typeof(sP.χ_helper) === BSE_SC_Helper
                @error "SC not fully implemented yet"
                @info "Using asymptotics improvement for large ν, ν' of χ_DMFT with shell size of $(sP.n_iν_shell)"
                improve_χ!(:m, ωi, view(χDMFT_m,:,:,ωi), view(χ₀Loc,1,:,ωi), mP.U, mP.β, sP.χ_helper);
                improve_χ!(:d, ωi, view(χDMFT_d,:,:,ωi), view(χ₀Loc,1,:,ωi), mP.U, mP.β, sP.χ_helper);
            end
            if typeof(sP.χ_helper) <: BSE_Asym_Helpers
                χLoc_m_ω[ωi] = χ_m_loc[ωi]
                χLoc_d_ω[ωi] = χ_d_loc[ωi]
            else
                χLoc_m_ω[ωi] = sum(view(χDMFT_m,:,:,ωi))/mP.β^2
                χLoc_d_ω[ωi] = sum(view(χDMFT_d,:,:,ωi))/mP.β^2
            end
        end
    end

    @timeit to "ranges/imp. dens." begin
        usable_loc_m = find_usable_χ_interval(real(χLoc_m_ω), reduce_range_prct=sP.usable_prct_reduction)
        usable_loc_d = find_usable_χ_interval(real(χLoc_d_ω), reduce_range_prct=sP.usable_prct_reduction)
        loc_range = intersect(usable_loc_m, usable_loc_d)

        iωn = 1im .* 2 .* (-sP.n_iω:sP.n_iω) .* π ./ mP.β

        χLoc_m = sum(subtract_tail(χLoc_m_ω[usable_loc_m], mP.Ekin_DMFT, iωn[usable_loc_m], 2))/mP.β -mP.Ekin_DMFT*mP.β/12
        χLoc_d = sum(subtract_tail(χLoc_d_ω[usable_loc_d], mP.Ekin_DMFT, iωn[usable_loc_d], 2))/mP.β -mP.Ekin_DMFT*mP.β/12

        χupup_DMFT_ω = 0.5 * (χLoc_m_ω + χLoc_d_ω)[loc_range]
        χupup_DMFT_ω_sub = subtract_tail(χupup_DMFT_ω, mP.Ekin_DMFT, iωn[loc_range], 2)

        imp_density_ntc = real(sum(χupup_DMFT_ω))/mP.β
        imp_density = real(sum(χupup_DMFT_ω_sub))/mP.β -mP.Ekin_DMFT*mP.β/12

        #TODO: update output
        @info """Inputs Read. Starting Computation.
          Local susceptibilities with ranges are:
          χLoc_m($(usable_loc_m)) = $(printr_s(χLoc_m)), χLoc_d($(usable_loc_d)) = $(printr_s(χLoc_d))
          sum χupup check (plain ?≈? tail sub ?≈? imp_dens ?≈? n/2 (1-n/2)): $(imp_density_ntc) ?=? $(0.5 .* real(χLoc_m + χLoc_d)) ?≈? $(imp_density) ≟ $(mP.n/2 * ( 1 - mP.n/2))"
          """
    end

    workerpool = get_workerpool()
    @sync begin
    for w in workers()
        @async remotecall_fetch(LadderDGA.update_wcache!,w,:G_fft, gLoc_fft; override=true)
        @async remotecall_fetch(LadderDGA.update_wcache!,w,:G_fft_reverse, gLoc_rfft; override=true)
        @async remotecall_fetch(LadderDGA.update_wcache!,w,:kG, kGridStr; override=true)
        @async remotecall_fetch(LadderDGA.update_wcache!,w,:mP, mP; override=true)
        @async remotecall_fetch(LadderDGA.update_wcache!,w,:sP, sP; override=true)
    end
    end

    return Σ_ladderLoc, Σ_loc, imp_density, kG, gLoc, gLoc_fft, gLoc_rfft, Γ_m, Γ_d, χDMFT_m, χDMFT_d, χ_m_loc, γ_m_loc, χ_d_loc, γ_d_loc, χ₀Loc, gImp
end

# ========================================== Index Functions =========================================
"""
    νnGrid(ωn::Int, sP::SimulationParameters)

Calculates grid of fermionic Matsubara frequencies for given bosonic frequency `ωn` (including shift, if set through `sP`).
"""
νnGrid(ωn::Int, sP::SimulationParameters) = ((-sP.n_iν-sP.n_iν_shell):(sP.n_iν+sP.n_iν_shell-1)) .- sP.shift*trunc(Int,ωn/2)

"""
    q0_index(kG::KGrid)   

Index of zero k-vector.
"""
q0_index(kG::KGrid) = findfirst(x -> all(x .≈ zeros(length(gridshape(kG)))), kG.kGrid)

"""
    ω0_index(sP::SimulationParameters)
    ω0_index(χ::[χT or AbstractMatrix])

Index of ω₀ frequency. 
"""
ω0_index(sP::SimulationParameters) = sP.n_iω+1
ω0_index(χ::χT) = ω0_index(χ.data)
ω0_index(χ::AbstractMatrix) = ceil(Int64, size(χ,2)/2)

"""
    OneToIndex_to_Freq(ωi::Int, νi::Int, sP::SimulationParameters)

Converts `(1:N,1:N)` index tuple for bosonic (`ωi`) and fermionic (`νi`) frequency to
Matsubara frequency number. If the array has a `ν` shell (for example for tail
improvements) this will also be taken into account by providing `Nν_shell`.
"""
function OneToIndex_to_Freq(ωi::Int, νi::Int, sP::SimulationParameters)
    ωn = ωi-sP.n_iω-1
    νn = (νi-sP.n_iν-1) - sP.shift*trunc(Int,ωn/2)
    return ωn, νn
end

"""
    ν0Index_of_ωIndex(ωi::Int[, sP])::Int

Calculates index of zero fermionic Matsubara frequency (which may depend on the bosonic frequency). 
`ωi` is the index (i.e. starting with 1) of the bosonic Matsubara frequency.
"""
ν0Index_of_ωIndex(ωi::Int, sP::SimulationParameters)::Int = sP.n_iν + sP.shift*(trunc(Int, (ωi - sP.n_iω - 1)/2)) + 1

"""
    νi_νngrid_pos(ωi::Int, νmax::Int, sP::SimulationParameters)

Indices for positive fermionic Matsubara frequencies, depinding on `ωi`, the index of the bosonic Matsubara frequency.
"""
function νi_νngrid_pos(ωi::Int, νmax::Int, sP::SimulationParameters)
    ν0Index_of_ωIndex(ωi, sP):νmax
end

# =========================================== Noise Filter ===========================================
"""
    filter_MA(m::Int, X::AbstractArray{T,1}) where T <: Number
    filter_MA!(res::AbstractArray{T,1}, m::Int, X::AbstractArray{T,1}) where T <: Number

Iterated moving average noise filter for inut data. See also [`filter_KZ`](@ref filter_KZ).
"""
function filter_MA(m::Int, X::AbstractArray{T,1}) where T <: Number
    res = deepcopy(X)
    offset = trunc(Int,m/2)
    res[1+offset] = sum(@view X[1:m])/m
    for (ii,i) in enumerate((2+offset):(length(X)-offset))
        res[i] = res[i-1] + (X[m+ii] - X[ii])/m
    end
    return res
end

function filter_MA!(res::AbstractArray{T,1}, m::Int, X::AbstractArray{T,1}) where T <: Number
    offset = trunc(Int,m/2)
    res[1+offset] = sum(@view X[1:m])/m
    for (ii,i) in enumerate((2+offset):(length(X)-offset))
        res[i] = res[i-1] + (X[m+ii] - X[ii])/m
    end
    return res
end

"""
    filter_KZ(m::Int, k::Int, X::AbstractArray{T,1}) where T <: Number

Iterated moving average noise filter for inut data. See also [`filter_MA`](@ref filter_MA).
"""
function filter_KZ(m::Int, k::Int, X::AbstractArray{T,1}) where T <: Number
    res = filter_MA(m, X)
    for ki in 2:k
        res = filter_MA!(res, m, res)
    end
    return res
end

# ======================================== Consistency Checks ========================================
"""
    log_q0_χ_check(kG::KGrid, sP::SimulationParameters, χ::AbstractArray{_eltype,2}, type::Symbol)

TODO: documentation
"""
function log_q0_χ_check(kG::KGrid, sP::SimulationParameters, χ::AbstractArray{Float64,2}, type::Symbol)
    q0_ind = q0_index(kG)
    if q0_ind != nothing
        #TODO: adapt for arbitrary ω indices
        ω_ind = setdiff(1:size(χ,2), sP.n_iω+1)
        @info "$type channel: |∑χ(q=0,ω≠0)| = $(round(abs(sum(view(χ,q0_ind,ω_ind))),digits=12)) ≟ 0"
    end
end

"""
    νi_health(νGrid::AbstractArray{Int}, sP::SimulationParameters)

Returns a list of available bosonic frequencies for each fermionic frequency, given in `νGrid`.
This can be used to estimate the maximum number of usefull frequencies for the equation of motion.
"""
function νi_health(νGrid::AbstractArray{Int}, sP::SimulationParameters)
    t = gen_ν_part(νGrid, sP, 1)[1]
    return [length(filter(x->x[4] == i, t)) for i in unique(getindex.(t,4))]
end
# ============================================== Misc. ===============================================

"""
    reduce_range(range::AbstractArray, red_prct::Float64)

Returns indices for 1D array slice, reduced by `red_prct` % (compared to initial `range`).
Range is symmetrically reduced fro mstart and end.
"""
function reduce_range(range::AbstractArray, red_prct::Float64)
    sub = floor(Int, length(range)/2 * red_prct)
    lst = maximum([last(range)-sub, ceil(Int,length(range)/2 + iseven(length(range)))])
    fst = minimum([first(range)+sub, ceil(Int,length(range)/2)])
    return fst:lst
end

"""
    G_fft(G::GνqT, kG::KGrid, mP::ModelParameters, sP::SimulationParameters)

Calculates fast Fourier transformed lattice Green's functions used for [`calc_bubble`](@ref calc_bubble).
"""
function G_fft(G::GνqT, kG::KGrid, sP::SimulationParameters)
    G_fft = OffsetArray(Array{ComplexF64,2}(undef, kG.Nk, length(sP.fft_range)), 1:kG.Nk, sP.fft_range)
    G_rfft = OffsetArray(Array{ComplexF64,2}(undef, kG.Nk, length(sP.fft_range)), 1:kG.Nk, sP.fft_range)
    G_fft!(G_fft, G, kG, sP.fft_range)
    G_rfft!(G_rfft, G, kG, sP.fft_range)
    return G_fft, G_rfft
end

function G_rfft!(G_rfft::GνqT, G::GνqT, kG::KGrid, fft_range::UnitRange)
    expand_flag = all(collect(axes(G_rfft,2)) .== fft_range)
    for νn in fft_range
        expand_flag && νn < 0 ? expandKArr!(kG, conj(G[:,-νn-1].parent)) : expandKArr!(kG, G[:,νn].parent)
        G_rfft[:,νn] .= fft(reverse(kG.cache1))[:]
    end
    return G_rfft
end

function G_fft!(G_fft::GνqT, G::GνqT, kG::KGrid, fft_range::UnitRange)
    for νn in fft_range
        νn < 0 ? expandKArr!(kG, conj(G[:,-νn-1].parent)) : expandKArr!(kG, G[:,νn].parent)
        G_fft[:,νn] .= fft(kG.cache1)[:]
    end
    return G_fft
end
