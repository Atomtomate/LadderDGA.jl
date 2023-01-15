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
    gLoc_fft,
    gLoc_rfft, 
    Γ_sp,
    Γ_ch, 
    χDMFTsp,
    χDMFTch,
    χ_sp_loc,
    γ_sp_loc,
    χ_ch_loc,
    γ_ch_loc,
    χ₀Loc,
    gImp
]
"""
function setup_LDGA(kGridStr::Tuple{String,Int}, mP::ModelParameters, sP::SimulationParameters, env::EnvironmentVars; local_correction=true)

    @info "Setting up calculation for kGrid $(kGridStr[1]) of size $(kGridStr[2])"
    @timeit to "gen kGrid" kG    = gen_kGrid(kGridStr[1], kGridStr[2])
    @timeit to "load f" χDMFTsp, χDMFTch, Γ_sp, Γ_ch, gImp_in, Σ_loc = jldopen(env.inputVars, "r") do f 
        #TODO: permute dims creates inconsistency between user input and LadderDGA.jl data!!
        Ns = typeof(sP.χ_helper) === BSE_SC_Helper ? sP.χ_helper.Nν_shell : 0
        χDMFTsp = zeros(_eltype, 2*sP.n_iν, 2*sP.n_iν, 2*sP.n_iω+1)
        χDMFTsp[(Ns+1):(end-Ns),(Ns+1):(end-Ns),:] = permutedims(_eltype === Float64 ? real.(f["χDMFTsp"]) : f["χDMFTsp"], (2,3,1))
        χDMFTch = zeros(_eltype, 2*sP.n_iν, 2*sP.n_iν, 2*sP.n_iω+1)
        χDMFTch[(Ns+1):(end-Ns),(Ns+1):(end-Ns),:] = permutedims(_eltype === Float64 ? real.(f["χDMFTch"]) : f["χDMFTch"], (2,3,1))
        Γ_sp = zeros(_eltype, 2*sP.n_iν, 2*sP.n_iν, 2*sP.n_iω+1)
        Γ_sp[(Ns+1):(end-Ns),(Ns+1):(end-Ns),:] = permutedims(_eltype === Float64 ? real.(-f["Γsp"]) : -f["Γsp"], (2,3,1))
        Γ_ch = zeros(_eltype, 2*sP.n_iν, 2*sP.n_iν, 2*sP.n_iω+1)
        Γ_ch[(Ns+1):(end-Ns),(Ns+1):(end-Ns),:] = permutedims(_eltype === Float64 ? real.(-f["Γch"]) : -f["Γch"], (2,3,1))
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
        χDMFTsp, χDMFTch, Γ_sp, Γ_ch, gImp, Σ_loc
    end

    @timeit to "Compute GLoc" begin
        rm = maximum(abs.(sP.fft_range))
        t = cat(conj(reverse(gImp_in[1:rm])),gImp_in[1:rm], dims=1)
        gImp = OffsetArray(reshape(t,1,length(t)),1:1,-length(gImp_in[1:rm]):length(gImp_in[1:rm])-1)
        gLoc_fft = OffsetArray(Array{ComplexF64,2}(undef, kG.Nk, length(sP.fft_range)), 1:kG.Nk, sP.fft_range)
        gLoc_rfft = OffsetArray(Array{ComplexF64,2}(undef, kG.Nk, length(sP.fft_range)), 1:kG.Nk, sP.fft_range)
        ϵk_full = expandKArr(kG, kG.ϵkGrid)[:]
        for νi in sP.fft_range
            Σ_loc_i = (νi < 0) ? conj(Σ_loc[-νi]) : Σ_loc[νi + 1]
            GLoc_νi  = reshape(map(ϵk -> G_from_Σ(νi, mP.β, mP.μ, ϵk, Σ_loc_i), ϵk_full), gridshape(kG))
            gLoc_fft[:,νi] .= fft(GLoc_νi)[:]
            gLoc_rfft[:,νi] .= fft(reverse(GLoc_νi))[:]
        end
    end

    @timeit to "local correction" begin
        #TODO: unify checks
        kGridLoc = gen_kGrid(kGridStr[1], 1)
        Fsp   = F_from_χ(χDMFTsp, gImp[1,:], sP, mP.β);
        χ₀Loc = calc_bubble(gImp, gImp, kGridLoc, mP, sP, local_tail=true);
        χ_sp_loc, γ_sp_loc = calc_χγ(:sp, Γ_sp, χ₀Loc, kGridLoc, mP, sP);
        χ_ch_loc, γ_ch_loc = calc_χγ(:ch, Γ_ch, χ₀Loc, kGridLoc, mP, sP);
        λ₀Loc = calc_λ0(χ₀Loc, Fsp, χ_sp_loc, γ_sp_loc, mP, sP)
        Σ_ladderLoc = calc_Σ(χ_sp_loc, γ_sp_loc, χ_ch_loc, γ_ch_loc, λ₀Loc, gImp, kGridLoc, mP, sP)
        any(isnan.(Σ_ladderLoc)) && @error "Σ_ladderLoc contains NaN"

        χLocsp_ω = similar(χDMFTsp, size(χDMFTsp,3))
        χLocch_ω = similar(χDMFTch, size(χDMFTch,3))
        for ωi in axes(χDMFTsp,ω_axis)
            if typeof(sP.χ_helper) === BSE_SC_Helper
                @error "SC not fully implemented yet"
                @info "Using asymptotics improvement for large ν, ν' of χ_DMFT with shell size of $(sP.n_iν_shell)"
                improve_χ!(:sp, ωi, view(χDMFTsp,:,:,ωi), view(χ₀Loc,1,:,ωi), mP.U, mP.β, sP.χ_helper);
                improve_χ!(:ch, ωi, view(χDMFTch,:,:,ωi), view(χ₀Loc,1,:,ωi), mP.U, mP.β, sP.χ_helper);
            end
            if typeof(sP.χ_helper) <: BSE_Asym_Helpers
                χLocsp_ω[ωi] = χ_sp_loc[ωi]
                χLocch_ω[ωi] = χ_ch_loc[ωi]
            else
                χLocsp_ω[ωi] = sum(view(χDMFTsp,:,:,ωi))/mP.β^2
                χLocch_ω[ωi] = sum(view(χDMFTch,:,:,ωi))/mP.β^2
            end
        end
    end

    @timeit to "ranges/imp. dens." begin
        usable_loc_sp = find_usable_χ_interval(real(χLocsp_ω), reduce_range_prct=sP.usable_prct_reduction)
        usable_loc_ch = find_usable_χ_interval(real(χLocch_ω), reduce_range_prct=sP.usable_prct_reduction)
        loc_range = intersect(usable_loc_sp, usable_loc_ch)

        iωn = 1im .* 2 .* (-sP.n_iω:sP.n_iω) .* π ./ mP.β

        χLocsp = sum(subtract_tail(χLocsp_ω[usable_loc_sp], mP.Ekin_DMFT, iωn[usable_loc_sp], 2))/mP.β -mP.Ekin_DMFT*mP.β/12
        χLocch = sum(subtract_tail(χLocch_ω[usable_loc_ch], mP.Ekin_DMFT, iωn[usable_loc_ch], 2))/mP.β -mP.Ekin_DMFT*mP.β/12

        χupup_DMFT_ω = 0.5 * (χLocsp_ω + χLocch_ω)[loc_range]
        χupup_DMFT_ω_sub = subtract_tail(χupup_DMFT_ω, mP.Ekin_DMFT, iωn[loc_range], 2)

        imp_density_ntc = real(sum(χupup_DMFT_ω))/mP.β
        imp_density = real(sum(χupup_DMFT_ω_sub))/mP.β -mP.Ekin_DMFT*mP.β/12

        #TODO: update output
        @info """Inputs Read. Starting Computation.
          Local susceptibilities with ranges are:
          χLoc_sp($(usable_loc_sp)) = $(printr_s(χLocsp)), χLoc_ch($(usable_loc_ch)) = $(printr_s(χLocch))
          sum χupup check (plain ?≈? tail sub ?≈? imp_dens ?≈? n/2 (1-n/2)): $(imp_density_ntc) ?=? $(0.5 .* real(χLocsp + χLocch)) ?≈? $(imp_density) ≟ $(mP.n/2 * ( 1 - mP.n/2))"
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

    return Σ_ladderLoc, Σ_loc, imp_density, kG, gLoc_fft, gLoc_rfft, Γ_sp, Γ_ch, χDMFTsp, χDMFTch, χ_sp_loc, γ_sp_loc, χ_ch_loc, γ_ch_loc, χ₀Loc, gImp
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
function log_q0_χ_check(kG::KGrid, sP::SimulationParameters, χ::AbstractArray{_eltype,2}, type::Symbol)
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
function G_fft(G::GνqT, kG::KGrid, mP::ModelParameters, sP::SimulationParameters)
    g_fft = OffsetArray(Array{ComplexF64,2}(undef, kG.Nk, length(sP.fft_range)), 1:kG.Nk, sP.fft_range)
    g_rfft = OffsetArray(Array{ComplexF64,2}(undef, kG.Nk, length(sP.fft_range)), 1:kG.Nk, sP.fft_range)
    G_νn = Array{ComplexF64, length(gridshape(kG))}(undef, gridshape(kG)...) 
    for νn in axes(G, 2)
        G_νn  = expandKArr(kG, G[:,νn].parent)
        g_fft[:,νn] .= fft(G_νn)[:]
        g_rfft[:,νn] .= fft(reverse(G_νn))[:]
    end
    return g_fft, g_rfft
end
