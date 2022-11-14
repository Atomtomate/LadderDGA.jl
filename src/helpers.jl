# ==================================================================================================== #
#                                           helpers.jl                                                 #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
#   Last Edit Date  : 01.09.22                                                                         #
# ----------------------------------------- Description ---------------------------------------------- #
#   General purpose helper functions for the ladder DΓA code.                                          #
# -------------------------------------------- TODO -------------------------------------------------- #
#   Add documentation for all functions                                                                #
#   Cleanup of setup function                                                                          #
# ==================================================================================================== #


# =============================================== Setup ==============================================

"""
    setup_LDGA(kGridStr::Tuple{String,Int}, mP::ModelParameters, sP::SimulationParameters, env::EnvironmentVars [; local_correction=true])

Computes all needed objects for DΓA calculations. Returns:
    Σ_ladderLoc, Σ_loc, imp_density, kGrid, gLoc_fft, gLoc_rfft, Γsp, Γch, χDMFTsp, χDMFTch, χsp_loc, γsp_loc, χch_loc, γch_loc, χ₀Loc, gImp
"""
function setup_LDGA(kGridStr::Tuple{String,Int}, mP::ModelParameters, sP::SimulationParameters, env::EnvironmentVars; local_correction=true)

    @info "Setting up calculation for kGrid $(kGridStr[1]) of size $(kGridStr[2])"
    @timeit to "gen kGrid" kG    = gen_kGrid(kGridStr[1], kGridStr[2])
    in_file = env.inputDir*"/"*env.inputVars
    @timeit to "load f" χDMFTsp, χDMFTch, Γsp, Γch, gImp_in, Σ_loc = jldopen(in_file, "r") do f 
        #TODO: permute dims creates inconsistency between user input and LadderDGA.jl data!!
        Ns = typeof(sP.χ_helper) === BSE_SC_Helper ? sP.χ_helper.Nν_shell : 0
        χDMFTsp = zeros(_eltype, 2*sP.n_iν, 2*sP.n_iν, 2*sP.n_iω+1)
        χDMFTsp[(Ns+1):(end-Ns),(Ns+1):(end-Ns),:] = permutedims(_eltype === Float64 ? real.(f["χDMFTsp"]) : f["χDMFTsp"], (2,3,1))
        χDMFTch = zeros(_eltype, 2*sP.n_iν, 2*sP.n_iν, 2*sP.n_iω+1)
        χDMFTch[(Ns+1):(end-Ns),(Ns+1):(end-Ns),:] = permutedims(_eltype === Float64 ? real.(f["χDMFTch"]) : f["χDMFTch"], (2,3,1))
        Γsp = zeros(_eltype, 2*sP.n_iν, 2*sP.n_iν, 2*sP.n_iω+1)
        Γsp[(Ns+1):(end-Ns),(Ns+1):(end-Ns),:] = permutedims(_eltype === Float64 ? real.(-f["Γsp"]) : -f["Γsp"], (2,3,1))
        Γch = zeros(_eltype, 2*sP.n_iν, 2*sP.n_iν, 2*sP.n_iω+1)
        Γch[(Ns+1):(end-Ns),(Ns+1):(end-Ns),:] = permutedims(_eltype === Float64 ? real.(-f["Γch"]) : -f["Γch"], (2,3,1))
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
        χDMFTsp, χDMFTch, Γsp, Γch, gImp, Σ_loc
    end

    @timeit to "Compute GLoc" begin
        rm = maximum(abs.(sP.fft_range))
        t = cat(conj(reverse(gImp_in[1:rm])),gImp_in[1:rm], dims=1)
        gImp = OffsetArray(reshape(t,1,length(t)),1:1,-length(gImp_in[1:rm]):length(gImp_in[1:rm])-1)
        gLoc_fft = OffsetArray(Array{ComplexF64,2}(undef, kG.Nk, length(sP.fft_range)), 1:kG.Nk, sP.fft_range)
        gLoc_rfft = OffsetArray(Array{ComplexF64,2}(undef, kG.Nk, length(sP.fft_range)), 1:kG.Nk, sP.fft_range)
        ϵk_full = expandKArr(kG, kG.ϵkGrid)[:]
        for νi in sP.fft_range
            GLoc_νi  = reshape(G_from_Σ(νi, Σ_loc, ϵk_full, mP.β, mP.μ), gridshape(kG))
            gLoc_fft[:,νi] .= fft(GLoc_νi)[:]
            gLoc_rfft[:,νi] .= fft(reverse(GLoc_νi))[:]
        end
    end

    @timeit to "local correction" begin
        #TODO: unify checks
        kGridLoc = gen_kGrid(kGridStr[1], 1)
        Fsp   = F_from_χ(χDMFTsp, gImp[1,:], sP, mP.β);
        χ₀Loc = calc_bubble(gImp, gImp, kGridLoc, mP, sP, local_tail=true);
        println(size(χ₀Loc.data))
        χsp_loc, γsp_loc = calc_χγ(:sp, Γsp, χ₀Loc, kGridLoc, mP, sP);
        χch_loc, γch_loc = calc_χγ(:ch, Γch, χ₀Loc, kGridLoc, mP, sP);
        λ₀Loc = calc_λ0(χ₀Loc, Fsp, χsp_loc, γsp_loc, mP, sP)
        Σ_ladderLoc = calc_Σ(χsp_loc, γsp_loc, χch_loc, γch_loc, λ₀Loc, gImp, kGridLoc, mP, sP)
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
                χLocsp_ω[ωi] = χsp_loc[ωi]
                χLocch_ω[ωi] = χch_loc[ωi]
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

        χLocsp = sum(subtract_tail(χLocsp_ω[usable_loc_sp], mP.Ekin_DMFT, iωn[usable_loc_sp]))/mP.β -mP.Ekin_DMFT*mP.β/12
        χLocch = sum(subtract_tail(χLocch_ω[usable_loc_ch], mP.Ekin_DMFT, iωn[usable_loc_ch]))/mP.β -mP.Ekin_DMFT*mP.β/12

        χupup_DMFT_ω = 0.5 * (χLocsp_ω + χLocch_ω)[loc_range]
        χupup_DMFT_ω_sub = subtract_tail(χupup_DMFT_ω, mP.Ekin_DMFT, iωn[loc_range])

        imp_density_ntc = real(sum(χupup_DMFT_ω))/mP.β
        imp_density = real(sum(χupup_DMFT_ω_sub))/mP.β -mP.Ekin_DMFT*mP.β/12

        #TODO: update output
        @info """Inputs Read. Starting Computation.
          Local susceptibilities with ranges are:
          χLoc_sp($(usable_loc_sp)) = $(printr_s(χLocsp)), χLoc_ch($(usable_loc_ch)) = $(printr_s(χLocch))
          sum χupup check (plain ?≈? tail sub ?≈? imp_dens ?≈? n/2 (1-n/2)): $(imp_density_ntc) ?=? $(0.5 .* real(χLocsp + χLocch)) ?≈? $(imp_density) ≟ $(mP.n/2 * ( 1 - mP.n/2))"
          """
    end
    return Σ_ladderLoc, Σ_loc, imp_density, kG, gLoc_fft, gLoc_rfft, Γsp, Γch, χDMFTsp, χDMFTch, χsp_loc, γsp_loc, χch_loc, γch_loc, χ₀Loc, gImp
end

# ========================================== Index Functions =========================================

q0_index(kG::KGrid) = findfirst(x -> all(x .≈ zeros(length(gridshape(kG)))), kG.kGrid)
ω0_index(sP::SimulationParameters) = sP.n_iω+1

"""
    OneToIndex_to_Freq(ωi::Int, νi::Int, sP::SimulationParameters [, Nν_shell])

Converts `(1:N,1:N)` index tuple for bosonic (`ωi`) and fermionic (`νi`) frequency to
Matsubara frequency number. If the array has a `ν` shell (for example for tail
improvements) this will also be taken into account by providing `Nν_shell`.
"""
@inline function OneToIndex_to_Freq(ωi::Int, νi::Int, sP::SimulationParameters, Nν_shell)
    ωn = ωi-sP.n_iω-1
    νn = (νi-sP.n_iν-Nν_shell-1) - sP.shift*trunc(Int,ωn/2)
    return ωn, νn
end

@inline function OneToIndex_to_Freq(ωi::Int, νi::Int, sP::SimulationParameters)
    ωn = ωi-sP.n_iω-1
    νn = (νi-sP.n_iν-1) - sP.shift*trunc(Int,ωn/2)
    return ωn, νn
end

@inline ν0Index_of_ωIndex(ωi::Int, sP)::Int = sP.n_iν + sP.shift*(trunc(Int, (ωi - sP.n_iω - 1)/2)) + 1

"""
    to_m_index(arr::AbstractArray{T,2/3}, sP::SimulationParameters)

Converts array with simpel `1:N` index to larger array, where the index matches the Matsubara
Frequency number. This function is not optimized!
"""
function to_m_index(arr::AbstractArray{T,3}, sP::SimulationParameters) where T
    ωrange = -sP.n_iω:sP.n_iω
    νrange = -2*sP.n_iν:2*sP.n_iν
    length(ωrange) != size(arr,3) && @error "Assumption -n_iω:n_iω for ω grid not fulfilled."
    ωl = length(ωrange)
    νl = length(νrange)
    res = OffsetArray(zeros(ComplexF64, size(arr,1), νl, ωl), 1:size(arr,1) ,νrange, ωrange)
    for qi in 1:size(arr,1)
        to_m_index!(view(res,qi,:,:),view(arr,qi,:,:), sP)
    end
    return res
end

function to_m_index(arr::AbstractArray{T,2}, sP::SimulationParameters) where T
    ωrange = -sP.n_iω:sP.n_iω
    νrange = -2*sP.n_iν:2*sP.n_iν
    length(ωrange) != size(arr,2) && @error "Assumption -n_iω:n_iω for ω grid not fulfilled."
    ωl = length(ωrange)
    νl = length(νrange)
    res = OffsetArray(zeros(ComplexF64, νl,ωl), νrange, ωrange)
    to_m_index!(res, arr, sP)
    return res
end

function to_m_index!(res::AbstractArray{T,2}, arr::AbstractArray{T,2}, sP::SimulationParameters) where T
    for ωi in 1:size(arr,2)
        for νi in 1:size(arr,1)
            ωn,νn = OneToIndex_to_Freq(ωi, νi, sP)
            @inbounds res[νn, ωn] = arr[νi,ωi]
        end
    end
    return res
end

function ωindex_range(sP::SimulationParameters)
    return 1:(2*sP.n_iω+1)
    # TODO: placeholder for reduced omega-range computations
end


# =========================================== Noise Filter ===========================================

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
