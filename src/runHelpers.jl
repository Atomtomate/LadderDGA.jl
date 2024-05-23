# ==================================================================================================== #
#                                          runHelpers.jl                                               #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe, Jan Frederik Weissler                                             #
# ----------------------------------------- Description ---------------------------------------------- #
#   Setup functions and definition of runHelpers for different methods                                 #
# -------------------------------------------- TODO -------------------------------------------------- #
# ==================================================================================================== #

# =============================================== Setup ==============================================
abstract type RunHelper end

"""
    lDΓAHelper <: RunHelper

Struct with data needed to run ladder DΓA calculations.

Constructor
-------------
    setup_LDGA(kGridStr::Tuple{String,Int}, mP::ModelParameters, sP::SimulationParameters, env::EnvironmentVars)

See [`setup_LDGA`](@ref setup_LDGA)

Fields
-------------
   TODO: documentation for fields
"""
mutable struct lDΓAHelper <: RunHelper
    sP::SimulationParameters
    mP::ModelParameters
    kG::KGrid
    Σ_ladderLoc::OffsetMatrix
    Σ_loc::OffsetVector
    imp_density::Float64
    gLoc::GνqT
    gLoc_fft::GνqT
    gLoc_rfft::GνqT
    Γ_m::ΓT
    Γ_d::ΓT
    χloc_m_sum::Float64
    χDMFT_m::Array{ComplexF64,3}
    χDMFT_d::Array{ComplexF64,3}
    χ_m_loc::χT
    γ_m_loc::γT
    χ_d_loc::χT
    γ_d_loc::γT
    χ₀Loc::χ₀T
    gImp::OffsetArray
    # Σ_tail_factor::OffsetVector{ComplexF64} 
end


"""
    RPAHelper <: RunHelper

Struct with data needed to run ladder RPA calculations.

Constructor
-------------
    setup_RPA(kGridStr::Tuple{String,Int}, mP::ModelParameters, sP::SimulationParameters, env::EnvironmentVars [; silent=false])

See [`setup_RPA`](@ref setup_RPA)

Fields
-------------
    TODO: documentation
"""
mutable struct RPAHelper <: RunHelper
    sP::SimulationParameters
    mP::ModelParameters
    kG::KGrid
    gLoc::GνqT
    gLoc_fft::GνqT
    gLoc_rfft::GνqT
end



"""
    setup_LDGA(kGridStr::Tuple{String,Int}, mP::ModelParameters, sP::SimulationParameters, env::EnvironmentVars [;silent=false])

Computes all needed objects for DΓA calculations.

Returns: [`lDΓAHelper`](@ref lDΓAHelper)
"""
function setup_LDGA(kGridStr::Tuple{String,Int}, mP::ModelParameters, sP::SimulationParameters, env::EnvironmentVars; silent::Bool = false)

    !silent && @info "Setting up calculation for kGrid $(kGridStr[1]) of size $(kGridStr[2])"
    @timeit to "gen kGrid" kG = gen_kGrid(kGridStr[1], kGridStr[2])
    @timeit to "load f" χDMFT_m, χDMFT_d, Γ_m, Γ_d, gImp_in, Σ_loc = jldopen(env.inputVars, "r") do f
        #TODO: permute dims creates inconsistency between user input and LadderDGA.jl data!!
        Ns = typeof(sP.χ_helper) === BSE_SC_Helper ? sP.χ_helper.Nν_shell : 0
        χDMFT_m = zeros(_eltype, 2 * sP.n_iν, 2 * sP.n_iν, 2 * sP.n_iω + 1)
        χDMFT_m[(Ns+1):(end-Ns), (Ns+1):(end-Ns), :] = permutedims(_eltype === Float64 ? real.(f["χDMFTsp"]) : f["χDMFTsp"], (2, 3, 1))
        χDMFT_d = zeros(_eltype, 2 * sP.n_iν, 2 * sP.n_iν, 2 * sP.n_iω + 1)
        χDMFT_d[(Ns+1):(end-Ns), (Ns+1):(end-Ns), :] = permutedims(_eltype === Float64 ? real.(f["χDMFTch"]) : f["χDMFTch"], (2, 3, 1))
        Γ_m = zeros(_eltype, 2 * sP.n_iν, 2 * sP.n_iν, 2 * sP.n_iω + 1)
        Γ_m[(Ns+1):(end-Ns), (Ns+1):(end-Ns), :] = permutedims(_eltype === Float64 ? real.(-f["Γsp"]) : -f["Γsp"], (2, 3, 1))
        Γ_d = zeros(_eltype, 2 * sP.n_iν, 2 * sP.n_iν, 2 * sP.n_iω + 1)
        Γ_d[(Ns+1):(end-Ns), (Ns+1):(end-Ns), :] = permutedims(_eltype === Float64 ? real.(-f["Γch"]) : -f["Γch"], (2, 3, 1))
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
        χDMFT_m, χDMFT_d, Γ_m, Γ_d, gImp, OffsetVector(Σ_loc, 0:(length(Σ_loc)-1))
    end

    @timeit to "Compute GLoc" begin
        rm = min(length(gImp_in), max(2048,maximum(abs.(sP.fft_range))))
        gs = gridshape(kG)
        kGdims = grid_dimension(kG)
        νdim = kGdims + 1
        gLoc = OffsetArray(Array{ComplexF64,2}(undef, length(kG.kMult), length(sP.fft_range)), 1:length(kG.kMult), sP.fft_range)
        gLoc_i = Array{ComplexF64,kGdims}(undef, gs...)
        gLoc_fft = OffsetArrays.Origin(repeat([1], kGdims)..., first(sP.fft_range))(Array{ComplexF64,kGdims + 1}(undef, gs..., length(sP.fft_range)))
        gLoc_rfft = OffsetArrays.Origin(repeat([1], kGdims)..., first(sP.fft_range))(Array{ComplexF64,kGdims + 1}(undef, gs..., length(sP.fft_range)))
        ϵk_full = expandKArr(kG, kG.ϵkGrid)[:]
        for νn in sP.fft_range
            Σ_loc_i = (νn < 0) ? conj(Σ_loc[-νn-1]) : Σ_loc[νn]
            gLoc_i = reshape(map(ϵk -> G_from_Σ(νn, mP.β, mP.μ, ϵk, Σ_loc_i), ϵk_full), gridshape(kG))
            gLoc[:, νn] = reduceKArr(kG, gLoc_i)
            selectdim(gLoc_fft, νdim, νn) .= fft(gLoc_i)
            selectdim(gLoc_rfft, νdim, νn) .= fft(reverse(gLoc_i))
        end
    end

    @timeit to "local correction" begin
        #TODO: unify checks
        kGridLoc = gen_kGrid(kGridStr[1], 1)
        t = cat(conj(reverse(gImp_in[1:rm])), gImp_in[1:rm], dims = 1)
        gImp = OffsetArray(reshape(t, 1, length(t)), 1:1, -length(gImp_in[1:rm]):length(gImp_in[1:rm])-1)
        F_m = F_from_χ(χDMFT_m, gImp[1, :], sP, mP.β)
        χ₀Loc = calc_bubble(:local, gImp, gImp, kGridLoc, mP, sP)
        χ_m_loc, γ_m_loc = calc_χγ(:m, Γ_m, χ₀Loc, kGridLoc, mP, sP)
        χ_d_loc, γ_d_loc = calc_χγ(:d, Γ_d, χ₀Loc, kGridLoc, mP, sP)
        λ₀Loc = calc_λ0(χ₀Loc, F_m, χ_m_loc, γ_m_loc, mP, sP)
        χloc_m_sum = real(sum_ω(χ_m_loc)[1])
        Σ_ladderLoc = calc_Σ(χ_m_loc, γ_m_loc, χ_d_loc, γ_d_loc, χloc_m_sum, λ₀Loc, Σ_loc, gImp, kGridLoc, mP, sP, tc = false)
        any(isnan.(Σ_ladderLoc)) && @error "Σ_ladderLoc contains NaN"

        χLoc_m_ω = similar(χDMFT_m, size(χDMFT_m, 3))
        χLoc_d_ω = similar(χDMFT_d, size(χDMFT_d, 3))
        for ωi in axes(χDMFT_m, 3)
            if typeof(sP.χ_helper) === BSE_SC_Helper
                @error "SC not fully implemented yet"
                @info "Using asymptotics improvement for large ν, ν' of χ_DMFT with shell size of $(sP.n_iν_shell)"
                improve_χ!(:m, ωi, view(χDMFT_m, :, :, ωi), view(χ₀Loc, 1, :, ωi), mP.U, mP.β, sP.χ_helper)
                improve_χ!(:d, ωi, view(χDMFT_d, :, :, ωi), view(χ₀Loc, 1, :, ωi), mP.U, mP.β, sP.χ_helper)
            end
            if typeof(sP.χ_helper) <: BSE_Asym_Helpers
                χLoc_m_ω[ωi] = χ_m_loc[ωi]
                χLoc_d_ω[ωi] = χ_d_loc[ωi]
            else
                χLoc_m_ω[ωi] = sum(view(χDMFT_m, :, :, ωi)) / mP.β^2
                χLoc_d_ω[ωi] = sum(view(χDMFT_d, :, :, ωi)) / mP.β^2
            end
        end
    end

    @timeit to "ranges/imp. dens." begin
        usable_loc_m = find_usable_χ_interval(real(χLoc_m_ω), reduce_range_prct = sP.usable_prct_reduction)
        usable_loc_d = find_usable_χ_interval(real(χLoc_d_ω), reduce_range_prct = sP.usable_prct_reduction)
        loc_range = intersect(usable_loc_m, usable_loc_d)

        iωn = 1im .* 2 .* (-sP.n_iω:sP.n_iω) .* π ./ mP.β

        χLoc_m = sum(subtract_tail(χLoc_m_ω[usable_loc_m], mP.Ekin_DMFT, iωn[usable_loc_m], 2)) / mP.β - mP.Ekin_DMFT * mP.β / 12
        χLoc_d = sum(subtract_tail(χLoc_d_ω[usable_loc_d], mP.Ekin_DMFT, iωn[usable_loc_d], 2)) / mP.β - mP.Ekin_DMFT * mP.β / 12

        χupup_DMFT_ω = 0.5 * (χLoc_m_ω+χLoc_d_ω)[loc_range]
        χupup_DMFT_ω_sub = subtract_tail(χupup_DMFT_ω, mP.Ekin_DMFT, iωn[loc_range], 2)

        imp_density_ntc = real(sum(χupup_DMFT_ω)) / mP.β
        imp_density = real(sum(χupup_DMFT_ω_sub)) / mP.β - mP.Ekin_DMFT * mP.β / 12

        #TODO: update output
        !silent && @info """Inputs Read. Starting Computation.
          Local susceptibilities with ranges are:
          χLoc_m($(usable_loc_m)) = $(printr_s(χLoc_m)), χLoc_d($(usable_loc_d)) = $(printr_s(χLoc_d))
          sum χupup check (plain ?≈? tail sub ?≈? imp_dens ?≈? n/2 (1-n/2)): $(imp_density_ntc) ?=? $(0.5 .* real(χLoc_m + χLoc_d)) ?≈? $(imp_density) ≟ $(mP.n/2 * ( 1 - mP.n/2))"
          """
    end

    workerpool = get_workerpool()
    @sync begin
        for w in workers()
            @async remotecall_fetch(LadderDGA.update_wcache!, w, :G_fft, gLoc_fft; override = true)
            @async remotecall_fetch(LadderDGA.update_wcache!, w, :G_fft_reverse, gLoc_rfft; override = true)
            @async remotecall_fetch(LadderDGA.update_wcache!, w, :kG, kGridStr; override = true)
            @async remotecall_fetch(LadderDGA.update_wcache!, w, :mP, mP; override = true)
            @async remotecall_fetch(LadderDGA.update_wcache!, w, :sP, sP; override = true)
        end
    end

    # Σlim = U^2 * n/2 * (1 - n/2)
    # DMFT_dff =  -imag(Σ_loc[0:length(iν)-1]) .* imag(iν) .- Σlim
    # tail_factor = - 2*U .* exp.(-(DMFT_dff).^2 ./ δ) ./ iν

    return lDΓAHelper(sP, mP, kG, Σ_ladderLoc, Σ_loc, imp_density, gLoc, gLoc_fft, gLoc_rfft, Γ_m, Γ_d, real(sum_ω(χ_m_loc)[1]), χDMFT_m, χDMFT_d, χ_m_loc, γ_m_loc, χ_d_loc, γ_d_loc, χ₀Loc, gImp)
end

"""
    setup_RPA(kGridStr::Tuple{String,Int}, mP::ModelParameters, sP::SimulationParameters; [silent=false])

Computes all needed objects for RPA calculations.

Returns: [`RPAHelper`](@ref RPAHelper)
"""
function setup_RPA(kGridStr::Tuple{String,Int}, mP::ModelParameters, sP::SimulationParameters; silent::Bool = false)

    !silent && @info "Setting up calculation for kGrid $(kGridStr[1]) of size $(kGridStr[2])"
    @timeit to "gen kGrid" kG = gen_kGrid(kGridStr[1], kGridStr[2])

    # TODO: refactor this as a function

    @timeit to "Compute GLoc" begin
        gs = gridshape(kG)
        kGdims = length(gs)
        νdim = kGdims + 1

        gLoc = OffsetArray(Array{ComplexF64,2}(undef, length(kG.kMult), length(sP.fft_range)), 1:length(kG.kMult), sP.fft_range)
        gLoc_i = Array{ComplexF64,kGdims}(undef, gs...)
        gLoc_fft = OffsetArrays.Origin(repeat([1], kGdims)..., first(sP.fft_range))(Array{ComplexF64,kGdims + 1}(undef, gs..., length(sP.fft_range)))
        gLoc_rfft = OffsetArrays.Origin(repeat([1], kGdims)..., first(sP.fft_range))(Array{ComplexF64,kGdims + 1}(undef, gs..., length(sP.fft_range)))
        ϵk_full = expandKArr(kG, kG.ϵkGrid)[:]
        for νn in sP.fft_range
            gLoc_i = reshape(map(ϵk -> G_from_Σ(νn, mP.β, mP.μ, ϵk, mP.U * mP.n / 2 + 0.0im), ϵk_full), gridshape(kG))
            gLoc[:, νn] = reduceKArr(kG, gLoc_i)
            selectdim(gLoc_fft, νdim, νn) .= fft(gLoc_i)
            selectdim(gLoc_rfft, νdim, νn) .= fft(reverse(gLoc_i))
        end
    end

    return RPAHelper(sP, mP, kG, gLoc, gLoc_fft, gLoc_rfft)
end
