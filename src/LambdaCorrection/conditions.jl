# return trace, Σ_ladder, G_ladder, E_kin, E_pot, μnew, λm, lhs_c1, E_pot_2, converged

"""
    λdm_correction(χ_m, γ_m, χ_d, γ_d, Σ_loc, gLoc_rfft, λ₀, kG, mP, sP; 
        maxit_root = 100, atol_root = 1e-8, λd_min_δ = 0.1, λd_max = 500,
        maxit::Int = 50, update_χ_tail=false, mixing=0.2, conv_abs=1e-6, par=false)

Calculates ``\\lambda_\\mathrm{dm}`` and associated quantities like the self-energy.

TODO: full documentation. Pack results into struct

Returns: 
-------------
    Σ_ladder : ladder self-energy
    G_ladder : ladder Green's function obtained from `Σ_ladder`
    E_kin    : kinetic energy, unless `update_χ_tail = true`, this will be not consistent with the susceptibility tail coefficients.
    E_pot    : one-particle potential energy, obtained through galitskii-migdal formula
    μnew:    : chemical potential of `G_ladder`
    λm       : λ-correction for the magnetic channel
    lhs_c1   : check-sum for the Pauli-principle value obtained from the susceptibilities (`λm` fixes this to ``n/2 \\cdot (1-n/2)``) 
    E_pot_2  : Potential energy obtained from susceptibilities. `λd` fixes this to `E_pot`
    converged: error flag. False if no `λd` was found. 
    λd       : λ-correction for the density channel.
"""
function λdm_correction(χ_m::χT, γ_m::γT, χ_d::χT, γ_d::γT, Σ_loc::Vector{ComplexF64},
                        gLoc_rfft::GνqT, χloc_m_sum::Union{Float64,ComplexF64}, λ₀::Array{ComplexF64,3},
                        kG::KGrid, mP::ModelParameters, sP::SimulationParameters; 
                        maxit_root = 50, atol_root = 1e-7,
                        νmax::Int = -1, λd_min_δ = 0.05, λd_max = 500,
                        maxit::Int = 50, update_χ_tail=false, mixing=0.2, conv_abs=1e-6, par=false, with_trace=false)



    ωindices, νGrid, iωn_f = gen_νω_indices(χ_m, χ_d, mP, sP)
    iωn = iωn_f[ωindices]
    iωn[findfirst(x->x ≈ 0, iωn)] = Inf
    iωn_m2 = 1 ./ iωn .^ 2
    gLoc_rfft_init = deepcopy(gLoc_rfft)
    par && initialize_EoM(gLoc_rfft_init, χloc_m_sum, λ₀, νGrid, kG, mP, sP, χ_m = χ_m, γ_m = γ_m, χ_d = χ_d, γ_d = γ_d)
    fft_νGrid= 0:last(sP.fft_range)
    G_ladder::OffsetMatrix{ComplexF64, Matrix{ComplexF64}} = OffsetArray(Matrix{ComplexF64}(undef, length(kG.kMult), length(fft_νGrid)), 1:length(kG.kMult), fft_νGrid) 
    Σ_ladder_work::OffsetMatrix{ComplexF64, Matrix{ComplexF64}} = OffsetArray(Matrix{ComplexF64}(undef, length(kG.kMult), length(νGrid)), 1:length(kG.kMult), νGrid)
    Σ_ladder::OffsetMatrix{ComplexF64, Matrix{ComplexF64}} = OffsetArray(Matrix{ComplexF64}(undef, length(kG.kMult), length(νGrid)), 1:length(kG.kMult), νGrid)


    traces = DataFrame[]

    trace_i = with_trace ? Ref(DataFrame(it = Int[], λm = Float64[], λd = Float64[], μ = Float64[], EKin = Float64[], EPot = Float64[], 
        lhs_c1 = Float64[], EPot_c2 = Float64[], cs_d = Float64[], cs_m = Float64[], cs_Σ = Float64[], cs_G = Float64[])) : Ref(nothing)
    function ff_seq(λd_i::Float64)
        copy!(gLoc_rfft_init, gLoc_rfft)
        trace_i, _, _, _, E_pot, _, _, _, E_pot_2, _ = run_sc(
                    χ_m, γ_m, χ_d, γ_d, χloc_m_sum, λ₀, gLoc_rfft, Σ_loc, λd_i, kG, mP, sP, 
                maxit=maxit, mixing=mixing, conv_abs=conv_abs, update_χ_tail=update_χ_tail)
        rhs_c2 = E_pot/mP.U - (mP.n/2) * (mP.n/2)
        push!(traces, trace_i)
        return rhs_c2 - E_pot_2
    end


    function ff_par(λd_i::Float64)
        copy!(gLoc_rfft_init, gLoc_rfft)
            _, E_pot, _, _, _, E_pot_2, _ = run_sc!(G_ladder, Σ_ladder_work,  Σ_ladder, gLoc_rfft_init, trace_i,
                         νGrid, iωn_f, iωn_m2, χ_m, χ_d, Σ_loc, λd_i, kG, mP, sP;
                         maxit=maxit, mixing=mixing, conv_abs=conv_abs, update_χ_tail=update_χ_tail, par=true)
        rhs_c2 = E_pot/mP.U - (mP.n/2) * (mP.n/2)
        with_trace && push!(traces, trace_i[])
        return rhs_c2 - E_pot_2
    end

    residual_f = par ? ff_par : ff_seq

    λd_min_tmp = get_λ_min(real(χ_d.data)) 
    track = Roots.Tracks()
    println("Bracket for λdm: [",λd_min_tmp + λd_min_δ*abs(λd_min_tmp), ",", λd_max, "]")
    root = try
        find_zero(residual_f, (λd_min_tmp + λd_min_δ*abs(λd_min_tmp), λd_max), Roots.A42(), maxiters=maxit_root, atol=atol_root, tracks=track)
    catch e
        println("Error: $e")
        println("Track: $track")
        println("Retrying with initial guess 0!")
        NaN
    end
    root = if isnan(root)
        try
            find_zero(residual_f, 0.0, maxiters=maxit_root, atol=atol_root, tracks=track)
        catch e
            println("Error: $e")
            println("Track: $track")
            NaN
        end
    else
        root
    end

    if isnan(root) || track.convergence_flag == :not_converged
        println("WARNING: No λd root was found! Track:")
        println(track)
        return nothing, nothing, NaN, NaN, NaN, NaN, NaN, NaN, false, NaN
    elseif root < λd_min_tmp
        println("WARNING: λd = $root outside region ($λd_min_tmp)!")
        return nothing, nothing, NaN, NaN, NaN, NaN, NaN, NaN, false, NaN
    else
        copy!(gLoc_rfft_init, gLoc_rfft)
        if par
            vars = run_sc!(G_ladder, Σ_ladder_work,  Σ_ladder, gLoc_rfft_init, Ref(nothing),
                         νGrid, iωn_f, iωn_m2, χ_m, χ_d, Σ_loc, root, kG, mP, sP;
                         maxit=maxit, mixing=mixing, conv_abs=conv_abs, update_χ_tail=update_χ_tail, par=true)
            return traces, Σ_ladder, G_ladder, vars..., root
        else
            vars = run_sc(
                    χ_m, γ_m, χ_d, γ_d, χloc_m_sum, λ₀, gLoc_rfft_init, Σ_loc, root, kG, mP, sP, 
                maxit=maxit, mixing=mixing, conv_abs=conv_abs, update_χ_tail=update_χ_tail)
            return traces, vars[2:end]..., root
        end
    end
end


# ============================================== Helpers =============================================
# --------------------------------- clean versions (slow/single core) --------------------------------
"""
TODO: refactor (especially _par version)
"""
function run_sc(χ_m::χT, γ_m::γT, χ_d::χT, γ_d::γT, χloc_m_sum::Union{Float64,ComplexF64}, 
                λ₀::AbstractArray{ComplexF64,3}, gLoc_rfft_init::GνqT, Σ_loc::Vector{ComplexF64},
                λd::Float64, kG::KGrid, mP::ModelParameters, sP::SimulationParameters;
                maxit::Int=100, mixing::Float64=0.2, conv_abs::Float64=1e-8, update_χ_tail::Bool=false)
    _, νGrid, iωn_f = gen_νω_indices(χ_m, χ_d, mP, sP)
    gLoc_rfft = deepcopy(gLoc_rfft_init)
    G_ladder::OffsetMatrix{ComplexF64, Matrix{ComplexF64}}     = OffsetArray(Matrix{ComplexF64}(undef, length(kG.kMult), length(νGrid)), 1:length(kG.kMult), νGrid) 
    Σ_ladder_old::OffsetMatrix{ComplexF64, Matrix{ComplexF64}} = OffsetArray(Matrix{ComplexF64}(undef, length(kG.kMult), length(νGrid)), 1:length(kG.kMult), νGrid)
    Σ_ladder::OffsetMatrix{ComplexF64, Matrix{ComplexF64}} = OffsetArray(Matrix{ComplexF64}(undef, length(kG.kMult), length(νGrid)), 1:length(kG.kMult), νGrid)
    E_pot    = NaN
    E_kin    = NaN
    cont     = true
    lhs_c1   = NaN
    E_pot_2  = NaN
    converged = maxit == 0

    trace = DataFrame(it = Int[], λm = Float64[], λd = Float64[], μ = Float64[], EKin = Float64[], EPot = Float64[], 
        lhs_c1 = Float64[], EPot_c2 = Float64[], cs_d = Float64[], cs_m = Float64[], cs_Σ = Float64[], cs_G = Float64[])

    rhs_λsp = λm_rhs(NaN, χ_m, χ_d, λd, kG, mP, sP)
    λm = λm_correction(χ_m, real(rhs_λsp), kG, mP, sP)
    if !isfinite(λm)
        @warn "no finite λm found!"
        cont = false
    end

    μbak = mP.μ
    it = 1
    while cont
        copy!(Σ_ladder_old, Σ_ladder)
        Σ_ladder = calc_Σ(χ_m, γ_m, χ_d, γ_d, χloc_m_sum, λ₀, gLoc_rfft, kG, mP, sP, νmax=last(νGrid)+1, λm=λm,λd=λd);
        mixing != 0 && it > 1 && (Σ_ladder[:,:] = (1-mixing) .* Σ_ladder .+ mixing .* Σ_ladder_old)
        μnew, G_ladder = G_from_Σladder(Σ_ladder, Σ_loc, kG, mP, sP; fix_n=true)
        isnan(μnew) && break
        _, gLoc_rfft = G_fft(G_ladder, kG, sP)
        # println("SEQ CS: ", sum(abs.(G_ladder[:,νGrid].parent)), " // ",  sum(abs.(Σ_ladder.parent)),  last(νGrid)+1, axes(G_ladder), axes(Σ_ladder))
        E_kin, E_pot = calc_E(G_ladder[:,νGrid].parent, Σ_ladder.parent, μnew, kG, mP, νmax = last(νGrid)+1)
        if update_χ_tail
            if isfinite(E_kin)
            update_tail!(χ_m, [0, 0, E_kin], iωn_f)
            update_tail!(χ_d, [0, 0, E_kin], iωn_f)
            rhs_λsp = λm_rhs(NaN, χ_m, χ_d, λd, kG, mP, sP)
            λm = λm_correction(χ_m, real(rhs_λsp), kG, mP, sP)
            else
                println("Warning: unable to update χ tail: E_kin not finite")
            end
        end
        if it != 1
            if abs(sum(Σ_ladder .- Σ_ladder_old))/(kG.Nk) < conv_abs  
                converged = true
                cont = false
            end
        end
        (it >= maxit) && (cont = false)

        χ_m_sum = sum_kω(kG, χ_m)
        χ_d_sum = sum_kω(kG, χ_d)
        lhs_c1  = real(χ_d_sum + χ_m_sum)/2
        E_pot_2 = real(χ_d_sum - χ_m_sum)/2
        row = [it, λm, λd, μnew, E_kin, E_pot, lhs_c1, E_pot_2, abs(χ_d_sum), abs(χ_m_sum), abs(sum(Σ_ladder)), abs(sum(G_ladder))]
        push!(trace, row)
        it += 1
    end

    if isfinite(E_kin)
        χ_m_sum = sum_kω(kG, χ_m, λ=λm)
        χ_d_sum = sum_kω(kG, χ_d, λ=λd)
        lhs_c1  = real(χ_d_sum + χ_m_sum)/2
        E_pot_2 = real(χ_d_sum - χ_m_sum)/2
    end
    update_tail!(χ_m, [0, 0, mP.Ekin_DMFT], iωn_f)
    update_tail!(χ_d, [0, 0, mP.Ekin_DMFT], iωn_f)
    μnew = mP.μ
    mP.μ = μbak
    converged = converged && all(isfinite.([lhs_c1, E_pot_2]))
    return trace, Σ_ladder, G_ladder, E_kin, E_pot, μnew, λm, lhs_c1, E_pot_2, converged
end

function run_sc_par(χ_m::χT, γ_m::γT, χ_d::χT, γ_d::γT, χloc_m_sum::Union{Float64,ComplexF64}, 
                    λ₀::AbstractArray{ComplexF64,3}, gLoc_rfft_init::GνqT, Σ_loc::Vector{ComplexF64},
                    λd::Float64, kG::KGrid, mP::ModelParameters, sP::SimulationParameters;
                    maxit::Int=100, mixing::Float64=0.2, conv_abs::Float64=1e-8, update_χ_tail::Bool=false, par = true)
    ωindices, νGrid, iωn_f = gen_νω_indices(χ_m, χ_d, mP, sP)
    iωn = iωn_f[ωindices]
    iωn[findfirst(x->x ≈ 0, iωn)] = Inf
    iωn_m2 = 1 ./ iωn .^ 2
    gLoc_rfft_init = deepcopy(gLoc_rfft_init)
    par && initialize_EoM(gLoc_rfft_init, χloc_m_sum, λ₀, νGrid, kG, mP, sP, χ_m = χ_m, γ_m = γ_m, χ_d = χ_d, γ_d = γ_d)
    trace = DataFrame(it = Int[], λm = Float64[], λd = Float64[], μ = Float64[], EKin = Float64[], EPot = Float64[], 
        lhs_c1 = Float64[], EPot_c2 = Float64[], cs_d = Float64[], cs_m = Float64[], cs_Σ = Float64[], cs_G = Float64[])
    fft_νGrid= sP.fft_range

                
    G_ladder::OffsetMatrix{ComplexF64, Matrix{ComplexF64}} = OffsetArray(Matrix{ComplexF64}(undef, length(kG.kMult), length(fft_νGrid)), 1:length(kG.kMult), fft_νGrid) 
    Σ_ladder_work::OffsetMatrix{ComplexF64, Matrix{ComplexF64}} = OffsetArray(Matrix{ComplexF64}(undef, length(kG.kMult), length(νGrid)), 1:length(kG.kMult), νGrid)
    Σ_ladder::OffsetMatrix{ComplexF64, Matrix{ComplexF64}} = OffsetArray(Matrix{ComplexF64}(undef, length(kG.kMult), length(νGrid)), 1:length(kG.kMult), νGrid)


    E_kin, E_pot, μnew, λm, lhs_c1, E_pot_2, converged = run_sc!(G_ladder, Σ_ladder_work,  Σ_ladder, gLoc_rfft_init, Ref(trace),
                         νGrid, iωn_f, iωn_m2, χ_m, χ_d, Σ_loc, λd, kG, mP, sP;
                         maxit=maxit, mixing=mixing, conv_abs=conv_abs, update_χ_tail=update_χ_tail, par=true)
    return trace, Σ_ladder, G_ladder, E_kin, E_pot, μnew, λm, lhs_c1, E_pot_2, converged
end

function run_sc!(G_ladder::OffsetMatrix, Σ_ladder_work::OffsetMatrix,  Σ_ladder::OffsetMatrix, gLoc_rfft::GνqT, trace::Ref, 
                νGrid::AbstractVector{Int}, iωn_f::Vector{ComplexF64}, iωn_m2::Vector{ComplexF64},
                 χ_m::χT, χ_d::χT, Σ_loc::Vector{ComplexF64},
                 λd::Float64, kG::KGrid, mP::ModelParameters, sP::SimulationParameters;
                 maxit::Int=100, mixing::Float64=0.2, conv_abs::Float64=1e-8, update_χ_tail::Bool=false, par = true)
    E_pot    = NaN
    E_kin    = NaN
    lhs_c1   = NaN
    E_pot_2  = NaN
    cont     = true
    converged= maxit == 0
    μbak     = mP.μ
    it       = 1
    fft_νGrid= sP.fft_range

    χ_λ!(χ_d, λd)
    rhs_λsp = λm_rhs(NaN, χ_m, χ_d, 0.0, kG, mP, sP)
    λm = λm_correction(χ_m, real(rhs_λsp), kG, mP, sP)
    if !isfinite(λm)
        @warn "no finite λm found!"
        cont = false
    end

    while cont
        par && update_wcaches_G_rfft!(gLoc_rfft)
        copy!(Σ_ladder_work, Σ_ladder)
        calc_Σ_par!(Σ_ladder, mP, λm=λm, λd=λd)
        mixing != 0 && it > 1 && (Σ_ladder[:,:] = (1-mixing) .* Σ_ladder .+ mixing .* Σ_ladder_work)
        μnew = G_from_Σladder!(G_ladder, Σ_ladder, Σ_loc, kG, mP; fix_n=true)
        isnan(μnew) && break
        G_rfft!(gLoc_rfft, G_ladder, kG, fft_νGrid)
        mP.μ = μnew
        # TODO: optimize calc_E (precompute tails) 
        # println("PAR CS: ", sum(abs.(G_ladder[:,νGrid].parent)), " // ",  sum(abs.(Σ_ladder.parent)),  last(νGrid)+1, axes(G_ladder), axes(Σ_ladder))
        E_kin, E_pot = calc_E(G_ladder[:,νGrid].parent, Σ_ladder.parent, μnew, kG, mP, νmax = last(νGrid)+1)
        if update_χ_tail
            if isfinite(E_kin)
                par && update_tail!([0, 0, E_kin])
                update_tail!(χ_d, [0, 0, E_kin], iωn_f)
                update_tail!(χ_m, [0, 0, E_kin], iωn_f)
                rhs_λsp = λm_rhs(NaN, χ_m, χ_d, 0.0, kG, mP, sP)
                λm = λm_correction(χ_m, real(rhs_λsp), kG, mP, sP)
            else
                println("Warning: unable to update χ tail: E_kin not finite")
            end
        end
        if Σ_ladder_work !== nothing
            if abs(sum(Σ_ladder .- Σ_ladder_work))/(kG.Nk) < conv_abs  
                converged = true
                cont = false
            end
        end
        (it >= maxit) && (cont = false)
        if !isnothing(trace[])
            χ_m_sum = sum_kω(kG, χ_m, λ=λm)
            χ_d_sum = sum_kω(kG, χ_d)
            lhs_c1  = real(χ_d_sum + χ_m_sum)/2
            E_pot_2 = real(χ_d_sum - χ_m_sum)/2
            row = [it, λm, λd, μnew, E_kin, E_pot, lhs_c1, E_pot_2, abs(sum(χ_d)), abs(sum(χ_m)), abs(sum(Σ_ladder)), abs(sum(G_ladder))]
            push!(trace[], row)
        end
        it += 1
    end
    if isfinite(E_kin)
        χ_m_sum = sum_kω(kG, χ_m, λ=λm)
        χ_d_sum = sum_kω(kG, χ_d)
        lhs_c1  = real(χ_d_sum + χ_m_sum)/2
        E_pot_2 = real(χ_d_sum - χ_m_sum)/2
        if update_χ_tail
            update_tail!(χ_m, [0, 0, mP.Ekin_DMFT], iωn_f)
            update_tail!(χ_d, [0, 0, mP.Ekin_DMFT], iωn_f)
            par && update_tail!([0, 0, mP.Ekin_DMFT])
            par && update_tail!([0, 0, mP.Ekin_DMFT])
        end
    end
    reset!(χ_d)
    μnew = mP.μ
    mP.μ = μbak
    converged = converged && all(isfinite.([lhs_c1, E_pot_2]))
    return E_kin, E_pot, μnew, λm, lhs_c1, E_pot_2, converged
end

# ------------------------------------------ fast versions  ------------------------------------------
#TODO: these need to be refactored. many code replications (use sum_kω instead)
function f_c1(χ::Matrix, λ::Float64, kMult::Vector{Float64}, 
                 tail::Vector{Float64})::Float64
    res  = 0.0
    resi = 0.0
    norm = sum(kMult)
    for (i,ωi) in enumerate(tail)
        resi = 0.0
        for (ki,km) in enumerate(kMult)
            resi += χ_λ(χ[ki,i],λ) * km
        end
        res += resi/norm - ωi
    end
    return res
end 

function df_c1(χ::Matrix{Float64}, λ::Float64, kMult::Vector{Float64}, 
                tail::Vector{Float64})::Float64
    res = 0.0
    resi = 0.0
    norm = sum(kMult)
    for (i,ωi) in enumerate(tail)
        resi = 0.0
        for (ki,km) in enumerate(kMult)
            resi += dχ_λ(χ[ki,i],λ) * km
        end
        res += resi/norm - ωi
    end
    return res
end

