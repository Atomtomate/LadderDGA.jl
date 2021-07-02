using IntervalArithmetic
using IntervalRootFinding

χ_λ(χ::AbstractArray, λ::Union{Float64,Interval{Float64}}) = map(χi -> 1.0 / ((1.0 / χi) + λ), χ)
χ_λ2(χ, λ) = map(χi -> 1.0 / ((1.0 / χi) + λ), χ)
function χ_λ!(χ_λ, χ, λ::Union{Float64,Interval{Float64}})
    for i in eachindex(χ_λ)
        χ_λ[i] = 1.0 / ((1.0 / χ[i]) + λ)
    end
end

dχ_λ(χ, λ::Union{Float64,Interval{Float64}}) = map(χi -> - ((1.0 / χi) + λ)^(-2), χ)

dΣch_λ_amp(G_plus_νq, γch, dχch_λ, qNorm) = -sum(G_plus_νq .* γch .* dχch_λ)*qNorm
dΣsp_λ_amp(G_plus_νq, γsp, dχsp_λ, qNorm) = -1.5*sum(G_plus_νq .* γsp .* dχsp_λ)*qNorm

function new_χλ(χ_in::SharedArray{Complex{Float64},2}, λ::Float64, sP::SimulationParameters)
    res = SharedArray{eltype(χ_in), ndims(χ_in)}(size(χ_in)...)
    if sP.χFillType == zero_χ_fill
        res[usable_ω,:] =  χ_λ(χ_in[uable_ω,:], λ) 
    elseif sP.χFillType == lambda_χ_fill
        res =  χ_λ(χ_in, λ) 
    else
        res[:] = deepcopy(χsp)
        res[usable_ω,:] =  χ_λ(χ_in[usable_ω,:], λ) 
    end
    return res
end

function calc_λsp_rhs_usable(impQ_sp::ImpurityQuantities, impQ_ch::ImpurityQuantities, nlQ_sp::NonLocalQuantities, nlQ_ch::NonLocalQuantities, kGrid::T1, mP::ModelParameters, sP::SimulationParameters) where T1 <: ReducedKGrid
    usable_ω = intersect(nlQ_sp.usable_ω, nlQ_ch.usable_ω)
    χch_ω = kintegrate(kGrid, nlQ_ch.χ[usable_ω,:], dim=2)[:,1]
    @warn "currently using min(usable_sp, usable_ch) = min($(nlQ_sp.usable_ω),$(nlQ_ch.usable_ω)) = $(usable_ω) for all calculations. relax this?"

    #sh = get_sum_helper(usable_ω, sP, :b)
    iωn = 1im .* 2 .* (-sP.n_iω:sP.n_iω)[usable_ω] .* π ./ mP.β
    χch_ω_sub = subtract_tail(χch_ω, impQ_ch.tailCoeffs[3], iωn)
    χch_sum = real(sum_freq(χch_ω_sub, [1], Naive(), mP.β, corr=-impQ_ch.tailCoeffs[3]*mP.β^2/12)[1])

    #χupup_ω = 0.5*(χLocsp_ω .+ χLocch_ω)
    #χupup_DMFT_ω_sub = subtract_tail(χupup_ω[loc_range], E_kin_ED, iωn)
    #imp_density = real(sum_freq(χupup_DMFT_ω_sub, [1], Naive(), mP.β, corr=-E_kin_ED*mP.β^2/12))

    rhs = ((sP.tc_type_b != :nothing && sP.λ_rhs == :native) || sP.λ_rhs == :fixed) ? mP.n * (1 - mP.n/2) - χch_sum : real(impQ_ch.χ_loc + impQ_sp.χ_loc - χch_sum)
    #@info "tc:  $(sP.tc_type), rhs =  $rhs , $χch_sum , $(real(impQ_ch.χ_loc)) , $(real(impQ_sp.χ_loc)), $(real(χch_sum))"

    @info """Found usable intervals for non-local susceptibility of length 
          sp: $(nlQ_sp.usable_ω), length: $(length(nlQ_sp.usable_ω))
          ch: $(nlQ_ch.usable_ω), length: $(length(nlQ_ch.usable_ω))
          usable: $(usable_ω), length: $(length(usable_ω))
          χch sum = $(χch_sum), rhs = $(rhs)"""

    return rhs, usable_ω
end

function λsp(χr::Array{Float64,2}, iωn::Array{Complex{Float64},1}, EKin::Float64,
                            rhs::Float64, kGrid::T1, mP::ModelParameters) where T1 <: ReducedKGrid
    #tmp = Array{Float64,2}(undef, size(χr)...)
    f(λint) = sum_freq(subtract_tail(kintegrate(kGrid, χ_λ(χr, λint), dim=2)[:,1],EKin, iωn), [1], Naive(), mP.β, corr=-EKin*mP.β^2/12)[1] - rhs
    df(λint) = sum_freq(kintegrate(kGrid, -χ_λ(χr, λint) .^ 2, dim=2)[:,1], [1], Naive(), mP.β)[1]

    nh    = ceil(Int64, size(χr,1)/2)
    χ_min    = -minimum(1 ./ χr[nh,:])
    λsp_old = newton_right(χr, f, df, χ_min)
    return λsp_old
end

function calc_λsp_correction(χ_in::SharedArray{Complex{Float64},2}, usable_ω::AbstractArray{Int64},
                            searchInterval::AbstractArray{Float64,1}, EKin::Float64,
                            rhs::Float64, kGrid::T1, mP::ModelParameters, sP::SimulationParameters) where T1 <: ReducedKGrid
    χr    = real.(χ_in[usable_ω,:])
    iωn = 1im .* 2 .* (-sP.n_iω:sP.n_iω)[usable_ω] .* π ./ mP.β
    f(λint) = sum_freq(subtract_tail(kintegrate(kGrid, χ_λ(χr, λint), dim=2)[:,1],EKin, iωn), [1], Naive(), mP.β, corr=-EKin*mP.β^2/12)[1] - rhs
    df(λint) = sum_freq(kintegrate(kGrid, -χ_λ(χr, λint) .^ 2, dim=2)[:,1], [1], Naive(), mP.β)[1]

    #TODO: new method needs testing
     X = @interval(0.02,20.0)
     tol = maximum([1e-7, (1e-4)*abs(searchInterval[2] - searchInterval[1])])
     #r = roots(f, df, X, Newton, tol)
     r2 = find_zeros(f, 0.00, 20.0, verbose=true)
     @info "Method 2 root:" r2
     #ln = empty(r2) ? 0.0 : mid(maximum(filter(x->!isempty(x),interval.(r))))
     #@info "Interval" ln
     

    nh    = ceil(Int64, size(χr,1)/2)
    χ_min    = -minimum(1 ./ χr[nh,:])
    λsp_old = newton_right(χr, f, df, χ_min)
    # if isempty(r) 
    #   @warn "   ---> WARNING: no lambda roots with new method found!!!"
    # end
    # λsp = mid(maximum(filter(x->!isempty(x),interval.(r))))
    # if !isempty(r) && abs2(λsp_old - λsp) > 1e-5
    #    @warn "   ---> WARNING: old and new λ not matching!!! $(λsp_old) != $(λsp)"
    # end
    @info "Found λsp " λsp_old
    return λsp_old, new_χλ(χ_in, λsp_old, sP)
end


function calc_λch_correction(nlQ_sp::NonLocalQuantities, nlQ_ch::NonLocalQuantities, bubble::BubbleT, 
        Gνω::AbstractArray{Complex{Float64},2}, FUpDo::AbstractArray{Complex{Float64},3}, λch::Float64,
        Σ_loc_pos, Σ_ladderLoc,kGrid::ReducedKGrid, tail_coeffs_upup, tail_coeffs_updo, mP::ModelParameters, sP::SimulationParameters; λsp_guess=0.1)
    # --- prepare auxiliary vars ---
    νZero = sP.n_iν
    ωZero = sP.n_iω
    ωindices = intersect(nlQ_sp.usable_ω, nlQ_ch.usable_ω)
    iωn = 1im .* 2 .* (-sP.n_iω:sP.n_iω)[ωindices] .* π ./ mP.β
    sh_f = get_sum_helper(2*sP.n_iν, sP, :f)

    tmp = SharedArray{Complex{Float64},3}(length(ωindices), size(bubble,2), size(bubble,3))
    Σ_ladder_ω = SharedArray{Complex{Float64},3}(length(ωindices), length(kGrid.kInd), trunc(Int,sP.n_iν-sP.shift*sP.n_iω/2))
    χupdo_ω = SharedArray{eltype(nlQ_sp.χ),1}(length(ωindices))
    χupup_ω = SharedArray{eltype(nlQ_sp.χ),1}(length(ωindices))
    χsp_λ = SharedArray{eltype(nlQ_sp.χ),2}(length(ωindices),size(nlQ_sp.χ,2))
    χch_λ = SharedArray{eltype(nlQ_sp.χ),2}(length(ωindices),size(nlQ_ch.χ,2))

    Σ_hartree = mP.n * mP.U/2
    E_pot_tail_c = (mP.U^2 * 0.5 * mP.n * (1-0.5*mP.n) .+ Σ_hartree .* (kGrid.ϵkGrid .+ Σ_hartree .- mP.μ))
    E_pot_tail = E_pot_tail_c' ./ (iν_array(mP.β, 0:sP.n_iν-1) .^ 2)
    E_pot_tail_inv = sum((mP.β/2)  .* [Σ_hartree .* ones(size(kGrid.ϵkGrid)), (-mP.β/2) .* E_pot_tail_c])

    Σ_internal2!(tmp, ωindices, bubble, FUpDo, sh_f)

    χsp_λ = nlQ_sp.χ

        χch_λ = SharedArray(χ_λ(nlQ_ch.χ, λch))
        Σ_internal!(Σ_ladder_ω, ωindices, ωZero, νZero, sP.shift, χsp_λ, χch_λ, nlQ_sp.γ, nlQ_ch.γ, Gνω, tmp, mP.U, kGrid)
        Σ_new = permutedims(mP.U .* sum_freq(Σ_ladder_ω, [1], Naive(), mP.β)[1,:,:], [2, 1])
        Σ_λ = Σ_new .- Σ_ladderLoc[1:size(Σ_new,1)] .+ Σ_loc_pos[1:size(Σ_new,1)]

        
        G_λ = G_from_Σ(Σ_λ .+ Σ_hartree, kGrid.ϵkGrid, 0:size(Σ_new, 1)-1, mP)
        E_pot_DGA = calc_E_pot(kGrid, flatten_2D(G_λ), Σ_λ, view(E_pot_tail,1:size(Σ_new, 1),:), E_pot_tail_inv)

        for (wi,w) in enumerate(ωindices)
            χupup_ω[wi] = kintegrate(kGrid, χch_λ[w,:] .+ χsp_λ[w,:])[1] ./ 2
            χupdo_ω[wi] = kintegrate(kGrid, χch_λ[w,:] .- χsp_λ[w,:])[1] ./ 2
        end

        χch_ω_sub = subtract_tail(χupup_ω, tail_coeffs_upup[3], iωn)
        lhs_c2 = real(sum_freq(χupdo_ω, [1], Naive(), mP.β)[1])

        rhs_c2 = E_pot_DGA/(mP.U*mP.β) + (mP.n/2) * (mP.n/2)
        #nlsolve(cond_both!, [ λsp_guess; 0.0]).zero
        return lhs_c2, rhs_c2
end

function extended_λ(nlQ_sp::NonLocalQuantities, nlQ_ch::NonLocalQuantities, bubble::BubbleT, 
        Gνω::AbstractArray{Complex{Float64},2}, FUpDo::AbstractArray{Complex{Float64},3}, 
        Σ_loc_pos, Σ_ladderLoc,kGrid::ReducedKGrid, tail_coeffs_upup, tail_coeffs_updo, mP::ModelParameters, sP::SimulationParameters; λsp_guess=0.1)
    # --- prepare auxiliary vars ---
    νZero = sP.n_iν
    ωZero = sP.n_iω
    ωindices = intersect(nlQ_sp.usable_ω, nlQ_ch.usable_ω)
    iωn = 1im .* 2 .* (-sP.n_iω:sP.n_iω)[ωindices] .* π ./ mP.β
    sh_f = get_sum_helper(2*sP.n_iν, sP, :f)
    sh_b = Naive() #get_sum_helper(length(ωindices), sP, :b)

    tmp = SharedArray{Complex{Float64},3}(length(ωindices), size(bubble,2), size(bubble,3))
    Σ_ladder_ω = SharedArray{Complex{Float64},3}(length(ωindices), size(bubble,2), sP.n_iν-sP.shift*(trunc(Int,sP.n_iω/2) + 1))
    χupdo_ω = SharedArray{eltype(nlQ_sp.χ),1}(length(ωindices))
    χupup_ω = SharedArray{eltype(nlQ_sp.χ),1}(length(ωindices))
    χsp_λ = SharedArray{eltype(nlQ_sp.χ),2}(length(ωindices),size(nlQ_sp.χ,2))
    χch_λ = SharedArray{eltype(nlQ_sp.χ),2}(length(ωindices),size(nlQ_ch.χ,2))

    Σ_hartree = mP.n * mP.U/2
    E_pot_tail_c = (mP.U^2 * 0.5 * mP.n * (1-0.5*mP.n) .+ Σ_hartree .* (kGrid.ϵkGrid .+ Σ_hartree .- mP.μ))
    E_pot_tail = E_pot_tail_c' ./ (iν_array(mP.β, 0:sP.n_iν-1) .^ 2)
    E_pot_tail_inv = sum((mP.β/2)  .* [Σ_hartree .* ones(size(kGrid.ϵkGrid)), (-mP.β/2) .* E_pot_tail_c])

    Σ_internal2!(tmp, ωindices, bubble, FUpDo, sh_f)

    function cond_both!(F, λ)
        χsp_λ = SharedArray(χ_λ(nlQ_sp.χ, λ[1]))
        χch_λ = SharedArray(χ_λ(nlQ_ch.χ, λ[2]))
        Σ_internal!(Σ_ladder_ω, ωindices, ωZero, νZero, sP.shift, χsp_λ, χch_λ, nlQ_sp.γ, nlQ_ch.γ, Gνω, tmp, mP.U, kGrid)
        Σ_new = permutedims(mP.U .* sum_freq(Σ_ladder_ω, [1], Naive(), mP.β)[1,:,:], [2, 1])
        Σ_λ = Σ_new .- Σ_ladderLoc[1:size(Σ_new,1)] .+ Σ_loc_pos[1:size(Σ_new,1)]

        
        G_λ = G_from_Σ(Σ_λ .+ Σ_hartree, kGrid.ϵkGrid, 0:size(Σ_new, 1)-1, mP)
        E_pot_DGA = calc_E_pot(kGrid, flatten_2D(G_λ), Σ_λ, view(E_pot_tail,1:size(Σ_new, 1),:), E_pot_tail_inv)

        for (wi,w) in enumerate(ωindices)
            χupup_ω[wi] = kintegrate(kGrid, χch_λ[w,:] .+ χsp_λ[w,:])[1] ./ 2
            χupdo_ω[wi] = kintegrate(kGrid, χch_λ[w,:] .- χsp_λ[w,:])[1] ./ 2
        end

        χch_ω_sub = subtract_tail(χupup_ω, tail_coeffs_upup[3], iωn)
        lhs_c1 = real(sum_freq(χupup_ω, [1], sh_b, mP.β, corr=-tail_coeffs_upup[3]*mP.β^2/12)[1])
        lhs_c2 = real(sum_freq(χupdo_ω, [1], sh_b, mP.β)[1])

        rhs_c1 = mP.n/2 * (1 - mP.n/2)
        rhs_c2 = E_pot_DGA/mP.U + mP.U * (mP.n/2) * (mP.n/2)
        F[1] = lhs_c1 - rhs_c1
        F[2] = lhs_c2 - rhs_c2
    end
    return nlsolve(cond_both!, [ λsp_guess; 0.0]).zero
end

function λsp_correction_search_int(χr::AbstractArray{Float64,2}, kGrid::ReducedKGrid, mP::ModelParameters; init=nothing, init_prct::Float64 = 0.1)
    nh    = ceil(Int64, size(χr,1)/2)
    χ_min    = -minimum(1 ./ χr[nh,:])
    int = if init === nothing
        rval = χ_min > 0 ? χ_min + 10/kGrid.Ns + 20/mP.β : minimum([20*abs(χ_min), χ_min + 10/kGrid.Ns + 20/mP.β])
        [χ_min, rval]
    else
        [init - init_prct*abs(init), init + init_prct*abs(init)]
    end
    @info "found " χ_min ". Looking for roots in intervall $(int)" 
    return int
end

function λ_correction!(impQ_sp, impQ_ch, FUpDo, Σ_loc_pos, Σ_ladderLoc, nlQ_sp::NonLocalQuantities, nlQ_ch::NonLocalQuantities, 
                      bubble::BubbleT, Gνω::SharedArray{Complex{Float64},2}, 
                      kGrid::ReducedKGrid,
                      mP::ModelParameters, sP::SimulationParameters; init_sp=nothing, init_spch=nothing)
    @info "Computing λsp corrected χsp, using " sP.χFillType " as fill value outside usable ω range."
    rhs,usable_ω_λc = calc_λsp_rhs_usable(impQ_sp, impQ_ch, nlQ_sp, nlQ_ch, kGrid, mP, sP)
    searchInterval_sp = λsp_correction_search_int(real.(nlQ_sp.χ[usable_ω_λc,:]), kGrid, mP, init=init_sp)
    searchInterval_spch = [-Inf, Inf]
    λsp,χsp_λ = calc_λsp_correction(nlQ_sp.χ, usable_ω_λc, searchInterval_sp, impQ_sp.tailCoeffs[3] , rhs, kGrid, mP, sP)
    #@info "Computing λsp corrected χsp, using " sP.χFillType " as fill value outside usable ω range."
    #λ_new = extended_λ(nlQ_sp, nlQ_ch, bubble, Gνω, FUpDo, Σ_loc_pos, Σ_ladderLoc, kGrid, impQ_sp.tailCoeffs, [0.0,0.0,0.0,0.0,0.0], mP, sP; λsp_guess=λsp - 0.001)
    λ_new = [0.0, 0.0]
    if sP.λc_type == :sp
        nlQ_sp.χ = χsp_λ
        nlQ_sp.λ = λsp
    elseif sP.λc_type == :sp_ch
        nlQ_sp.χ = SharedArray(χ_λ(nlQ_sp.χ, λ_new[1]))
        nlQ_sp.λ = λ_new[1]
        nlQ_ch.χ = SharedArray(χ_λ(nlQ_ch.χ, λ_new[2]))
        nlQ_ch.λ = λ_new[2]
    end
    @info "new lambda correction: λsp=$(λ_new[1]) and λch=$(λ_new[2])"
    return λsp, λ_new, searchInterval_sp, searchInterval_spch
end

function newton_right(χr::Array{Float64,2}, f::Function, df::Function,
                            start::Float64; nsteps=5000, atol=1e-11)
    done = false
    δ = 0.1
    x0 = start + δ
    xi = x0
    i = 1
    while !done
        fi = f(xi)
        dfi = df(xi)
        xlast = xi
        xi = x0 - fi / dfi
        (abs2(xi-x0) < atol) && break
        if xi < x0               # only ever search to the right!
            δ  = δ/2.0
            x0  = start + δ      # reset with smaller delta
            xi = x0
        else
            x0 = xi
        end
        (i >= nsteps ) && (done = true)
        i += 1
    end
    return xi
end
