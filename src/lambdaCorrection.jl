using IntervalArithmetic
using IntervalRootFinding
χ_λ(χ, λ::Union{Float64,Interval{Float64}}) = map(χi -> 1.0 / ((1.0 / χi) + λ), χ)
χ_λ!(χ_λ, χ, λ::Union{Float64,Interval{Float64}}) = (χ_λ = map(χi -> 1.0 / ((1.0 / χi) + λ), χ))
dχ_λ(χ, λ::Union{Float64,Interval{Float64}}) = - ((1 ./ χ) .+ λ)^(-2)

dΣch_λ_amp(G_plus_νq, γch, dχch_λ, qNorm) = -sum(G_plus_νq .* γch .* dχch_λ)*qNorm
dΣsp_λ_amp(G_plus_νq, γsp, dχsp_λ, qNorm) = -1.5*sum(G_plus_νq .* γsp .* dχsp_λ)*qNorm

function calc_λsp_rhs_usable(impQ_sp::ImpurityQuantities, impQ_ch::ImpurityQuantities, nlQ_sp::NonLocalQuantities,
                      nlQ_ch::NonLocalQuantities, sP::SimulationParameters, mP::ModelParameters)
    @warn "currently using min(usable_sp, usable_ch) for all calculations. relax this?"
    χch_ω = sum_q(nlQ_ch.χ, qMultiplicity, dims=[2])[:,1]
    usable_ω = intersect(nlQ_sp.usable_ω, nlQ_ch.usable_ω)
    @info """Found usable intervals for non-local susceptibility of length 
          sp: $(nlQ_sp.usable_ω), length: $(length(nlQ_sp.usable_ω))
          ch: $(nlQ_ch.usable_ω), length: $(length(nlQ_ch.usable_ω))
          usable: $(usable_ω), length: $(length(usable_ω))"""
    χch_sum = real(sum_freq(χch_ω[usable_ω], [1], sP.tail_corrected, mP.β)[1])
    @info sP.tail_corrected , mP.n/2 , χch_sum , real(impQ_ch.χ_loc) , real(impQ_sp.χ_loc), real(χch_sum)
    rhs = sP.tail_corrected ? mP.n * (1 - mP.n/2) - χch_sum : real(impQ_ch.χ_loc + impQ_sp.χ_loc - χch_sum)
    return rhs, usable_ω
end

function calc_λsp_correction!(nlQ_sp::NonLocalQuantities, usable_ω::UnitRange{Int64}, 
                             rhs::Float64, qMult::Array{Float64,1}, β::Float64, tc::Bool, χFillType)
    λsp,χsp_λ = calc_λsp_correction(nlQ_sp.χ, usable_ω, rhs, qMultiplicity, 
                                    modelParams.β, simParams.tail_corrected, simParams.χFillType)
    nlQ_sp = NonLocalQuantities(χsp_λ, nlQ_sp.γ, nlQ_sp.usable_ω, λsp)
end

function calc_λsp_correction(χ_in::SharedArray{Complex{Float64},2}, usable_ω::UnitRange{Int64}, 
                             rhs::Float64, qMult::Array{Float64,1}, β::Float64, tc::Bool, χFillType)
    @info "Using rhs for lambda correction: " rhs " with tc = " tc
    res = zeros(eltype(χ_in), size(χ_in)...)
    χr    = real.(χ_in[usable_ω,:])
    nh    = ceil(Int64, size(χr,1)/2)
    W = tc ? build_weights(floor(Int64,size(χr, 1)/4), floor(Int64,size(χr, 1)/2), [0,1,2,3]) : nothing
    f(λint) = sum_freq(sum_q(χ_λ(χr, λint), qMult, dims=[2])[:,1], [1], tc, β, weights=W)[1] - rhs
    df(λint) = sum_freq(sum_q(-χ_λ(χr, λint) .^ 2, qMult, dims=[2])[:,1], [1], tc, β, weights=W)[1]
    χ_min    = -minimum(1 ./ χr[nh,:])
    int = [χ_min, χ_min + 10/length(qMult)]
    @info "found " χ_min ". Looking for roots in intervall " int
    X = @interval(int[1],int[2])
    r = roots(f, df, X, Newton, 1e-10)
    @info "possible roots: " r

    if isempty(r)
       @warn "   ---> WARNING: no lambda roots found!!!"
       return 0, χ_λ(χr, 0)
    else
        λsp = mid(maximum(interval.(r)))
        @info "Found λsp " λsp
        if χFillType == zero_χ_fill
            res[usable_ω,:] =  χ_λ(χ_in[uable_ω,:], λsp) 
        elseif χFillType == lambda_χ_fill
            res =  χ_λ(χ_in, λsp) 
        else
            copy!(res, χsp) 
            res[usable_ω,:] =  χ_λ(χ_in[usable_sp,:], λsp) 
        end
        return λsp, convert(SharedArray,res)
    end
end


function calc_E_pot_cond(λsp::Float64, λch::Float64, nlQ_sp::NonLocalQuantities, nlQ_ch::NonLocalQuantities, bubble::BubbleT, 
                         ϵkGrid::Base.Generator, FUpDo::Array{Complex{Float64},3}, 
                         Σ_loc_pos, Σ_ladderLoc,
                         qIndices::qGridT, qM, mP::ModelParameters, sP::SimulationParameters, tc::Bool)
    # --- prepare auxiliary vars ---
    gridShape = repeat([sP.Nk], mP.D)
    transform = reduce_kGrid ∘ ifft_cut_mirror ∘ ifft 
    transformK(x) = fft(expand_kGrid(qIndices, x))
    transformG(x) = reshape(x, gridShape...)

    Gνω = convert(SharedArray,Gfft_from_Σ(Σ_loc_pos, ϵkGrid, -simParams.n_iω:(simParams.n_iν+simParams.n_iω-1), mP))
    ϵqGrid = reduce_kGrid(cut_mirror(collect(ϵkGrid)));
    νGrid = 0:sP.n_iν-1
    usable_ω = intersect(nlQ_sp.usable_ω, nlQ_ch.usable_ω)
    Σ_hartree = mP.n * mP.U/2
    norm = mP.β * sP.Nk^mP.D
    E_pot_tail_c = (mP.U^2 * 0.5 * mP.n * (1-0.5*mP.n) .+ Σ_hartree .* (ϵqGrid .+ Σ_hartree .- mP.μ))
    E_pot_tail = E_pot_tail_c' ./ (iν_array(mP.β, νGrid) .^ 2)
    E_pot_tail_inv = sum((mP.β/2)  .* [Σ_hartree .* ones(size(ϵqGrid)), (-mP.β/2) .* E_pot_tail_c])

    Wν    = tc ? build_weights(Int(floor(sP.n_iν*3/5)), sP.n_iν, [0,1,2,3]) : nothing
    Wω    = tc ? build_weights(Int(floor(last(usable_ω .- sP.n_iω)*3/5)), Int(last(usable_ω) - sP.n_iω), [0,1,2,3]) : nothing

    Σ_ladder_ω = SharedArray{Complex{Interval{Float64}},3}(length(usable_ω), length(1:sP.n_iν), length(qIndices))
    tmp = SharedArray{Complex{Float64},3}(length(usable_ω), size(bubble,2), sP.n_iν)
    Σ_internal2!(tmp, usable_ω, bubble, view(FUpDo,:,(sP.n_iν+1):size(FUpDo,2),:), tc, Wν)

    # --- Calculate new Sigma ---
    function find_roots(λspi, λchi)
        χsp_λ = SharedArray(χ_λ(nlQ_sp.χ, λspi))
        χch_λ = SharedArray(χ_λ(nlQ_ch.χ, λchi))
        Σ_internal!(Σ_ladder_ω, usable_ω, χsp_λ, χch_λ,
                    view(nlQ_sp.γ,:,:,(sP.n_iν+1):size(nlQ_sp.γ,3)), view(nlQ_ch.γ,:,:,(sP.n_iν+1):size(nlQ_ch.γ,3)),
                    Gνω, tmp, mP.U, transformG, transformK, transform)
        Σ_new = mP.U .* sum_freq(Σ_ladder_ω, [1], tc, 1.0, weights=Wω)[1,:,:] ./ norm
        Σ_corr = Σ_new .- Σ_ladderLoc .+ Σ_loc_pos[eachindex(Σ_ladderLoc)] .+ Σ_hartree
        G_corr = flatten_2D(G_from_Σ(Σ_corr, ϵqGrid, νGrid, mP));
        E_pot  = calc_E_pot(G_corr, Σ_corr, E_pot_tail, E_pot_tail_inv, qM, norm)
        println("internal: ", λspi, ", ", λchi)
        return mP.U * sum(real.(χch_λ[usable_ω,:] .- χsp_λ[usable_ω,:])) ./ norm  - E_pot
    end
    Xsp = @interval(-1,1)
    res_roots = roots(x->find_roots(x,0.0), Xsp, Bisection, 1e-6);
    return res_roots
end