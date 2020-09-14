using IntervalArithmetic
using IntervalRootFinding
χ_λ(χ, λ::Union{Float64,Interval{Float64}}) = map(χi -> 1.0 / ((1.0 / χi) + λ), χ)
χ_λ!(χ_λ, χ, λ::Union{Float64,Interval{Float64}}) = (χ_λ = map(χi -> 1.0 / ((1.0 / χi) + λ), χ))
dχ_λ(χ, λ::Union{Float64,Interval{Float64}}) = - ((1 ./ χ) .+ λ)^(-2)

#println("possible roots: ", Optim.minimizer(r))
dΣch_λ_amp(G_plus_νq, γch, dχch_λ, qNorm) = -sum(G_plus_νq .* γch .* dχch_λ)*qNorm
dΣsp_λ_amp(G_plus_νq, γsp, dχsp_λ, qNorm) = -1.5*sum(G_plus_νq .* γsp .* dχsp_λ)*qNorm

function calc_λsp_rhs_usable(impQ_sp::ImpurityQuantities, impQ_ch::ImpurityQuantities, nlQ_sp::NonLocalQuantities,
                      nlQ_ch::NonLocalQuantities, sP::SimulationParameters, mP::ModelParameters)
    usable_ω = sP.fullRange ? (1:length(nlQ_sp.χ_physical)) : intersect(nlQ_sp.usable_ω, nlQ_ch.usable_ω)
    @info """Found usable intervals for non-local susceptibility of length 
          sp: $(nlQ_sp.usable_ω), length: $(length(nlQ_sp.usable_ω))
          ch: $(nlQ_ch.usable_ω), length: $(length(nlQ_ch.usable_ω))
          usable: $(usable_ω), length: $(length(usable_ω))"""
    χch_sum = real(sum_freq(nlQ_ch.χ_physical[usable_ω], [1], sP.tail_corrected, mP.β)[1])
    rhs = sP.tail_corrected ? mP.n/2 - χch_sum : real(impQ_ch.χ_loc + impQ_sp.χ_loc - χch_sum)
    return rhs, usable_ω
end

function calc_λsp_correction(χ_in, usable_ω, rhs::Float64, qMult::Array{Float64,1}, β::Float64, tc::Bool, χFillType)
    @info "Using rhs for lambda correction: " rhs " with tc = " tc
    res = zeros(eltype(χ_in), size(χ_in)...)
    W = tc ? build_weights(floor(Int64,size(χ_in, 1)/4), floor(Int64,size(χ_in, 1)/2), [0,1,2,3]) : nothing
    χr    = real.(χ_in)
    nh    = ceil(Int64, size(χr,1)/2)
    f(λint) = sum_freq(sum_q(χ_λ(χr, λint), qMult', dims=[2])[:,1], [1], tc, β, weights=W)[1] - rhs
    df(λint) = sum_freq(sum_q(-χ_λ(χr, λint) .^ 2, qMult', dims=[2])[:,1], [1], tc, β, weights=W)[1]
    χ_min    = -minimum(1 ./ χr[nh,:])
    int = [χ_min - 1/length(qMult)^3, χ_min + 1/length(qMult)]
    @info "found " χ_min ". Looking for roots in intervall " int
    X = @interval(int[1],int[2])
    r = roots(f, df, X, Newton)
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
        return λsp, res
    end
end
