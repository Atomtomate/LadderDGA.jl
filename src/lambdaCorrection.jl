using IntervalArithmetic
using IntervalRootFinding
χ_λ(χ, λ) = map(χi -> 1.0 / ((1.0 / χi) + λ), χ)
χ_λ!(χ_λ, χ, λ) = (χ_λ = map(χi -> 1.0 / ((1.0 / χi) + λ), χ))
dχ_λ(χ, λ) = - ((1 ./ χ) .+ λ)^(-2)

#println("possible roots: ", Optim.minimizer(r))
dΣch_λ_amp(G_plus_νq, γch, dχch_λ, qNorm) = -sum(G_plus_νq .* γch .* dχch_λ)*qNorm
dΣsp_λ_amp(G_plus_νq, γsp, dχsp_λ, qNorm) = -1.5*sum(G_plus_νq .* γsp .* dχsp_λ)*qNorm


function calc_λsp_correction(χ_in, rhs, qMult,  simParams, modelParams)
    χr    = real.(χ_in)
    rhsr  = real(rhs)
    nh    = ceil(Int64, size(χr,1)/2)
    qNorm = sum(qMult)*modelParams.β
    χ_new(λ) = map(χi -> 1.0 / ((1.0 / χi) + λ), χr)
    println("TODO: lambda tc")
    f(λint)  = sum(sum(χ_new(λint), dims=1)[1,:] .* qMult) / qNorm - rhsr
    df(λint) = sum(sum(- χ_new(λint) .^ 2, dims=1)[1,:] .* qMult) / qNorm
    χ_min    = -minimum(1 ./ χr[nh,:])
    int = [χ_min - 1/simParams.Nk^modelParams.D, χ_min + 1/simParams.Nk]
    @info "found " χ_min ". Looking for roots in intervall " int
    X = @interval(int[1],int[2])
    r = roots(f, df, X, Newton)
    @info "possible roots: " r
    if isempty(r)
       @warn "   ---> WARNING: no lambda roots found!!!"
       return 0, χ_new(0)
    else
        max_int = maximum(interval.(r))
        return mid(max_int)
    end
end


function cond_Epot(λsp, λch, χsp, χch, trilexsp, trilexch, bubble, GLoc, FUpDo, 
                   Σ_loc, Σ_ladderLoc, ϵkGrid, qIndices, usable_ω, usable_ν, mP, sP)
	χsp_λ = χ_λ(χsp, λsp)
	χch_λ = χ_λ(χch, λch)
    Σ_λ = calc_DΓA_Σ_fft(χsp_λ, χch_λ, trilexsp, trilexch, bubble, GLoc, FUpDo, 
                         ϵkGrid, qIndices, usable_ω, 1:sP.n_iν, sP.Nk,
                         mP, sP, sP.tail_corrected)
    Σ_λ_corrected = Σ_λ .- Σ_ladderLoc .+ Σ_loc[eachindex(Σ_ladderLoc)]
	tmp = mapslices(x -> 1 ./ GLoc[1:size(Σ_λ,2)].- x, Σ_λ; dims=[2])
	sum(χch_λ .- χsp_λ) ./ (2*mP.β) .- sum(Σ_λ ./ (tmp))
end

function eval_f(x, G0, χch, χsp, γch, γsp,
                             bubble, Σ_loc, FUpDo, qMult, qGrid, kGrid,
                             modelParams::ModelParameters, simParams::SimulationParameters)

	χch_λ = χ_λ(χch, x[1])
	χsp_λ = χ_λ(χsp, x[2])
	Σ_λ = calc_DΓA_Σ(χch_λ, χsp_λ, γch, γsp, bubble, Σ_loc, FUpDo, qMult, qGrid, kGrid, modelParams, simParams)
	res1 = sum(χch_λ + χsp_λ)/(2*modelParams.β) - (1-modelParams.n/2)*(modelParams.n/2)
	tmp = mapslices(x -> 1 ./ G0[1:size(Σ_λ,2)].- x, Σ_λ; dims=[2])
	res2 = sum(χch_λ .- χsp_λ) ./ (2*modelParams.β) .- sum(Σ_λ ./ (tmp))
    return SVector(res1, res2)
end

function find_λ(G0, χch, χsp, trilexch, trilexsp,  bubble, Σ_loc, FUpDo, 
                ϵkGrid, qIndices, qMult, usable_ω, rhs1,
                modelParams::ModelParameters, simParams::SimulationParameters)

    rhs1 = real(rhs1)
    χchr = real.(χch)
    χspr = real.(χsp)
    function construct_f( (λch, λsp))
		χch_λ = χ_λ(χchr, (λch.hi + λch.lo)/2 )
		χsp_λ = χ_λ(χspr, (λsp.hi + λsp.lo)/2 )

        Σ_λ = Interval.(calc_DΓA_Σ_fft(χsp_λ, χch_λ, trilexsp, trilexch, bubble, Σ_loc, FUpDo, 
                                       ϵkGrid, qIndices, usable_ω, 1:simParams.n_iν, 
                                       modelParams, simParams, false))
    f(λint)  = sum(sum(χ_new(λint), dims=1)[1,:] .* qMult) / qNorm - rhsr
        #TODO: subtract local self
        #TODO: LOCAL SUM, from write.f90!!!
        res1 = real(sum(χch_λ + χsp_λ)/(2*modelParams.β)) - rhs1
		tmp = mapslices(x -> 1 ./ G0[1:size(Σ_λ,2)].- x, Σ_λ; dims=[2])
        #TODO: fix q-sum
		res2 = real(sum(χch_λ .- χsp_λ) ./ (2*modelParams.β) .- sum(Σ_λ ./ (tmp)))
        #Kriterien: χsp/ch(ω=0) > 0, TODO: checken
        #U>0 => χch, χsp >= 0 UND χch soll kleiner, χsp groesser werden
        #U<0 => λsp < 0, λch > 0
        return SVector(res1, res2)
	end

    #J = gradient(construct_f!, [0.0, 0.0])
    #optimize(construct_f, init) #, Newton(); autodiff = :forward
    X = -1..1
    IntervalRootFinding.roots(construct_f, X × X)
end
#find_λ(G0, χch, χsp, trilexch, trilexsp, bubble, Σ_loc, FUpDo, ϵkGrid, qIndices, usable_ω, modelParams, simParams)
