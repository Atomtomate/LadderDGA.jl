using IntervalArithmetic
using IntervalRootFinding
χ_λ(χ, λ) = map(χi -> 1.0 / ((1.0 / χi) + λ), χ)
χ_λ!(χ_λ, χ, λ) = (χ_λ = map(χi -> 1.0 / ((1.0 / χi) + λ), χ))

dχ_λ(χ, λ) = - ((1 ./ χ) .+ λ)^(-2)

#println("possible roots: ", Optim.minimizer(r))
dΣch_λ_amp(G_plus_νq, γch, dχch_λ, qNorm) = -sum(G_plus_νq .* γch .* dχch_λ)*qNorm
dΣsp_λ_amp(G_plus_νq, γsp, dχsp_λ, qNorm) = -1.5*sum(G_plus_νq .* γsp .* dχsp_λ)*qNorm


function calc_λ_correction(χr, rhs, qMult,  simParams, modelParams)
    @assert typeof(χr) <: Array{Float64}
    @assert typeof(rhs) <: Real

    qNorm = sum(qMult)*modelParams.β
    χ_new(λ) = map(χi -> 1.0 / ((1.0 / χi) + λ), χr)
    f(λint)  = sum(sum(χ_new(λint), dims=1)[1,:] .* qMult) / qNorm - rhs
    df(λint)  = sum(sum(- χ_new(λint) .^ 2, dims=1)[1,:] .* qMult) / qNorm - rhs
    nh       = ceil(Int64, size(χr,1)/2)
    χ_min    = -minimum(1 ./ χr[nh,:])
    #
    int = [χ_min - 1/simParams.Nk^modelParams.D, χ_min + 1/simParams.Nk]
    println("found χ_min = ", printr_s(χ_min), ". Looking for roots in intervall [", 
            printr_s(int[1]), ", ", printr_s(int[2]), "]")
    X = @interval(int[1],int[2])
    r = roots(f, df, X, Newton)
    #r = roots(f, df, int[1]..int[2], IntervalRootFinding.Newton, 1e-10)
    println("possible roots: ", r)
    if isempty(r)
       println(stderr, "   ---> WARNING: no lambda roots found!!!")
       return 0, χ_new(0)
    else
        max_int = maximum(interval.(r))
        λ = mid(max_int)
        return λ, χ_new(λ)
    end
end


function eval_f(x, G0, χch, χsp, γch, γsp,
                             bubble, Σ_loc, FUpDo, qMult, qGrid, kGrid,
                             modelParams::ModelParameters, simParams::SimulationParameters)

	χch_λ = χ_λ(χch, x[1])
	χsp_λ = χ_λ(χsp, x[2])
	Σ_λ = calc_DΓA_Σ(χch_λ, χsp_λ, γch, γsp, bubble, Σ_loc, FUpDo, qMult, qGrid, kGrid, modelParams, simParams)
	res1 = sum(χch_λ + χsp_λ)/(2*modelParams.β) - (1-modelParams.n/2)*(modelParams.n/2)
	tmp = mapslices(x -> 1 ./ G0[1:size(Σ_λ,2)].- x, Σ_λ; dims=[2])
	res2 += sum(χch_λ .- χsp_λ) ./ (2*modelParams.β) .- sum(Σ_λ ./ (tmp))
    return SVector(res1, res2)
end

function find_λ(G0, χch, χsp, trilexch, trilexsp,
                             bubble, Σ_loc, FUpDo, ϵkGrid, qIndices, usable_ω, 
                             modelParams::ModelParameters, simParams::SimulationParameters)

    function construct_f( (λch, λsp))
		χch_λ = χ_λ(χch, (λch.hi + λch.lo)/2 )
		χsp_λ = χ_λ(χsp, (λsp.hi + λsp.lo)/2 )

        Σ_λ = Interval.(calc_DΓA_Σ_fft(χsp_λ, χch_λ, trilexsp, trilexch, bubble, Σ_loc, FUpDo, 
                                       ϵkGrid, qIndices, usable_ω, modelParams, simParams))
		res1 = real(sum(χch_λ + χsp_λ)/(2*modelParams.β) - (1-modelParams.n/2)*(modelParams.n/2))
		tmp = mapslices(x -> 1 ./ G0[1:size(Σ_λ,2)].- x, Σ_λ; dims=[2])
		res2 = real(sum(χch_λ .- χsp_λ) ./ (2*modelParams.β) .- sum(Σ_λ ./ (tmp)))
        #Kriterien: χsp/ch(ω=0) > 0, TODO: checken
        #U>0 => χch, χsp >= 0 UND χch soll kleiner, χsp groesser werden
        #U<0 => λsp < 0, λch > 0
        return SVector(res1, res2)
	end

    function df(x)
		χch_λ = χ_λ(χch, x[1])
		dχch_λ = dχ_λ(χch, x[1])

		χsp_λ = χ_λ(χsp, x[2])
		dχsp_λ = dχ_λ(χsp, x[2])
        res_ch = res/β
    end
    #J = gradient(construct_f!, [0.0, 0.0])
    #optimize(construct_f, init) #, Newton(); autodiff = :forward
    X = -1..1
    IntervalRootFinding.roots(construct_f, X × X)
end
#find_λ(G0, χch, χsp, trilexch, trilexsp, bubble, Σ_loc, FUpDo, ϵkGrid, qIndices, usable_ω, modelParams, simParams)
