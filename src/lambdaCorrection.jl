χ_λ(χ, λ) = 1 ./ ((1 ./ χ) .+ λ)
χ_λ!(χ_λ, χ, λ) = (χ_λ = 1 ./ ((1 ./ χ) .+ λ))

dχ_λ(χ, λ) = - ((1 ./ χ) .+ λ)^(-2)
dΣch_λ_amp(G_plus_νq, γch, dχch_λ, qNorm) = -sum(G_plus_νq .* γch .* dχch_λ)*qNorm
dΣsp_λ_amp(G_plus_νq,γsp, dχsp_λ, qNorm) = -1.5*sum(G_plus_νq .* γsp .* dχsp_λ)*qNorm


function custom_newton(x0; steps=1000)
end

function calc_λ_correction(χ, χloc, qMult,  simParams, modelParams)
    #qNorm      = sum(qMult)*modelParams.β
    qNorm       = sum(qMult)*modelParams.β
    χ_new(λ)  = 1.0  ./ (1 ./ χ .+ λ)
    χr = real.(χ)
    χlocr = real(χloc)
    f(λint)  = sum([sum(((1 ./ χr[i,:]) .+ λint).^(-1) .* qMult) for i in 1:size(χr,1)])/qNorm - χlocr
    af(λint)  = abs(sum([sum(((1 ./ χr[i,:]) .+ λint).^(-1) .* qMult) for i in 1:size(χr,1)])/qNorm - χlocr)
    df(λint)  = sum([sum(-((1 ./ χr[i,:]) .+ λint).^(-2) .* qMult) for i in 1:size(χr,1)])/qNorm
    ddf(λint)  = sum([sum(2 .* ((1 ./ χr[i,:]) .+ λint).^(-3) .* qMult) for i in 1:size(χr,1)])/qNorm
    nh  =ceil(Int64, size(χr,1)/2)
    χ_min =  -minimum(1 ./ real(χr)[nh,:]) #TODO ??????
    interval = [χ_min-1.0/length(qMult), χ_min + 1.0/length(qMult)]
    println("found χ_min = ", χ_min, ". Looking for roots in intervall [", interval[1], ", ", interval[2], "]")
    #r = Optim.optimize(af,[χ_min+0.001],  Newton(); inplace=false, autodiff = :forward)
    r = find_zeros(f, interval[1], interval[2])
    #println("possible roots: ", r)
    println("possible roots: ", r)
    #println("possible roots: ", Optim.minimizer(r))
    if isempty(r)
       println(stderr, "   ---> WARNING: no lambda roots found!!!")
       return 0, χ_new(0)
    else
        λ = r[end]
        return λ, χ_new(λ)
    end
end


function eval_f(x, G0, χch, χsp, γch, γsp,
                             bubble, Σ_loc, FUpDo, qMult, qGrid, kGrid,
                             modelParams::ModelParameters, simParams::SimulationParameters)

	χch_λ = χ_λ(χch, x[1])
	χsp_λ = χ_λ(χsp, x[2])
	Σ_λ = calc_DΓA_Σ(χch_λ, χsp_λ, γch, γsp, bubble, Σ_loc, FUpDo, qMult, qGrid, kGrid, modelParams, simParams)
	res = abs2(sum(χch_λ + χsp_λ)/(2*modelParams.β) - (1-modelParams.n/2)*(modelParams.n/2))
	tmp = mapslices(x -> 1 ./ G0[1:size(Σ_λ,2)].- x, Σ_λ; dims=[2])
	res += abs2(sum(χch_λ .- χsp_λ) ./ (2*modelParams.β) .- sum(Σ_λ ./ (tmp)))
	return res
end

function find_λ(G0, χch, χsp, γch, γsp,
                             bubble, Σ_loc, FUpDo, qMult, qGrid, kGrid,
                             modelParams::ModelParameters, simParams::SimulationParameters,
							init, method=Newton())

	function construct_f(x)
		χch_λ = χ_λ(χch, x[1])
		χsp_λ = χ_λ(χsp, x[2])
		Σ_λ = calc_DΓA_Σ(χch_λ, χsp_λ, γch, γsp, bubble, Σ_loc, FUpDo, qMult, qGrid, kGrid, modelParams, simParams)
		res = abs2(sum(χch_λ + χsp_λ)/(2*modelParams.β) - (1-modelParams.n/2)*(modelParams.n/2))
		tmp = mapslices(x -> 1 ./ G0[1:size(Σ_λ,2)].- x, Σ_λ; dims=[2])
		res += abs2(sum(χch_λ .- χsp_λ) ./ (2*modelParams.β) .- sum(Σ_λ ./ (tmp)))
		return res
	end

    function df(x)
		χch_λ = χ_λ(χch, x[1])
		dχch_λ = dχ_λ(χch, x[1])

		χsp_λ = χ_λ(χsp, x[2])
		dχsp_λ = dχ_λ(χsp, x[2])
        
        res_ch = -(n/2)*(1-(n/2)) + sum(dχch_λ)*sum(χch_λ .+ χsp_λ)/(2β) +\
                 #TODO: check consistemcy of nu andk sums with simplified Σ function evaluation
        res_ch = res/β
    end
    #J = gradient(construct_f!, [0.0, 0.0])
    optimize(construct_f, init) #, Newton(); autodiff = :forward
end
