χ_λ(χ, λ) = 1 ./ ((1 ./ χ) .+ λ)
χ_λ!(χ_λ, χ, λ) = (χ_λ = 1 ./ ((1 ./ χ) .+ λ))

dχ_λ(χ, λ) = - ((1 ./ χ) .+ λ)^(-2)
dΣch_λ_amp(G_plus_νq, γch, dχch_λ, qNorm) = -sum(G_plus_νq .* γch .* dχch_λ)*qNorm
dΣsp_λ_amp(G_plus_νq,γsp, dχsp_λ, qNorm) = -1.5*sum(G_plus_νq .* γsp .* dχsp_λ)*qNorm



#TODO: compute start point according to fortran code
function calc_λ_correction(χ, χloc, qMult, modelParams)
    qMult_tmp = reshape(qMult, 1, (size(qMult)...))
    qNorm      = sum(qMult)*modelParams.β
    χ_new(λ)  = (1.0 + 0.0im) ./ (1 ./ χ .+ λ)
    println("aa")
    println(size(sum([sum((1 ./ χ[i,:] .+ 1).^(-1) .* qMult) for i in 1:size(χ,1)])/qNorm))
    f(λ)  = real(sum([sum((1 ./ χ[i,:] .+ λ).^(-1) .* qMult) for i in 1:size(χ,1)])/qNorm - χloc)
    df(λ) = -1*real(sum([sum((1 ./ χ[i,:] .+ λ).^(-2) .* qMult) for i in 1:size(χ,1)])/qNorm) 
    ddf(λ) = 2*real(sum([sum((1 ./ χ[i,:] .+ λ).^(-3) .* qMult) for i in 1:size(χ,1)])/qNorm) 
    start_val = -1.0 #maximum(real(χ[ceil(Int64, size(χ,1)/2), :])) - 1.0
    λ    = Optim.minimizer(Optim.optimize(f, -10, 10))
    return λ, χ_new(λ)
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
    println(construct_f(init))
    optimize(construct_f, init) #, Newton(); autodiff = :forward
end
