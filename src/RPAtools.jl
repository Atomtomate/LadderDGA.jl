#TODO: impement easy way to generate an  ϵqGrid, doesn't work until this is done

function calc_bubble_analytic_frequency_sum(q::NTuple,kG::KGrid,β::Float64,μ::Float64,ω::Float64)
	nk = fermi_function.(β, kG.ϵkGrid, μ)
	nkq = fermi_function.(β, kG.ϵkGrid, μ) #add eqGrid here
	return -kintegrate(kG,(nk-nkq)/(1im*ω-kG.ϵkGrid+kg.ϵkGrid)) #add eqGrid here
end

function calc_RPA(bubble,U::Float64)
	return bubble/(1-U*bubble)
end

function fermi_function(β::Float64,ϵ::Float64,μ::Float64)
	return 1/(1+exp(β*(ϵ-μ)))
end