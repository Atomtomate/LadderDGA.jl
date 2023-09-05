
function calc_bubble(h::RPAHelper; local_tail=false)
    calc_bubble(h.gLoc_fft, h.gLoc_rfft, h.kG, h.mP, h.sP, local_tail=local_tail)
end

function calc_bubble_RPA(kG::KGrid, mP::ModelParameters, sP::SimulationParameters)
    #kG.kGrid .*
end

function calc_χγ(type::Symbol, h::RPAHelper, χ₀::χ₀T)
    calc_χγ(type, χ₀, h.kG, h.mP, h.sP)
end

function calc_χγ(type::Symbol, χ₀::χ₀T, kG::KGrid, mP::ModelParameters, sP::SimulationParameters)
    s = if type == :d 
        1
    elseif type == :m
        -1
    else
        error("Unkown type")
    end
    χ₀_qω = dropdims(sum(χ₀.data, dims=χ₀.axis_types[:ν]), dims=χ₀.axis_types[:ν]) ./ mP.β^2
    Nq  = length(kG.kMult)
    Nω  = size(χ₀.data, χ₀.axis_types[:ω])
    Nν  = 2*sP.n_iν+1
    γ = ones(ComplexF64, Nq, Nν, Nω)
    χ = real(χ₀_qω  ./ (1 .+ s * mP.U .* χ₀_qω))
    return χT(χ, mP.β, full_range=true; tail_c=[0.0,0.0,0.0]), γT(γ)
end