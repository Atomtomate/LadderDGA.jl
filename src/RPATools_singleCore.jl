
function calc_bubble_test(h::RPAHelper)
    calc_bubble(h.mP.β, h.kG, h.sP)
end

"""
    calc_bubble(h::RPAHelper)
    calc_bubble(β::Float64, kG::KGrid, sP::SimulationParameters)

Calc RPA-bubble term.

TODO: So far 3d hardcoded. Generalize to d dimensions...

χ_0(q,ω)=-Σ_{k} Σ_ν G(ν, k) * G(ν+ω, k+q)

where
    ν  : Fermionic Matsubara frequencies
    ω  : Bosonic Matsubara frequencies
    k,q: Element of the first Brilluoin zone

    This is a real-valued quantity.

Parameters
----------
    β     :: Float64  Inverse temperature in natural units
    kG    :: KGrid    The k-grid on which to perform the calculation
    sP    :: SimulationParameters (to construct a frequency range)

"""
function calc_bubble(β::Float64, kG::KGrid, sP::SimulationParameters)
    data_qνω = Array{ComplexF64,3}(undef, length(kG.kMult), 2*sP.n_iν, 2 * sP.n_iω + 1) # shell indices?
    ωrange = (-sP.n_iω : sP.n_iω)
    νrange = (-sP.n_iν : sP.n_iν - 1) # shift ?
    for (iω, ωn) = enumerate(ωrange)
        for (iν, νn) = enumerate(νrange) 
            gν = gf(νn, β, dispersion(kG))
            gνω = gf(νn + ωn, β, dispersion(kG))
            data_qνω[:, iν, iω] = conv(kG, gν, gνω) # prefactor of 1/β is attached to frequency sums. Omit them here.
        end
    end
    
    if maximum(abs.(imag(data_qνω))) > 1e-10
        error("Non vanishing imaginary part!")
    end
    data_qω = Array{Float64,2}(undef, length(kG.kMult), 2 * sP.n_iω + 1)
    data_qω = -dropdims(real(sum(data_qνω,dims=2)),dims=2)/β; # naive sum over fermionic matsubara frequencies (1\β corresponds to this sum)

    return χ₀RPA_T(data_qω, ωrange, β)
end

function calc_bubble(h::RPAHelper; local_tail=false)
    calc_bubble(h.gLoc_fft, h.gLoc_rfft, h.kG, h.mP, h.sP, local_tail=false)
end

"""
    gf(n::Int,β::Float64,ϵk)

Evaluates the RPA greensfunction.
G(ν,k) = \frac{1}{i ν - ϵ_k}

Parameters
----------
    n     :: Integer that corresponds to the fermionic matsubara frequency
    β     :: Float64  Inverse temperature in natural units
    ϵk    :: evaluated dispersion relation ... (ϵk-μ)
"""
function gf(n::Int, β::Float64, ϵk)
    ν = (2n + 1) * π / β # fermionic matsubara frequency
    return 1.0 ./ (im * ν .- ϵk)
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
    χ₀_qω = dropdims(sum(χ₀.data, dims=χ₀.axis_types[:ν]), dims=χ₀.axis_types[:ν]) ./ mP.β
    Nq  = length(kG.kMult)
    Nω  = size(χ₀.data, χ₀.axis_types[:ω])
    Nν  = 2*sP.n_iν+1
    γ = ones(ComplexF64, Nq, Nν, Nω)
    χ = real(χ₀_qω  ./ (1 .+ s * mP.U .* χ₀_qω))
    return χT(χ, mP.β, full_range=true; tail_c=[0.0,0.0,0.0]), γT(γ)
end