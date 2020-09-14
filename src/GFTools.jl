# TODO: test everything for single orbital case
iν_array(β::Real, grid::Array{Int64}) = [1.0im*((2.0 *el + 1)* π/β) for el in grid]
iν_array(β::Real, grid::UnitRange{Int64}) = [1.0im*((2.0 *el + 1)* π/β) for el in grid]
iν_array(β::Real, size::Integer)    = [1.0im*((2.0 *i + 1)* π/β) for i in 0:size-1]
iω_array(β::Real, grid::Array{Int64}) = [1.0im*((2.0 *el)* π/β) for el in grid]
iω_array(β::Real, grid::UnitRange{Int64}) = [1.0im*((2.0 *el)* π/β) for el in grid]
iω_array(β::Real, size::Integer)    = [1.0im*((2.0 *i)* π/β) for i in 0:size-1]


function tail_func(n::Array{Int}, β, c::Array{Float64})
    iνn = iν_array(β, n)
    tail_func(iνn, β, c)
end

function tail_τ_func(τ::Array, β, c::Array{Float64})
    res = [c[1] for i = 1:length(τ)]
    for  i = 2:length(c)
        if i == 2
            res = res .- (c[2]/2)
        elseif i == 3 
            res = res .+ (c[3]/4) .* (2 .* τ .- β)
        elseif i == 4 
            res = res .+ (c[4]/4) .* (τ .* (β .- τ))
        elseif i == 5 
            res = res .+ (c[5]/48) .* (2 .* τ .- β) .* (2 .* τ .* τ .- 2 .* β .* τ .- (β*β))
        else  
            # TODO: include @printf("Warning: only 4 tail coefficients implemented, supplied: %d\n", length(c))
        end

    end
    return res
end

function tail(GF, β, n_tail = 5, nFreq = 20, stop = :end)
    stop = stop == :end ? size(GF, 1) : stop
    start = stop - nFreq + 1
    iνn = iν_array(β, collect(start:stop))
    g_grid = view(G.f_grid, start:stop)
    cost(c) = sum(abs2.(imag(GF) - imag(tail_func(iνn, β, c))))
    res = Optim.minimizer(Optim.optimize(cost, zeros(n_tail), Optim.BFGS()))
    return res
end

# TODO: implement tail correction here
# nFreq = size(GBath, 1)
# tail = tail(GImp, 5, 20) 
#new_grid = GImp
#for i = 1:nFreq
#    new_grid[i] = view(new_grid,i) ./ view(tail,2)
#end
function FUpDo_from_χDMFT(χdo, GImp, ωGrid, νGrid1, νGrid2, β)  
    FUpDo = zeros(eltype(χdo), length(ωGrid), length(νGrid1), length(νGrid2))
    for (ωi, ωₙ) in enumerate(ωGrid)
        for (νi, νₙ) in enumerate(νGrid1)
            for (νj, νpₙ) in enumerate(νGrid2)
                FUpDo[ωi, νi, νj] = χdo[ωi, νi, νj]/(β^2 * get_symm_f(GImp,νₙ) * get_symm_f(GImp,ωₙ + νₙ)
                                        * get_symm_f(GImp,νpₙ) * get_symm_f(GImp,ωₙ + νpₙ))
            end
        end
    end
    return FUpDo
end

function Σ_Dyson(GBath::Array{Complex{Float64},1}, GImp::Array{Complex{Float64},1}, eps = 1e-3) 
    @inbounds Σ::Array{Complex{Float64},1} =  1 ./ GBath .- 1 ./ GImp
    return Σ
end

@inline @fastmath G_from_Σ(n::Int64, β::Float64, μ::Float64, ϵₖ::T, Σ::Complex{T}) where T <: Real =
                    1/((π/β)*(2*n + 1)*1im + μ - ϵₖ - Σ)

"""
    G(ind::Int64, Σ::Array{Complex{Float64},1}, ϵkGrid, β::Float64, μ)
Constructs GF from k-independent self energy, using the Dyson equation
and the dispersion relation of the lattice.
"""
@inline function G(ind::Int64, Σ::Array{Complex{Float64},1}, 
                   ϵkGrid::Base.Generator, β::Float64, μ::Float64)
    Σν = get_symm_f(Σ,ind)
    return map(ϵk -> G_from_Σ(ind, β, μ, ϵk, Σν), ϵkGrid)
end

@inline function G(ind::Int64, Σ::Array{Complex{Float64},2}, 
                   ϵkGrid, β::Float64, μ::Float64)
    Σνk = get_symm_f(Σ,ind)
    return reshape(map(((ϵk, Σνk_i),) -> G_from_Σ(ind, β, μ, ϵk, Σνk_i), zip(ϵkGrid, Σνk)), size(ϵkGrid)...)
end


@inline Gfft_from_Σ(Σ::Array{Complex{Float64}}, ϵkGrid, 
                    range::UnitRange{Int64}, mP::ModelParameters) = flatten_2D(fft.(G_from_Σ(Σ, ϵkGrid, range, mP)))

@inline G_from_Σ(Σ::Array{Complex{Float64}}, ϵkGrid, 
                 range::UnitRange{Int64}, mP::ModelParameters) = [G(ind, Σ, ϵkGrid, mP.β, mP.μ) for ind in range]

"""
    extend_Σ(Σ_ladder, Σ_loc, range)

Builds new  self energy with `Σ_loc` tail, extending the frequency
range of `Σ_ladder`. If `range` contains negative frequencies, the
results are obtaind from `get_symm_f(Σ, i)`.
"""
function extend_Σ(Σ_ladder, Σ_loc, range)
    res = zeros(eltype(Σ_ladder), length(range), size(Σ_ladder,2))
    Σ_length = size(Σ_ladder, 1)
    for (ind, ν) in enumerate(range)
        res[ind,:] .= (ν < Σ_length && ν >= -Σ_length) ? get_symm_f(Σ_ladder,ν) : get_symm_f(Σ_loc,ν)
    end
    return res
end
