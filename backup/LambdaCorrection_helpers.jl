
"""
    λ_seach_range(χ::Matrix{Float64}; λ_max_default = 50)

Calculates reasonable interval for the search of the ``\\lambda``-correction parameter. 

The interval is chosen with ``\\lambda_\\text{min}``, such that all unphysical poles are excluded and
``\\lambda_\\text{max} = \\lambda_\\text{default} / \\max_{q,\\omega} \\chi(q,\\omega)``. The `λ_max_default` parameter may need to be
adjusted, depending on the model, since in principal arbitrarily large maximum values are possible.
"""
function λ_seach_range(χ::Matrix{Float64}; λ_max_default = 50)
    λ_min = get_λ_min(χ)
    λ_max = λ_max_default / maximum(χ)
    if λ_min > 1000
        @warn "found very large λ_min ( = $λ_min). This indicates problems with the susceptibility."
    end
    return λ_min, λ_max
end

