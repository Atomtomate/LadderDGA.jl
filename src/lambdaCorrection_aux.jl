dχ_λ(χ, λ::Float64) = map(χi -> - ((1.0 / χi) + λ)^(-2), χ)
χ_λ2(χ, λ) = map(χi -> 1.0 / ((1.0 / χi) + λ), χ)

@inline χ_λ(χ::Float64, λ::Float64)::Float64 = 1.0/(1.0/χ + λ)
@inline dχ_λ(χ::Float64, λ::Float64)::Float64 = -1.0/(1.0/χ + λ)^2

function χ_λ(nlQ::NonLocalQuantities, λ::Float64) where T <: Union{ComplexF64, Float64}
    nlQ_new = deepcopy(nlQ)
    χ_λ!(nlQ_new.χ, nlQ.χ, λ)
    return nlQ_new
end

function χ_λ(χ::AbstractArray{T}, λ::Float64) where T <: Union{ComplexF64, Float64}
    res = similar(χ)
    χ_λ!(res, χ, λ)
    return res
end

function χ_λ!(χ_λ::AbstractArray{ComplexF64}, χ::AbstractArray{ComplexF64}, λ::Float64)
    @simd for i in eachindex(χ_λ)
        @inbounds χ_λ[i] = 1.0 / ((1.0 / χ[i]) + λ)
    end
end

function χ_λ!(χ_λ::AbstractArray{Float64}, χ::AbstractArray{Float64}, λ::Float64)
    @simd for i in eachindex(χ_λ)
        @inbounds χ_λ[i] = 1.0 / ((1.0 / χ[i]) + λ)
    end
end

function χ_λ!(χ_λ::AbstractArray{T,2}, χ::AbstractArray{T,2}, λ::Float64, ωindices::AbstractArray{Int,1}) where T <: Number
    for i in ωindices
        χ_λ!(view(χ_λ, :, i),view(χ,:,i), λ)
    end
end

function get_χ_min(χr::AbstractArray{Float64,2})
    nh  = ceil(Int64, size(χr,2)/2)
    -minimum(1 ./ view(χr,:,nh))
end


