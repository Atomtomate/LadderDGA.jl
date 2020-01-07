#TODO: this should be a macro
@inline get_symm_f(f::Array{Complex{Float64},1}, i::Int64) = @inbounds if i < 0 conj(f[-i]) else f[i+1] end

# This function exploits, that χ(ν, ω) = χ*(-ν, -ω) and a storage of χ with only positive fermionic frequencies
# TODO: For now a fixed order of axis is assumed
@inline function get_symm_χ(f::Array, ωₙ::Int64, νₙ::Int64, k::Int64) 
    @inbounds n_iωₙ = Int((size(f,1)-1)/2)
    @inbounds if νₙ < 0 f[n_iωₙ - ωₙ + 1, -νₙ, k] else f[ωₙ + n_iωₙ + 1,νₙ + 1, k] end
end

function convert_to_real(f; eps=10E-12)
    if maximum(imag.(f)) > eps
        throw(InexactError("Imaginary part too large for conversion!"))
    end
    return real.(f)
end

sum_limits(a, b, e) = if (ndims(a) == 1) sum(a[b:e]) else sum(mapslices(x -> sum_limits(x,b,e), a; dims=2:ndims(a))[b:e]) end

"""
    Sums first νmax entries of any array along given dimension.
    Warning: This has NOT been tested for multiple dimensions.
"""
sum_νmax(a, to; dims) = mapslices(x -> sum_limits(x, 1, to), a; dims=dims)
