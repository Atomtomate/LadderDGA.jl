#TODO: this should be a macro
@inline get_symm_f(f::Array{Complex{Float64},1}, i::Int64) = @inbounds if i < 0 conj(f[-i]) else f[i+1] end

# This function exploits, that χ(ν, ω) = χ*(-ν, -ω) and a storage of χ with only positive fermionic frequencies
# TODO: For now a fixed order of axis is assumed

function convert_to_real(f; eps=10E-12)
    if maximum(imag.(f)) > eps
        throw(InexactError("Imaginary part too large for conversion!"))
    end
    return real.(f)
end

sum_limits(a, b, e) = if (ndims(a) == 1) sum(a[b:e]) else sum(mapslices(x -> sum_limits(x,b,e), a; dims=2:ndims(a))[b:e]) end
        
sum_inner(a, cut) =  if (ndims(a) == 1) sum(a[cut:(end-cut+1)]) else 
                        sum(mapslices(x -> sum_inner(x,cut), a; dims=2:ndims(a))[cut:(end-cut+1)]) end

"""
    Sums first νmax entries of any array along given dimension.
    Warning: This has NOT been tested for multiple dimensions.
"""
sum_νmax(a, cut; dims) = mapslices(x -> sum_inner(x, (cut)), a; dims=dims)

"""
    Returns index of the maximum which is closest to the mid point of the array
"""
function find_inner_maximum(arr)
    darr = diff(arr; dims=1)
    mid_index = Int(floor(size(arr,1)/2))
    intervall_range = 1

    # find interval
    while (intervall_range < mid_index) &&
        (darr[(mid_index-intervall_range)] * darr[(mid_index+intervall_range-1)] > 0)
            intervall_range = intervall_range+1
    end

    index_maximum = mid_index-intervall_range+1
    # find index
    while darr[(mid_index-intervall_range)]*darr[index_maximum] > 0
        index_maximum = index_maximum + 1
    end
    return index_maximum
end

"""
    Returns rang of indeces that are usable under 2 conditions.
    TODO: This is temporary and should be replace with a function accepting general predicates.
"""
function find_usable_interval(arr; reduce_range_prct = 0.1)
    darr = diff(arr; dims=1)
    index_maximum = find_inner_maximum(arr)
    mid_index = Int(ceil(size(arr,1)/2))

    # interval for condition 1 (positive values)
    cond1_intervall_range = 1
    # find range for positive values
    while (cond1_intervall_range < mid_index) &&
        (arr[(mid_index-cond1_intervall_range)] > 0) &&
        (arr[(mid_index+cond1_intervall_range)] > 0)
        cond1_intervall_range = cond1_intervall_range + 1
    end

    # interval for condition 2 (monotonicity)
    cond2_intervall_range = 1
    # find range for first turning point
    while (cond2_intervall_range < mid_index) &&
        (darr[(mid_index-cond2_intervall_range)] > 0) &&
        (darr[(mid_index+cond2_intervall_range)] < 0)
        cond2_intervall_range = cond2_intervall_range + 1
    end

    intervall_range = minimum([cond1_intervall_range, cond2_intervall_range])
    range = floor(Int64, intervall_range*(1-reduce_range_prct))
    res = ((mid_index-range+1):(mid_index+range-2) .+ 1)
    println("res: $(res) = $(mid_index) +- $(range)")
    if length(res) < 1
        println(stderr, "   ---> WARNING: could not determine usable range. Defaulting to single frequency!")
        res = [mid_index]
    end
    return res
end

function compute_Ekin(iνₙ, ϵₖ, Vₖ, GImp, β; full=true)
    Ekin = 0.0 + 0.0*1im
    fak = if full sum(Vₖ .^ 2)*(β^2)/4 else sum(Vₖ .^ 2)*(β^2)/8 end
    for n in 1:length(GImp)
        for l in 1:length(Vₖ)
            Ekin += (GImp[n] * (Vₖ[l]^2) / (iνₙ[n] - ϵₖ[l])) - (Vₖ[l] ^ 2)/(iνₙ[n]^2)
        end
    end
    if (full)
        return  ((Ekin - fak)/(2*β))
    else
        return (Ekin - fak)/β
    end
end

iω(n) = 1im*2*n*π/(modelParams.β);


split_n(str, n) = [str[(i-n+1):(i)] for i in n:n:length(str)]
split_n(str, n, len) = [str[(i-n+1):(i)] for i in n:n:len]
