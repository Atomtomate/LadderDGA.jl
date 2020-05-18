"""
    gen_kGrid(Nk, D[; min = 0, max = π, include_min=true]) = 

Generates an Iterator for the Cartesian product of k vectors. 
This can be collected to reduce into a `Nk` times `Nk` array, containing
tuples of length `D`.

# Examples
```
julia> gen_kGrid(2, 2; min = 0, max = 2π, include_min = false)
Base.Iterators.ProductIterator{Tuple{Array{Float64,1},Array{Float64,1}}}(([3.141592653589793, 6.283185307179586], [3.141592653589793, 6.283185307179586]))
```
"""
function gen_kGrid(Nk::Int64, D::Int64; min = 0, max = π, include_min=true)
    kx::Array{Float64} = [((max-min)/(Nk - Int(include_min))) * j + min for j in (1:Nk) .- Int(include_min)]
    indArr = Base.product([1:(Nk) for Di in 1:D]...)
    kGrid  = Base.product([kx for Di in 1:D]...)
    return indArr, kGrid
end

"""
    reduce_kGrid(kGrid) 

Filters the grid, so that only the lower triangle remains, i.e.
for any (x_1, x_2, ...) the condition x_1 >= x_2 >= x_3 ... 
is fulfilled.
"""
function reduce_kGrid(kGrid) 
    kGrid_arr = collect(kGrid)
    Nk = size(kGrid_arr,1)
    if ndims(kGrid_arr) == 2
        index = [[x,y] for x=1:Nk for y=1:x]
    elseif ndims(kGrid_arr) == 3
        index = [[x,y,z] for x=1:Nk for y=1:x for z = 1:y]
    else
        throw(BoundsError("Number of dimensions for grid must be 2 or 3"))
    end
    grid_red = Array{eltype(kGrid_arr)}(undef, length(index))
    for (i,ti) in enumerate(index)
        grid_red[i] = kGrid_arr[ti...]
    end
    #isMonotonic(x) = all(sort(collect(x), rev=true) .≈ collect(x))        # A list is monotonic, iff sorting does not change the order
    #grid = Iterators.filter(isMonotonic, kGrid)
    return grid_red
end

function kGrid_multiplicity(kIndices)
    min_ind = minimum(kIndices)
    max_ind = maximum(kIndices)
    function borderFactor(el) 
        val = 1.0
        for i in 1:length(el)
            val = if (el[i] == min_ind[i] || el[i] == max_ind[i]) val*0.5 else val end
        end
        return val
    end
    #TODO: for arbitraty dim
    if length(min_ind) == 2
        res = map(el -> borderFactor(el)*8/((el[2]==el[1]) + 1), kIndices)
    elseif length(min_ind) == 3
        res = map(el -> borderFactor(el)*48/( (el[2]==el[1]) + (el[3]==el[2]) + 3*(el[3]==el[1]) + 1), kIndices)
    else
        res = []
        print(stderr, "   ---> Warning! arbitrary dimensions for kGrid multiplicity not implemented!\n")
    end
    return res
end

function gen_reduced_kGrid(Nk::Int64, D::Int64; min = 0, max = π, include_min=true)
    return gen_reduced_kGrid(gen_kGrid(Nk::Int64, D::Int64; min = 0, max = π, include_min=true))
end


"""
    kGrid_to_array(kGrid)

Collects the kGrid (given as an iterator) and returns a 
N times D array (N being the number of k-points)
"""
function kGrid_to_array(kGrid::Base.Iterators.ProductIterator{Tuple{Array{Float64,1},Array{Float64,1}}})
    permutedims(hcat(collect.(kGrid)...),[2,1])
end

function kGrid_to_array(kGrid::Base.Iterators.ProductIterator{Tuple{Array{Float64,1},Array{Float64,1},Array{Float64,1}}})
    permutedims(cat(collect.(kGrid)..., dims=3)[:,1,:],[2,1])
end
kGrid_to_array(kGrid::Array{Tuple{Float64,Float64}}) = permutedims(hcat(collect.(kGrid)...),[2,1])
kGrid_to_array(kGrid::Array{Tuple{Float64,Float64,Float64}}) = permutedims(cat(collect.(kGrid)..., dims=3)[:,1,:],[2,1])
kGrid_to_array(kGrid::Array{Float64}) = kGrid

"""
    squareLattice_ek_grid(kgrid)

Computes 0.5 [cos(k_x) + ... + cos(k_D)] and returns a grid with Nk points.
"""
function squareLattice_ekGrid(kgrid, tsc)
    ek(k) = tsc*sum([cos(kᵢ) for kᵢ in k])
    res = [ek(k) for k in kgrid]
end

function gen_squareLattice_ekq_grid(kList::Any, qList::Any, tsc)
    gen_squareLattice_ekq_grid(collect.(kList), collect.(qList), tsc)
end

function gen_squareLattice_ekq_grid(kList::Array, qList::Array, tsc)
    res = zeros(length(kList),length(qList))
    for (ki,k) in enumerate(kList)
        for (qi,q) in enumerate(qList)
            @inbounds res[ki,qi] = tsc.*sum(cos.(k .+ q))
        end
    end
    return res
end

#TODO: generalize to 3D, better abstraction
function gen_squareLattice_full_ekq_grid(kList::Array{Tuple{Float64,Float64},1}, qList::Array{Tuple{Float64,Float64},1})
    res = zeros(length(kList),length(qList), 8) # There are 8 additional 
    for (ki,k) in enumerate(kList)
        for (qi,q) in enumerate(qList)
            @inbounds res[ki,qi,1] = 0.5*(cos(k[1] + q[1]) + cos(k[2] + q[2]))
            @inbounds res[ki,qi,2] = 0.5*(cos(k[1] + q[1]) + cos(k[2] - q[2]))
            @inbounds res[ki,qi,3] = 0.5*(cos(k[1] - q[1]) + cos(k[2] + q[2]))
            @inbounds res[ki,qi,4] = 0.5*(cos(k[1] - q[1]) + cos(k[2] - q[2]))
            @inbounds res[ki,qi,5] = 0.5*(cos(k[1] + q[2]) + cos(k[2] + q[1]))
            @inbounds res[ki,qi,6] = 0.5*(cos(k[1] + q[2]) + cos(k[2] - q[1]))
            @inbounds res[ki,qi,7] = 0.5*(cos(k[1] - q[2]) + cos(k[2] + q[1]))
            @inbounds res[ki,qi,8] = 0.5*(cos(k[1] - q[2]) + cos(k[2] - q[1]))
        end
    end
    return res
end
