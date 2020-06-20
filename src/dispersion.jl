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
function gen_kGrid(Nk::Int64, D::Int64; min=-π, max = π, include_min=false)
    kx::Array{Float64} = [((max-min)/(Nk - Int(include_min))) * j + min for j in (1:Nk) .- Int(include_min)]
    indArr = Base.product([1:(Nk) for Di in 1:D]...)
    kGrid  = Base.product([kx for Di in 1:D]...)
    return indArr, kGrid
end

#TODO: Rotation fertig
#      Mirror Symmetry
#TODO: define data type for grids, this should include type of grid to match symmetries, collect should not be necessary
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



"""
    expand_kGrid(reducedInd, reducedArr)

Expands arbitrary Array on reduced k-Grid back to full grid.
Results are shaped as 1D array.

#Examples
```
    mapslices(x->expand_kGrid(qIndices, x, simParams.Nk),sdata(bubble), dims=[2])
```
"""
function expand_kGrid(reducedInd, reducedArr::Array)
    D = length(reducedInd[1])
    Nk = maximum(maximum.(reducedInd))
    newArr = Array{eltype(reducedArr)}(undef, (Nk*ones(Int64, D))...)
    for (ri,redInd) in enumerate(reducedInd)
        perms = unique(collect(permutations(redInd)))
        #println(perms)
        for p in perms
            newArr[p...] = reducedArr[ri]
        end
    end
    return newArr
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
squareLattice_ekGrid(kgrid)  = ((length(first(kgrid)) == 3 ? -0.40824829046386301636 : -0.5) * 
                                 sum([cos(kᵢ) for kᵢ in k]) for k in kgrid)


function gen_squareLattice_ekq_grid(kList::Any, qList::Any)
    gen_squareLattice_ekq_grid(collect.(kList), collect.(qList))
end

function gen_squareLattice_ekq_grid(kList::Array, qList::Array)
    tsc =  length(first(klist)) == 3 ? -0.40824829046386301636 : -0.5
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
    tsc = -0.5
    for (ki,k) in enumerate(kList)
        for (qi,q) in enumerate(qList)
            @inbounds res[ki,qi,1] = tsc*(cos(k[1] + q[1]) + cos(k[2] + q[2]))
            @inbounds res[ki,qi,2] = tsc*(cos(k[1] + q[1]) + cos(k[2] - q[2]))
            @inbounds res[ki,qi,3] = tsc*(cos(k[1] - q[1]) + cos(k[2] + q[2]))
            @inbounds res[ki,qi,4] = tsc*(cos(k[1] - q[1]) + cos(k[2] - q[2]))
            @inbounds res[ki,qi,5] = tsc*(cos(k[1] + q[2]) + cos(k[2] + q[1]))
            @inbounds res[ki,qi,6] = tsc*(cos(k[1] + q[2]) + cos(k[2] - q[1]))
            @inbounds res[ki,qi,7] = tsc*(cos(k[1] - q[2]) + cos(k[2] + q[1]))
            @inbounds res[ki,qi,8] = tsc*(cos(k[1] - q[2]) + cos(k[2] - q[1]))
        end
    end
    return res
end

function gen_squareLattice_full_ekq_grid(kList::Array{Tuple{Float64,Float64,Float64},1}, qList::Array{Tuple{Float64,Float64,Float64},1})
    perm = permutations([1,2,3])
    res = zeros(length(kList),length(qList), 8*length(perm)) # There are 8 additional 
    tsc = -0.40824829046386301636
    for (ki,k) in enumerate(kList)
        for (qi,q) in enumerate(qList)
            ind = 0
            for p in perm
                res[ki,qi,ind+1] = tsc*(cos(k[p[1]] + q[1]) + cos(k[p[2]] + q[2]) + cos(k[p[3]] + q[3]))
                res[ki,qi,ind+2] = tsc*(cos(k[p[1]] + q[1]) + cos(k[p[2]] + q[2]) + cos(k[p[3]] - q[3]))
                res[ki,qi,ind+3] = tsc*(cos(k[p[1]] + q[1]) + cos(k[p[2]] - q[2]) + cos(k[p[3]] + q[3]))
                res[ki,qi,ind+4] = tsc*(cos(k[p[1]] + q[1]) + cos(k[p[2]] - q[2]) + cos(k[p[3]] - q[3]))
                res[ki,qi,ind+5] = tsc*(cos(k[p[1]] - q[1]) + cos(k[p[2]] + q[2]) + cos(k[p[3]] + q[3]))
                res[ki,qi,ind+6] = tsc*(cos(k[p[1]] - q[1]) + cos(k[p[2]] + q[2]) + cos(k[p[3]] - q[3]))
                res[ki,qi,ind+7] = tsc*(cos(k[p[1]] - q[1]) + cos(k[p[2]] - q[2]) + cos(k[p[3]] + q[3]))
                res[ki,qi,ind+8] = tsc*(cos(k[p[1]] - q[1]) + cos(k[p[2]] - q[2]) + cos(k[p[3]] - q[3]))
                ind += 8
            end
        end
    end
    return res
end


function cut_mirror(arr)
    res = nothing

    Nk_cut = Int(size(arr,1)/2)
    if ndims(arr) == 2
        res = arr[Nk_cut:end, Nk_cut:end]
    elseif ndims(arr) == 3
        res = arr[Nk_cut:end, Nk_cut:end, Nk_cut:end]
    else
        println(stderr, "Error trying to reduce grid! Number of dimensions not recognized")
    end
    return res
end

function expand_mirror(arr)
    res = nothing
    Nk_old = size(arr,1)
    Nk_expand = Nk_old - 2
    size_new = size(arr) .+ Nk_expand

    res = Array{eltype(arr)}(undef, size_new...)
    if ndims(arr) == 2
        res[1:Nk_old,1:Nk_old] = arr
        res[Nk_old+1:end,1:Nk_old] = res[Nk_old-1:-1:2,1:Nk_old]
        res[1:end,Nk_old+1:end] = res[1:end,Nk_old-1:-1:2]
        for (i,ai) in enumerate((Nk_old-1):-1:2)
            res[Nk_old+i,Nk_old+i] = arr[ai,ai]
        end
    elseif ndims(arr) == 3
        res[1:Nk_old,1:Nk_old,1:Nk_old] = arr
        res[Nk_old+1:end,1:Nk_old,1:Nk_old] = res[Nk_old-1:-1:2,1:Nk_old,1:Nk_old]
        res[1:end,Nk_old+1:end,1:Nk_old] = res[1:end,Nk_old-1:-1:2,1:Nk_old]
        for (i,ai) in enumerate((Nk_old-1):-1:2)
            res[Nk_old+i,Nk_old+i,1:Nk_old] = arr[ai,ai,1:Nk_old]
        end
        res[1:end,1:end,Nk_old+1:end] .= res[1:end,1:end,Nk_old-1:-1:2]
        for (i,ai) in enumerate((Nk_old-1):-1:2)
            res[Nk_old+i,Nk_old+i,Nk_old+i] = arr[ai,ai,ai]
        end
    else
        println(stderr, "Error trying to expand grid! Number of dimensions not recognized")
    end
    return res
end
