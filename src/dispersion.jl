struct KGrid
    indices::Base.Iterators.ProductIterator
    kGrid::Base.Iterators.ProductIterator
    ϵkGrid::Base.Generator
end

struct Reduced_KGrid
    Nq::Int
    indices::Vector{Tuple}
    multiplicity::Vector{Float64}
    kGrid::Vector{Tuple}
    ϵkGrid::Vector{Float64}
end

function squareLattice_kGrid(Nk::Int, D::Int)
    ind, grid = gen_kGrid(Nk, D)
    ϵkGrid = squareLattice_ekGrid(grid)
    return KGrid(ind,grid,ϵkGrid)
end

function reduce_squareLattice(kGrid::KGrid)
    qIndices, qGrid, ϵqGrid = reduce_kGrid.(cut_mirror.((kGrid.indices, kGrid.kGrid, collect(kGrid.ϵkGrid))));
    qMult   = kGrid_multiplicity(qIndices);
    return Reduced_KGrid(length(qIndices), qIndices, qMult, qGrid, ϵqGrid)
end

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
This includes restoration of mirror symmetry if the index array 
reducedInd indicates uncomplete grid (by having no (1,1,1) index).
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
        for p in perms
            newArr[p...] = reducedArr[ri]
        end
    end
    minimum(minimum.(reducedInd)) > 1 && expand_mirror!(newArr)
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
    if length(min_ind) == 2
        res = map(el -> borderFactor(el)*8/((el[2]==el[1]) + 1), kIndices)
    elseif length(min_ind) == 3
        res = map(el -> borderFactor(el)*48/( (el[2]==el[1]) + (el[3]==el[2]) + 3*(el[3]==el[1]) + 1), kIndices)
    else
        throw("Multiplicity of k points only implemented for 2D and 3D")
    end
    return res
end


"""
    squareLattice_ek_grid(kgrid)

Computes 0.5 [cos(k_x) + ... + cos(k_D)] and returns a grid with Nk points.
"""
squareLattice_ekGrid(kgrid)  = ((length(first(kgrid)) == 3 ? -0.40824829046386301636 : -0.5) * 
                                 sum([cos(kᵢ) for kᵢ in k]) for k in kgrid)


@inbounds cut_mirror(arr::Base.Iterators.ProductIterator) = cut_mirror(collect(arr))
@inbounds cut_mirror(arr::Array{T, 2}) where T = arr[Int(size(arr,1)/2):end, Int(size(arr,2)/2):end]
@inbounds cut_mirror(arr::Array{T, 3}) where T = arr[Int(size(arr,1)/2):end, Int(size(arr,2)/2):end, Int(size(arr,3)/2):end]
#reverse cut. This is a helper function to avoid reversing the array after fft-convolution trick. assumes reserved input and returns correct array including cut
@inbounds ifft_cut_mirror(arr::Base.Iterators.ProductIterator) = ifft_cut_mirror(collect(arr))
@inbounds ifft_cut_mirror(arr::Array{T, 2}) where T = arr[end-1:-1:Int64(size(arr,1)/2-1), 
                                                          end-1:-1:Int64(size(arr,2)/2-1)]
@inbounds ifft_cut_mirror(arr::Array{T, 3}) where T = arr[end-1:-1:Int64(size(arr,1)/2-1), 
                                                          end-1:-1:Int64(size(arr,2)/2-1), 
                                                          end-1:-1:Int64(size(arr,3)/2-1)]


function expand_mirror!(arr::Array{T, 2}) where T <: Any
    al = Int(size(arr,1)/2) - 1

    arr[1:al,al+1:end] = arr[end-1:-1:al+2,al+1:end]
    arr[1:end,1:al] = arr[1:end,end-1:-1:al+2,]
    for i in 1:al
        arr[i,i] = arr[end-i,end-i]
    end
end

function expand_mirror(arr::Array{T, 2}) where T <: Any
    al = size(arr,1) - 2
    size_new = size(arr) .+ al
    res = Array{T}(undef, size_new...)

    res[al+1:end, al+1:end] = arr
    expand_mirror!(res)
    return res
end

function expand_mirror!(arr::Array{T, 3}) where T <: Any
    al = Int(size(arr,1)/2) - 1

    arr[1:al,al+1:end,al+1:end] = arr[end-1:-1:al+2,al+1:end,al+1:end]
    arr[1:end,1:al,al+1:end] = arr[1:end,end-1:-1:al+2,al+1:end]
    for i in 1:al
        arr[i,i,al+1:end] = arr[end-i,end-i,al+1:end]
    end
    arr[1:end,1:end,1:al] .= arr[1:end,1:end,end-1:-1:al+2]
    for i in 1:al
        arr[i,i,i] = arr[end-i,end-i,end-i]
    end
end

function expand_mirror(arr::Array{T, 3}) where T <: Any
    add_length = size(arr,1) - 2
    size_new = size(arr) .+ add_length
    res = Array{T}(undef, size_new...)

    res[add_length:end, add_length:end, add_length:end] = arr
    expand_mirror!(res)
    return res
end

"""
    sum_q(arr, qMult)
Computes normalized sum over all q-Points.
"""
function sum_q(arr, qMult; dims::Int64=1) 
    @assert all(size(arr, dims) .== size(qMult, 1))
    convert(stripped_type(arr), mapslices(x-> sum(x .* qMult), arr, dims=dims) ./ sum(qMult))
end
sum_q_test(arr, qMult; dims=1) = stripped_type(arr)(sum(mapslices(x-> x .* qMult, arr, dims=dims), dims=dims) ./ sum(qMult))

sum_q_drop(arr, qMult; dims=1) = stripped_type(arr)(sum_drop(mapslices(x-> x .* qMult, arr, dims=dims), dims=dims) ./ sum(qMult))


reduce_kGrid_ifft = reduce_kGrid ∘ ifft_cut_mirror ∘ ifft

