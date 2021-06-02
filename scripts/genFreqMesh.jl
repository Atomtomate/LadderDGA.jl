using EquivalenceClassesConstructor
using Printf, DataStructures
using JLD2

include("$(@__DIR__)/vertexIntTriple.jl")

#const nBose = 20
#const nFermi = 20
#const shift = 0
#path = "."
function gen_mesh(nBose, nFermi, shift, path)
# ===== Test with integers =====
println("Integer test")
nB = UInt32(2*maximum((2*nFermi+shift*ceil(nBose/2),2*nBose+1))+1)
nBh = floor(UInt32, nB/2)
(nB^2 > typemax(UInt32)) && throw(ArgumentError("nBose or nFermi too large for UInt32 operations."))
nB2 = UInt32(nB*nB)
nBm1 = UInt32(nB-1)
oobConst = tripleToInt(nB,0,0,nB,nB2)

@fastmath @inline function reverseSymm(z::UInt32)
    r,k = divrem(z,nB)
    i,j = divrem(r,nB)
    return tripleToInt(UInt32(-i+nBm1), UInt32(-j+nBm1-1),UInt32(-k+nBm1-1),nB,nB2)
end

@fastmath @inline function symm_map_identity(z::UInt32)
    (z,)
end

@fastmath @inline function symm_map_int(z::UInt32)
    r,k = divrem(z,nB)
    i,j = divrem(r,nB)
    iu = UInt32(i)
    ju = UInt32(j)
    ku = UInt32(k)
    ni = UInt32(-i+nBm1)
    nj = UInt32(-j+nBm1)
    nk = UInt32(-k+nBm1)
    t1 = ku+nj-nBh #i and j both contain nBh, we need to subtract the double shift
    t2 = iu+ju-nBh
    t3 = iu+ku-nBh
    t4 = ju+nk-nBh
    r1 = tripleToInt(ni, nj-UInt32(1), nk-UInt(1),nB,nB2) # c.c.: Op 1 (*)
    r2 = tripleToInt(iu, ku, ju,nB,nB2)                   # time reversal: Op 2 (1)
    r3 = oobConst                                         # double Crossing: Op 3 (-)
    r4 = oobConst                                         # crossing c: Op 4 (-)
    r5 = oobConst                                         # crossing c^t: Op 5 (+) 
    if t2 < nB 
        if t1 < nB 
            r3 = tripleToInt(t1, ju, t2,nB,nB2)
        end
        if t3 < nB
            r5 = tripleToInt(ni, t3, t2,nB,nB2)
        end
    end
    if (t3 < nB && t4 < nB)
        r4 = tripleToInt(t4, t3, ku,nB,nB2)
    end
    return r1,r2,r3,r4,r5
end

function uint_to_index(parents::Dict{UInt32,UInt32}, ops::Dict{UInt32,UInt32}, vl::Array{UInt32,1})
    parents_new = Array{Int64}(undef, length(vl))
    ops_new = Array{Int64}(undef, length(vl))
    lookup = Dict(zip(vl,1:length(vl)))
    for (i,el) in enumerate(vl)
        parents_new[i] = lookup[parents[el]]
        ops_new[i] = Int64(ops[el])
    end
    return parents_new, ops_new
end


println("Constructing Array")
freqList = [(i,j,k) for i in (-nBose:nBose) for j in (-nFermi:nFermi-1) .- trunc(Int64,shift*i/2) for k in (-nFermi:nFermi-1) .- trunc(Int64,shift*i/2)]
freqList_int = map(x->tripleToInt(x..., nBh,nB,nB2), freqList)
len_freq = (2*nBose+1)*(2*nFermi)^2

mm_2 = Mapping(symm_map_int)
println("Starting Computation 3")
maxF = nBose + 2*nFermi + shift*ceil(Int,nBose/2) + 5
headerstr= @sprintf("%10d", maxF)

@time parents_int, ops_int = find_classes(mm_2.f, freqList_int, UInt32.([1, 2, 3, 4, 5]), vl_len=len_freq);
@time freqRed_map, freqList_min_int = minimal_set(parents_int, freqList_int)
freqList_min = intToTriple.(freqList_min_int,nB,nBh)
@time parents,ops = uint_to_index(parents_int, ops_int, freqList_int)

mkpath(path)
write_fixed_width(path*"/freqList.dat", freqList_min, sorted=true, header_add=headerstr);
#write_JLD("freqList_2.jld", rm_2, expMap)
#save("freqList.jld", "ExpandMap", expMap, "ReduceMap", redMap, "base", nB, "nFermi", 2*nFermi, "nBose", 2*nBose+1, "shift", shift, "offset", nBh)
base = nB
offset = nBh
@save path*"/freqList.jld2" freqRed_map freqList freqList_min parents ops nFermi nBose shift base offset
end
