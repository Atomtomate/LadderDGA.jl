using Pkg
using EquivalenceClassesConstructor
using Printf, DataStructures
using JLD2

include("./vertexIntTriple.jl")

#const nBose = 150
#const nFermi = 150
#const shift = 0
#path = "/scratch/usr/hhpstobb/grids/b$(nBose)_f$(nFermi)_s$(shift)"

# ===== Test with integers =====
const nB = UInt32(2*maximum((nFermi,nBose))+1)
const nBh = floor(UInt32, nB/2)
(nB^2 > typemax(UInt32)) && throw(ArgumentError("nBose or nFermi too large for UInt32 operations."))
const nB2 = UInt32(nB*nB)
const nBm1 = UInt32(nB-1)
const oobConst = tripleToInt(nB,0,0,nB,nB2)

@fastmath @inline function reverseSymm(z::UInt32)
    r,k = divrem(z,nB)
    i,j = divrem(r,nB)
    return tripleToInt(UInt32(-i+nBm1), UInt32(-j+nBm1-1),UInt32(-k+nBm1-1),nB,nB2)
end

@fastmath @inline function symm_map_identity(z::UInt32)
    (z,)
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
freqList = [(i,j,k) for i in bGrid for j in fGrid for k in fGrid]
const freqList_int = map(x->tripleToInt(x..., nBh,nB,nB2), freqList)
const len_freq = nBose*(nFermi^2)

const mm_2 = Mapping(symm_map_identity)
println("Starting Computation 3")
maxF = nBose + nFermi
headerstr= @sprintf("%10d", maxF)

@time parents_int, ops_int = find_classes(mm_2.f, freqList_int, UInt32.([1]), vl_len=len_freq);
@time freqRed_map, freqList_min_int = minimal_set(parents_int, freqList_int)
freqList_min = intToTriple.(freqList_min_int)
@time parents,ops = uint_to_index(parents_int, ops_int, freqList_int)

println("skipping .dat")
#write_fixed_width(outdir*"/freqList.dat", freqList_min, sorted=true, header_add=headerstr);
#write_JLD("freqList_2.jld", rm_2, expMap)
base = nB
offset = nBh
nFermi = ceil(Int,nFermi/2)
nBose = floor(Int,nBose/2)
println("writing $(outdir*"/freqList.jld2") nBose = $(nBose), nF = $(nFermi)")
@save outdir*"/freqList.jld2" freqRed_map freqList freqList_min parents ops nFermi nBose shift base offset
