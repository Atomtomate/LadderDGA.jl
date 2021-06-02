using HDF5
using JLD2
using Pkg
using SparseVertex

include("./triqs_conv_functions.jl")


file = ARGS[1]
outdir = ARGS[2]
# Gimp
gImp = triqs_read_gf(file, "G_imp")

# chi
chiupdo_mesh = h5read(file, "chi_updn_ph_imp/mesh")
mesh = triqs_build_freqGrid(chiupdo_mesh);
chiupdo_raw = h5read(file, "chi_updn_ph_imp/data")
chiupup_raw = h5read(file, "chi_upup_ph_imp/data")
χupdo = permutedims(chiupdo_raw[1,:,:,:] .+ 1im .* chiupdo_raw[2,:,:,:],[3,1,2])
χupup = permutedims(chiupup_raw[1,:,:,:] .+ 1im .* chiupup_raw[2,:,:,:],[3,1,2])
χch = χupup .+ χupdo
χsp = χupup .- χupdo
freqList = triqs_linearize_mesh(mesh)

bGrid = freqList[1][1]:freqList[end][1]
fGrid = freqList[1][2]:freqList[end][2]
nBose = length(bGrid)
nFermi = length(fGrid)
shift = 0
println("Generating freqency mesh with bosonic $(bGrid), fermionic $(fGrid) indices")
β = h5read(file, "chi_updn_ph_imp/mesh/MeshComponent0/domain/beta")

χ0_full = computeχ0(bGrid, fGrid, gImp, β)
Γch = computeΓ(freqList, χch, χ0_full, bGrid, fGrid)
Γsp = computeΓ(freqList, χsp, χ0_full, bGrid, fGrid)
Σ = triqs_read_gf(file, "Sigma_imp")


mkpath(outdir)
mkpath(outdir*"/chi_dir")
mkpath(outdir*"/gamma_dir")
SparseVertex.write_fort_dir("gamma", freqList, Γch, Γsp, outdir*"/gamma_dir", nBose, nFermi)
SparseVertex.write_fort_dir("chi", freqList, χupup, χupdo, outdir*"/chi_dir", nBose, nFermi)


# Grid stuff
include("./genFreqMesh.jl")
gen_mesh(freqList[end][1], freqList[end][2], shift, outdir)

JLD2.save(outdir*"/triqs_out.jld2", "Γch", Γch, "Γsp", Γsp, "χDMFTch", χch, "χDMFTsp", χsp, "gImp", gImp, "SigmaLoc", Σ, "beta", β)
