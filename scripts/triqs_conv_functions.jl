# ======================================================= #
#                                                         #
#                  TRIQS Covnersion                       #
#                                                         #
# ======================================================= #
function triqs_read_gf(file, name)
    freqSize = h5read(file, name)["mesh"]["size"]
    pos_only = h5read(file, name)["mesh"]["positive_freq_only"]
    arr_tmp = h5read(file, name)["data"]
    n0 = (1-pos_only)*(trunc(Int, freqSize/2))+1
    res = arr_tmp[1,n0:end] + 1im .* arr_tmp[2,n0:end]
end

function triqs_mesh_to_range(mesh)
	s = mesh["size"]
	p = mesh["positive_freq_only"]
	n0 = (1-p)*(trunc(Int, s/2))
	return (0:s-1) .- n0
end

function triqs_build_freqGrid(mesh)
	nu_range = triqs_mesh_to_range(mesh["MeshComponent2"])
	nup_range = triqs_mesh_to_range(mesh["MeshComponent1"])
	w_range = triqs_mesh_to_range(mesh["MeshComponent0"])
	return collect(Base.product(nu_range, nup_range, w_range))
end

function triqs_linearize_mesh(mesh)
	new_arr = Array{eltype(mesh),1}(undef, length(mesh))
	ind = 1
	for wi in 1:size(mesh,3)
		for nui in 1:size(mesh,1)
			for nupi in 1:size(mesh,2)
				new_arr[ind] = (mesh[nui, nupi, wi][3], mesh[nui, nupi, wi][1], mesh[nui, nupi, wi][2])
				ind += 1
			end
		end
	end
	return new_arr
end

function triqs_linearize(raw_arr)
	new_arr = Array{Complex{Float64},1}(undef, size(raw_arr,4)*size(raw_arr,3)*size(raw_arr,2))
	ind = 1
	for wi in 1:size(raw_arr,4)
		for nui in 1:size(raw_arr,2)
			for nupi in 1:size(raw_arr,3)
				new_arr[ind] = raw_arr[1,nui, nupi, wi] + 1im .* raw_arr[2,nui, nupi, wi]
				ind += 1
			end
		end
	end
	return new_arr
end


# ======================================================= #
#                                                         #
#                      GF Stuff                           #
#                                                         #
# ======================================================= #
@inline get_symm_f(f::Array{Complex{Float64},1}, i::Int64) = (i < 0) ? conj(f[-i]) : f[i+1]

function computeχ0(ω_range::AbstractArray{Int,1}, ν_range::AbstractArray{Int,1}, gImp::Array{Complex{Float64}, 1}, β::Float64)
    χ0 = Dict{Tuple{Int,Int},Complex{Float64}}()
    for ω in ω_range, ν in ν_range
        χ0[(ω,ν)] = -β*get_symm_f(gImp, ν)*get_symm_f(gImp, ν+ω)
    end
    return χ0
end

function computeΓ(freqList::Array, χ::Array{T,3}, χ0::Dict{Tuple{Int,Int},Complex{Float64}}, bGrid, fGrid) where T
    res = Array{T}(undef, length(bGrid), length(fGrid), length(fGrid))
    for (ωn,ω) in enumerate(bGrid)
        res[ωn,:,:] = inv(χ[ωn,:,:])
        for (νn,ν) in enumerate(fGrid)
            res[ωn,νn,νn] -= 1.0/χ0[(ω,ν)]
        end
    end
    return res
end

function gen_config(U::Float64, μ::Float64, β::Float64, nden::Float64, t::Float64)
cfg="""[Model]
U    = $U
mu   = $μ
beta = $β
nden = $nden
kGrid = "2Dsc-$t"

[Simulation]
Nk        = [4,20]                      # number of k points in each dimension
tail_correction = "Nothing"             # improvement algorithm for sum extrapolations. possible values: Nothing, Richardson, Shanks
lambda_correction = "sp"                # nothing, sp, sp_ch
bosonic_sum = "common"                  # common (intersection of individual ranges), individual (max range in each channel; fortran default), full (all frequencies), fixed:N:M (always sum from N to (including) M, indexing starts at 0)
force_full_bosonic_chi = true           # compute all omega frequencies for chi and trilex
chi_unusable_fill_value = "chi_lambda"  # can be "0", "chi_lambda" or "chi". sets either 0, lambda corrected or non lambda corrected values outside usable omega range
rhs  = "native"                         # native (fixed for tc, error_comp for naive), fixed (n/2 (1 - n/2) - sum(chi_ch)), error_comp (chi_loc_ch + chi_loc_sp - chi_ch)
fermionic_tail_coeffs = [0,1,2,3,4]     # internal parameter for richardson sum extrapolation
bosonic_tail_coeffs = [0,1,2,3,4]       # internal parameter for richardson sum extrapolation
usable_prct_reduction = 0.1             # safety cutoff for usable ranges, 0.0 means all values where chi is positive and strictly decreasing


[Environment]
inputDataType = "jld2"                  # LEGACY, do not change
writeFortran = false                    # LEGACY, do not change
loadAsymptotics = false                 # LEGACY, do not change
inputDir = ""                           # absolute path to input dir
freqFile = ""                           # absolute path to freqList.jld2 file
inputVars = "triqs_out.jld2"
asymptVars = "vars_asympt_sums.jld"     # LEGACY, do not change
cast_to_real = false                    # TODO: not implemented. cast all arrays with vanishing imaginary part to real
loglevel = "debug"                      # error, warn, info, debug
logfile = "stderr"                      # STDOUT, STDERR, filename
progressbar = false                     # LEGACY, do not change

[legacy]
lambda_correction = true                # Should a lambda-correction be performed only in the spin-channel?

[Debug]
read_bubble = false                     # LEGACY, do not change
"""
    return cfg
end
