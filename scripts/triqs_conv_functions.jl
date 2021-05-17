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
        res[ωn,:,:] = -1.0 .* inv(χ[ωn,:,:])
        for (νn,ν) in enumerate(fGrid)
            res[ωn,νn,νn] += 1.0/χ0[(ω,ν)]
        end
    end
    return res
end

