q_axis = 1
ω_axis = 3

function gen_ν_part_slices(data::Array{T,3}, index_list::Vector{NTuple{4,Int}}) where T
    νn = map(x->x[4], index_list)
    res = zeros(size(data,q_axis), size(data,ω_axis), length(unique(νn)))
    νn_list = unique(νn)
    ωn_ranges = Vector{UnitRange{Int}}(undef, length(νn_list))
    for (i,νn) in enumerate(νn_list)
        slice = map(x->x[2], filter(x->x[4]==νn, index_list))
        ωn_ranges[i] = first(slice):last(slice)
    end
    νi = 1
    νn = index_list[1][4]
    for i in 1:length(index_list)
        ωi_i, _, νi_i, νn_i = index_list[i]
        if νn_i != νn
            νn = νn_i
            νi += 1
        end
        res[:, ωi_i, νi] = data[:,νi_i,ωi_i]
    end
    return res, νn_list, ωn_ranges
end
