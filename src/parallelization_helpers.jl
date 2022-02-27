macro send_to_all(ex)
    println(ex)
    println(typeof(ex.args[1]))
    println(:(ex.args[1] = 3))
    eval(:($(ex.args[1]) = $(ex.args[1])))
    #esc(:(a=3))
    #println("$(ex.args[1]) = \$$($(ex.args[1]))")
    #esc($(ex.args[1])) = 4#Meta.quot(ex.args[1])
end

macro vars_clear()

    #elseif ex.head == :tuple    # multiple return values
    #else                        # could not parse expression
end

#TODO: use init function instead of zero initializer
#TODO: size as parameter?
@inline function create_wkn(sP::SimulationParameters, Nk::Int)
    b_size = (Nk, 2*sP.n_iν, 2*sP.n_iω+1)
    return Array{ComplexF64, 3}(undef, b_size)
end

@inline function create_wk(sP::SimulationParameters, Nk::Int)
    b_size = (Nk, 2*sP.n_iω+1)
    return Array{ComplexF64, 2}(undef, b_size)
end

@inline function create_w(sP::SimulationParameters, Nk::Int)
    b_size = (2*sP.n_iω+1,)
    return Array{ComplexF64, 1}(undef, b_size)
end


@inline _parallel_decision(Niω::Int, Nk::Int)::Bool = false# (Niω < 10 || Nk < 100) ? false : true

function parallel_χγ(ωindices::AbstractVector{Int}, type::Symbol, Γr::ΓT, χ₀_data::Array{ComplexF64,3}, χ₀_asym::Array{ComplexF64,2}, mP::ModelParameters, sP::SimulationParameters)
    s  = type === :ch ? -1 : 1
    Nν = 2*sP.n_iν
    Nq = size(χ₀.data,1)
    Nω = length(ωindices)
    γ  = γT(undef, Nq, Nν, Nω)
    χ  = χT(undef, Nq, Nω)
    νi_range = 1:Nν
    qi_range = 1:Nq

    χννpω = Matrix{_eltype}(undef, Nν, Nν)
    ipiv = Vector{Int}(undef, Nν)
    work = _gen_inv_work_arr(χννpω, ipiv)

    for ωi in ωindices
        ωn = (ωi - sP.n_iω) - 1
        for qi in qi_range
            χννpω[:,:] = -deepcopy(Γr[:,:,ωi])
            for l in νi_range
                χννpω[l,l] += 1.0/χ₀.data[qi,sP.n_iν_shell+l,ωi]
            end
            inv!(χννpω, ipiv, work)
            χ[qi, ωi], λ_out = calc_χλ_impr(type, ωn, χννpω, view(χ₀.data,qi,:,ωi), 
                                           mP.U, mP.β, χ₀.asym[qi,ωi], sP.χ_helper);
            γ[qi, :, ωi] = (1 .- s*λ_out) ./ (1 .+ s*mP.U .* χ[qi, ωi])
        end
    end

end
