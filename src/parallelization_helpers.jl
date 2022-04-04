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

function par_partition(set, N)
    N = N <= 0 ? 1 : N
    s,r = divrem(length(set),N)
    [(i*s+1+(i<r)*(i)+(i>=r)*r):(i+1)*s+(i<r)*(i+1)+(i>=r)*r for i in 0:(N-1)]
end
