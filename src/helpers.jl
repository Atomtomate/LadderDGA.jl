#TODO: this should be a macro
@everywhere @inline get_symm_f(f::Array{Complex{Float64},1}, i::Int64) = @inbounds if i < 0 conj(f[-i]) else f[i+1] end

# This function exploits, that χ(ν, ω) = χ*(-ν, -ω) and a storage of χ with only positive fermionic frequencies
# TODO: For now a fixed order of axis is assumed

function convert_to_real(f; eps=10E-12)
    if maximum(imag.(f)) > eps
        throw(InexactError("Imaginary part too large for conversion!"))
    end
    return real.(f)
end

sum_limits(a, b, e) = if (ndims(a) == 1) sum(a[b:e]) else sum(mapslices(x -> sum_limits(x,b,e), a; dims=2:ndims(a))[b:e]) end
        
sum_inner(a, cut) =  if (ndims(a) == 1) sum(a[cut:(end-cut+1)]) else 
                        sum(mapslices(x -> sum_inner(x,cut), a; dims=2:ndims(a))[cut:(end-cut+1)]) end

"""
    Sums first νmax entries of any array along given dimension.
    Warning: This has NOT been tested for multiple dimensions.
"""
sum_νmax(a, cut; dims) = mapslices(x -> sum_inner(x, (cut)), a; dims=dims)

"""
    Returns index of the maximum which is closest to the mid point of the array
"""
function find_inner_maximum(arr)
    darr = diff(arr; dims=1)
    mid_index = Int(floor(size(arr,1)/2))
    intervall_range = 1

    # find interval
    while (intervall_range < mid_index) &&
        (darr[(mid_index-intervall_range)] * darr[(mid_index+intervall_range-1)] > 0)
            intervall_range = intervall_range+1
    end

    index_maximum = mid_index-intervall_range+1
    # find index
    while darr[(mid_index-intervall_range)]*darr[index_maximum] > 0
        index_maximum = index_maximum + 1
    end
    return index_maximum
end

"""
    Returns rang of indeces that are usable under 2 conditions.
    TODO: This is temporary and should be replace with a function accepting general predicates.
"""
function find_usable_interval(arr; reduce_range_prct = 0.0)
    darr = diff(arr; dims=1)
    #index_maximum = find_inner_maximum(arr)
    mid_index = Int(ceil(size(arr,1)/2))

    if arr[mid_index] < 0.0
        res = [mid_index]
        return res
    end
    # interval for condition 1 (positive values)
    cond1_intervall_range = 1
    # find range for positive values
    while (cond1_intervall_range < mid_index) &&
        (arr[(mid_index-cond1_intervall_range)] > 0) &&
        (arr[(mid_index+cond1_intervall_range)] > 0)
        cond1_intervall_range = cond1_intervall_range + 1
    end

    # interval for condition 2 (monotonicity)
    cond2_intervall_range = 1
    # find range for first turning point
    #println(cond1_intervall_range)
    while (cond2_intervall_range < mid_index-1) &&
        (darr[(mid_index-cond2_intervall_range)] > 0) &&
        (darr[(mid_index+cond2_intervall_range)] < 0)
        cond2_intervall_range = cond2_intervall_range + 1
    end

    #println(cond2_intervall_range)
    intervall_range = minimum([cond1_intervall_range, cond2_intervall_range])
    #println(intervall_range)
    range = ceil(Int64, intervall_range*(1-reduce_range_prct))
    #println(range)
    if length(arr)%2 == 1
        res = ((mid_index-range+1):(mid_index+range-2) .+ 1)
    else
        res = ((mid_index-range+1):(mid_index+range-2) .+ 2)
    end

    #println("res: $(res) = $(mid_index) +- $(range)")
    if length(res) < 1
        println(stderr, "   ---> WARNING: could not determine usable range. Defaulting to single frequency!")
        res = [mid_index]
        println(res)
    end
    return res
end

function compute_Ekin(iνₙ, ϵₖ, Vₖ, GImp, β; full=true)
    Ekin = 0.0 + 0.0*1im
    fak = if full sum(Vₖ .^ 2)*(β^2)/4 else sum(Vₖ .^ 2)*(β^2)/8 end
    for n in 1:length(GImp)
        for l in 1:length(Vₖ)
            Ekin += (GImp[n] * (Vₖ[l]^2) / (iνₙ[n] - ϵₖ[l])) - (Vₖ[l] ^ 2)/(iνₙ[n]^2)
        end
    end
    if (full)
        return  ((Ekin - fak)/(2*β))
    else
        return (Ekin - fak)/β
    end
end

iω(n) = 1im*2*n*π/(modelParams.β);


split_n(str, n) = [str[(i-n+1):(i)] for i in n:n:length(str)]
split_n(str, n, len) = [str[(i-n+1):(i)] for i in n:n:len]

"""
    padlength(a,b)

computes the length of zero-padding required for convolution, using fft
This is the next larger or equally large number to max(a,b)
TODO: does only support padding for cube like arrays (i.e. all dimension have the same size).

# Examples
```
julia> padlength(1:5,1:14)
8
julia> padlength(1:4,1:13)
4
```
"""
padlength(a,b) = 2^floor(Int, log(2,size(a,1)+size(b,1)-1))


function fft_conv(a, b)
    zero_pad_length = padlength(a,b)
    PaddedView(0, collect(a), Tuple(repeat([pad(a,b)], ndims(a))))
end

"""
    print 4 digits of the real part of `x`
"""
printr_s(x::Complex{Float64}) = round(real(x), digits=4)
printr_s(x::Float64) = round(x, digits=4)


function setup_LDGA(configFile)
    modelParams, simParams, env = readConfig(configFile)#
    if env.loadFortran == "text"
        convert_from_fortran(simParams, env, loadFromBak)
        if env.loadAsymptotics
            readEDAsymptotics(env)
        end
    elseif env.loadFortran == "parquet"
        convert_from_fortran_pq(simParams, env)
        if env.loadAsymptotics
            readEDAsymptotics_parquet(env)
        end
    end
    println("loading from ", env.inputVars)
    vars    = load(env.inputVars) 
    G0      = vars["g0"]
    GImp    = vars["gImp"]
    Γch     = vars["GammaCharge"]
    Γsp     = vars["GammaSpin"]
    χDMFTch = vars["chiDMFTCharge"]
    χDMFTsp = vars["chiDMFTSpin"]
    println("TODO: check beta consistency, config <-> g0man, chi_dir <-> gamma dir")
    ωGrid   = (-simParams.n_iω):(simParams.n_iω)
    νGrid   = (-simParams.n_iν):(simParams.n_iν-1)
    if env.loadAsymptotics
        asympt_vars = load(env.asymptVars)
        χchAsympt = asympt_vars["chi_ch_asympt"]
        χspAsympt = asympt_vars["chi_sp_asympt"]
    end
    #TODO: unify checks
    (simParams.Nk % 2 != 0) && throw("For FFT, q and integration grids must be related in size!! 2*Nq-2 == Nk")

    Σ_loc = Σ_Dyson(G0, GImp)
    FUpDo = FUpDo_from_χDMFT(0.5 .* (χDMFTch - χDMFTsp), GImp, ωGrid, νGrid, νGrid, modelParams.β)

    χLocsp_ω = sum_freq(χDMFTsp, [2,3], simParams.tail_corrected, modelParams.β)[:,1,1]
    usable_loc_sp = simParams.fullSums ? (1:length(χLocsp_ω)) : find_usable_interval(real(χLocsp_ω))
    χLocsp = sum_freq(χLocsp_ω[usable_loc_sp], [1], simParams.tail_corrected, modelParams.β)[1]

    χLocch_ω = sum_freq(χDMFTch, [2,3], simParams.tail_corrected, modelParams.β)[:,1,1]
    usable_loc_ch = simParams.fullSums ? (1:length(χLocch_ω)) : find_usable_interval(real(χLocch_ω))
    χLocch = sum_freq(χLocch_ω[usable_loc_ch], [1], simParams.tail_corrected, modelParams.β)[1]

    return modelParams, simParams, env, Γch, Γsp, Σ_loc, FUpDo, χLocch, χLocsp, usable_loc_ch, usable_loc_sp
end
