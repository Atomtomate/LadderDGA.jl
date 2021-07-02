#TODO: this should be a macro
@inline get_symm_f(f::Array{Complex{Float64},1}, i::Int64) = (i < 0) ? conj(f[-i]) : f[i+1]
@inline get_symm_f(f::Array{Complex{Float64},2}, i::Int64) = (i < 0) ? conj(f[-i,:]) : f[i+1,:]
@inline get_symm_f(f::Array{Complex{Interval{Float64}},2}, i::Int64) = (i < 0) ? conj(f[-i,:]) : f[i+1,:]
store_symm_f(f::Array{T, 1}, range::UnitRange{Int64}) where T <: Number = [get_symm_f(f,i) for i in range]
store_symm_f(f::Array{T, 2}, range::UnitRange{Int64}) where T <: Number = [get_symm_f(f,i) for i in range]

# This function exploits, that χ(ν, ω) = χ*(-ν, -ω) and a storage of χ with only positive fermionic frequencies
# TODO: For now a fixed order of axis is assumed


function default_sum_range(mid_index::Int, lim_tuple::Tuple{Int,Int}) where T
    return union((mid_index - lim_tuple[2]):(mid_index - lim_tuple[1]), (mid_index + lim_tuple[1]):(mid_index + lim_tuple[2]))
end


function reduce_range(range::AbstractArray, red_prct::Float64)
    sub = floor(Int, length(range)/2 * red_prct)
    lst = maximum([last(range)-sub, ceil(Int,length(range)/2 + iseven(length(range)))])
    fst = minimum([first(range)+sub, ceil(Int,length(range)/2)])
    return fst:lst
end


split_n(str, n) = [str[(i-n+1):(i)] for i in n:n:length(str)]
split_n(str, n, len) = [str[(i-n+1):(i)] for i in n:n:len]

"""
    print 4 digits of the real part of `x`
"""
printr_s(x::Complex{Float64}) = round(real(x), digits=4)
printr_s(x::Float64) = round(x, digits=4)


function setup_LDGA(kGrid::ReducedKGrid, freqList::AbstractArray, mP::ModelParameters, sP::SimulationParameters, env::EnvironmentVars)
    fft_range = -(sP.n_iν+sP.n_iω):(sP.n_iν+sP.n_iω-1)
    in_file = env.inputVars
    if(myid() == 1)
        if env.inputDataType == "text"
            convert_from_fortran(sP, env, false)
            if env.loadAsymptotics
                readEDAsymptotics(env, mP)
            end
        elseif env.inputDataType == "parquet"
            convert_from_fortran_pq(sP, env)
            if env.loadAsymptotics
                readEDAsymptotics_parquet(env)
            end
        elseif env.inputDataType == "jld2"
            if env.loadAsymptotics
                readEDAsymptotics_julia(env)
            end
            in_file = env.inputDir*"/"*env.inputVars
        end
        f = load(in_file)
        Γch = f["Γch"]
        Γsp = f["Γsp"]
        χDMFTch = f["χDMFTch"]
        χDMFTsp = f["χDMFTsp"]
        gImp_in, Σ_loc = if haskey(f, "g0")
            gImp_in = copy(f["gImp"])
            g0 = copy(f["g0"])
            Σ_loc = Σ_Dyson(g0, gImp_in)
            gImp_in, Σ_loc
        else
            gImp_in = copy(f["gImp"])
            Σ_loc = copy(f["SigmaLoc"])
            gImp_in, Σ_loc
        end
        FUpDo_in = FUpDo_from_χDMFT(0.5 .* (χDMFTch - χDMFTsp), gImp_in, freqList, mP, sP)
        gImp_sym = store_symm_f(gImp_in, fft_range)
        gImp = reshape(gImp_sym, (length(gImp_sym),1))
        gLoc = G_from_Σ(Σ_loc, expandKArr(kGrid, kGrid.ϵkGrid)[:], fft_range, mP);
        gLoc_fft_in = flatten_2D(map(x->fft(reshape(x, gridshape(kGrid)...)), gLoc))
    end
    FUpDo = SharedArray{Complex{Float64},3}(size(FUpDo_in),pids=procs());copy!(FUpDo, FUpDo_in)
    gImp_fft = SharedArray{Complex{Float64}}(size(gImp),pids=procs());copy!(gImp_fft, gImp)
    gLoc_fft = SharedArray{Complex{Float64}}(size(gLoc_fft_in),pids=procs());copy!(gLoc_fft, gLoc_fft_in)
    χDMFTch_new = SharedArray{Complex{Float64},3}(size(χDMFTch),pids=procs());copy!(χDMFTch_new, χDMFTch)
    χDMFTsp_new = SharedArray{Complex{Float64},3}(size(χDMFTsp),pids=procs());copy!(χDMFTsp_new, χDMFTsp)
    Γch_new = SharedArray{Complex{Float64},3}(size(Γch),pids=procs());copy!(Γch_new, Γch)
    Γsp_new = SharedArray{Complex{Float64},3}(size(Γsp),pids=procs());copy!(Γsp_new, Γsp)
    @warn "TODO: check beta consistency, config <-> g0man, chi_dir <-> gamma dir"
    if env.loadAsymptotics
        asympt_vars = load(env.asymptVars)
        χchAsympt = asympt_vars["chi_ch_asympt"]
        χspAsympt = asympt_vars["chi_sp_asympt"]
    end
    #TODO: unify checks
    (sP.ωsum_type == :full && (sP.tc_type_b != :nothing)) && @warn "Full Sums combined with tail correction will probably yield wrong results due to border effects."
    sP.ωsum_type == :individual && println(stderr, "Individual ranges not tested yet")
    ((sP.n_iν < 30 || sP.n_iω < 15) && (sP.tc_type_f != :nothing)) && @warn "Improved sums usually require at least 30 positive fermionic frequencies"


    #TODO: this should no assume consecutive frequencies
    #νGrid = [(i,j) for i in 1:(2*sP.n_iω+1) for j in (1:2*sP.n_iν) .- trunc(Int64,sP.shift*(i-sP.n_iω-1)/2)]
    νGrid = Array{AbstractArray}(undef, 2*sP.n_iω+1);
    for i in 1:length(νGrid)
        νGrid[i] = (1:2*sP.n_iν) .- trunc(Int64,sP.shift*(i-1-sP.n_iω)/2)
    end
    #TODO: fix this! do not assume anything about freqGrid without reading from file

    sh_f = get_sum_helper(2*sP.n_iν, sP, :f)

    χLocsp_ω = sum_freq(χDMFTsp, [2,3], sh_f, mP.β)[:,1,1]
    χLocch_ω = sum_freq(χDMFTch, [2,3], sh_f, mP.β)[:,1,1]
    ωZero = sP.n_iω
    χLocsp_ω_tmp = deepcopy(χLocsp_ω)
    χLocch_ω_tmp = deepcopy(χLocch_ω)


    if sP.ω_smoothing == :full
        filter_MA!(χLocsp_ω[1:ωZero],3,χLocsp_ω[1:ωZero])
        filter_MA!(χLocsp_ω[ωZero:end],3,χLocsp_ω[ωZero:end])
        filter_MA!(χLocch_ω[1:ωZero],3,χLocch_ω[1:ωZero])
        filter_MA!(χLocch_ω[ωZero:end],3,χLocch_ω[ωZero:end])
        χLocsp_ω_tmp = deepcopy(χLocsp_ω)
        χLocch_ω_tmp = deepcopy(χLocch_ω)
    elseif sP.ω_smoothing == :range
        χLocsp_ω_tmp[1:ωZero]   = filter_MA(3,χLocsp_ω[1:ωZero])
        χLocsp_ω_tmp[ωZero:end] = filter_MA(3,χLocsp_ω[ωZero:end])
        χLocch_ω_tmp[1:ωZero]   = filter_MA(3,χLocch_ω[1:ωZero])
        χLocch_ω_tmp[ωZero:end] = filter_MA(3,χLocch_ω[ωZero:end])
    end


    usable_loc_sp = find_usable_interval(real(χLocsp_ω_tmp), sum_type=sP.ωsum_type)
    usable_loc_ch = find_usable_interval(real(χLocch_ω_tmp), sum_type=sP.ωsum_type)
    #if sP.tc_type_f != :nothing
    #    usable_loc_sp = reduce_range(usable_loc_sp, 1.0)
    #    usable_loc_ch = reduce_range(usable_loc_ch, 1.0)
    #end
    loc_range = intersect(usable_loc_sp, usable_loc_ch)
    if sP.ωsum_type == :common
        @info "setting usable ranges of sp and ch channel from $usable_loc_sp and $usable_loc_ch to the same range of $loc_range"
        usable_loc_ch = loc_range
        usable_loc_sp = loc_range
    end

    E_kin_ED, E_pot_ED = 0.0, 0.0
    if sP.tc_type_b == :coeffs
    if isfile(env.inputDir * "/gm_wim") && isfile(env.inputDir * "/hubb.andpar")
        @info "Computing kinetic energie for improved bosonic sums."
        iνₙ, GImp    = readGImp(env.inputDir * "/gm_wim", only_positive=true)
        ϵₖ, Vₖ, μ    = read_anderson_parameters(env.inputDir * "/hubb.andpar");
        E_kin_ED, E_pot_ED  = calc_E_ED(iνₙ[1:length(GImp)], ϵₖ, Vₖ, GImp, mP)
    else
        @warn "Could not find hubb.andpar and gm_wim for kinetic energy. proceding without improved bosonic sums!"
    end
    end

    sh_b_sp = get_sum_helper(usable_loc_sp, sP, :b)
    sh_b_ch = get_sum_helper(usable_loc_ch, sP, :b)

    χLocsp = sum_freq(χLocsp_ω[usable_loc_sp], [1], sh_b_sp, mP.β)[1]
    χLocch = sum_freq(χLocch_ω[usable_loc_ch], [1], sh_b_ch, mP.β)[1]

    impQ_sp = ImpurityQuantities(Γsp_new, χDMFTsp_new, χLocsp_ω, χLocsp, usable_loc_sp, [0,0,E_kin_ED])
    impQ_ch = ImpurityQuantities(Γch_new, χDMFTch_new, χLocch_ω, χLocch, usable_loc_ch, [0,0,E_kin_ED])

    χupup_ω = 0.5 * (χLocsp_ω + χLocch_ω)
    iωn = 1im .* 2 .* (-sP.n_iω:sP.n_iω)[loc_range] .* π ./ mP.β
    χupup_DMFT_ω_sub = subtract_tail(χupup_ω[loc_range], E_kin_ED, iωn)

    sh_b = get_sum_helper(loc_range, sP, :b)
    imp_density_pure = real(sum_freq(χupup_DMFT_ω_sub, [1], Naive(), mP.β, corr=-E_kin_ED*mP.β^2/12))
    imp_density = real(sum_freq(χupup_DMFT_ω_sub, [1], sh_b, mP.β, corr=-E_kin_ED*mP.β^2/12))

    @info """Inputs Read. Starting Computation.
      Local susceptibilities with ranges are:
      χLoc_sp($(impQ_sp.usable_ω)) = $(printr_s(impQ_sp.χ_loc)), χLoc_ch($(impQ_ch.usable_ω)) = $(printr_s(impQ_ch.χ_loc)) 
      sum χupup check (fit, tail sub, tail sub + fit, expected): $(0.5 .* real(χLocsp + χLocch)) ?≈? $(imp_density_pure) ?=? $(imp_density) ?≈? $(mP.n/2 * ( 1 - mP.n/2))"
      """
    return νGrid, sh_f, impQ_sp, impQ_ch, gImp_fft, gLoc_fft, Σ_loc, FUpDo, gImp, gLoc
end


function flatten_2D(arr)
    res = zeros(eltype(arr[1]),length(arr), length(arr[1]))
    for i in 1:length(arr)
        res[i,:] = arr[i][:]
    end
    return res
end

# ================== Noise Filter ==================

function filter_MA(m::Int, X::AbstractArray{T,1}) where T <: Number
    res = deepcopy(X) 
    offset = trunc(Int,m/2)
    res[1+offset] = sum(@view X[1:m])/m
    for (ii,i) in enumerate((2+offset):(length(X)-offset))
        res[i] = res[i-1] + (X[m+ii] - X[ii])/m
    end
    return res
end

function filter_MA!(res::AbstractArray{T,1}, m::Int, X::AbstractArray{T,1}) where T <: Number 
    offset = trunc(Int,m/2)
    res[1+offset] = sum(@view X[1:m])/m
    for (ii,i) in enumerate((2+offset):(length(X)-offset))
        res[i] = res[i-1] + (X[m+ii] - X[ii])/m
    end
    return res
end

function filter_KZ(m::Int, k::Int, X::AbstractArray{T,1}) where T <: Number
    res = filter_MA(m, X)
    for ki in 2:k
        res = filter_MA!(res, m, res)
    end
    return res
end

# ================== FFT + Intervals Workaround ==================
lo(arr::Array{Interval{Float64}}) = map(x->x.lo,arr)
hi(arr::Array{Interval{Float64}}) = map(x->x.hi,arr) 
lo(arr::Array{Complex{Interval{Float64}}}) = map(x->x.lo,real.(arr)) .+ map(x->x.lo,imag.(arr)) .* im
hi(arr::Array{Complex{Interval{Float64}}}) = map(x->x.hi,real.(arr)) .+ map(x->x.hi,imag.(arr)) .* im
cmplx_interval(x::Tuple{Complex{Float64},Complex{Float64}}) = Complex(interval(minimum(real.(x)),maximum(real.(x))),
                                                                      interval(minimum(imag.(x)),maximum(imag.(x))))
AbstractFFTs.fft(arr::Array{Interval{Float64}}) = map(x->interval(minimum(x),maximum(x)),zip(fft(lo(arr)), fft(lo(arr))))
AbstractFFTs.fft(arr::Array{Complex{Interval{Float64}}}) = map(x->cmplx_interval(x),zip(fft(lo(arr)), fft(hi(arr))))
AbstractFFTs.ifft(arr::Array{Interval{Float64}}) = map(x->interval(minimum(x),maximum(x)),zip(ifft(lo(arr)), ifft(lo(arr))))
AbstractFFTs.ifft(arr::Array{Complex{Interval{Float64}}}) = map(x->cmplx_interval(x),zip(ifft(lo(arr)), ifft(hi(arr))))
