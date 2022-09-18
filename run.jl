using JLD2, FileIO

println("Modules loaded")
flush(stdout)
flush(stderr)


function find_lin_interp_root(xdata::AbstractVector{Float64}, ydata::AbstractVector{Float64})
    ind = findlast(x -> x < 0, ydata)
    println(ind)
    (ind == nothing) && return NaN
    (ind == 1) && return NaN
    (ind == length(xdata)) && return NaN
    Δx = xdata[ind+1] - xdata[ind]
    Δy = ydata[ind+1] - ydata[ind]
    m = Δy/Δx
    println("m=$m; yd=$(ydata[ind]); yd1=$(ydata[ind+1]); xd=$(xdata[ind]); xd1=$(xdata[ind+1])")
    x₀ = Δx > 0 ? -(ydata[ind]-xdata[ind])/m : (ydata[ind]-xdata[ind+1])/m 
    return x₀
end

function find_root(c2_data::Array{Float64,2})
    ydata = (c2_data[5,end] .- c2_data[6,end]) > 0 ? c2_data[5,:] .- c2_data[6,:] : c2_data[6,:] .- c2_data[5,:]
    xdata_sp = c2_data[1,:]
    xdata_ch = c2_data[2,:]
    λsp = find_lin_interp_root(xdata_sp, ydata)
    λch = find_lin_interp_root(xdata_ch, ydata)
    check = (isnan(λsp) && isnan(λch)) ? Inf : 0.0 
    λsp, λch, check
end

function run_sim(; run_c2_curve=false, fname="", descr="", cfg_file=nothing, res_prefix="", res_postfix="", save_results=true, log_io=devnull)
    @warn "assuming linear, continuous nu grid for chi/trilex"

    @timeit LadderDGA.to "input" wp, mP, sP, env, kGridsStr = readConfig(cfg_file);


    last_λ = [-Inf, -Inf]
    for kIteration in 1:length(kGridsStr)
        cfg_string = read(cfg_file, String)
        @info "Running calculation for $(kGridsStr[kIteration])"
        tc_s = (sP.tc_type_f != :nothing) ? "rtc" : "ntc"
        fname_out =  res_prefix*"lDGA_"*tc_s*"_k$(kGridsStr[kIteration][2])_"*res_postfix*".jld2" 
        if isfile(fname_out)
            @warn "Skipping existing file " fname_out
            continue
        end
        @timeit LadderDGA.to "setup" Σ_ladderLoc, Σ_loc, imp_density, kG, gLoc_fft, gLoc_rfft, Γsp, Γch, χDMFTsp, χDMFTch, χ_sp_loc, γ_sp_loc, χ_ch_loc, γ_ch_loc, χ₀Loc, gImp = setup_LDGA(kGridsStr[1], mP, sP, env);

        @info "non local bubble"
        flush(log_io)
        @timeit LadderDGA.to "nl bblt par" bubble = calc_bubble_par(gLoc_fft, gLoc_rfft, kG, mP, sP, workerpool=wp);
        @info "chi sp"
        flush(log_io)
        @timeit LadderDGA.to "nl xsp par" χ_sp, γ_sp = LadderDGA.calc_χγ_par(:sp, Γsp, bubble, kG, mP, sP, workerpool=wp);
        @info "chi ch"
        flush(log_io)
        @timeit LadderDGA.to "nl xch par" χ_ch, γ_ch  = LadderDGA.calc_χγ_par(:ch, Γch, bubble, kG, mP, sP, workerpool=wp);
        flush(stdout)
        flush(stderr)
        ωindices = intersect(χ_sp.usable_ω, χ_ch.usable_ω)

        @timeit LadderDGA.to "λ₀" begin
            Fsp = F_from_χ(χDMFTsp, gImp[1,:], sP, mP.β);
            λ₀ = calc_λ0(bubble, Fsp, χ_sp_loc, γ_sp_loc, mP, sP)
        end

        @info "λsp"
        flush(log_io)
        λsp = λ_correction(:sp, imp_density, χ_sp, γ_sp, χ_ch, γ_ch, gLoc_rfft, λ₀, kG, mP, sP)

        @timeit LadderDGA.to "c2" c2_res = run_c2_curve ? c2_curve(15, 15, last_λ, χ_sp, γ_sp, χ_ch, γ_ch, gLoc_rfft, λ₀, kG, mP, sP) : Matrix{Float64}(undef, 6,0)
        c2_root_res = find_root(c2_res)
        @info "c2 root result:  $c2_root_res"

        λspch, λspch_z = try
            @timeit LadderDGA.to "new λ par" λspch = λ_correction(:sp_ch, imp_density, χ_sp, γ_sp, χ_ch, γ_ch, gLoc_rfft, λ₀, kG, mP, sP, x₀=[c2_root_res[1], c2_root_res[2]], parallel=true, workerpool=wp)
            println("extended lambda: ", λspch)
            @info λspch
            λspch, λspch.zero
        catch e
            @warn e
            @warn "new lambda correction did non converge, resetting lambda to zero"
            nothing, [c2_root_res[1], c2_root_res[2]]
        end
        sp_pos = false
        ch_pos = false

        @timeit LadderDGA.to "pp" begin

            iωn = 1im .* 2 .* collect(-sP.n_iω:sP.n_iω)[ωindices] .* π ./ mP.β
            nh = ceil(Int,size(χ_sp, 2)/2)
            νmax::Int = min(sP.n_iν,floor(Int,3*length(ωindices)/8))

            # DMFT
            @timeit LadderDGA.to "Σ" Σ_ladder_DMFT = LadderDGA.calc_Σ(χ_sp, γ_sp, χ_ch, γ_ch, λ₀, gLoc_rfft, kG, mP, sP);
            sp_pos = all(kintegrate(kG,real(χ_sp.data),1)[1,ωindices] .>= 0)
            ch_pos = all(kintegrate(kG,real(χ_ch.data),1)[1,ωindices] .>= 0)
            χsp_sum = sum(kintegrate(kG,real(χ_sp.data),1)[1,ωindices])/mP.β
            χch_sum = sum(kintegrate(kG,real(χ_ch.data),1)[1,ωindices])/mP.β
            @info "check for positivity of susceptibilities, DMFT. spin channel: " sp_pos ", charge channel: " ch_pos
            @info "sum check: " χsp_sum χch_sum
            χAF_DMFT = real(1 / (1 / χ_sp[end,nh]))
            #LadderDGA.writeFortranΣ("klist_DMFT", Σ_ladder_DMFT.parent .- mP.n * mP.U/2, mP.β)
            E_kin_ED, E_pot_ED = LadderDGA.calc_E_ED(env.inputDir*"/"*env.inputVars)

            # Pauli
            χ_λ!(χ_sp, λsp);
            #gLoc = G_from_Σ(Σ_loc, kG.ϵkGrid, sP.fft_range, mP);
            E_kin_DMFT_1, E_pot_DMFT_1 = 0.0, 0.0 #calc_E(gLoc, Σ_loc, kG, mP, νmax = length(Σ_loc))
            E_pot_DMFT_2 = 0.5 * mP.U * real(sum(kintegrate(kG,χ_ch.data .- χ_sp.data,1)[1,ωindices])/mP.β) + mP.U * mP.n^2/4
            sp_pos = all(kintegrate(kG,real(χ_sp.data),1)[1,ωindices] .>= 0)
            ch_pos = all(kintegrate(kG,real(χ_ch.data),1)[1,ωindices] .>= 0)
            χsp_sum = sum(kintegrate(kG,real(χ_sp.data),1)[1,ωindices])/mP.β
            χch_sum = sum(kintegrate(kG,real(χ_ch.data),1)[1,ωindices])/mP.β
            @info "check for positivity of susceptibilities, λsp. spin channel: " sp_pos ", charge channel: " ch_pos
            @info "sum check: " χsp_sum χch_sum
            Σ_ladder_λsp = LadderDGA.calc_Σ(χ_sp, γ_sp, χ_ch, γ_ch, λ₀, gLoc_rfft, kG, mP, sP);
            #LadderDGA.writeFortranΣ("klist_l_sp", Σ_ladder_λsp.parent  .- mP.n * mP.U/2, mP.β)
            E_kin_λsp_1, E_pot_λsp_1 = calc_E(Σ_ladder_λsp.parent, kG, mP, νmax = νmax)
            E_pot_λsp_2 = 0.5 * mP.U * real(sum(kintegrate(kG,χ_ch.data .- χ_sp.data,1)[1,ωindices])/mP.β)  + mP.U * mP.n^2/4
            χAF_λsp = real(1 / (1 / χ_sp[end,nh]))
            χ_λ!(χ_sp, -λsp); 

            # Both
            χ_λ!(χ_sp, λspch_z[1]); 
            χ_λ!(χ_ch, λspch_z[2]); 
            sp_pos = all(kintegrate(kG,real(χ_sp.data),1)[1,ωindices] .>= 0)
            ch_pos = all(kintegrate(kG,real(χ_ch.data),1)[1,ωindices] .>= 0)
            @info "check for positivity of susceptibilities, λspch. spin channel: " sp_pos ", charge channel: " ch_pos
            χsp_sum = sum(kintegrate(kG,real(χ_sp.data),1)[1,ωindices])/mP.β
            χch_sum = sum(kintegrate(kG,real(χ_ch.data),1)[1,ωindices])/mP.β
            @info "sum check: $χsp_sum + $χch_sum  = $(χsp_sum + χch_sum) ?=? 1/2" 

            Σ_ladder_λspch = LadderDGA.calc_Σ(χ_sp, γ_sp, χ_ch, γ_ch, λ₀, gLoc_rfft, kG, mP, sP);
            #LadderDGA.writeFortranΣ("klist_lspch", Σ_ladder_λspch.parent  .- mP.n * mP.U/2, mP.β)
            E_kin_λspch_1, E_pot_λspch_1 = calc_E(Σ_ladder_λspch.parent, kG, mP, νmax = νmax)
            E_pot_λspch_2 = 0.5 * mP.U * real(sum(kintegrate(kG,χ_ch.data .- χ_sp.data,1)[1,ωindices])/mP.β) + mP.U * mP.n^2/4
            χAF_λspch = real(1 / (1 / χ_sp[end,nh]))
            χ_λ!(χ_sp, -λspch_z[1]); 
            χ_λ!(χ_ch, -λspch_z[2]); 
        end

    # Prepare data
        if kG.Nk >= 27000
            @warn "Large number of k-points (Nk = $(kG.Nk)). χ₀, γ will not be saved!"
            γ_sp = nothing
            γ_ch = nothing
        end

        @info "Writing to $(fname_out)"
        flush(log_io)
        flush(stdout)
        flush(stderr)
        @timeit LadderDGA.to "write" jldopen(fname_out, "w") do f
            f["Description"] = descr
            f["config"] = cfg_string 
            f["kIt"] = kIteration  
            f["Nk"] = kG.Ns
            f["sP"] = sP
            f["mP"] = mP
            f["imp_density"] = imp_density
            f["Sigma_loc"] = Σ_ladderLoc
            (kG.Nk < 27000) && (f["λ₀_sp"] = λ₀)
            f["χch"] = χ_sp
            f["χsp"] = χ_ch
            f["γch"] = γ_sp
            f["γsp"] = γ_ch
            f["χAF_DMFT"] = χAF_DMFT
            f["χAF_λsp"] = χAF_λsp
            f["χAF_λspch"] = χAF_λspch
            f["Σ_ladder_DMFT"] = Σ_ladder_DMFT
            f["Σ_ladder_λsp"] = Σ_ladder_λsp
            f["Σ_ladder_λspch"] = Σ_ladder_λspch
            f["E_kin_ED"] = E_kin_ED
            f["E_pot_ED"] = E_pot_ED
            f["E_kin_DMFT_1"] = E_kin_DMFT_1
            f["E_pot_DMFT_1"] = E_pot_DMFT_1
            f["E_pot_DMFT_2"] = E_pot_DMFT_2
            f["E_kin_λsp_1"] = E_kin_λsp_1
            f["E_pot_λsp_1"] = E_pot_λsp_1
            f["E_pot_λsp_2"] = E_pot_λsp_2
            f["E_kin_λspch_1"] = E_kin_λspch_1
            f["E_pot_λspch_1"] = E_pot_λspch_1
            f["E_pot_λspch_2"] = E_pot_λspch_2
            f["λsp"] = λsp
            f["λspch"] = λspch
            f["c2_λspch"] = c2_root_res
            f["positiv_check"] = [sp_pos, ch_pos]
            f["Γsp"] = Γsp 
            f["Γch"] = Γch 
            f["gImp"] = gImp
            f["kG"] = kG
            f["gLoc_fft"] = gLoc_fft
            f["gLoc_rfft"] = gLoc_rfft
            f["Sigma_DMFT"] = Σ_loc 
            f["χ₀Loc"] = χ₀Loc
            f["log"] = LadderDGA.get_log()
            f["c2_curve"] = c2_res
            #TODO: save log string
            #f["log"] = string()
        end
        @info "Runtime for iteration:"
        @info LadderDGA.to
    end
    @info "Done! Runtime:"
    print(LadderDGA.to)
end

function run2(cfg_file)
    run_sim(cfg_file=cfg_file)
end
