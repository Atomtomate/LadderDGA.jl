using JLD2, FileIO

println("Modules loaded")
flush(stdout)
flush(stderr)

function run_sim(; fname="", descr="", cfg_file=nothing, res_prefix="", res_postfix="", save_results=true, log_io=devnull)
    @warn "assuming linear, continuous nu grid for chi/trilex"

    @timeit LadderDGA.to "input" wp, mP, sP, env, kGridsStr = readConfig(cfg_file);


    for kIteration in 1:length(kGridsStr)
        @info "Running calculation for $(kGridsStr[kIteration])"
        @timeit LadderDGA.to "setup" Σ_ladderLoc, Σ_loc, imp_density, kG, gLoc_fft, gLoc_rfft, Γsp, Γch, χDMFTsp, χDMFTch, locQ_sp, locQ_ch, χ₀Loc, gImp = setup_LDGA(kGridsStr[1], mP, sP, env);

        @info "non local bubble"
        flush(log_io)
        @timeit LadderDGA.to "nl bblt par" bubble = calc_bubble_par(gLoc_fft, gLoc_rfft, kG, mP, sP, workerpool=wp);
        @info "chi sp"
        flush(log_io)
        @timeit LadderDGA.to "nl xsp par" nlQ_sp = LadderDGA.calc_χγ_par(:sp, Γsp, bubble, kG, mP, sP, workerpool=wp);
        @info "chi ch"
        flush(log_io)
        @timeit LadderDGA.to "nl xch par" nlQ_ch = LadderDGA.calc_χγ_par(:ch, Γch, bubble, kG, mP, sP, workerpool=wp);

        @timeit LadderDGA.to "λ₀" begin
            Fsp = F_from_χ(χDMFTsp, gImp[1,:], sP, mP.β);
            λ₀ = calc_λ0(bubble, Fsp, locQ_sp, mP, sP)
        end

        @info "λsp"
        flush(log_io)
        λsp = λ_correction(:sp, imp_density, nlQ_sp, nlQ_ch, gLoc_rfft, λ₀, kG, mP, sP)

        λspch = try
            @timeit LadderDGA.to "new λ par" λspch = λ_correction(:sp_ch, imp_density, nlQ_sp, nlQ_ch, gLoc_rfft, λ₀, kG, mP, sP, parallel=true, workerpool=wp)
            @info λspch
            λspch.zero
        catch e
            @warn e
            @warn "new lambda correction did non converge, resetting lambda to zero"
            [0.0,0.0]
        end
        #@timeit LadderDGA.to "new λ" λspch = λ_correction(:sp_ch, imp_density, nlQ_sp, nlQ_ch, gLoc_rfft, λ₀, kG, mP, sP)

        @timeit LadderDGA.to "nl Σ par" Σ_ladder_DMFT = LadderDGA.calc_Σ_parts(nlQ_sp, nlQ_ch, λ₀, gLoc_rfft, kG, mP, sP);
        χ_λ!(nlQ_sp.χ, nlQ_sp.χ, λsp); nlQ_sp.λ = λsp;
        @timeit LadderDGA.to "nl Σ par" Σ_ladder_λsp = LadderDGA.calc_Σ_parts(nlQ_sp, nlQ_ch, λ₀, gLoc_rfft, kG, mP, sP);
        χ_λ!(nlQ_sp.χ, nlQ_sp.χ, -λsp); 
        χ_λ!(nlQ_sp.χ, nlQ_sp.χ, λspch[1]); 
        χ_λ!(nlQ_ch.χ, nlQ_ch.χ, λspch[2]); 
        nlQ_sp.λ = λspch[1];
        nlQ_ch.λ = λspch[2];
        @timeit LadderDGA.to "nl Σ par" Σ_ladder_λspch = LadderDGA.calc_Σ_parts(nlQ_sp, nlQ_ch, λ₀, gLoc_rfft, kG, mP, sP);
        χ_λ!(nlQ_sp.χ, nlQ_sp.χ, -λspch[1]); 
        χ_λ!(nlQ_ch.χ, nlQ_ch.χ, -λspch[2]); 
        nlQ_sp.λ = 0.0;
        nlQ_ch.λ = 0.0;

    # Prepare data
        if kG.Nk >= 27000
            @warn "Large number of k-points (Nk = $(kG.Nk)). χ₀, γ will not be saved!"
            nlQ_sp.γ = similar(nlQ_sp.γ, 0,0,0)
            nlQ_ch.γ = similar(nlQ_ch.γ, 0,0,0)
        end

        flush(log_io)
        tc_s = (sP.tc_type_f != :nothing) ? "rtc" : "ntc"
        fname = fname == "" ? res_prefix*"lDGA_"*tc_s*"_k$(kG.Ns)_ext_l"*res_postfix*".jld2" : fname * ".jld2"
        @info "Writing to $(fname)"
        @timeit LadderDGA.to "write" jldopen(fname, "w") do f
            f["Description"] = descr
            f["config"] = read(cfg_file, String)
            f["kIt"] = kIteration  
            f["Nk"] = kG.Ns
            f["sP"] = sP
            f["mP"] = mP
            f["imp_density"] = imp_density
            f["Sigma_loc"] = Σ_ladderLoc
            f["λ₀_sp"] = λ₀
            f["nlQ_sp"] = nlQ_sp
            f["nlQ_ch"] = nlQ_ch
            f["Σ_ladder_DMFT"] = Σ_ladder_DMFT
            f["Σ_ladder_λsp"] = Σ_ladder_λsp
            f["Σ_ladder_λspch"] = Σ_ladder_λspch
            f["λsp"] = λsp
            f["λspch"] = λspch
            f["Γsp"] = Γsp 
            f["Γch"] = Γch 
            f["gImp"] = gImp
            f["kG"] = kG
            f["gLoc_fft"] = gLoc_fft
            f["gLoc_rfft"] = gLoc_rfft
            f["Sigma_DMFT"] = Σ_loc 
            f["χ₀Loc"] = χ₀Loc
            f["log"] = LadderDGA.get_log()
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
