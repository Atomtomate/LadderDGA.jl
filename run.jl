using JLD2, FileIO

println("Modules loaded")
flush(stdout)
flush(stderr)

function run_sim(; descr="", cfg_file=nothing, res_prefix="", res_postfix="", save_results=true, log_io=devnull)
    @warn "assuming linear, continuous nu grid for chi/trilex"

    @timeit LadderDGA.to "read" mP, sP, env, kGridsStr = readConfig(cfg_file);


    for kIteration in 1:length(kGridsStr)
        @info "Running calculation for $(kGridsStr[kIteration])"
        @timeit LadderDGA.to "setup" Σ_ladderLoc, Σ_loc, imp_density, kG, gLoc, gLoc_fft, Γsp, Γch, χDMFTsp, χDMFTch, locQ_sp, locQ_ch, χ₀Loc, gImp = setup_LDGA(kGridsStr[kIteration], mP, sP, env);
        Fsp = F_from_χ(χDMFTsp, gImp[1,:], sP, mP.β);

        flush(log_io)
        # ladder quantities
        @info "non local bubble"
        flush(log_io)
        @timeit LadderDGA.to "nl bblt" bubble = calc_bubble(gLoc_fft, kG, mP, sP);
        @info "chi sp"
        flush(log_io)
        @timeit LadderDGA.to "nl xsp" nlQ_sp = calc_χγ(:sp, Γsp, bubble, kG, mP, sP);
        @info "chi ch"
        flush(log_io)
        @timeit LadderDGA.to "nl xch" nlQ_ch = calc_χγ(:ch, Γch, bubble, kG, mP, sP);
        
        λ₀ = calc_λ0(bubble, Fsp, locQ_sp, mP, sP)

        @info "λsp"
        flush(log_io)
        λsp_old = λ_correction(:sp, imp_density, nlQ_sp, nlQ_ch, gLoc_fft, λ₀, kG, mP, sP)
        @info "found $λsp_old\nextended λ"
        λnew_nls = λ_correction(:sp_ch, imp_density, nlQ_sp, nlQ_ch, gLoc_fft, λ₀, kG, mP, sP)
        @info "found $λnew_nls\n"
        λnew = λnew_nls.f_converged ? λnew_nls.zero : [NaN, NaN]
        flush(log_io)

        #@timeit LadderDGA.to "lsp(lch)" λch_range, spOfch = λsp_of_λch(nlQ_sp, nlQ_ch, kG, mP, sP; λch_max=20.0, n_λch=100)

        #@timeit LadderDGA.to "c2" λsp_of_λch_res = c2_along_λsp_of_λch(λch_range, spOfch, nlQ_sp, nlQ_ch, bubble,
        #               Σ_ladderLoc, Σ_loc, gLoc_fft, Fsp, locQ_sp, kG, mP, sP)
    # Prepare data

        flush(log_io)
        tc_s = (sP.tc_type_f != :nothing) ? "rtc" : "ntc"
        fname = res_prefix*"lDGA_"*tc_s*"_k$(kG.Ns)_ext_l"*res_postfix*".jld2"
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
            f["bubble"] = bubble
            f["λ₀_sp"] = λ₀
            f["nlQ_sp"] = nlQ_sp
            f["nlQ_ch"] = nlQ_ch
            f["λsp_old"] = λsp_old
            #f["λch_range"] = λch_range
            #f["spOfch"] = spOfch
            #f["λsp_of_λch_res"] = λsp_of_λch_res
            f["Γsp"] = Γsp 
            f["Γch"] = Γch 
            f["gImp"] = gImp
            f["kG"] = kG
            f["gLoc_fft"] = gLoc_fft
            f["Sigma_DMFT"] = Σ_loc 
            f["χ₀Loc"] = χ₀Loc
            f["λnew_nls"] = λnew_nls
            f["λnew"] = λnew
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
