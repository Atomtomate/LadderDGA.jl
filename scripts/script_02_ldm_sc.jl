using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using LadderDGA
using JLD2


tc_choice_i = parse(Int, ARGS[3])
tc = if tc_choice_i == 1
    tc=ΣTail_EoM
elseif tc_choice_i == 2
    δ = if length(ARGS) > 3 && !isnothing(tryparse(Float64, ARGS[4]))
        tryparse(Float64, ARGS[4])
    else
        error("ARGS[3] expected to have scaling for ExpStep!")
    end
    tc=ΣTail_ExpStep{δ}
elseif tc_choice_i == 3
    tc=ΣTail_Full
elseif tc_choice_i == 4
    tc=ΣTail_Plain
else
    error("Unrecognized tc choice $tc_choice_i")
end

function check_and_load(fname, key, err_key, dm_method)
    check, res, err = if isfile(fname) 
        jldopen(fname, "r") do f
            if haskey(f,key) && !isnothing(f[key])
                res = f[key]
                ep = abs(res.EPot_p1 .- res.EPot_p2) < res.eps_abs
                pp = abs(res.PP_p1 .- res.PP_p2) < res.eps_abs
                nh = ω0_index(χm)
                χd = fname["chi_m"]
                χm = fname["chi_d"]
                positive_m = all(χ_λ(χm, res.λm)[:, nh] .> 0)  
                positive_d = all(χ_λ(χd, res.λd)[:, nh] .> 0)
                check = dm_method ? (ep && pp && positive_m && positive_d) : (pp && positive_m, res)
                println("DBG: ", keys(f))
                check, f[key], f[err_key]
            else
                false, nothing, nothing
            end
        end
    else
        false, nothing, nothing
    end
    return check, res, err
end

function run(ARGS, tc; use_cache::Bool = true, cache_name::String = "lDGA_cache.jld2")
    cfg_file = ARGS[1]
    fname = ARGS[2]
    cfg_dir,_ = splitdir(cfg_file)

    wp, mP, sP, env, kGridsStr = readConfig(cfg_file);
    lDGAhelper = setup_LDGA(kGridsStr[1], mP, sP, env, silent=false);

    χm = nothing
    χd = nothing
    γm = nothing
    γd = nothing
    restored_run = false
    cfg_content = readlines(cfg_file)


    res_m = nothing 
    err_m = nothing
    res_m_sc = nothing 
    err_m_sc = nothing
    dm_done, res_dm, err_dm = check_and_load(fname, "res_dm", "err_dm", true)
    dmsc_done, res_dm_sc, err_dm_sc = check_and_load(fname, "res_dm_sc", "err_dm_sc", true)
    

    bubble = calc_bubble(:DMFT, lDGAhelper);
    λ₀ = calc_λ0(bubble, lDGAhelper)
    if use_cache
        cache_file = joinpath(cfg_dir, cache_name)
        run_id = hash(string(readlines(cfg_file)...))
        run_nk = lDGAhelper.kG.Ns
        if isfile(cache_file) 
            jldopen(cache_file, "r") do cache_f
                if haskey(cache_f, "$run_id")
                    χm = cache_f["$run_id/chi_m"]
                    χd = cache_f["$run_id/chi_d"]
                    γm = cache_f["$run_id/gamma_m"]
                    γd = cache_f["$run_id/gamma_d"]
                    restored_run = true
                end
            end
        end
    end
    if isnothing(χm)
        println("Did not find $run_id, restoring (mP: $(lDGAhelper.mP))")
        @time χm, γm = calc_χγ(:m, lDGAhelper, bubble);
        @time χd, γd = calc_χγ(:d, lDGAhelper, bubble);
    end
    if use_cache && !restored_run
        cache_file = joinpath(cfg_dir, cache_name)
        run_id = hash(string(readlines(cfg_file)...))
        run_nk = lDGAhelper.kG.Ns
        jldopen(cache_file, "a+") do cache_f
            if !haskey(cache_f, "$run_id")
                cache_f["$run_id/Nk"] = run_nk
                cache_f["$run_id/config"] = cfg_content
                cache_f["$run_id/chi_m"] = χm
                cache_f["$run_id/chi_d"] = χd
                cache_f["$run_id/gamma_m"] = γm
                cache_f["$run_id/gamma_d"] = γd
            end
        end
    end

    # msc_done = check_done(fname, "res_m_sc", false)

    # res_m_sc, err_m_sc = if msc_done
    #     jldopen(fname, "r") do f
    #         f["res_m_sc"], f["err_m_sc"]
    #     end
    # else
    #     nothing, nothing
    # end

    # done and vlaid do the same thing... this is duplicated because I don't want to compute χr if all results are done
    println("In $mP : dm valid = $dm_done / dm_sc valid = $dmsc_done")
        

    δ_name = (tc <: ΣTail_ExpStep) ? string(round(tc.parameters[1],digits=2)) : ""
    tc_name = string(tc.name.name)
    Epot_DMFT_p2 = EPot_p2(χm, χd, 0.0, 0.0, lDGAhelper.mP.n, lDGAhelper.mP.U, lDGAhelper.kG)
    Ekin_DMFT_p2 = χm.tail_c[3]

    try
        res_m = λm_correction(χm, γm, χd, γd, λ₀, lDGAhelper, tc=tc)
        err_m = nothing
    catch e
        res_m = nothing
        err_m = e
    end;
    jldopen(fname, "w") do f
        f["Nk"] = lDGAhelper.kG.Ns
        f["Sigma_tc"] = "$tc_name$δ_name"
        f["KGrid"] =kGridsStr[1] 
        f["U"] = lDGAhelper.mP.U
        f["μ_DMFT"] = lDGAhelper.mP.μ
        f["β"] = lDGAhelper.mP.β
        f["n"] = lDGAhelper.mP.n
        f["Epot_1Pt"] = lDGAhelper.mP.Epot_1Pt
        f["Epot_2Pt"] = Epot_DMFT_p2
        f["Ekin_2Pt"] = Ekin_DMFT_p2
        f["Ekin_1Pt"] = lDGAhelper.mP.Ekin_1Pt
        f["chi_m"] = χm
        f["chi_d"] = χd
        f["res_m"] = res_m
        f["res_dm"] = res_dm
        f["res_m_sc"] = res_m_sc
        f["res_dm_sc"] = res_dm_sc
        f["err_m"] = err_m
        f["err_dm"] = err_dm
        f["err_m_sc"] = err_m_sc
        f["err_dm_sc"] = err_dm_sc
        f["res_m_tsc"] = nothing
        f["res_dm_tsc"] = nothing
        f["err_m_tsc"] = nothing
        f["err_dm_tsc"] = nothing
    end
    if !dm_done
        try 
            res_dm = λdm_correction(χm, γm, χd, γd, λ₀, lDGAhelper; λd_δ=1e-1, tc=tc, λd_max=40.0)
            err_dm = nothing
        catch e 
            res_dm = nothing
            err_dm = e
        end;
    end

    jldopen(fname, "w") do f
        f["Nk"] = lDGAhelper.kG.Ns
        f["Sigma_tc"] = "$tc_name$δ_name"
        f["KGrid"] =kGridsStr[1] 
        f["U"] = lDGAhelper.mP.U
        f["μ_DMFT"] = lDGAhelper.mP.μ
        f["β"] = lDGAhelper.mP.β
        f["n"] = lDGAhelper.mP.n
        f["Epot_1Pt"] = lDGAhelper.mP.Epot_1Pt
        f["Epot_2Pt"] = Epot_DMFT_p2
        f["Ekin_1Pt"] = lDGAhelper.mP.Ekin_1Pt
        f["chi_m"] = χm
        f["chi_d"] = χd
        f["res_m"] = res_m
        f["res_dm"] = res_dm
        f["res_m_sc"] = res_m_sc
        f["res_dm_sc"] = res_dm_sc
        f["err_m"] = err_m
        f["err_dm"] = err_dm
        f["err_m_sc"] = err_m_sc
        f["err_dm_sc"] = err_dm_sc
        f["res_m_tsc"] = nothing
        f["res_dm_tsc"] = nothing
        f["err_m_tsc"] = nothing
        f["err_dm_tsc"] = nothing
    end
    if !dmsc_done
        try
            res_dm_sc = λdm_sc_correction(χm, γm, χd, γd, λ₀, lDGAhelper; max_steps_sc=250, max_steps_dm=200, validation_threshold=1e-7, λd_δ=1e-1, tc=tc, λd_max=30.0)
            err_dm_sc = nothing
        catch e 
            res_dm_sc = nothing
            err_dm_sc = e
        end;
    end
    jldopen(fname, "w") do f
        f["Nk"] = lDGAhelper.kG.Ns
        f["Sigma_tc"] = "$tc_name$δ_name"
        f["KGrid"] =kGridsStr[1] 
        f["U"] = lDGAhelper.mP.U
        f["μ_DMFT"] = lDGAhelper.mP.μ
        f["β"] = lDGAhelper.mP.β
        f["n"] = lDGAhelper.mP.n
        f["Epot_1Pt"] = lDGAhelper.mP.Epot_1Pt
        f["Epot_2Pt"] = Epot_DMFT_p2
        f["Ekin_2Pt"] = Ekin_DMFT_p2
        f["Ekin_1Pt"] = lDGAhelper.mP.Ekin_1Pt
        f["chi_m"] = χm
        f["chi_d"] = χd
        f["res_m"] = res_m
        f["res_dm"] = res_dm
        f["res_m_sc"] = res_m_sc
        f["res_dm_sc"] = res_dm_sc
        f["err_m"] = err_m
        f["err_dm"] = err_dm
        f["err_m_sc"] = err_m_sc
        f["err_dm_sc"] = err_dm_sc
        f["res_m_tsc"] = nothing
        f["res_dm_tsc"] = nothing
        f["err_m_tsc"] = nothing
        f["err_dm_tsc"] = nothing
    end
    if !msc_done
        try
            res_m_sc = λm_sc_correction(χm, γm, χd, γd, λ₀, lDGAhelper; λd_δ=1e-1, tc=tc)
            err_m_sc = nothing
        catch e 
            res_m_sc = nothing
            err_m_sc = e
        end;
    else
        jldopen(fname, "r") do f
            res_m_sc = f["res_m_sc"]
            err_m_sc = f["err_m_sc"]
        end
    end
    jldopen(fname, "w") do f
        f["Nk"] = lDGAhelper.kG.Ns
        f["Sigma_tc"] = "$tc_name$δ_name"
        f["KGrid"] =kGridsStr[1] 
        f["U"] = lDGAhelper.mP.U
        f["μ_DMFT"] = lDGAhelper.mP.μ
        f["β"] = lDGAhelper.mP.β
        f["n"] = lDGAhelper.mP.n
        f["Epot_1Pt"] = lDGAhelper.mP.Epot_1Pt
        f["Epot_2Pt"] = Epot_DMFT_p2
        f["Ekin_2Pt"] = Ekin_DMFT_p2
        f["Ekin_1Pt"] = lDGAhelper.mP.Ekin_1Pt
        f["chi_m"] = χm
        f["chi_d"] = χd
        f["res_m"] = res_m
        f["res_dm"] = res_dm
        f["res_m_sc"] = res_m_sc
        f["res_dm_sc"] = res_dm_sc
        f["err_m"] = err_m
        f["err_dm"] = err_dm
        f["err_m_sc"] = err_m_sc
        f["err_dm_sc"] = err_dm_sc
        f["res_m_tsc"] = nothing
        f["res_dm_tsc"] = nothing
        f["err_m_tsc"] = nothing
        f["err_dm_tsc"] = nothing
    end
end

run(ARGS, tc)
