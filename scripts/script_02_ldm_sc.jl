using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using LadderDGA
using JLD2
using Logging


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
elseif tc_choice_i == 5
    tc = ΣTail_λm
else
    error("Unrecognized tc choice $tc_choice_i")
end

function check_and_load(fname, key, err_key, dm_method; check_converged = false)
    check = false
    res = nothing
    err = nothing
    if isfile(fname) 
        try
            jldopen(fname, "r") do f
                if haskey(f,key) && !isnothing(f[key])
                    res = f[key]
                    err = f[err_key]
                    ep = abs(res.EPot_p1 .- res.EPot_p2) < res.eps_abs
                    pp = abs(res.PP_p1 .- res.PP_p2) < res.eps_abs
                    if !isnothing(f["chi_m"]) && !isnothing(f["chi_d"]) && !isnothing(res)
                        χd = f["chi_m"]
                        χm = f["chi_d"]
                        nh = ω0_index(χm)
                        if isfinite(res.λm) && isfinite(res.λd)
                            positive_m = all(χ_λ(χm, res.λm)[:, nh] .> 0)  
                            positive_d = all(χ_λ(χd, res.λd)[:, nh] .> 0)
                            check = if check_converged
                                dm_method ? (ep && pp && positive_m && positive_d) : (pp && positive_m)
                            else
                                true
                            end
                        end
                    end
                end
            end
        catch e
            println("could not open $fname , caught $e == DELETING FILE!")
            rm(fname)
        end
    end
    return check, res, err
end

function gen_run_id(mP, sP, kgrid_str)
    run_id_str = "$(round(mP.U,digits=8))_$(round(mP.n,digits=8))_$(round(mP.β,digits=8))_$(round(mP.μ,digits=8))"
    run_id_str = run_id_str*"$(sP.n_iν)_$(sP.n_iω)_$(sP.n_iν_shell)_$(sP.shift)_$(round(sP.usable_prct_reduction,digits=8))_$(sP.dbg_full_eom_omega)_$(sP.dbg_full_chi_omega)"
    run_id_str = run_id_str*"$(kgrid_str)"
    return hash(run_id_str)
end

function run(ARGS, tc; use_cache::Bool = true, cache_name::String = "lDGA_cache.jld2")
    cfg_file = ARGS[1]
    fname = ARGS[2]
    cfg_dir,_ = splitdir(cfg_file)
    cfg_content = readlines(cfg_file)

    wp, mP, sP, env, kGridsStr = readConfig(cfg_file);

    χm = nothing
    χd = nothing
    γm = nothing
    γd = nothing
    restored_run = false


    res_m = nothing 
    err_m = nothing
    res_m_sc = nothing 
    err_m_sc = nothing
    m_done, res_m, err_m = check_and_load(fname, "res_m", "err_m", false)
    dm_done, res_dm, err_dm = check_and_load(fname, "res_dm", "err_dm", true)
    dmsc_done, res_dm_sc, err_dm_sc = check_and_load(fname, "res_dm_sc", "err_dm_sc", true)
    msc_done, res_m_sc, err_m_sc = check_and_load(fname, "res_m_sc", "err_m_sc", false)

    run_id = gen_run_id(mP, sP, kGridsStr)

    println("[$run_id]: ======================================"); 
    println("[$run_id]: = beta = $(round(mP.β,digits=4)), U = $(round(mP.U,digits=4)), n = $(round(mP.n,digits=4)), Nk = $(kGridsStr[1][2])  ");
    println("[$run_id]: ======================================"); 
    if m_done && dm_done && dmsc_done 
        println("[$run_id]: Found completed runs for m/dm/dmsc. Aborting!"); flush(stdout)
        return true
    end

    lDGAhelper = setup_LDGA(kGridsStr[1], mP, sP, env, silent=false);
    if use_cache
        cache_file = joinpath(cfg_dir, cache_name)
        println("[$run_id]: Trying to open cache file $cache_file."); flush(stdout)
        run_nk = lDGAhelper.kG.Ns
        if isfile(cache_file) 
            try
                jldopen(cache_file, "r") do cache_f
                    if haskey(cache_f, "$run_id")
                        χm = cache_f["$run_id/chi_m"]
                        χd = cache_f["$run_id/chi_d"]
                        γm = cache_f["$run_id/gamma_m"]
                        γd = cache_f["$run_id/gamma_d"]
                        println("[$run_id]: restored χ_r and γ_r."); flush(stdout)
                        restored_run =  (!isnothing(χm) && !isnothing(χd)) ? true : false
                    end
                end
            catch e
                println("[$run_id]: WARNING! opening $cache_file resulted in $e"); 
                # println("[$run_id]: WARNING! deleting $cache_file"); flush(stdout)
                # rm(cache_file)
            end
        end
    end

    bubble = calc_bubble(:DMFT, lDGAhelper; mode=:ph, use_threads=true);
    λ₀ = calc_λ0(bubble, lDGAhelper)

    if isnothing(χm)
        println("[$run_id]: Inverting BSE, no cached values found."); flush(stdout)
        χm, γm = calc_χγ(:m, lDGAhelper, bubble; ω_symmetric=true, use_threads=true);
        χd, γd = calc_χγ(:d, lDGAhelper, bubble; ω_symmetric=true, use_threads=true);
    end
    if use_cache && !restored_run
        cache_file = joinpath(cfg_dir, cache_name)
        run_nk = lDGAhelper.kG.Ns
        println("[$run_id]: Trying to store χ_r and γ_r in $cache_file.");
        faulty_cache = false
        if isfile(cache_file) 
            jldopen(cache_file, "r") do cache_f
                if !haskey(cache_f, "$run_id")
                    println("[$run_id]: WARNING! run_id found in $cache_file but restored run = $restored_run. This indicates a faulty previous run.!"); flush(stdout)
                    faulty_cache = true
                end
                if haskey(cache_f, "$run_id") && cache_f["$run_id/Nk"] != lDGAhelper.kG.Ns
                    println("[$run_id]: WARNING! run_id found in $cache_file but number of k-points do not match $(cache_f["$run_id/Nk"]) != $(lDGAhelper.kG.Ns)!"); flush(stdout)
                    faulty_cache = true
                end
            end
            # Removing groupds is not supported by JLD2, creating new fiile
            if faulty_cache
                bak_file = joinpath(cfg_dir, "bak_$run_id"*cache_name)
                mv(cache_file, bak_file, force=true)
                if isfile(bak_file)
                    jldopen(cache_file, "w") do cache_f
                        jldopen(bak_file, "r") do bak_cache_f
                            for ki in keys(bak_cache_f)
                                if ki != run_id
                                    for kj in keys(bak_cache_f["$ki"])
                                        cache_f["$ki/$kj"] = bak_cache_f["$ki/$kj"]
                                    end
                                end
                            end
                        end
                    end
                else
                    error("[$run_id]: ERROR! Could not create backup file for cache rebuild!")
                end
                rm(bak_file)
            end
        end

        jldopen(cache_file, "a+") do cache_f
            cache_f["$run_id/Nk"] = run_nk
            cache_f["$run_id/config"] = cfg_content
            cache_f["$run_id/chi_m"] = χm
            cache_f["$run_id/chi_d"] = χd
            cache_f["$run_id/gamma_m"] = γm
            cache_f["$run_id/gamma_d"] = γd
            println("[$run_id]: Saved data."); flush(stdout)
        end
    end

    println("[$run_id]: Fixing χ"); 
    LadderDGA.log_q0_χ_check(lDGAhelper.kG, lDGAhelper.sP, χm, :m; verbose=true)
    LadderDGA.log_q0_χ_check(lDGAhelper.kG, lDGAhelper.sP, χd, :d; verbose=true)
    fix_χr!(χm; negative_eps = 1e-2)
    fix_χr!(χd; negative_eps = 1e-2)
    LadderDGA.log_q0_χ_check(lDGAhelper.kG, lDGAhelper.sP, χm, :m; verbose=true)
    LadderDGA.log_q0_χ_check(lDGAhelper.kG, lDGAhelper.sP, χd, :d; verbose=true)
    println("[$run_id]: Done fixing χ"); flush(stdout)

    δ_name = (tc <: ΣTail_ExpStep) ? string(round(tc.parameters[1],digits=2)) : ""
    tc_name = string(tc.name.name)
    Epot_DMFT_p2 = EPot_p2(χm, χd, 0.0, 0.0, lDGAhelper.mP.n, lDGAhelper.mP.U, lDGAhelper.kG)
    Ekin_DMFT_p2 = χm.tail_c[3]

    if !m_done
        try
            res_m = λm_correction(χm, γm, χd, γd, λ₀, lDGAhelper, tc=tc)
            err_m = nothing
        catch e
            res_m = nothing
            err_m = e
        end;
    end
    println("[$run_id]: lDΓA_m result = $(res_m). Storing in $fname."); flush(stdout)
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
            res_dm = λdm_correction(χm, γm, χd, γd, λ₀, lDGAhelper; λd_δ=1e-1, tc=tc, λd_max=200.0)
            err_dm = nothing
        catch e 
            res_dm = nothing
            err_dm = e
        end;
    end
    println("[$run_id]: lDΓA_dm result = $(res_dm). Storing in $fname."); flush(stdout)
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
            res_dm_sc = λdm_sc_correction(χm, γm, χd, γd, λ₀, lDGAhelper; max_steps_sc=150, max_steps_dm=200, λd_δ=1e-1, tc=tc, λd_max=200.0)
            err_dm_sc = nothing
        catch e 
            res_dm_sc = nothing
            err_dm_sc = e
        end;
    end
    println("[$run_id]: lDΓA_dm_sc result = $(res_dm_sc). Storing in $fname."); flush(stdout)
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
            res_m_sc = nothing #λm_sc_correction(χm, γm, χd, γd, λ₀, lDGAhelper; λd_δ=1e-1, tc=tc)
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

# Set to Logging.Warn or Info for more verbose output
logger = ConsoleLogger(stdout, Logging.Error)
with_logger(logger) do
    run(ARGS, tc)
end
