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

function check_done(fname, key, dm_method)
    check= if isfile(fname) 
        jldopen(fname, "r") do f
            if !isnothing(f[key])
                ep = abs(f[key].EPot_p1 .- f[key].EPot_p2) < f[key].eps_abs
                pp = abs(f[key].PP_p1 .- f[key].PP_p2) < f[key].eps_abs
                dm_dmethod ? (ep && pp) : pp
            else
                false
            end
        end
    else
        false
    end
    return check
end

function run(ARGS, tc; use_cache::Bool = true, cache_name::String = "lDGA_cache.jld2")
    cfg_file = ARGS[1]
    fname = ARGS[2]
    cfg_dir,_ = splitdir(cfg_file)
    m_done = check_done(fname, "res_m", false)
    dm_done = check_done(fname, "res_dm", true)
    dmsc_done = check_done(fname, "res_dm_sc", true)
    msc_done = check_done(fname, "res_m_sc", false)
    if m_done && dm_done && dmsc_done && msc_done
        println("All calculations found for $fname, skipping!")
        return
    end

    wp, mP, sP, env, kGridsStr = readConfig(cfg_file);
    lDGAhelper = setup_LDGA(kGridsStr[1], mP, sP, env, silent=false);

    χm = nothing
    χd = nothing
    γm = nothing
    γd = nothing
    restored_run = false

    bubble = calc_bubble(:DMFT, lDGAhelper);
    λ₀ = calc_λ0(bubble, lDGAhelper)
    if use_cache
        cache_file = joinpath(cfg_dir, cache_name)
        run_id = hash(readlines(cfg_file))
        run_nk = lDGAhelper.kG.Ns
        if isfile(cache_file) 
            println("restoring from $cache_file")
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
        @time χm, γm = calc_χγ(:m, lDGAhelper, bubble);
        @time χd, γd = calc_χγ(:d, lDGAhelper, bubble);
    end
    if use_cache && !restored_run
        cache_file = joinpath(cfg_dir, cache_name)
        cfg_content = readlines(cfg_file)
        run_id = hash(cfg_content)
        run_nk = lDGAhelper.kG.Ns
        jldopen(cache_file, "a+") do cache_f
            cache_f["$run_id/Nk"] = run_nk
            cache_f["$run_id/config"] = cfg_content
            cache_f["$run_id/chi_m"] = χm
            cache_f["$run_id/chi_d"] = χd
            cache_f["$run_id/gamma_m"] = γm
            cache_f["$run_id/gamma_d"] = γd
        end
    end

    res_m = nothing 
    err_m = nothing
    res_dm = nothing 
    err_dm = nothing
    res_m_sc = nothing 
    err_m_sc = nothing
    res_dm_sc = nothing 
    err_dm_sc = nothing
    δ_name = (tc <: ΣTail_ExpStep) ? string(round(tc.parameters[1],digits=2)) : ""
    tc_name = string(tc.name.name)
    Epot_DMFT_p2 = EPot_p2(χm, χd, 0.0, 0.0, lDGAhelper.mP.n, lDGAhelper.mP.U, lDGAhelper.kG)
    Ekin_DMFT_p2 = χm.tail_c[3]

    res_m, err_m = try
        λm_correction(χm, γm, χd, γd, λ₀, lDGAhelper, tc=tc), nothing
    catch e
        nothing, e
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
    res_dm, err_dm = if !dm_done
        try 
            λdm_correction(χm, γm, χd, γd, λ₀, lDGAhelper; λd_δ=1e-1, tc=tc, λd_max=30.0), nothing
        catch e 
            nothing, e
        end;
    else
        jldopen(fname, "r") do f
            f["res_dm"], f["err_dm"]
        end
    end
    res_m, err_m = jldopen(fname,"r") do f
        f["res_m"], f["err_m"]
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
    res_dm_sc, err_dm_sc = if !dmsc_done
        try
            λdm_sc_correction(χm, γm, χd, γd, λ₀, lDGAhelper; max_steps_sc=250, max_steps_dm=200, validation_threshold=1e-7, λd_δ=1e-1, tc=tc, λd_max=30.0), nothing
        catch e 
            nothing, e
        end;
    else
        jldopen(fname, "r") do f
            f["res_dm_sc"], f["err_dm_sc"]
        end
    end
    res_m, err_m, res_dm, err_dm  = jldopen(fname,"r") do f
        f["res_m"], f["err_m"], f["res_dm"], f["err_dm"]
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
    res_m_sc, err_m_sc = if !msc_done
    try
        λm_sc_correction(χm, γm, χd, γd, λ₀, lDGAhelper; λd_δ=1e-1, tc=tc), nothing
    catch e 
        nothing, e
    end;
    else
        jldopen(fname, "r") do f
            f["res_m_sc"], f["err_m_sc"]
        end
    end
    res_m, err_m, res_dm, err_dm, res_dm_sc, err_dm_sc  = jldopen(fname,"r") do f
        f["res_m"], f["err_m"], f["res_dm"], f["err_dm"], f["res_dm_sc"], f["err_dm_sc"]
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
