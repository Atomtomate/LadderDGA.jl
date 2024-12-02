using LadderDGA
using JLD2

tc_choice_i = parse(Int, ARGS[3])
tc = if tc_choice_i == 1
    tc=ΣTail_EoM
elseif tc_choice_i == 2
    tc=ΣTail_ExpStep{0.8}
elseif tc_choice_i == 3
    tc=ΣTail_Full
else
    error("Unrecognized tc choice $tc_choice_i")
end

function check_done(fname, key)
    check= if isfile(fname) 
        jldopen(fname, "r") do f
            if !isnothing(f["res_dm"])
                abs(f["res_dm"].EPot_p1 .- f["res_dm"].EPot_p2) < f["res_dm"].eps_abs
            else
                false
            end
        end
    else
        false
    end
    return check
end

function run(ARGS, tc)
    cfg_file = ARGS[1]
    fname = ARGS[2]
    wp, mP, sP, env, kGridsStr = readConfig(cfg_file);
    lDGAhelper = setup_LDGA(kGridsStr[1], mP, sP, env, silent=false);
    bubble     = calc_bubble(:DMFT, lDGAhelper);
    χm, γm = calc_χγ(:m, lDGAhelper, bubble);
    χd, γd = calc_χγ(:d, lDGAhelper, bubble);
    λ₀ = calc_λ0(bubble, lDGAhelper);
    res_m = nothing 
    err_m = nothing
    res_dm = nothing 
    err_dm = nothing
    res_m_sc = nothing 
    err_m_sc = nothing
    res_dm_sc = nothing 
    err_dm_sc = nothing

    res_m, err_m = try
        λm_correction(χm, γm, χd, γd, λ₀, lDGAhelper, tc=tc), nothing
    catch e
        nothing, e
    end;
    jldopen(fname, "w") do f
        f["Nk"] = lDGAhelper.kG.Ns
        f["Sigma_tc"] = "EoM"
        f["KGrid"] =kGridsStr[1] 
        f["U"] = lDGAhelper.mP.U
        f["μ_DMFT"] = lDGAhelper.mP.μ
        f["β"] = lDGAhelper.mP.β
        f["n"] = lDGAhelper.mP.n
        f["Epot_1Pt"] = lDGAhelper.mP.Epot_1Pt
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
    if !check_done(fname, "res_dm")
    res_dm, err_dm = try 
        λdm_correction(χm, γm, χd, γd, λ₀, lDGAhelper; λd_δ=1e-2, tc=tc), nothing
    catch e 
        nothing, e
    end;
    else
        jldopen(fname, "w") do f
            f["res_dm"], f["err_dm"]
        end
    end
    res_m, err_m = jldopen(fname,"r") do f
        f["res_m"], f["err_m"]
    end
    jldopen(fname, "w") do f
        f["Nk"] = lDGAhelper.kG.Ns
        f["Sigma_tc"] = "EoM"
        f["KGrid"] =kGridsStr[1] 
        f["U"] = lDGAhelper.mP.U
        f["μ_DMFT"] = lDGAhelper.mP.μ
        f["β"] = lDGAhelper.mP.β
        f["n"] = lDGAhelper.mP.n
        f["Epot_1Pt"] = lDGAhelper.mP.Epot_1Pt
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
    if !check_done(fname, "res_dm_sc")
    res_dm_sc, err_dm_sc = try
        λdm_sc_correction(χm, γm, χd, γd, λ₀, lDGAhelper; max_steps_sc=250, max_steps_dm=200, validation_threshold=1e-7, λd_δ=1e-2, tc=tc), nothing
    catch e 
        nothing, e
    end;
    else
        jldopen(fname, "w") do f
            f["res_dm_sc"], f["err_dm_sc"]
        end
    end
    res_m, err_m, res_dm, err_dm  = jldopen(fname,"r") do f
        f["res_m"], f["err_m"], f["res_dm"], f["err_dm"]
    end
    jldopen(fname, "w") do f
        f["Nk"] = lDGAhelper.kG.Ns
        f["Sigma_tc"] = "EoM"
        f["KGrid"] =kGridsStr[1] 
        f["U"] = lDGAhelper.mP.U
        f["μ_DMFT"] = lDGAhelper.mP.μ
        f["β"] = lDGAhelper.mP.β
        f["n"] = lDGAhelper.mP.n
        f["Epot_1Pt"] = lDGAhelper.mP.Epot_1Pt
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
    if !check_done(fname, "res_m_sc")
    res_m_sc, err_m_sc = try
        λm_sc_correction(χm, γm, χd, γd, λ₀, lDGAhelper; λd_δ=1e-2, tc=tc), nothing
    catch e 
        nothing, e
    end;
    else
        jldopen(fname, "w") do f
            f["res_m_sc"], f["err_m_sc"]
        end
    end
    res_m, err_m, res_dm, err_dm, res_dm_sc, err_dm_sc  = jldopen(fname,"r") do f
        f["res_m"], f["err_m"], f["res_dm"], f["err_dm"], f["res_dm_sc"], f["err_dm_sc"]
    end
    jldopen(fname, "w") do f
        f["Nk"] = lDGAhelper.kG.Ns
        f["Sigma_tc"] = "EoM"
        f["KGrid"] =kGridsStr[1] 
        f["U"] = lDGAhelper.mP.U
        f["μ_DMFT"] = lDGAhelper.mP.μ
        f["β"] = lDGAhelper.mP.β
        f["n"] = lDGAhelper.mP.n
        f["Epot_1Pt"] = lDGAhelper.mP.Epot_1Pt
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
