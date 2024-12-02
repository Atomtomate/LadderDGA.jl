using LadderDGA
using JLD2

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


function run(ARGS)
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
    tc=ΣTail_Full

    try
        res_m = λm_correction(χm, γm, χd, γd, λ₀, lDGAhelper, tc=tc)
    catch e
        err_m = e
    end;
    if !check_done(fname, "res_dm")
    try 
        res_dm = λdm_correction(χm, γm, χd, γd, λ₀, lDGAhelper; λd_δ=1e-2, tc=tc)
        res_m, err_m = jldopen(fname,"r") do f
            f["res_m"], f["err_m"]
        end
    catch e 
        err_dm = nothing
    end;
    end
    if !check_done(fname, "res_dm_sc")
    try
    res_dm_sc = λdm_sc_correction(χm, γm, χd, γd, λ₀, lDGAhelper; max_steps_sc=250, max_steps_dm=200, validation_threshold=1e-7, λd_δ=1e-2, tc=tc)
        res_m, err_m, res_dm, err_dm = jldopen(fname,"r") do f
            f["res_m"], f["err_m"], f["res_dm"], f["err_dm"]
        end
    catch e 
        err_dm_sc = nothing
    end;
    end
    if !check_done(fname, "res_m_sc")
    try
        res_m_sc = λm_sc_correction(χm, γm, χd, γd, λ₀, lDGAhelper; λd_δ=1e-2, tc=tc)
        res_m, err_m, res_dm, err_dm, res_dm_sc, err_dm_sc  = jldopen(fname,"r") do f
            f["res_m"], f["err_m"], f["res_dm"], f["err_dm"], f["res_dm_sc"], f["err_dm_sc"]
        end
    catch e 
        err_m_sc = nothing
    end;
    end


end

run(ARGS)
