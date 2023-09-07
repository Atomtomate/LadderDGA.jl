using TimerOutputs
using Pkg
using JLD2
Pkg.activate(@__DIR__)
using LadderDGA


cfg_file = ARGS[1]



wp, mP, sP, env, kGridsStr = readConfig(cfg_file);
Σ_ladderLoc, Σ_loc, imp_density, kG, gLoc, gLoc_fft, gLoc_rfft, Γsp, Γch, χDMFTsp, χDMFTch, χ_m_loc, γ_m_loc, χ_d_loc, γ_d_loc, χ₀Loc, gImp = setup_LDGA(kGridsStr[1], mP, sP, env);
bubble = calc_bubble(gLoc_fft, gLoc_rfft, kG, mP, sP);
χ_m, γ_m = calc_χγ(:sp, Γsp, bubble, kG, mP, sP);
χ_d, γ_d = calc_χγ(:ch, Γch, bubble, kG, mP, sP);

Fsp = F_from_χ(χDMFTsp, gImp[1,:], sP, mP.β);
λ₀ = calc_λ0(bubble, Fsp, χ_m_loc, γ_m_loc, mP, sP)

λm = LadderDGA.λ_correction(:sp, imp_density, χ_m, γ_m, χ_d, γ_d, gLoc_rfft, λ₀, kG, mP, sP)

include(joinpath(@__DIR__,"scripts/chi0t.jl"))
_, νGrid, _ = LadderDGA.LambdaCorrection.gen_νω_indices(χ_m, χ_d, mP, sP)
# ========================================== DMFT ==========================================
χ0_inv_dmft = χ0_inv(gLoc, kG, mP, sP)
χ0_inv_dmft_0 = χ0_inv_dmft[qi_0, ωi]
χ0_inv_dmft_π = χ0_inv_dmft[qi_π, ωi]
# ========================================= lDΓA_m =========================================
Σ_ladder_m = calc_Σ(χ_m, γ_m, χ_d, γ_d, λ₀, gLoc_rfft, kG, mP, sP, λm = λm);
χ0_inv_m_0, χ0_inv_m_π, E_kin_m, E_pot_m, μ_m, converged_m = if isfinite(λm)
    μnew, gLoc_m = G_from_Σladder(Σ_ladder_m, Σ_loc, kG, mP, sP; fix_n=true)
    χ0_inv_m = χ0_inv(gLoc_m, kG, mP, sP)
    E_kin_m, E_pot_m = calc_E(gLoc_m[:,νGrid].parent, Σ_ladder_m.parent, kG, mP, νmax = last(νGrid)+1)
    χ0_inv_m[qi_0, ωi], χ0_inv_m[qi_π, ωi], E_kin_m, E_pot_m, μnew, true
else
    @warn "No finite λm found!"
    NaN, NaN, NaN, NaN, NaN, false 
end

# ======================================= lDΓA_dm ==========================================

@timeit LadderDGA.to "λdm" Σ_ladder_dm, gLoc_dm, E_kin_dm, E_pot_dm, μ_dm, λdm_m, _, _, converged_dm, λdm_d  = λdm_correction(χ_m, γ_m, χ_d, γ_d, Σ_loc, gLoc_rfft, λ₀, kG, mP, sP; maxit=0, par=false)
λdm = [λdm_m, λdm_d]
χ0_inv_dm_0, χ0_inv_dm_π = if all(isfinite.(λdm))
    χ0_inv_dm = χ0_inv(gLoc_dm, kG, mP, sP)
    χ0_inv_dm[qi_0, ωi], χ0_inv_dm[qi_π, ωi]
else
    @warn "No finite λdm found!"
    NaN, NaN
end

# ======================================= lDΓA_m_sc ========================================
Σ_ladder_m_sc, χ0_inv_m_sc_0, χ0_inv_m_sc_π, E_kin_m_sc, E_pot_m_sc, μ_m_sc, λm_sc, converged_m_sc = if isfinite(λm)
    Σ_ladder_m_sc, gLoc_m_sc, E_kin_m_sc, E_pot_m_sc, μ_m_sc, λm_sc, _, _, converged_m_sc  = run_sc(χ_m, γ_m, χ_d, γ_d, λ₀, gLoc_rfft, Σ_loc, 0.0, kG, mP, sP)
    χ0_inv_m_sc = χ0_inv(gLoc_m_sc, kG, mP, sP)
    Σ_ladder_m_sc, χ0_inv_m_sc[qi_0, ωi], χ0_inv_m_sc[qi_π, ωi], E_kin_m_sc, E_pot_m_sc, μ_m_sc, λm_sc, converged_m_sc
else
    @warn "No finite λdm_sc found!"
    nothing, NaN, NaN, NaN, NaN, NaN, NaN, false 
end

# ===================================== lDΓA_dm_sc =========================================
@timeit LadderDGA.to "λdm sc" Σ_ladder_dm_sc, gLoc_dm_sc, E_kin_dm_sc, E_pot_dm_sc, μ_dm_sc, λdm_sc_m, _, _, converged_dm_sc, λdm_sc_d  = λdm_correction(χ_m, γ_m, χ_d, γ_d, Σ_loc, gLoc_rfft, λ₀, kG, mP, sP; maxit=100, par=false)
λdm_sc = [λdm_sc_m, λdm_sc_d]
χ0_inv_dm_sc_0, χ0_inv_dm_sc_π = if all(isfinite.(λdm_sc))
    χ0_inv_dm_sc = χ0_inv(gLoc_dm_sc, kG, mP, sP)
    χ0_inv_dm_sc[qi_0, ωi], χ0_inv_dm_sc[qi_π, ωi]
else
    @warn "No finite λdm_sc found!"
    NaN, NaN
end

# ======================================= lDΓA_m_tsc ========================================
println("m_tsc:")
Σ_ladder_m_tsc, χ0_inv_m_tsc_0, χ0_inv_m_tsc_π, E_kin_m_tsc, E_pot_m_tsc, μ_m_tsc, λm_tsc, converged_m_tsc = if isfinite(λm)
    Σ_ladder_m_tsc, gLoc_m_tsc, E_kin_m_tsc, E_pot_m_tsc, μ_m_tsc, λm_tsc, _, _, converged_m_tsc  = run_sc(χ_m, γ_m, χ_d, γ_d, λ₀, gLoc_rfft, Σ_loc, 0.0, kG, mP, sP, update_χ_tail=true)
    χ0_inv_m_tsc = χ0_inv(gLoc_m_tsc, kG, mP, sP)
    Σ_ladder_m_tsc, χ0_inv_m_tsc[qi_0, ωi], χ0_inv_m_tsc[qi_π, ωi], E_kin_m_tsc, E_pot_m_tsc, μ_m_tsc, λm_tsc, converged_m_tsc
else
    @warn "No finite λdm_sc found!"
    nothing, NaN, NaN, NaN, NaN, NaN, NaN, false 
end


# ===================================== lDΓA_dm_tsc ========================================
println("dm_tsc:")
@timeit LadderDGA.to "λdm tsc" Σ_ladder_dm_tsc, gLoc_dm_tsc, E_kin_dm_tsc, E_pot_dm_tsc, μ_dm_tsc, λdm_tsc_m, _, _, converged_dm_tsc, λdm_tsc_d  = λdm_correction(χ_m, γ_m, χ_d, γ_d, Σ_loc, gLoc_rfft, λ₀, kG, mP, sP; update_χ_tail=true, maxit=100, par=false)
λdm_tsc = [λdm_tsc_m, λdm_tsc_d]
χ0_inv_dm_tsc_0, χ0_inv_dm_tsc_π = if all(isfinite.(λdm_tsc))
    χ0_inv_dm_tsc = χ0_inv(gLoc_dm_tsc, kG, mP, sP)
    χ0_inv_dm_tsc[qi_0, ωi], χ0_inv_dm_tsc[qi_π, ωi]
else
    @warn "No finite λdm_tsc found!"
    NaN, NaN
end
