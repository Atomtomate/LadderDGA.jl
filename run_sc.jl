using TimerOutputs
using Pkg
using Distributed
using JLD2
Pkg.activate(@__DIR__)


cfg_file = ARGS[1]
out_path = ARGS[2]
nprocs_in   = parse(Int,ARGS[3]) # TODO: use slurm
fname_out =  out_path*"/lDGA_c2.jld2" 

nprocs() == 1 && addprocs(nprocs_in, exeflags="--project=$(Base.active_project())")
@everywhere using LadderDGA


@timeit LadderDGA.to "input" wp, mP, sP, env, kGridsStr = readConfig(cfg_file);
@timeit LadderDGA.to "setup" Σ_ladderLoc, Σ_loc, imp_density, kG, gLoc, gLoc_fft, gLoc_rfft, Γsp, Γch, χDMFTsp, χDMFTch, χ_m_loc, γ_m_loc, χ_d_loc, γ_d_loc, χ₀Loc, gImp = setup_LDGA(kGridsStr[1], mP, sP, env);
@info "Bubble"
@timeit LadderDGA.to "bubble" bubble = calc_bubble_par(kG, mP, sP, collect_data=true);
@info "BSE sp"
@timeit LadderDGA.to "BSE sp" χ_m, γ_m = calc_χγ_par(:sp, Γsp, kG, mP, sP, collect_data=true);
@info "BSE ch"
@timeit LadderDGA.to "BSE ch" χ_d, γ_d = calc_χγ_par(:ch, Γch, kG, mP, sP, collect_data=true);

@info "EoM Correction"
@timeit LadderDGA.to "EoM Correction" begin
    Fsp = F_from_χ(χDMFTsp, gImp[1,:], sP, mP.β);
    λ₀ = calc_λ0(bubble, Fsp, χ_m_loc, γ_m_loc, mP, sP)
end

@info "λsp"
@timeit LadderDGA.to "λsp" λm = LadderDGA.λ_correction(:sp, imp_density, χ_m, γ_m, χ_d, γ_d, gLoc_rfft, λ₀, kG, mP, sP)
@info "c2 curve sc"
@timeit LadderDGA.to "c2 sc" c2_res_sc = residuals(3, 3, Float64[], χ_m, γ_m, χ_d, γ_d, Σ_loc, gLoc_rfft, λ₀, kG, mP, sP, conv_abs=1e-6, maxit=100, par=true)
# @info "c2 curve sc"
# @timeit LadderDGA.to "c2 sc" c2_res_sc_seq = residuals(3, 3, Float64[], χ_m, γ_m, χ_d, γ_d, Σ_loc, gLoc_rfft, λ₀, kG, mP, sP, conv_abs=1e-6, maxit=100, par=false)
@info "c2 curve sc tail"
@timeit LadderDGA.to "c2 sc" c2_res_tsc = residuals(3, 3, Float64[], χ_m, γ_m, χ_d, γ_d, Σ_loc, gLoc_rfft, λ₀, kG, mP, sP, conv_abs=1e-6, maxit=100, par=true, update_χ_tail=true)
@info "c2 curve"
@timeit LadderDGA.to "c2" c2_res = residuals(3, 3, Float64[], χ_m, γ_m, χ_d, γ_d, Σ_loc, gLoc_rfft, λ₀, kG, mP, sP; maxit=0, par=false)

@info "calc Σ"
ind_tsc = findall(x->x≈0, c2_res_tsc[2,:])
ind_sc = findall(x->x≈0, c2_res_sc[2,:])
λdm = find_root(c2_res)
λdm_sc = find_root(c2_res_sc)
λdm_tsc = find_root(c2_res_tsc)
λm_sc = c2_res_sc[1,ind_sc][1]
λm_tsc = c2_res_tsc[1,ind_tsc][1]

Σ_ladder = calc_Σ(χ_m, γ_m, χ_d, γ_d, λ₀, gLoc_rfft, kG, mP, sP);
Σ_ladder_m = calc_Σ(χ_m, γ_m, χ_d, γ_d, λ₀, gLoc_rfft, kG, mP, sP, λsp = λm);
Σ_ladder_dm = if all(isfinite.(λdm))
    calc_Σ(χ_m, γ_m, χ_d, γ_d, λ₀, gLoc_rfft, kG, mP, sP, λsp = λdm[1], λch = λdm[2]);
else
    @warn "No finite λdm found!"
    Σ_ladder
end

@info "bubble with form factor"
include(joinpath(@__DIR__,"scripts/chi0t.jl"))
_, νGrid, _ = LadderDGA.LambdaCorrection.gen_νω_indices(χ_m, χ_d, mP, sP)
# ========================================== DMFT ==========================================
χ0_inv_dmft = χ0_inv(gLoc, kG, mP, sP)
χ0_inv_dmft_0 = χ0_inv_dmft[qi_0, ωi]
χ0_inv_dmft_π = χ0_inv_dmft[qi_π, ωi]
# ========================================= lDΓA_m =========================================
Σ_ladder_m, χ0_inv_m_0, χ0_inv_m_π, E_kin_m, E_pot_m, μ_m, converged_m = if isfinite(λm)
    μnew, gLoc_m = G_from_Σladder(Σ_ladder_m, Σ_loc, kG, mP, sP; fix_n=true)
    χ0_inv_m = χ0_inv(gLoc_m, kG, mP, sP)
    E_kin_m, E_pot_m = calc_E(gLoc_m[:,νGrid].parent, Σ_ladder.parent, kG, mP, νmax = last(νGrid)+1)
    Σ_ladder_m, χ0_inv_m[qi_0, ωi], χ0_inv_m[qi_π, ωi], E_kin_m, E_pot_m, μnew, true
else
    @warn "No finite λm found!"
    nothing, NaN, NaN, NaN, NaN, NaN, false 
end

# ======================================= lDΓA_dm ==========================================
Σ_ladder_dm, χ0_inv_dm_0, χ0_inv_dm_π, E_kin_dm, E_pot_dm, μ_dm, converged_dm = if all(isfinite.(λdm))
    μnew, gLoc_dm = G_from_Σladder(Σ_ladder_dm, Σ_loc, kG, mP, sP; fix_n=true)
    χ0_inv_dm = χ0_inv(gLoc_dm, kG, mP, sP)
    E_kin_dm, E_pot_dm = calc_E(gLoc_dm[:,νGrid].parent, Σ_ladder.parent, kG, mP, νmax = last(νGrid)+1)
    Σ_ladder_dm, χ0_inv_dm[qi_0, ωi], χ0_inv_dm[qi_π, ωi], E_kin_dm, E_pot_dm, μnew, true
else
    @warn "No finite λdm found!"
    nothing, NaN, NaN, NaN, NaN, NaN, false 
end

# ======================================= lDΓA_m_sc ========================================
Σ_ladder_m_sc, χ0_inv_m_sc_0, χ0_inv_m_sc_π, E_kin_m_sc, E_pot_m_sc, μ_m_sc, converged_m_sc = if isfinite(λm)
    Σ_ladder_m_sc, gLoc_m_sc, E_kin_m_sc, E_pot_m_sc, μ_m_sc, converged_m  = run_sc(χ_m, γ_m, χ_d, γ_d, λ₀, gLoc_rfft, Σ_loc, 0.0, kG, mP, sP)
    χ0_inv_m_sc = χ0_inv(gLoc_m_sc, kG, mP, sP)
    Σ_ladder_m_sc, χ0_inv_m_sc[qi_0, ωi], χ0_inv_m_sc[qi_π, ωi], E_kin_m_sc, E_pot_m_sc, μ_m_sc, converged_m
else
    @warn "No finite λdm_sc found!"
    nothing, NaN, NaN, NaN, NaN, NaN, false 
end

# ===================================== lDΓA_dm_sc =========================================
Σ_ladder_dm_sc, χ0_inv_dm_sc_0, χ0_inv_dm_sc_π, E_kin_dm_sc, E_pot_dm_sc, μ_dm_sc, converged_dm_sc = if all(isfinite.(λdm_sc))
    Σ_ladder_dm_sc, gLoc_dm_sc, E_kin_dm_sc, E_pot_dm_sc, μ_dm_sc, converged_dm  = run_sc(χ_m, γ_m, χ_d, γ_d, λ₀, gLoc_rfft, Σ_loc, λdm_sc[2], kG, mP, sP)
    χ0_inv_dm_sc = χ0_inv(gLoc_dm_sc, kG, mP, sP)
    Σ_ladder_dm_sc, χ0_inv_dm_sc[qi_0, ωi], χ0_inv_dm_sc[qi_π, ωi], E_kin_dm_sc, E_pot_dm_sc, μ_dm_sc, converged_dm
else
    @warn "No finite λdm_sc found!"
    nothing, NaN, NaN, NaN, NaN, NaN, false 
end

# ======================================= lDΓA_m_tsc ========================================
Σ_ladder_m_tsc, χ0_inv_m_tsc_0, χ0_inv_m_tsc_π, E_kin_m_tsc, E_pot_m_tsc, μ_m_tsc, converged_m_tsc = if isfinite(λm)
    Σ_ladder_m_tsc, gLoc_m_tsc, E_kin_m_tsc, E_pot_m_tsc, μ_m_tsc, converged_m  = run_sc(χ_m, γ_m, χ_d, γ_d, λ₀, gLoc_rfft, Σ_loc, 0.0, kG, mP, sP, update_χ_tail=true)
    χ0_inv_m_tsc = χ0_inv(gLoc_m_tsc, kG, mP, sP)
    Σ_ladder_m_tsc, χ0_inv_m_tsc[qi_0, ωi], χ0_inv_m_tsc[qi_π, ωi], E_kin_m_tsc, E_pot_m_tsc, μ_m_tsc, converged_m
else
    @warn "No finite λdm_tsc found!"
    nothing, NaN, NaN, NaN, NaN, NaN, false 
end


# ===================================== lDΓA_dm_tsc ========================================
Σ_ladder_dm_tsc, χ0_inv_dm_tsc_0, χ0_inv_dm_tsc_π, E_kin_dm_tsc, E_pot_dm_tsc, μ_dm_tsc, converged_dm_tsc = if all(isfinite.(λdm_tsc))
    Σ_ladder_dm_tsc, gLoc_dm_tsc, E_kin_dm_tsc, E_pot_dm_tsc, μ_dm_tsc, converged_dm  = run_sc(χ_m, γ_m, χ_d, γ_d, λ₀, gLoc_rfft, Σ_loc, λdm_tsc[2], kG, mP, sP, update_χ_tail=true)
    χ0_inv_dm_tsc = χ0_inv(gLoc_dm_tsc, kG, mP, sP)
    Σ_ladder_dm_tsc, χ0_inv_dm_tsc[qi_0, ωi], χ0_inv_dm_tsc[qi_π, ωi], E_kin_dm_tsc, E_pot_dm_tsc, μ_dm_tsc, converged_dm
else
    @warn "No finite λdm_tsc found!"
    nothing, NaN, NaN, NaN, NaN, NaN, false 
end

@info "Bubble after sc"

function lin_fit(ν, Σ)
    m = (Σ[2] - Σ[1])/(ν[2] - ν[1])
    return Σ[1] - m * ν[1]
end

function get_ef(Σ_ladder)
    νGrid = [1im * (2*n+1)*π/mP.β for n in 0:1];
    s_r0 = [lin_fit(imag(νGrid), real.(Σ_ladder[i,0:2])) for i in 1:size(Σ_ladder,1)];
    Σ0_full = LadderDGA.expandKArr(kG, s_r0);
    ekf = LadderDGA.expandKArr(kG,mP.μ .- kG.ϵkGrid)
    ek_diff = ekf .- Σ0_full
    min_diff = minimum(abs.(ekf .- Σ0_full))
    return ef_ind = abs.(ek_diff) .< kG.Ns*min_diff
end

@info "Output"
@timeit LadderDGA.to "write" jldopen(fname_out, "w") do f
    cfg_string = read(cfg_file, String)
    f["config"] =  cfg_string
    f["sP"] = sP
    f["mP"] = mP
    f["χsp"] = χ_m
    f["χch"] = χ_d

    f["λm"] = λm
    f["λdm"] = λdm
    f["λm_sc"] = λm_sc
    f["λdm_sc"] = λdm_sc
    f["λm_tsc"] = λm_tsc
    f["λdm_tsc"] = λdm_tsc

    f["Σ_loc"] = Σ_loc
    f["Σ_ladder"] = Σ_ladder
    f["Σ_ladder_m"] = Σ_ladder_m
    f["Σ_ladder_dm"] = Σ_ladder_dm
    f["Σ_ladder_m_sc"] = Σ_ladder_m_sc
    f["Σ_ladder_dm_sc"] = Σ_ladder_dm_sc
    f["Σ_ladder_m_tsc"] = Σ_ladder_m_tsc
    f["Σ_ladder_dm_tsc"] = Σ_ladder_dm_tsc

    f["μ_dmft"] = mP.μ
    f["μ_m"] = μ_m
    f["μ_m_sc"] = μ_m_sc
    f["μ_dm"] = μ_dm
    f["μ_dm_sc"] = μ_dm_sc
    f["μ_m_tsc"] = μ_m_tsc
    f["μ_dm_tsc"] = μ_dm_tsc

    f["E_kin_DMFT"] = mP.Ekin_DMFT
    f["E_kin_m"] = E_kin_m
    f["E_kin_m_sc"] = E_kin_m_sc
    f["E_kin_dm"] = E_kin_dm
    f["E_kin_dm_sc"] = E_kin_dm_sc
    f["E_kin_m_tsc"] = E_kin_m_tsc
    f["E_kin_dm_tsc"] = E_kin_dm_tsc

    f["E_pot_DMFT"] = mP.Epot_DMFT
    f["E_pot_m"] = E_pot_m
    f["E_pot_m_sc"] = E_pot_m_sc
    f["E_pot_dm"] = E_pot_dm
    f["E_pot_dm_sc"] = E_pot_dm_sc
    f["E_pot_m_tsc"] = E_pot_m_tsc
    f["E_pot_dm_tsc"] = E_pot_dm_tsc

    f["χ0_inv_DMFT_0"] = χ0_inv_dmft_0
    f["χ0_inv_m_0"] = χ0_inv_m_0
    f["χ0_inv_m_sc_0"] = χ0_inv_m_sc_0
    f["χ0_inv_dm_0"] = χ0_inv_dm_0
    f["χ0_inv_dm_sc_0"] = χ0_inv_dm_sc_0
    f["χ0_inv_m_tsc_0"] = χ0_inv_m_tsc_0
    f["χ0_inv_dm_tsc_0"] = χ0_inv_dm_tsc_0

    f["χ0_inv_DMFT_π"] = χ0_inv_dmft_π
    f["χ0_inv_m_π"] = χ0_inv_m_π
    f["χ0_inv_m_sc_π"] = χ0_inv_m_sc_π
    f["χ0_inv_dm_π"] = χ0_inv_dm_π
    f["χ0_inv_dm_sc_π"] = χ0_inv_dm_sc_π
    f["χ0_inv_m_tsc_π"] = χ0_inv_m_tsc_π
    f["χ0_inv_dm_tsc_π"] = χ0_inv_dm_tsc_π

    f["converged_m"] = converged_m
    f["converged_m_sc"] = converged_m_sc
    f["converged_dm"] = converged_dm
    f["converged_dm_sc"] = converged_dm_sc
    f["converged_m_tsc"] = converged_m_tsc
    f["converged_dm_tsc"] = converged_dm_tsc

    f["ef_m"] = Σ_ladder_m !== nothing ? get_ef(Σ_ladder_m) : nothing
    f["ef_dm"] = Σ_ladder_dm !== nothing ? get_ef(Σ_ladder_dm) : nothing
    f["ef_dm_sc"] = Σ_ladder_dm_sc !== nothing ? get_ef(Σ_ladder_dm_sc) : nothing
    f["ef_dm_tsc"] = Σ_ladder_dm_tsc !== nothing ? get_ef(Σ_ladder_dm_tsc) : nothing
end

println(LadderDGA.to)
