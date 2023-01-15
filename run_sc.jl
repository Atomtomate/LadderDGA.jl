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
@timeit LadderDGA.to "setup" Σ_ladderLoc, Σ_loc, imp_density, kG, gLoc_fft, gLoc_rfft, Γsp, Γch, χDMFTsp, χDMFTch, χ_sp_loc, γ_sp_loc, χ_ch_loc, γ_ch_loc, χ₀Loc, gImp = setup_LDGA(kGridsStr[1], mP, sP, env);
@info "Bubble"
@timeit LadderDGA.to "bubble" bubble = calc_bubble_par(kG, mP, sP, collect_data=true);
@info "BSE sp"
@timeit LadderDGA.to "BSE sp" χ_sp, γ_sp = calc_χγ_par(:sp, Γsp, kG, mP, sP, collect_data=true);
@info "BSE ch"
@timeit LadderDGA.to "BSE ch" χ_ch, γ_ch = calc_χγ_par(:ch, Γch, kG, mP, sP, collect_data=true);

@info "EoM Correction"
@timeit LadderDGA.to "EoM Correction" begin
    Fsp = F_from_χ(χDMFTsp, gImp[1,:], sP, mP.β);
    λ₀ = calc_λ0(bubble, Fsp, χ_sp_loc, γ_sp_loc, mP, sP)
end

@info "λsp"
@timeit LadderDGA.to "λsp" λsp_old = LadderDGA.λ_correction(:sp, imp_density, χ_sp, γ_sp, χ_ch, γ_ch, gLoc_rfft, λ₀, kG, mP, sP)
@info "c2 curve sc"
@timeit LadderDGA.to "c2 sc" c2_res_sc = residuals(6, 6, Float64[], χ_sp, γ_sp, χ_ch, γ_ch, Σ_loc, gLoc_rfft, λ₀, kG, mP, sP, conv_abs=1e-6, maxit=100, par=true)
@info "c2 curve"
@timeit LadderDGA.to "c2" c2_res = residuals(6, 6, Float64[], χ_sp, γ_sp, χ_ch, γ_ch, Σ_loc, gLoc_rfft, λ₀, kG, mP, sP; maxit=0, par=false)

@info "calc Σ"
λspch_sc = find_root(c2_res_sc)
λspch = find_root(c2_res)
Σ_ladder = calc_Σ(χ_sp, γ_sp, χ_ch, γ_ch, λ₀, gLoc_rfft, kG, mP, sP);
Σ_ladder_m = calc_Σ(χ_sp, γ_sp, χ_ch, γ_ch, λ₀, gLoc_rfft, kG, mP, sP, λsp = λsp_old);
Σ_ladder_dm = calc_Σ(χ_sp, γ_sp, χ_ch, γ_ch, λ₀, gLoc_rfft, kG, mP, sP, λsp = λspch[1], λch = λspch[2]);

@info "Bubble after sc"
Σ_ladder_dm_sc, gLoc_sc, E_pot_sc, μsc, converged = run_sc(χ_sp, γ_sp, χ_ch, γ_ch, λ₀, gLoc_rfft, Σ_loc, λspch_sc[1], λspch_sc[2], kG, mP, sP)
gLoc_sc_fft, gLoc_sc_rfft = G_fft(gLoc_sc, kG, mP, sP)
bubble_sc = calc_bubble(gLoc_sc_fft, gLoc_sc_rfft, kG, mP, sP)

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

f_d(q) = cos(q[1]) - cos(q[2])
ff = (f_d.(kG.kGrid)).^2
χ0_i_ff = LadderDGA.kintegrate(kG, ff .* sum(1 ./ bubble.data[:,sP.n_iν_shell+1:end-sP.n_iν_shell,:], dims=(2,3))/mP.β^2, 1)[1,1,1]
χ0_i = LadderDGA.kintegrate(kG, sum(1 ./ bubble.data[:,sP.n_iν_shell+1:end-sP.n_iν_shell,:], dims=(2,3))/mP.β^2, 1)[1,1,1]

χ0_i_ff_sc = LadderDGA.kintegrate(kG, ff .* sum(1 ./ bubble_sc.data[:,sP.n_iν_shell+1:end-sP.n_iν_shell,:], dims=(2,3))/mP.β^2, 1)[1,1,1]
χ0_i_sc = LadderDGA.kintegrate(kG, sum(1 ./ bubble_sc.data[:,sP.n_iν_shell+1:end-sP.n_iν_shell,:], dims=(2,3))/mP.β^2, 1)[1,1,1]

@info "Output"
@timeit LadderDGA.to "write" jldopen(fname_out, "w") do f
    cfg_string = read(cfg_file, String)
    f["config"] =  cfg_string
    f["sP"] = sP
    f["mP"] = mP
    f["χsp"] = χ_sp
    f["χch"] = χ_ch
    f["λsp_old"] = λsp_old
    f["λspch"] = λspch
    f["λspch_sc"] = λspch_sc
    f["Σ_ladder"] = Σ_ladder
    f["Σ_ladder_m"] = Σ_ladder_m
    f["Σ_ladder_dm"] = Σ_ladder_dm
    f["Σ_ladder_dm_sc"] = Σ_ladder_dm_sc
    f["gLoc_sc"] = gLoc_sc
    f["E_pot_sc"] = E_pot_sc
    f["μsc"] = μsc
    f["sc_converged"] = converged
    f["ef_dmft"] = get_ef(Σ_ladder)
    f["ef_m"] = get_ef(Σ_ladder_m)
    f["ef_dm"] = get_ef(Σ_ladder_dm)
    f["ef_dm_sc"] = get_ef(Σ_ladder_dm_sc)
    f["χ0_i_ff_DMFT"] = χ0_i_ff
    f["χ0_i_DMFT"] = χ0_i
end

println(LadderDGA.to)
