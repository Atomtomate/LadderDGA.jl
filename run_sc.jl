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


@info "Reading Input"
@timeit LadderDGA.to "input" wp, mP, sP, env, kGridsStr = readConfig(cfg_file);
@info "Setup"
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
@timeit LadderDGA.to "c2 sc" c2_res_sc = residuals(6, 6, Float64[], χ_sp, γ_sp, χ_ch, γ_ch, Σ_loc, gLoc_rfft, λ₀, kG, mP, sP, conv_abs=1e-7, maxit=100)
@info "c2 curve"
@timeit LadderDGA.to "c2" c2_res = residuals(6, 6, Float64[], χ_sp, γ_sp, χ_ch, γ_ch, Σ_loc, gLoc_rfft, λ₀, kG, mP, sP; maxit=0)

λspch_sc = find_root(c2_res_sc)
λspch = find_root(c2_res)

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
end

println(LadderDGA.to)
