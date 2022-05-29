using Base.GC
using TimerOutputs

using Pkg
Pkg.activate(@__DIR__)

using Distributed
@everywhere using Pkg
@everywhere Pkg.activate(@__DIR__)
using LadderDGA
@everywhere using LadderDGA

cfg_file_dir = (@isdefined config_dir) ? config_dir : "/scratch/projects/hhp00048/lDGA/PD/data_PD_hf_sb_01/U4.0/b15.0/lDGA_julia"
cfg_list = filter(f->endswith(f,"toml"), readdir(cfg_file_dir))
for cfg_file_i in cfg_list
    cfg_file = joinpath(cfg_file_dir, cfg_file_i)
    @info "Processing cfg: " cfg_file
    @timeit LadderDGA.to "input" wp, mP, sP, env, kGridsStr = readConfig(cfg_file);
    #TODO: loop over kGrids
    @timeit LadderDGA.to "setup" Σ_ladderLoc, Σ_loc, imp_density, kG, gLoc_fft, gLoc_rfft, Γsp, Γch, χDMFTsp, χDMFTch, locQ_sp, locQ_ch, χ₀Loc, gImp = setup_LDGA(kGridsStr[1], mP, sP, env);

    # ladder quantities
    @info "bubble"
    @timeit LadderDGA.to "nl bblt" bubble = calc_bubble(gLoc_fft, gLoc_rfft, kG, mP, sP);

    @timeit LadderDGA.to "λ₀" begin
        Fsp = F_from_χ(χDMFTsp, gImp[1,:], sP, mP.β);
        λ₀ = calc_λ0(bubble, Fsp, locQ_sp, mP, sP)
    end

    @info "chi"
    @timeit LadderDGA.to "nl xsp" nlQ_sp = calc_χγ(:sp, Γsp, bubble, kG, mP, sP);
    @timeit LadderDGA.to "nl xch" nlQ_ch = calc_χγ(:ch, Γch, bubble, kG, mP, sP);

    λsp_old = λ_correction(:sp, imp_density, nlQ_sp, nlQ_ch, gLoc_rfft, λ₀, kG, mP, sP)
    @timeit LadderDGA.to "new λ" λsp_new = λ_correction(:sp_ch, imp_density, nlQ_sp, nlQ_ch, gLoc_rfft, λ₀, kG, mP, sP, parallel=false)
    println("=========================================================================")
    #@timeit LadderDGA.to "new λ par" λsp_new_par = λ_correction(:sp_ch, imp_density, nlQ_sp, nlQ_ch, gLoc_rfft, λ₀, kG, mP, sP, parallel=true, workerpool=wp)
    #@info "parallel data agrees with sequential: " all(λsp_new.zero .≈ λsp_new_par.zero)

    @info "Σ"
    @timeit LadderDGA.to "nl Σ" Σ_ladder = calc_Σ(nlQ_sp, nlQ_ch, λ₀, gLoc_rfft, kG, mP, sP);
    println("=========================================================================")
    @timeit LadderDGA.to "nl Σ par" Σ_ladder_par = LadderDGA.calc_Σ_par(nlQ_sp, nlQ_ch, λ₀, gLoc_rfft, kG, mP, sP);
    @info "parallel data agrees with sequential: " all(Σ_ladder .≈ Σ_ladder_par)

    @timeit LadderDGA.to "nl bblt par" bubble_par = calc_bubble_par(gLoc_fft, gLoc_rfft, kG, mP, sP, workerpool=wp);
    @timeit LadderDGA.to "nl xsp par" nlQ_sp_par = LadderDGA.calc_χγ_par(:sp, Γsp, bubble, kG, mP, sP, workerpool=wp);
    @info "parallel data agrees with sequential: " all(nlQ_sp_par.γ .≈ nlQ_sp.γ)
    !all(nlQ_sp_par.γ .≈ nlQ_sp.γ) && error("Sequential and parallel computation of the susceptibilities do not yield the same result.")
    @timeit LadderDGA.to "nl Σ par" Σ_ladder_par = LadderDGA.calc_Σ_par(nlQ_sp, nlQ_ch, λ₀, gLoc_rfft, kG, mP, sP);
    @info "parallel data agrees with sequential: " all(Σ_ladder .≈ Σ_ladder_par)
    !all(Σ_ladder .≈ Σ_ladder_par) && error("Sequential and parallel computation of the self energy do not yield the same result.")
    #Σ_ladder = Σ_loc_correction(Σ_ladder, Σ_ladderLoc, Σ_loc);
    true
end
println(LadderDGA.to)
println(stderr, LadderDGA.to)
true
