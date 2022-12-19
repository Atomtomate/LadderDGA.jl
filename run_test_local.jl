using TimerOutputs
# using Pkg
# Pkg.activate(@__DIR__)

using Distributed
using LadderDGA
using JLD2
@everywhere using LadderDGA
out_path = ARGS[1]

cfg_file = "/home/julian/Hamburg/Julia_lDGA/LadderDGA.jl/config.toml"
@timeit LadderDGA.to "input" wp, mP, sP, env, kGridsStr = readConfig(cfg_file);
#TODO: loop over kGrids
@timeit LadderDGA.to "setup" Σ_ladderLoc, Σ_loc, imp_density, kG, gLoc_fft, gLoc_rfft, Γsp, Γch, χDMFTsp, χDMFTch, χ_sp_loc, γ_sp_loc, χ_ch_loc, γ_ch_loc, χ₀Loc, gImp = setup_LDGA(kGridsStr[1], mP, sP, env);

# ladder quantities
@info "bubble"
@timeit LadderDGA.to "nl bblt" bubble = calc_bubble(gLoc_fft, gLoc_rfft, kG, mP, sP);

@timeit LadderDGA.to "λ₀" begin
    Fsp = F_from_χ(χDMFTsp, gImp[1,:], sP, mP.β);
    λ₀ = calc_λ0(bubble, Fsp, χ_sp_loc, γ_sp_loc, mP, sP)
end

@info "chi"
@timeit LadderDGA.to "nl xsp" χ_sp, γ_sp = calc_χγ(:sp, Γsp, bubble, kG, mP, sP);
@timeit LadderDGA.to "nl xch" χ_ch, γ_ch = calc_χγ(:ch, Γch, bubble, kG, mP, sP);
@timeit LadderDGA.to "nl Σ" Σ_ladder = calc_Σ(χ_sp, γ_sp, χ_ch, γ_ch, λ₀, gLoc_rfft, kG, mP, sP);


#λsp_old = λ_correction(:sp, imp_density, χ_sp, γ_sp, χ_ch, γ_ch, gLoc_rfft, λ₀, kG, mP, sP)
#@timeit LadderDGA.to "new λ" λsp_new = λ_correction(:sp_ch, imp_density, χ_sp, γ_sp, χ_ch, γ_ch, gLoc_rfft, λ₀, kG, mP, sP, parallel=false)
#println("=========================================================================")
#@timeit LadderDGA.to "new λ clean" λsp_new_clean = LadderDGA.extended_λ_clean(χ_sp, γ_sp, χ_ch, γ_ch, gLoc_rfft, λ₀, kG, mP, sP)
# @info "parallel data agrees with sequential: " all(isapprox.(λsp_new.zero , λsp_new_par.zero, atol=1e-5))
# @info "clean data agrees with fast: " all(isapprox.(λsp_new.zero , λsp_new_clean.zero, atol=1e-5))

# @info "parallel"
# println("=========================================================================")
# @timeit LadderDGA.to "nl bblt par" bubble_par = calc_bubble_par(gLoc_fft, gLoc_rfft, kG, mP, sP, workerpool=wp);
# @timeit LadderDGA.to "nl xsp par" χ_sp_par, γ_sp_par = LadderDGA.calc_χγ_par(:sp, Γsp, bubble, kG, mP, sP, workerpool=wp);
# @info "parallel data agrees with sequential: " all(γ_sp_par .≈ γ_sp)
# !all(γ_sp_par .≈ γ_sp) && error("Sequential and parallel computation of the susceptibilities do not yield the same result.")

# @timeit LadderDGA.to "nl Σ par" Σ_ladder_par = LadderDGA.calc_Σ_par(χ_sp, γ_sp, χ_ch, γ_ch, λ₀, gLoc_rfft, kG, mP, sP);
# @info "parallel data agrees with sequential: " all(Σ_ladder .≈ Σ_ladder_par)
# !all(Σ_ladder .≈ Σ_ladder_par) && error("Sequential and parallel computation of the self energy do not yield the same result.")
# #Σ_ladder = Σ_loc_correction(Σ_ladder, Σ_ladderLoc, Σ_loc);

# @timeit LadderDGA.to "nl Σ par" Σ_ladder_parts = LadderDGA.calc_Σ_parts(χ_sp, γ_sp, χ_ch, γ_ch, λ₀, gLoc_rfft, kG, mP, sP);
# Σ_ladder2 = sum(Σ_ladder_parts, dims=3)[:,:,1]
# @info "Channel split data agrees with sequential: " all(Σ_ladder .≈ Σ_ladder2)
# !all(Σ_ladder .≈ Σ_ladder2) && error("Channel split and normal computation of the self energy do not yield the same result.")
# #LadderDGA.writeFortranΣ("klist_parts_test", Σ_ladder_parts.parent, mP.β)
# #LadderDGA.writeFortranΣ("klist_summed_test", Σ_ladder.parent, mP.β)

c2_res = residuals(15, 15, [0.0, 0.0], χ_sp, γ_sp, χ_ch, γ_ch, Σ_loc, gLoc_rfft, λ₀, kG, mP, sP)

fname_out =  out_path*"/lDGA_c2.jld2" 
jldopen(fname_out, "a+") do f
    cfg_string = read(cfg_file, String)
    f["config"] = cfg_string 
    f["c2_res"] = c2_res
end
true
