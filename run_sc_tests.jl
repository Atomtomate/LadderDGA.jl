using TimerOutputs
using Pkg
using Distributed
Pkg.activate(@__DIR__)
@everywhere using Pkg
@everywhere Pkg.activate(@__DIR__)
using LadderDGA
@everywhere using LadderDGA

function run_sc_test(; run_c2_curve=false, fname="", descr="", cfg_file=nothing, res_prefix="", res_postfix="", save_results=true, log_io=devnull)
    @timeit LadderDGA.to "input" wp, mP, sP, env, kGridsStr = readConfig(cfg_file);
    @timeit LadderDGA.to "setup" Σ_ladderLoc, Σ_loc, imp_density, kG, gLoc_fft, gLoc_rfft, Γsp, Γch, χDMFTsp, χDMFTch, χ_sp_loc, γ_sp_loc, χ_ch_loc, γ_ch_loc, χ₀Loc, gImp = setup_LDGA(kGridsStr[1], mP, sP, env);

    # ladder quantities
    @info "bubble"
    @timeit LadderDGA.to "nl bblt" bubble = calc_bubble_par(gLoc_fft, gLoc_rfft, kG, mP, sP, workerpool=wp);

    @timeit LadderDGA.to "λ₀" begin
        Fsp = F_from_χ(χDMFTsp, gImp[1,:], sP, mP.β);
        λ₀ = calc_λ0(bubble, Fsp, χ_sp_loc, γ_sp_loc, mP, sP)
    end

    @info "chi"
    @timeit LadderDGA.to "nl xsp" χ_sp, γ_sp = calc_χγ_par(:sp, Γsp, bubble, kG, mP, sP, workerpool=wp);
    @timeit LadderDGA.to "nl xch" χ_ch, γ_ch = calc_χγ_par(:ch, Γch, bubble, kG, mP, sP, workerpool=wp);
    c2_res = c2_curve(20, 20, χ_sp, γ_sp, χ_ch, γ_ch, gLoc_rfft, λ₀, kG, mP, sP)

    @timeit LadderDGA.to "nl Σ" Σ_ladder = LadderDGA.calc_Σ_par(χ_sp, γ_sp, χ_ch, γ_ch, λ₀, gLoc_rfft, kG, mP, sP);

    true
end
