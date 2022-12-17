using TimerOutputs
using Pkg
using Distributed
Pkg.activate(@__DIR__)
@everywhere using Pkg
@everywhere Pkg.activate(@__DIR__)
using LadderDGA
@everywhere using LadderDGA

function run_sc_test_full(cfg_file, fname, Nit)
    @timeit LadderDGA.to "input" wp, mP, sP, env, kGridsStr = readConfig(cfg_file);
    @timeit LadderDGA.to "setup" Σ_ladderLoc, Σ_loc, imp_density, kG, gLoc_fft, gLoc_rfft, Γsp, Γch, χDMFTsp, χDMFTch, χ_sp_loc, γ_sp_loc, χ_ch_loc, γ_ch_loc, χ₀Loc, gImp = setup_LDGA(kGridsStr[1], mP, sP, env);
    for it in 1:Nit

        # ladder quantities
        bubble = calc_bubble_par(gLoc_fft, gLoc_rfft, kG, mP, sP, workerpool=wp);

        Fsp = F_from_χ(χDMFTsp, gImp[1,:], sP, mP.β);
        λ₀ = calc_λ0(bubble, Fsp, χ_sp_loc, γ_sp_loc, mP, sP)

        χ_sp, γ_sp = calc_χγ_par(:sp, Γsp, bubble, kG, mP, sP, workerpool=wp);
        χ_ch, γ_ch = calc_χγ_par(:ch, Γch, bubble, kG, mP, sP, workerpool=wp);
        c2_res = c2_curve(20, 20, [0.0,0.0], χ_sp, γ_sp, χ_ch, γ_ch, gLoc_rfft, λ₀, kG, mP, sP)
        λsp,λch, _ = find_root(c2_res)

        @timeit LadderDGA.to "nl Σ" Σ_ladder = LadderDGA.calc_Σ(χ_sp, γ_sp, χ_ch, γ_ch, λ₀, gLoc_rfft, kG, mP, sP);

        @timeit LadderDGA.to "write" jldopen(fname, "a+") do f
            f["$it/c2_curve"] = c2_res
            f["$it/Σ_ladder"] = Σ_ladder
            f["$it/chi_sp"] = χ_sp 
            f["$it/chi_ch"] = χ_ch 
            f["$it/mu"] = mP.μ 
            f["$it/lambda_sp"] = λsp
            f["$it/lambda_ch"] = λch
        end

        println("gLoc_rfft checksum = ", sum(gLoc_rfft))
        μ, gLoc_red, gLoc_fft, gLoc_rfft = GLoc_from_Σladder(Σ_ladder, Σ_loc, kG, mP, sP)
        mP.μ = μ  
    end
end

function run_sc_test(cfg_file, fname, Nit)
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

    for it in 1:Nit
        c2_res = c2_curve(20, 20, [0.0,0.0], χ_sp, γ_sp, χ_ch, γ_ch, gLoc_rfft, λ₀, kG, mP, sP)
        λsp,λch, _ = find_root(c2_res)

        println("gLoc_rfft checksum = ", sum(gLoc_rfft))
        @timeit LadderDGA.to "nl Σ" Σ_ladder = LadderDGA.calc_Σ(χ_sp, γ_sp, χ_ch, γ_ch, λ₀, gLoc_rfft, kG, mP, sP);
        @timeit LadderDGA.to "write" jldopen(fname, "a+") do f
            f["$it/c2_curve"] = c2_res
            f["$it/Σ_ladder"] = Σ_ladder
            f["$it/chi_sp"] = χ_sp 
            f["$it/chi_ch"] = χ_ch 
            f["$it/mu"] = mP.μ 
            f["$it/lambda_sp"] = λsp
            f["$it/lambda_ch"] = λch
        end

        μ, gLoc_red, gLoc_fft, gLoc_rfft = GLoc_from_Σladder(Σ_ladder, Σ_loc, kG, mP, sP)
        mP.μ = μ  
    end
end
