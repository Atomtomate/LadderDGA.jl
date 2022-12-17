using TimerOutputs
using Logging
using Pkg
Pkg.activate(@__DIR__)
using LadderDGA
using NLsolve

using Distributed, SlurmClusterManager
if length(ARGS) >= 3 
    np =  parse(Int,ARGS[3])
    addprocs(np, topology=:master_worker)
else
    addprocs(SlurmManager(), topology=:master_worker)
end
@everywhere using Pkg
@everywhere Pkg.activate(@__DIR__)
using LadderDGA
@everywhere using LadderDGA

cfg_file = ARGS[1]
out_path = ARGS[2]
continue_on_error = true

@timeit LadderDGA.to "input" wp, mP, sP, env, kGridsStr = readConfig(cfg_file);
println("using workerpool: ", wp)
name = "lDGA_sc_01"
logfile_path = out_path*"/"*name*".log"
i = 1
while isfile(logfile_path)
    global i
    global logfile_path
    postf = "_$i.log"
    logfile_path = i > 1 ? logfile_path[1:end-4-(length("_$(i-1)"))]*postf : logfile_path*postf
    i += 1
end

println("config path: $cfg_file\noutput path: $out_path\nlogging to $(logfile_path)")
include("./run.jl")
include("/scratch/projects/hhp00048/codes/scripts/LadderDGA_utils/new_lambda_analysis.jl")

open(logfile_path,"w") do io

    cfg_string = read(cfg_file, String)
    @timeit LadderDGA.to "input" wp, mP, sP, env, kGridsStr = readConfig(cfg_file);

    Nk   = 20
    tc_s = (sP.tc_type_f != :nothing) ? "rtc" : "ntc"
    fname_out =  out_path*"/lDGA_"*tc_s*"_sc_01.jld2" 
    λ_root_guess = [0.0, 0.0]

    @timeit LadderDGA.to "write" jldopen(fname_out, "w") do f
        E_kin_ED, E_pot_ED = LadderDGA.calc_E_ED(env.inputDir*"/"*env.inputVars)
        f["config"] = cfg_string 
        f["sP"] = sP
        f["mP"] = mP
        f["E_kin_ED"] = E_kin_ED
        f["E_pot_ED"] = E_pot_ED
        f["Nk"] = Nk
    end

    @info "Running k-grid convergence calculation for Nk = $Nk"
    @timeit LadderDGA.to "setup" Σ_ladderLoc, Σ_loc, imp_density, kG, gLoc_fft, gLoc_rfft, Γsp, Γch, χDMFTsp, χDMFTch, locQ_sp, locQ_ch, χ₀Loc, gImp = setup_LDGA((kGridsStr[1][1], Nk), mP, sP, env);
    @timeit LadderDGA.to "nl bblt par" bubble = calc_bubble_par(gLoc_fft, gLoc_rfft, kG, mP, sP, workerpool=wp);
    @timeit LadderDGA.to "nl xsp par" nlQ_sp = LadderDGA.calc_χγ_par(:sp, Γsp, bubble, kG, mP, sP, workerpool=wp);
    @timeit LadderDGA.to "nl xch par" nlQ_ch = LadderDGA.calc_χγ_par(:ch, Γch, bubble, kG, mP, sP, workerpool=wp);
    
    sc_it = 0

    while sc_it < 10

        @timeit LadderDGA.to "λ₀" begin
            Fsp = F_from_χ(χDMFTsp, gImp[1,:], sP, mP.β);
            λ₀ = calc_λ0(bubble, Fsp, locQ_sp, mP, sP)
        end
        λsp = λ_correction(:sp, imp_density, nlQ_sp, nlQ_ch, gLoc_rfft, λ₀, kG, mP, sP)

        @timeit LadderDGA.to "c2" c2_res = c2_curve(30, 30, [-Inf, -Inf], nlQ_sp, nlQ_ch, gLoc_rfft, λ₀, kG, mP, sP)
        c2_root_res = find_root(c2_res)
        @info "c2 root result:  $c2_root_res"
        λ_root_guess[:] = [c2_root_res[1], c2_root_res[2]]

        @info "trying to obtain root with guess: $λ_root_guess"
        λspch = nothing

        @timeit LadderDGA.to "new λ" λspch = λ_correction(:sp_ch, imp_density, nlQ_sp, nlQ_ch, gLoc_rfft, λ₀, kG, mP, sP)
        @info "found root $(λspch.zero). log: \n$(λspch)\n"

        ωindices = intersect(nlQ_sp.usable_ω, nlQ_ch.usable_ω)
        iωn = 1im .* 2 .* collect(-sP.n_iω:sP.n_iω)[ωindices] .* π ./ mP.β
        nh = ceil(Int,size(nlQ_sp.χ, 2)/2)
        νmax::Int = min(sP.n_iν,floor(Int,3*length(ωindices)/8))
        
        println("DMFT Epot")
        E_pot_DMFT_2 = calc_Epot2(nlQ_sp, nlQ_ch, kG, sP, mP) + mP.U * mP.n^2/4

        χAF_DMFT = real(1 / (1 / nlQ_sp.χ[end,nh]))


        println("λ_m Epot")
        χ_λ!(nlQ_sp.χ, nlQ_sp.χ, λsp); nlQ_sp.λ = λsp;
        E_kin_λsp_1, E_pot_λsp_1 = calc_E(nlQ_sp, nlQ_ch, λ₀, gLoc_rfft, kG, mP, sP, νmax=νmax)
        E_pot_λsp_2 = calc_Epot2(nlQ_sp, nlQ_ch, kG, sP, mP) + mP.U * mP.n^2/4
        println("Epot_2 λsp: $E_pot_λsp_2")
        χAF_λsp = real(1 / (1 / nlQ_sp.χ[end,nh]))
        sp_m_pos = all(kintegrate(kG,real(nlQ_sp.χ),1)[1,ωindices] .>= 0)
        ch_m_pos = all(kintegrate(kG,real(nlQ_ch.χ),1)[1,ωindices] .>= 0)
        χ_λ!(nlQ_sp.χ, nlQ_sp.χ, -λsp); 

        # Both
        println("λ_md Epot")
        χ_λ!(nlQ_sp.χ, nlQ_sp.χ, λspch.zero[1]); 
        χ_λ!(nlQ_ch.χ, nlQ_ch.χ, λspch.zero[2]); 
        E_kin_λspch_1, E_pot_λspch_1 = calc_E(nlQ_sp, nlQ_ch, λ₀, gLoc_rfft, kG, mP, sP, νmax=νmax)
        E_pot_λspch_2 = calc_Epot2(nlQ_sp, nlQ_ch, kG, sP, mP) + mP.U * mP.n^2/4
        χAF_λspch = real(1 / (1 / nlQ_sp.χ[end,nh]))
        χ_λ!(nlQ_sp.χ, nlQ_sp.χ, -λspch.zero[1]); 
        χ_λ!(nlQ_ch.χ, nlQ_ch.χ, -λspch.zero[2]); 


        @timeit LadderDGA.to "write" jldopen(fname_out, "a+") do f
            f["$sc_it/χAF_DMFT"] = χAF_DMFT
            f["$sc_it/χAF_λsp"] = χAF_λsp
            f["$sc_it/χAF_λspch"] = χAF_λspch
            f["$sc_it/c2_res"] = c2_res
            f["$sc_it/E_pot_DMFT_2"] = E_pot_DMFT_2
            f["$sc_it/E_kin_λsp_1"] = E_kin_λsp_1
            f["$sc_it/E_pot_λsp_1"] = E_pot_λsp_1
            f["$sc_it/E_pot_λsp_2"] = E_pot_λsp_2
            f["$sc_it/E_kin_λspch_1"] = E_kin_λspch_1
            f["$sc_it/E_pot_λspch_1"] = E_pot_λspch_1
            f["$sc_it/E_pot_λspch_2"] = E_pot_λspch_2
            f["$sc_it/λsp"] = λsp
            f["$sc_it/λspch"] = λspch
            f["$sc_it/log"] = LadderDGA.get_log()
        end


        @info "Runtime for iteration:"
        @info LadderDGA.to
        flush(stdout)
        flush(stderr)
    end
    @info "Done! Runtime:"
    print(LadderDGA.to)
end
