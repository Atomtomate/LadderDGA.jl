using TimerOutputs
using Logging
using Pkg
Pkg.activate(@__DIR__)
using LadderDGA

using Distributed
np = parse(Int,ARGS[3])
addprocs(np, topology=:master_worker)
@everywhere using Pkg
@everywhere Pkg.activate(@__DIR__)
using LadderDGA
@everywhere using LadderDGA

cfg_file = ARGS[1]
out_path = ARGS[2]

@timeit LadderDGA.to "input" wp, mP, sP, env, kGridsStr = readConfig(cfg_file);
println("using workerpool: ", wp)
name = "lDGA_kConv"
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

    conv = false
    conv_error = false
    Nk   = 10
    tc_s = (sP.tc_type_f != :nothing) ? "rtc" : "ntc"
    fname_out =  out_path*"/lDGA_"*tc_s*"_kConv.jld2" 
    λ_root_guess = [0.0, 0.0]

    if isfile(fname_out)
        jldopen(fname_out,"r") do f
            max_prev_Nk = maximum(union(filter(x->x !== nothing, tryparse.(Int,keys(f))),[0]))
            λ_root_guess[:] = max_prev_Nk > 10 ? f["$max_prev_Nk/λspch"] : [0.0, 0.0]
            Nk = max_prev_Nk + 10;
        end
        @info "Found existing kConv file. Continuing at Nk = $Nk"
    else
        @timeit LadderDGA.to "write" jldopen(fname_out, "w") do f
            E_kin_ED, E_pot_ED = LadderDGA.calc_E_ED(env.inputDir*"/"*env.inputVars)
            f["config"] = cfg_string 
            f["sP"] = sP
            f["mP"] = mP
            f["E_kin_ED"] = E_kin_ED
            f["E_pot_ED"] = E_pot_ED
        end
    end

    while !conv && !conv_error
        @info "Running k-grid convergence calculation for Nk = $Nk"
        @timeit LadderDGA.to "setup" Σ_ladderLoc, Σ_loc, imp_density, kG, gLoc_fft, gLoc_rfft, Γsp, Γch, χDMFTsp, χDMFTch, locQ_sp, locQ_ch, χ₀Loc, gImp = setup_LDGA((kGridsStr[1][1], Nk), mP, sP, env);

        @timeit LadderDGA.to "nl bblt par" bubble = calc_bubble_par(gLoc_fft, gLoc_rfft, kG, mP, sP, workerpool=wp);
        @timeit LadderDGA.to "nl xsp par" nlQ_sp = LadderDGA.calc_χγ_par(:sp, Γsp, bubble, kG, mP, sP, workerpool=wp);
        @timeit LadderDGA.to "nl xch par" nlQ_ch = LadderDGA.calc_χγ_par(:ch, Γch, bubble, kG, mP, sP, workerpool=wp);
        ωindices = intersect(nlQ_sp.usable_ω, nlQ_ch.usable_ω)

        @timeit LadderDGA.to "λ₀" begin
            Fsp = F_from_χ(χDMFTsp, gImp[1,:], sP, mP.β);
            λ₀ = calc_λ0(bubble, Fsp, locQ_sp, mP, sP)
        end
        λsp = λ_correction(:sp, imp_density, nlQ_sp, nlQ_ch, gLoc_rfft, λ₀, kG, mP, sP)

        if Nk == 10
            @timeit LadderDGA.to "c2" c2_res = c2_curve(15, 15, [Inf, Inf], nlQ_sp, nlQ_ch, gLoc_rfft, λ₀, kG, mP, sP)
            c2_root_res = find_root(c2_res)
            λ_root_guess[:] = [c2_root_res[1], c2_root_res[2]]
        end

        λspch, λspch_z = try
            @timeit LadderDGA.to "new λ par" λspch = λ_correction(:sp_ch, imp_density, nlQ_sp, nlQ_ch, gLoc_rfft, λ₀, kG, mP, sP, x₀=[λ_root_guess[1], λ_root_guess[2]], parallel=true, workerpool=wp)
            println("extended lambda: ", λspch)
            @info λspch
            λ_root_guess = λspch.zero
            λspch, λspch.zero
        catch e
            @warn e
            @warn "new lambda correction did non converge, resetting lambda to zero"
            nothing, [λ_root_guess[1], λ_root_guess[2]]
            conv_error = true
        end
        #@timeit LadderDGA.to "new λ" λspch = λ_correction(:sp_ch, imp_density, nlQ_sp, nlQ_ch, gLoc_rfft, λ₀, kG, mP, sP)
        ωindices = intersect(nlQ_sp.usable_ω, nlQ_ch.usable_ω)
        iωn = 1im .* 2 .* collect(-sP.n_iω:sP.n_iω)[ωindices] .* π ./ mP.β
        nh = ceil(Int,size(nlQ_sp.χ, 2)/2)
        νmax::Int = min(sP.n_iν,floor(Int,3*length(ωindices)/8))
        χAF_DMFT = real(1 / (1 / nlQ_sp.χ[end,nh]))

        χ_λ!(nlQ_sp.χ, nlQ_sp.χ, λsp); nlQ_sp.λ = λsp;
        E_pot_DMFT_2 = 0.5 * mP.U * real(sum(kintegrate(kG,nlQ_ch.χ .- nlQ_sp.χ,1)[1,ωindices])/mP.β) + mP.U * mP.n^2/4
        Σ_ladder_λsp = LadderDGA.calc_Σ(nlQ_sp, nlQ_ch, λ₀, gLoc_rfft, kG, mP, sP);
        E_kin_λsp_1, E_pot_λsp_1 = calc_E(Σ_ladder_λsp.parent, kG, mP, νmax = νmax)
        E_pot_λsp_2 = 0.5 * mP.U * real(sum(kintegrate(kG,nlQ_ch.χ .- nlQ_sp.χ,1)[1,ωindices])/mP.β)  + mP.U * mP.n^2/4
        χAF_λsp = real(1 / (1 / nlQ_sp.χ[end,nh]))
        χ_λ!(nlQ_sp.χ, nlQ_sp.χ, -λsp); 
        Σ_ladder_λsp = nothing

        # Both
        χ_λ!(nlQ_sp.χ, nlQ_sp.χ, λspch_z[1]); 
        χ_λ!(nlQ_ch.χ, nlQ_ch.χ, λspch_z[2]); 
        Σ_ladder_λspch = LadderDGA.calc_Σ(nlQ_sp, nlQ_ch, λ₀, gLoc_rfft, kG, mP, sP);
        E_kin_λspch_1, E_pot_λspch_1 = calc_E(Σ_ladder_λspch.parent, kG, mP, νmax = νmax)
        E_pot_λspch_2 = 0.5 * mP.U * real(sum(kintegrate(kG,nlQ_ch.χ .- nlQ_sp.χ,1)[1,ωindices])/mP.β) + mP.U * mP.n^2/4
        χAF_λspch = real(1 / (1 / nlQ_sp.χ[end,nh]))
        sp_pos = all(kintegrate(kG,real(nlQ_sp.χ),1)[1,ωindices] .>= 0)
        ch_pos = all(kintegrate(kG,real(nlQ_ch.χ),1)[1,ωindices] .>= 0)
        nlQ_sp = nothing
        nlQ_ch = nothing

        @timeit LadderDGA.to "write" jldopen(fname_out, "a+") do f
            relC = Inf
            if Nk > 10
                println("χAF = $χAF_λspch")
                old_val = f["$(Nk-10)/χAF_λspch"]
                println("old χAF = $old_val")
                relC = abs((χAF_λspch - old_val)/χAF_λspch)
                println("Change: $(100*relC) %")
                if relC < 0.005 
                    conv = true
                     open(out_path*"/kConv.txt","w") do f_conv
                         write(f_conv, "Ns = $Nk\n")
                         write(f_conv, "relative error between last iterations: $(relC)")
                     end
                end
            end
            if !sp_pos || !ch_pos
                println("ERROR: negative χ. check sp: $sp_pos, ch: $ch_pos")
            end
            f["$Nk/χAF_DMFT"] = χAF_DMFT
            f["$Nk/χAF_λsp"] = χAF_λsp
            f["$Nk/χAF_λspch"] = χAF_λspch
            f["$Nk/E_pot_DMFT_2"] = E_pot_DMFT_2
            f["$Nk/E_kin_λsp_1"] = E_kin_λsp_1
            f["$Nk/E_pot_λsp_1"] = E_pot_λsp_1
            f["$Nk/E_pot_λsp_2"] = E_pot_λsp_2
            f["$Nk/E_kin_λspch_1"] = E_kin_λspch_1
            f["$Nk/E_pot_λspch_1"] = E_pot_λspch_1
            f["$Nk/E_pot_λspch_2"] = E_pot_λspch_2
            f["$Nk/λsp"] = λsp
            f["$Nk/λspch"] = λspch
            f["$Nk/ΔχAF"] = relC
            f["$Nk/log"] = LadderDGA.get_log()
            f["$Nk/sp_pos"] = sp_pos
            f["$Nk/ch_pos"] = ch_pos
            f["$Nk/conv_error"] = conv_error
            f["$Nk/conv"] = conv
        end


        @info "Runtime for iteration:"
        @info LadderDGA.to
        Nk += 10;
        flush(stdout)
        flush(stderr)
    end
    @info "Done! Runtime:"
    print(LadderDGA.to)
end
