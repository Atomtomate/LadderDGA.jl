# ==================== Includes ====================
#nprocs() == 1 && addprocs(8)
using Pkg
path = joinpath(abspath(@__DIR__),"..")
println("activating: ", path)
Pkg.activate(path)
using LadderDGA
using JLD2

wp, mP, sP, env, kGridsStr = readConfig(cfg);
lDGAhelper = setup_LDGA(kGridsStr[1], mP, sP, env);
Nk = lDGAhelper.kG.Ns
Nω = 2*lDGAhelper.sP.n_iω

tmp_file = "tmp_lDGA_Nk$(Nk)_Nw$Nω.jld2" 

# ====================== lDGA ======================
bubble, χm, χd, γm, γd, λ₀ = nothing, nothing, nothing, nothing, nothing, nothing

tmp_f = joinpath(tmp_path,tmp_file)
if isfile(tmp_f)
    println("Loading Data")
    jldopen(tmp_f, "r") do f
        global bubble = f["bubble"]
        global χm  = f["chi_m"]
        global χd  = f["chi_d"]
        global γm  = f["gamma_m"]
        global γd  = f["gamma_d"]
        global λ₀  = f["lambda_0"]
    end
else
    println("Calculating Data")
    bubble     = calc_bubble(:DMFT, lDGAhelper);
    χm, γm = calc_χγ(:m, lDGAhelper, bubble);
    χd, γd = calc_χγ(:d, lDGAhelper, bubble);
    λ₀ = calc_λ0(bubble, lDGAhelper)
    jldopen(tmp_f, "w") do f
        f["bubble"] = bubble
        f["chi_m"] = χm 
        f["chi_d"] = χd 
        f["gamma_m"] = γm 
        f["gamma_d"] = γd 
        f["lambda_0"] = λ₀ 
    end
end



# ==================== Results =====================
# println("RUNNING lDGA_m")
# res_m = LadderDGA.λ_correction(:m, χm, γm, χd, γd, λ₀, lDGAhelper)
println("RUNNING lDGA_dm")
res_dm = λ_correction(:dm, χm, γm, χd, γd, λ₀, lDGAhelper; fit_μ=true, verbose=true)
# println("RUNNING lDGA_dm_sc")
# res_dm_sc = run_sc(χm, γm, χd, γd, λ₀, lDGAhelper.mP.μ, lDGAhelper; type=:pre_dm, fit_μ=true, maxit=100, mixing=0.2, conv_abs=1e-4, trace=true);
# println("RUNNING lDGA_m (no tail corrections)")
# res_m_ntc = LadderDGA.λ_correction(:m, χm, γm, χd, γd, λ₀, lDGAhelper, tc=false)
# println("RUNNING lDGA_dm (no tail corrections)")
# res_dm_ntc = λ_correction(:dm, χm, γm, χd, γd, λ₀, lDGAhelper; fit_μ=true, tc=false)
# println("RUNNING lDGA_dm (no tail corrections, no μ fit)")
# res_dm_ntc_no_mu_fit  = λ_correction(:dm, χm, γm, χd, γd, λ₀, lDGAhelper; fit_μ=false, tc=false)
# println("RUNNING lDGA_dm_sc (no tail corrections)")
# res_dm_sc_ntc = run_sc(χm, γm, χd, γd, λ₀, lDGAhelper.mP.μ, lDGAhelper; type=:pre_dm, fit_μ=true, maxit=100, mixing=0.2, conv_abs=1e-4, tc=false, trace=true);
