using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using LadderDGA
using JLD2

cfg_file = "/home/julisn/Hamburg/ED_data/2D_data/b20_mu0.5.toml" #ARGS[1]
fOutName = ""#ARGS[2]
 #parse(Int, ARGS[3]))

wp, mP, sP, env, kGridsStr = readConfig(cfg_file);
lDGAhelper = setup_LDGA(kGridsStr[1], mP, sP, env, silent=true);

function run_sc(maxIt::Int, h; tc=ΣTail_ExpStep{0.2}, eps = 1e-7, maxIt = 1)

    converged = false
    it = 0
    G_fft_it = h.gLoc_fft
    G_rfft_it = h.gLoc_rfft
    G_bak = nothing
    μ_it = NaN 
    G_it = nothing
    Σ_it = nothing
    χm_list = []; χd_list = []; 
    χm = nothing; γm = nothing 
    χd = nothing; γd = nothing 
    χ₀ = nothing; λ₀ = nothing

    while !converged && it < maxIt
      println(" ========== IT = $it / $maxIt ========== ")
        χ₀ = calc_bubble(:DMFT, G_fft_it, G_rfft_it, h.kG, h.mP, h.sP);
        println("cs bubble = $(sum(χ₀))")
        χm, γm = calc_χγ(:m, lDGAhelper, χ₀; ω_symmetric=true);
        χd, γd = calc_χγ(:d, lDGAhelper, χ₀; ω_symmetric=true);
        push!(χm_list, χm)
        push!(χd_list, χd)
        check_χ_health(χm, :m, lDGAhelper; q0_check_eps = 0.1, λmin_check_eps = 1000)
        check_χ_health(χd, :d, lDGAhelper; q0_check_eps = 0.1, λmin_check_eps = 1000)
        λ₀ = calc_λ0(χ₀, lDGAhelper);

        G_bak = deepcopy(G_it)
        μ_it, G_it, Σ_it = calc_G_Σ(χm, γm, χd, γd, λ₀, 0.0, 0.0, h; gLoc_rfft=G_rfft_it, tc = tc)
        # println("b:", axes(G_it))
        G_fft_it, G_rfft_it = LadderDGA.G_fft(G_it, h.kG, h.sP)
        converged = it > 0 && sum(abs.(G_it .- G_bak)) < eps
        it += 1
    end
    return χm, γm, χd, γd, χ₀, λ₀, G_it, μ_it, χm_list, χd_list
end

χm, γm, χd, γd, χ₀, λ₀, G_it, μ_it, χm_list, χd_list = run_sc(5, lDGAhelper);

# jldopen(fOutName, "w") do f
#     f["chi_m"] = χm
#     f["chi_d"] = χd
#     f["res_m_nat"] = res_m_nat
#     f["res_m_fix"] = res_m_fix
#     f["res_dm"] = res_dm
# end
