using Plots, LaTeXStrings
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
using LadderDGA, OffsetArrays
using JLD2
using Test

cfg_file = ARGS[1]

wp, mP, sP, env, kGridsStr = readConfig(cfg_file);
lDGAhelper = setup_LDGA(kGridsStr[1], mP, sP, env, silent=true);
bubble     = calc_bubble(:DMFT, lDGAhelper);
χm, γm = calc_χγ(:m, lDGAhelper, bubble);
χd, γd = calc_χγ(:d, lDGAhelper, bubble);
λ₀ = calc_λ0(bubble, lDGAhelper);

h = lDGAhelper; 
λm = 0.1
λd = 0.2
λd_min = LadderDGA.LambdaCorrection.get_λ_min(χd)
ωn2_tail = LadderDGA.ω2_tail(χm)
Nq, Nω = size(χm)
νmax::Int = LadderDGA.eom_ν_cutoff(h.sP)
fft_νGrid= h.sP.fft_range



Kνωq_pre    = Vector{ComplexF64}(undef, length(h.kG.kMult))
G_ladder_1 = OffsetArray(Matrix{ComplexF64}(undef, Nq, length(fft_νGrid)), 1:Nq, fft_νGrid) 
G_ladder_bak = similar(G_ladder_1)
Σ_ladder_1 = OffsetArray(Matrix{ComplexF64}(undef, Nq, νmax), 1:Nq, 0:νmax-1)
iν = LadderDGA.iν_array(h.mP.β, collect(axes(Σ_ladder_1, 2)))
tc_factor = (true ? LadderDGA.tail_factor(h.mP.U, h.mP.β, h.mP.n, h.Σ_loc, iν) : 0.0 ./ iν)
tc_term  = LadderDGA.tail_correction_term(sum_kω(h.kG, χm, λ=λm), h.χloc_m_sum, tc_factor)
_, gLoc_rfft = G_fft(G_ladder_1, h.kG, h.sP)

μ_0, G_ladder_0, Σ_ladder_0 = LadderDGA.LambdaCorrection.calc_G_Σ(χm, γm, χd, γd, λ₀, λm, λd, h; gLoc_rfft = h.gLoc_rfft, tc = true, fix_n = true)

μ_1 = LadderDGA.LambdaCorrection.calc_G_Σ!(G_ladder_1, Σ_ladder_1, Kνωq_pre, tc_term, χm, γm, χd, γd, λ₀, λm, λd, h; gLoc_rfft=h.gLoc_rfft)

@test μ_0 ≈ μ_1 
@test sum(abs.(G_ladder_0 .- G_ladder_1)) < 1e-12
@test sum(abs.(Σ_ladder_0 .- Σ_ladder_1)) < 1e-12

_, gLoc_rfft = G_fft(G_ladder_0, h.kG, h.sP)
gLoc_rfft_test = deepcopy(gLoc_rfft)
LadderDGA.G_rfft!(gLoc_rfft_test, G_ladder_0, h.kG, h.sP.fft_range)
@test sum(abs.(gLoc_rfft .- gLoc_rfft_test)) < 1e-12

tr_1 = []
println(sum(χm), ", ", sum(χd), ", ", sum(γm), ", ", sum(γd), ", ", sum(λ₀), ", ", sum(h.gLoc_rfft))

converged_chk, μ_it_chk, G_ladder_it_chk, Σ_ladder_it_chk, tr_0 = LadderDGA.LambdaCorrection.run_sc(χm, γm, χd, γd, λ₀, λm, λd, h; 
                                trace=true)
println(sum(χm), ", ", sum(χd), ", ", sum(γm), ", ", sum(γd), ", ", sum(λ₀), ", ", sum(h.gLoc_rfft))

converged, μ_new = LadderDGA.LambdaCorrection.run_sc!(G_ladder_1, Σ_ladder_1, G_ladder_bak, Kνωq_pre, tc_factor, 
                            χm, γm, χd, γd, λ₀, λm, λd, h; 
                            mixing=0.3, trace=tr_1)
println(sum(χm), ", ", sum(χd), ", ", sum(γm), ", ", sum(γd), ", ", sum(λ₀), ", ", sum(h.gLoc_rfft))

@test sum(abs.(G_ladder_it_chk .- G_ladder_1)) < 1e-12