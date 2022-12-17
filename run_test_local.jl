using Base.GC
using TimerOutputs

using Pkg
Pkg.activate(@__DIR__)

using Distributed
@everywhere using Pkg
@everywhere Pkg.activate(@__DIR__)
using LadderDGA
@everywhere using LadderDGA

cfg_file = "/home/julian/Hamburg/Julia_lDGA/LadderDGA_stable/config.toml"
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
@timeit LadderDGA.to "new λ clean" λsp_new_clean = LadderDGA.extended_λ_clean(nlQ_sp, nlQ_ch, gLoc_rfft, λ₀, kG, mP, sP)
# @timeit LadderDGA.to "new λ par" λsp_new_par = λ_correction(:sp_ch, imp_density, nlQ_sp, nlQ_ch, gLoc_rfft, λ₀, kG, mP, sP, parallel=true, workerpool=wp)
# @info "parallel data agrees with sequential: " all(isapprox.(λsp_new.zero , λsp_new_par.zero, atol=1e-5))
# @info "clean data agrees with fast: " all(isapprox.(λsp_new.zero , λsp_new_clean.zero, atol=1e-5))

@info "parallel"
println("=========================================================================")
@timeit LadderDGA.to "nl bblt par" bubble_par = calc_bubble_par(gLoc_fft, gLoc_rfft, kG, mP, sP, workerpool=wp);
@timeit LadderDGA.to "nl xsp par" nlQ_sp_par = LadderDGA.calc_χγ_par(:sp, Γsp, bubble, kG, mP, sP, workerpool=wp);
@info "parallel data agrees with sequential: " all(nlQ_sp_par.γ .≈ nlQ_sp.γ)
!all(nlQ_sp_par.γ .≈ nlQ_sp.γ) && error("Sequential and parallel computation of the susceptibilities do not yield the same result.")

@timeit LadderDGA.to "nl Σ" Σ_ladder = calc_Σ(nlQ_sp, nlQ_ch, λ₀, gLoc_rfft, kG, mP, sP);
@timeit LadderDGA.to "nl Σ par" Σ_ladder_par = LadderDGA.calc_Σ_par(nlQ_sp, nlQ_ch, λ₀, gLoc_rfft, kG, mP, sP);
@info "parallel data agrees with sequential: " all(Σ_ladder .≈ Σ_ladder_par)
!all(Σ_ladder .≈ Σ_ladder_par) && error("Sequential and parallel computation of the self energy do not yield the same result.")
#Σ_ladder = Σ_loc_correction(Σ_ladder, Σ_ladderLoc, Σ_loc);

@timeit LadderDGA.to "nl Σ par" Σ_ladder_parts = LadderDGA.calc_Σ_parts(nlQ_sp, nlQ_ch, λ₀, gLoc_rfft, kG, mP, sP);
Σ_ladder2 = sum(Σ_ladder_parts, dims=3)[:,:,1]
@info "Channel split data agrees with sequential: " all(Σ_ladder .≈ Σ_ladder2)
!all(Σ_ladder .≈ Σ_ladder2) && error("Channel split and normal computation of the self energy do not yield the same result.")
LadderDGA.writeFortranΣ("klist_parts_test", Σ_ladder_parts.parent, mP.β)
LadderDGA.writeFortranΣ("klist_summed_test", Σ_ladder.parent, mP.β)

c2_res = c2_curve(1000, 20, [0.0, 0.0], nlQ_sp, nlQ_ch, gLoc_rfft, λ₀, kG, mP, sP)

# lch_list = -2:0.01:2
# lsp_lch = Array{Float64,1}(undef, length(lch_list))
# c2_vals = Array{Float64,1}(undef, length(lch_list))
# nlQ_ch_rr = deepcopy(nlQ_ch)
# nlQ_sp_rr = deepcopy(nlQ_sp)
# νmax::Int = minimum([sP.n_iν,floor(Int,3*length(nlQ_sp.χ[1,:])/8)])
# νGrid::UnitRange{Int} = 0:(νmax-1)
# iωn = 1im .* 2 .* (-sP.n_iω:sP.n_iω) .* π ./ mP.β
# Σ_hartree::Float64 = mP.n * mP.U/2.0;
# E_pot_tail_c = [zeros(size(kG.ϵkGrid)),
#         (mP.U^2 * 0.5 * mP.n * (1-0.5*mP.n) .+ Σ_hartree .* (kG.ϵkGrid .+ Σ_hartree .- mP.μ))]
# tail = [1 ./ (LadderDGA.iν_array(mP.β, νGrid) .^ n) for n in 1:length(E_pot_tail_c)]
# E_pot_tail::Matrix{ComplexF64} = sum(E_pot_tail_c[i] .* transpose(tail[i]) for i in 1:length(tail))
# E_pot_tail_inv::Vector{Float64} = sum((mP.β/2)  .* [Σ_hartree .* ones(size(kG.ϵkGrid)), (-mP.β/2) .* E_pot_tail_c[2]])
# for (i,lch) in enumerate(lch_list)
#     nlQ_ch_rr.χ = 1 ./ (1 ./ deepcopy(nlQ_ch.χ) .+ lch)
#     lsp_lch[i] = λ_correction(:sp, imp_density, nlQ_sp, nlQ_ch_rr, gLoc_rfft, λ₀, kG, mP, sP)
#     nlQ_sp_rr.χ = 1 ./ (1 ./ deepcopy(nlQ_sp.χ) .+ lsp_lch[i])
#     Σ_ladder_i = calc_Σ(nlQ_sp_rr, nlQ_ch_rr, λ₀, gLoc_rfft, kG, mP, sP).parent[:,1:νmax]
#     χupup_ω = LadderDGA.subtract_tail(0.5 * kintegrate(kG,nlQ_ch_rr.χ .+ nlQ_sp_rr.χ,1)[1,:], mP.Ekin_DMFT, iωn)
#     χupdo_ω = 0.5 * kintegrate(kG,nlQ_ch_rr.χ .- nlQ_sp_rr.χ,1)[1,:]
#     E_kin, E_pot = calc_E(Σ_ladder_i, kG, mP)
#     G_corr = transpose(LadderDGA.flatten_2D(LadderDGA.G_from_Σ(Σ_ladder_i, kG.ϵkGrid, νGrid, mP)));
#     E_pot2 = LadderDGA.calc_E_pot(kG, G_corr, Σ_ladder_i, E_pot_tail, E_pot_tail_inv, mP.β)
#     lhs_c1 = real(sum(χupup_ω))/mP.β - mP.Ekin_DMFT*mP.β/12
#     lhs_c2 = real(sum(χupdo_ω))/mP.β
#     rhs_c1 = mP.n/2 * (1 - mP.n/2)
#     rhs_c2 = E_pot/mP.U - (mP.n/2) * (mP.n/2)
#     c2_vals[i] = lhs_c2 - rhs_c2
# end

true
