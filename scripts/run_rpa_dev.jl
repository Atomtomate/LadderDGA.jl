using Pkg
using Plots
using TimerOutputs
path = joinpath(abspath(@__DIR__), "..")
println("activating: ", path)
Pkg.activate(path)
Pkg.instantiate()
using LadderDGA

χ₀, wP, mP, sP, env, kGridStr = readConfig_RPA("/home/coding/LadderDGA.jl/test/test_data/config_rpa_example.toml");

mP.U = 1.2
mP.μ = mP.U * mP.n / 2 # enforce half filling
mP.n = 1.0

# --------- pull into setup_RPA --------- 
kG = gen_kGrid(kGridStr, χ₀.Nq)

if length(LadderDGA.gridPoints(kG)) ≠ size(χ₀.data)[χ₀.axis_types[:q]]
    error("Number of q points in kGrid does not match number of q-points in χ₀!")
end
# ---------------------------------------

χm, γm = calc_χγ(:m, χ₀, mP, sP);
χd, γd = calc_χγ(:d, χ₀, mP, sP);
helper = setup_RPA(kG, mP, sP, χm; silent=false)

# ---------- pull into a test -----------
println(all(γm .== 1))               # triangular vertex is identity
println(all(γd .== 1))               # triangular vertex is identity

println(count(χd[begin, :] ≠ 0) == 1)            # χd(Γ, ω≠0) = 0
println(count(χm[begin, :] ≠ 0) == 1)            # χm(Γ, ω≠0) = 0
println(χm[begin, LadderDGA.ω0_index(sP)] ≠ 0)   # χd(Γ, ω=0) ≠ 0
println(χm[begin, LadderDGA.ω0_index(sP)] ≠ 0)   # χm(Γ, ω=0) ≠ 0
# ---------------------------------------

λ₀ = calc_λ0(χ₀, helper)
# ---------- pull into a test -----------
println(maximum(real.(-λ₀[begin, begin, :])) / mP.U ≈ χ₀[begin, LadderDGA.ω0_index(sP)])    # check value of λ₀(q, ν₁, ω) = -U⋅χ₀(q, ω)
println(count(abs.(λ₀[begin, begin, :]) ≠ 0) == 1)                                          # check that for ω=0 and at the Γ-point only one contribution does not vanish 
# ---------------------------------------

# λ_result = LadderDGA.λm_correction_full_RPA(χm, χd, helper; verbose=true, validate_threshold=1e-8)

# Σ_tc_true  = calc_Σ(χm, γm, χd, γd, sum_kω(kG, χm), λ₀, helper.gLoc_fft, kG, mP, sP; νmax=sP.n_iν, λm=0.0, λd=0.0, tc=true)
Σ_tc_false = calc_Σ(χm, γm, χd, γd, sum_kω(kG, χm), λ₀, helper.gLoc_fft, kG, mP, sP; νmax=sP.n_iν, λm=0.0, λd=0.0, tc=false)


println("done.")
plot(imag.(Σ_tc_false[begin,begin+5:end]), marker=:circle) # for mP.U = 1.2 mP.μ = mP.U * mP.n / 2 # enforce half filling mP.n = 1.0

res_dm = λdm_correction(χm, γm, χd, γd, helper.Σ_loc, helper.gLoc_rfft, helper.χloc_m_sum, λ₀, kG, mP, sP; fit_μ=true, tc=false, verbose=true)

