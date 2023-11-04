using Pkg
using TimerOutputs
using DataFrames
using CSV
path = "/home/coding/LadderDGA.jl/" # joinpath(abspath(@__DIR__),"..")
println("activating: ", path)
Pkg.activate(path)
# Pkg.instantiate()
using LadderDGA

cfg = ARGS[1]
output_file = splitext(cfg)[1] * ".csv"

"""
    max_index(kG::KGrid)

Max integer position along a dimension
"""
function max_index(kG::LadderDGA.KGrid)
    return Int(kG.Ns/2)
end

"""
    to_1d_index(n₁::Int, n₂::Int, n₃::Int, kG::KGrid)

Maps integer position to 1d index in reduced grid array
"""
function to_1d_index(n₁::Int, n₂::Int, n₃::Int, kG::LadderDGA.KGrid)
    if (max_index(kG) ≥ n₁) && (n₁ ≥ n₂) && (n₂ ≥ n₃) && (n₃ ≥ 0)
        return Int(n₁*(n₁ + 1)*(n₁ + 2) / 6 + n₂*(n₂ + 1) / 2 + n₃ + 1)
    else
        error("require n₁ ≥ n₂ ≥ n₃")
    end
end

function R_index(kG)
    return to_1d_index(max_index(kG), max_index(kG), max_index(kG), kG)
end

χ₀, wP, mP, sP, env, kGridStr = readConfig_RPA(cfg);

println("u=$(mP.U)\nμ=$(mP.μ)")

# --------- pull into setup_RPA --------- 
kG = gen_kGrid(kGridStr, χ₀.Nq)

if length(LadderDGA.gridPoints(kG)) ≠ size(χ₀.data)[χ₀.axis_types[:q]]
    error("Number of q points in kGrid does not match number of q-points in χ₀!")
end
# ---------------------------------------

χm, γm = calc_χγ(:m, χ₀, mP, sP);
χd, γd = calc_χγ(:d, χ₀, mP, sP);
helper = setup_RPA(kG, mP, sP, χ₀)

if count(χd[begin, :] ≠ 0) ≠ 1 || count(χm[begin, :] ≠ 0) ≠ 1 || χm[begin, LadderDGA.ω0_index(sP)] ≈ 0 || χd[begin, LadderDGA.ω0_index(sP)] ≈ 0
   error("Wrong input data. Stop!")
end

if isfile(output_file)
    println("Output file exists, aborting.")
    exit(1)
end
println("output file location: ", output_file)
flush(stdout)

λ₀ = calc_λ0(χ₀, helper)

λm_result = LadderDGA.λm_correction_full_RPA(χm, χd, helper; verbose=true, validate_threshold=1e-8)
λdm_result = λdm_correction(χm, γm, χd, γd, helper.Σ_loc, helper.gLoc_rfft, helper.χloc_m_sum, λ₀, kG, mP, sP; fit_μ=true , tc=true , verbose=true, rpa=true)

# helper
r_point = R_index(kG)
ω_0     = LadderDGA.ω0_index(sP)

# technical stuff
ngl = χ₀.Ngl
Nq = kG.Ns

# thermodynamics
β = mP.β
T = 1.0/β
μ = λdm_result.μ
n = λdm_result.n

# rpa result
chi0 = χ₀[r_point, ω_0]
chim = χm[r_point, ω_0]
chid = χd[r_point, ω_0]

# lambda m correction results
λm_converged = λm_result.converged

λm_m = λm_result.λm
λd_m = 0.0

χm_λm = χ_λ(χm, λm_m)[r_point, ω_0]
χd_λm = χd[r_point, ω_0]

# lambda dm correction results
λdm_converged = λdm_result.converged

λm_dm = λdm_result.λm
λd_dm = λdm_result.λd

if isfinite(λdm_result.λm) && isfinite(λdm_result.λd)
    χm_dm = χ_λ(χm, λm_dm)[r_point, ω_0]
    χd_dm = χ_λ(χd, λd_dm)[r_point, ω_0]

    EKin_dm    = λdm_result.EKin
    EPot_p1_dm = λdm_result.EPot_p1
    EPot_p2_dm = λdm_result.EPot_p2
    PP_p1_dm   = λdm_result.PP_p1
    PP_p2_dm   = λdm_result.PP_p2
else
    χm_dm = NaN
    χd_dm = NaN

    EKin_dm    = NaN
    EPot_p1_dm = NaN
    EPot_p2_dm = NaN
    PP_p1_dm   = NaN
    PP_p2_dm   = NaN
end

df = DataFrame(
    # macro state
    "T" => T,
    "μ" => μ,
    "n" => n,
    # rpa
    "chi0" => chi0,
    "chim" => chim,
    "chid" => chid,
    # λm-correction
    "m_converged"    => λm_converged,
    "lambda_m_m"     => λm_m,
    "lambda_d_m"     => λd_m,
    "chi_m_lambda_m" => χm_λm,
    "chi_d_lambda_m" => χd_λm,
    # λdm-correction
    "dm_converged" => λdm_converged,
    "lambda_m_dm"  => λm_dm,
    "lambda_d_dm"  => λd_dm,
    "chi_m_dm"     => χm_dm,
    "chi_d_dm"     => χd_dm,
    # additional info about λdm-correction
    "EKin_dm"    => EKin_dm,
    "EPot_p1_dm" => EPot_p1_dm,
    "EPot_p2_dm" => EPot_p2_dm,
    "PP_p1_dm"   => PP_p1_dm,
    "PP_p2_dm"   => PP_p2_dm,
    # simulation parameters
    "Ngl"           => ngl,
    "Nq"            => Nq,
    "max_omega_int" => sP.n_iω
)

CSV.write(output_file, df)