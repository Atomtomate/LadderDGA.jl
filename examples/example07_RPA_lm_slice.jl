using Pkg
Pkg.activate("$(@__DIR__)/..")
using LadderDGA
using DataFrames, CSV

wp, mP, sP, env, kGridsStr = readConfig_RPA(ARGS[1])

df = DataFrame(U = Float64[], beta = Float64[], chi_m = Float64[], lm = Float64[], chi_m_lm = Float64[])

for beta in LinRange(0.1, 50.0, 2)
    mP.β = beta
    RPAhelper = setup_RPA!(kGridsStr, mP, sP);
    bubble     = calc_bubble(:RPA, RPAhelper);
    #bubble_RPA = χ₀RPA_T()
    χm, γm = calc_χγ(:m, RPAhelper, bubble);
    χd, γd = calc_χγ(:d, RPAhelper, bubble);
    λ₀ = calc_λ0(bubble, RPAhelper)

    res_m = LadderDGA.λ_correction(:m, χm, χd, RPAhelper)
    nh = ceil(Int,size(χm,2)/2)
    chi_m = χm[end, nh] 
    chi_m_lm = χ_λ(χm, res_m[1])[end, nh] 
    push!(df, [mP.U, mP.β, chi_m, res_m[1], chi_m_lm])
end

CSV.write(ARGS[2], df)
