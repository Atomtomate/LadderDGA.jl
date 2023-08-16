# ==================== Includes ====================
#nprocs() == 1 && addprocs(8)
using Pkg
path = joinpath(abspath(@__DIR__), "..")
println("activating: ", path)
Pkg.activate(path)
Pkg.instantiate()
using LadderDGA
# using Plots
# using LaTeXStrings
using JLD2

cfg = ARGS[1]
out_dir = splitdir(cfg)[1]

β = 20.0;
U = 1.0;
μ = 0.0;
nden = 1.0;
Nk = 16
mP = ModelParameters(U, μ, β, nden, NaN, NaN, NaN)

n_iω = 1;                # number of bosonic frequencies
n_iν = 400;                # number of fermionic frequencies
n_iν_shell = 0;           # Number of fermionic frequencies used for asymptotic sum improvement
shift = false             # shift of center for interval of bosonic frequencies
χ_helper = undef;       # Helper for χ asymptotics improvement
fft_range = Array{Float64}(undef, 2);
usable_prct_reduction = NaN;      # safety cutoff for usable ranges
dbg_full_eom_omega = false;
sP = SimulationParameters(n_iω,n_iν,n_iν_shell,shift,χ_helper,fft_range,usable_prct_reduction,dbg_full_eom_omega)


RPAhelper = setup_RPA(("3dsc-" * string(1/(2*sqrt(6))), Nk), mP, sP);
Nk = RPAhelper.kG.Ns
Nω = 2 * RPAhelper.sP.n_iω + 1

file_name = "test_res_rpa_NK$(Nk)_Nw$(Nω).jld2"
output_file = joinpath(out_dir, file_name)
if isfile(output_file)
    println("Output file exists, aborting.")
    exit(1)
end
println("output file location: ", output_file)
flush(stdout)


# ====================== lDGA ======================
bubble = calc_bubble(RPAhelper);
bubble.data; # sum over fermionic matsubara frequencies
println("χ_0(Γ-point,ω):")
println(bubble.data[begin,:])
println("χ_0(R-point,ω):")
println(bubble.data[end,:])

χm, γm = calc_χγ(:m, RPAhelper, bubble); # delta vertex = 1
χd, γd = calc_χγ(:d, RPAhelper, bubble);
println("done")
# # λ₀ = calc_λ0(bubble, RPAhelper)


# # ==================== Results =====================

# res_m = LadderDGA.λ_correction(:m, χm, γm, χd, γd, λ₀, RPAhelper)
# # res_dm = λdm_correction(χm, γm, χd, γd, λ₀, RPAhelper; fit_μ=true)
# # res_dm_sc = run_sc(χm, γm, χd, γd, λ₀, RPAhelper.mP.μ, RPAhelper; type=:pre_dm, fit_μ=true, maxit=100, mixing=0.2, conv_abs=1e-8, trace=true);
# # if isfinite(res_dm.λm) && isfinite(res_dm.λd)
# jldopen(joinpath(out_dir, file_name), "w") do f
#     f["RPAHelper"] = RPAhelper
#     f["χ0"] = bubble
#     f["χm"] = χm
#     f["χd"] = χd
#     f["res_m"] = res_m
#     # f["res_dm"] = res_dm
#     # f["res_dm_sc"] = res_dm_sc
# end
# # end
