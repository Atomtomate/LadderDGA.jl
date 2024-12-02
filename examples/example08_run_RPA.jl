# ==================== Includes ====================
#nprocs() == 1 && addprocs(8)
using Pkg
using TimerOutputs
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

to2 = TimerOutput()

β = 26.790697674418606;
U = 1.8;
μ = 0.0;
nden = 1.0;
Nk = 40

mP = ModelParameters(U, μ, β, nden, NaN, NaN, NaN)

n_iω = 100;               # number of bosonic frequencies
n_iν = 100;               # number of fermionic frequencies
n_iν_shell = 0;           # Number of fermionic frequencies used for asymptotic sum improvement
shift = true             # shift of center for interval of bosonic frequencies
χ_helper = undef;         # Helper for χ asymptotics improvement

freq_r = 2*(n_iν + n_iω)
fft_range = -freq_r:freq_r
usable_prct_reduction = NaN;      # safety cutoff for usable ranges
dbg_full_eom_omega = true;
sP = SimulationParameters(n_iω,n_iν,n_iν_shell,shift,χ_helper,fft_range,usable_prct_reduction,dbg_full_eom_omega)


RPAhelper = setup_RPA(("3dsc-$(1/(2*sqrt(6)))", Nk), mP, sP);
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
# bubblett = calc_bubble(RPAhelper);
# bubble_testtt = LadderDGA.calc_bubble_test(RPAhelper);

@timeit to2 "intern" bubble = calc_bubble(RPAhelper);
@timeit to2 "test" bubble_test = LadderDGA.calc_bubble_test(RPAhelper);
bbl = dropdims(sum(bubble.data,dims=bubble.axis_types[:ν]) ./ mP.β, dims=2)


χm, γm = calc_χγ(:m, RPAhelper, bubble); # delta vertex = 1
χd, γd = calc_χγ(:d, RPAhelper, bubble);

λm, validation = LadderDGA.λ_correction(:m, χm, χd, RPAhelper);

# bubble.data; # sum over fermionic matsubara frequencies
# println("χ_0(Γ-point,ω):")
# println(bubble.data[begin,:])
# println("χ_0(R-point,ω):")
# println(bubble.data[end,:])

# for U in [1.5 1.6 1.7 1.8 1.9 2.0]
#     RPAhelper.mP = ModelParameters(U, μ, β, nden, NaN, NaN, NaN)
#     χm, γm = calc_χγ(:m, RPAhelper, bubble); # delta vertex = 1
#     χd, γd = calc_χγ(:d, RPAhelper, bubble);
#     # # λ₀ = calc_λ0(bubble, RPAhelper)
    
    
#     # # ==================== Results =====================
    
#     λm, validation = LadderDGA.λ_correction(:m, χm, χd, RPAhelper);
#     println("u=" * string(U) * ",λm=" * string(λm))
#     # # res_dm = λdm_correction(χm, γm, χd, γd, λ₀, RPAhelper; fit_μ=true)
#     # res_dm_sc = run_sc(χm, γm, χd, γd, λ₀, RPAhelper.mP.μ, RPAhelper; type=:pre_dm, fit_μ=true, maxit=100, mixing=0.2, conv_abs=1e-8, trace=true);
#     # # if isfinite(res_dm.λm) && isfinite(res_dm.λd)
#     # jldopen(joinpath(out_dir, file_name), "w") do f
#     #     f["RPAHelper"] = RPAhelper
#     #     f["χ0"] = bubble
#     #     f["χm"] = χm
#     #     f["χd"] = χd
#     #     f["res_m"] = res_m
#     #     # f["res_dm"] = res_dm
#     #     # f["res_dm_sc"] = res_dm_sc
#     # end
#     # # end
# end
