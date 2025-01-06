using Pkg
Pkg.activate("/scratch/projects/hhp00048/codes/LadderDGA.jl")
using LadderDGA, JLD2
using DataFrames, CSV
using LinearAlgebra

prefix = ARGS[1]
out_fn = ARGS[2]

function findmax_vec(vec::AbstractVector)
    any(vec .< 0) ? findmin(vec) : findmax(vec)
end

function find_k_N_AN(kG) #LadderDGA.lin_fit
    _, ki_AN  = findmin(x->norm(x .- [π, 0]), kG.kGrid)
    _, ki_N  = findmin(x->norm(x .- [π/2,π/2]), kG.kGrid)
    return ki_N, ki_AN
end

function find_k_max_min(Σ_ladder, efmask, kG, mP) #LadderDGA.lin_fit
    tmp    =  Σ_ladder[:,0] .- (Inf + Inf*im) .* (.! efmask)
    Σ_AN, ki_AN  = findmax(imag(tmp))
    tmp[:]  =  Σ_ladder[:,0] .+ (Inf + Inf*im) .* (.! efmask)
    Σ_N, ki_N  = findmin(imag(tmp))
    return Σ_ladder[ki_N,0], ki_N, Σ_ladder[ki_AN,0], ki_AN
end

function run_gather(prefix, out_fn)
    nrows = 0
    kgrids = Dict{Int, LadderDGA.KGrid}()
    df = DataFrame(U=Float64[], β=Float64[], Nk=Int[], Nf=Int[], μ_DMFT=Float64[], n=Float64[], EKin_DMFT_p1=Float64[], EKin_DMFT_p2=Float64[], EPot_DMFT_p1=Float64[], EPot_DMFT_p2=Float64[], Σ_tc=String[],
                   qi_χm_max=Int[], qi_χd_max=Int[], χm_max=Float64[], χd_max=Float64[], χm_AF=Float64[], χm_FM=Float64[], χd_0=Float64[],
                   qi_χm_min=Int[], qi_χd_min=Int[], χm_min=Float64[], χd_min=Float64[],
    #
                   λm_m=Float64[], λm_d=Float64[], λm_m_broken=Bool[], λm_d_broken=Bool[], qi_χm_λm_max=Int[], qi_χd_λm_max=Int[], μ_m=Float64[],
                   EKin_p1_m=Float64[], EKin_p2_m=Float64[], EPot_p1_m=Float64[], EPot_p2_m=Float64[], PP_p1_m=Float64[], PP_p2_m=Float64[], 
                   ki_Σ_m_0_N=Int[], ki_Σ_m_0_AN=Int[], Σ_m_0_N=ComplexF64[], Σ_m_0_AN=ComplexF64[], 
                   ki_Σ_m_lf_N=Int[], ki_Σ_m_lf_AN=Int[], Σ_m_lf_N=ComplexF64[], Σ_m_lf_AN=ComplexF64[], conn_m=Float64[],
    #
                   λdm_m=Float64[], λdm_d=Float64[], λdm_m_broken=Bool[], λdm_d_broken=Bool[], qi_χm_λdm_max=Int[], qi_χd_λdm_max=Int[], μ_dm=Float64[], EKin_p1_dm=Float64[],EKin_p2_dm=Float64[],  EPot_p1_dm=Float64[], EPot_p2_dm=Float64[], PP_p1_dm=Float64[], PP_p2_dm=Float64[], 
                   ki_Σ_dm_0_N=Int[], ki_Σ_dm_0_AN=Int[], Σ_dm_0_N=ComplexF64[], Σ_dm_0_AN=ComplexF64[], 
                   ki_Σ_dm_lf_N=Int[], ki_Σ_dm_lf_AN=Int[], Σ_dm_lf_N=ComplexF64[], Σ_dm_lf_AN=ComplexF64[], conn_dm=Float64[], 
    #
                   λm_sc_m=Float64[], λm_sc_d=Float64[], λm_sc_m_broken=Bool[], λm_sc_d_broken=Bool[], qi_χm_λm_sc_max=Int[], qi_χd_λm_sc_max=Int[], μ_m_sc=Float64[], EKin_p1_m_sc=Float64[],EKin_p2_m_sc=Float64[],  EPot_p1_m_sc=Float64[], EPot_p2_m_sc=Float64[], PP_p1_m_sc=Float64[], PP_p2_m_sc=Float64[], 
                   ki_Σ_m_sc_0_N=Int[], ki_Σ_m_sc_0_AN=Int[], Σ_m_sc_0_N=ComplexF64[], Σ_m_sc_0_AN=ComplexF64[],
                   ki_Σ_m_sc_lf_N=Int[], ki_Σ_m_sc_lf_AN=Int[], Σ_m_sc_lf_N=ComplexF64[], Σ_m_sc_lf_AN=ComplexF64[], conn_m_sc=Float64[],
    #
                   λdm_sc_m=Float64[], λdm_sc_d=Float64[], λdm_sc_m_broken=Bool[], λdm_sc_d_broken=Bool[],  qi_χm_λdm_sc_max=Int[], qi_χd_λdm_sc_max=Int[], μ_dm_sc=Float64[], EKin_p1_dm_sc=Float64[], EKin_p2_dm_sc=Float64[], EPot_p1_dm_sc=Float64[], EPot_p2_dm_sc=Float64[], PP_p1_dm_sc=Float64[], PP_p2_dm_sc=Float64[], 
                   ki_Σ_dm_sc_0_N=Int[], ki_Σ_dm_sc_0_AN=Int[], Σ_dm_sc_0_N=ComplexF64[], Σ_dm_sc_0_AN=ComplexF64[], 
                   ki_Σ_dm_sc_lf_N=Int[], ki_Σ_dm_sc_lf_AN=Int[], Σ_dm_sc_lf_N=ComplexF64[], Σ_dm_sc_lf_AN=ComplexF64[], conn_dm_sc=Float64[],)
    #
                   #λm_tsc_m=Float64[], λm_tsc_d=Float64[], qi_χm_λm_tsc_max=Int[], qi_χd_λm_tsc_max=Int[], μ_m_tsc=Float64[], EKin_p1_m_tsc=Float64[], EKin_p2_m_tsc=Float64[], EPot_p1_m_tsc=Float64[], EPot_p2_m_tsc=Float64[], PP_p1_m_tsc=Float64[], PP_p2_m_tsc=Float64[], 
                   #ki_Σ_m_tsc_0_N=Int[], ki_Σ_m_tsc_0_AN=Int[], Σ_m_tsc_0_N=ComplexF64[], Σ_m_tsc_0_AN=ComplexF64[],
                   #ki_Σ_m_tsc_lf_N=Int[], ki_Σ_m_tsc_lf_AN=Int[], Σ_m_tsc_lf_N=ComplexF64[], Σ_m_tsc_lf_AN=ComplexF64[], conn_m_tsc=Float64[],
    ##
                   #λdm_tsc_m=Float64[], λdm_tsc_d=Float64[], qi_χm_λdm_tsc_max=Int[], qi_χd_λdm_tsc_max=Int[], μ_dm_tsc=Float64[], EKin_p1_dm_tsc=Float64[], EKin_p2_dm_tsc=Float64[], EPot_p1_dm_tsc=Float64[], EPot_p2_dm_tsc=Float64[], PP_p1_dm_tsc=Float64[], PP_p2_dm_tsc=Float64[], 
                   #ki_Σ_dm_tsc_0_N=Int[], ki_Σ_dm_tsc_0_AN=Int[], Σ_dm_tsc_0_N=ComplexF64[], Σ_dm_tsc_0_AN=ComplexF64[],
                   #ki_Σ_dm_tsc_lf_N=Int[], ki_Σ_dm_tsc_lf_AN=Int[], Σ_dm_tsc_lf_N=ComplexF64[], Σ_dm_tsc_lf_AN=ComplexF64[], conn_dm_tsc=Float64[],
    ##
                   #)  

                   #         [res.λm, res.λd, qi_λ_m_max, qi_λ_d_max, res.μ, res.EKin_p1, res.EPot_p1, res.EPot_p2, res.PP_p1, res.PP_p2, 
                   #             ki_N, ki_AN, Σ_N, Σ_AN, ki_N_lf, ki_AN_lf, Σ_N_lf, Σ_AN_lf, rc]
    # "res_ldga.jld2"
    for (root, dirs, files) in walkdir(".")
        if endswith(root, "lDGA_julia")
            filelist = filter(f-> startswith(f, prefix) && endswith(f, ".jld2"), files)
            for ldga_fn in filelist
                println("[$nrows] adding dir: ", root, "/", ldga_fn)
                ki_N = 0
                ki_AN = 0

                file_ok = false
                try
                    jldopen(joinpath(root, ldga_fn))
                    file_ok = true
                catch e
                    println("Error $e , skipping file")
                    file_ok = false
                end
                if file_ok
                nrows += 1
                # try
                row = jldopen(joinpath(root, ldga_fn), "r") do f
                    nh = ceil(Int,size(f["chi_d"],2)/2)
                    sl = findfirst("Nk", ldga_fn)
                    tc_type = f["Sigma_tc"]
                    sle = findfirst(".jld2", ldga_fn)
                    Nk = parse(Int, ldga_fn[last(sl)+1:first(sle)-1])
                    kG = if haskey(kgrids, Nk) 
                        kgrids[Nk]
                        else
                            kGi = LadderDGA.gen_kGrid("2Dsc-0.25-0.05-0.025", Nk)
                            kgrids[Nk] = kGi
                            kGi
                    end
                    mP = ModelParameters(f["U"], f["μ_DMFT"], f["β"], f["n"], f["Epot_1Pt"], f["Ekin_1Pt"])
                    χm = f["chi_m"]
                    χd = f["chi_d"]
                    χm_max, qi_m_max = findmax(χm[:,nh])
                    χd_max, qi_d_max = findmax(χd[:,nh])
                    χm_λ_min, qi_m_min = findmin(1 ./ χm[:,nh])
                    χd_λ_min, qi_d_min = findmin(1 ./ χd[:,nh])

                    χm_AF   = χm[end,nh]
                    χm_FM   = χm[1,nh]
                    χd_0    = χd[1,nh]
                    Nf = size(χm, 2)
                    Ekin_p2, Epot_p2 = if haskey(f, "Ekin_2Pt")
                        f["Ekin_2Pt"], f["Epot_2Pt"]
                    else
                        NaN, NaN
                    end

                    row = [mP.U, mP.β, Nk, Nf, mP.μ, mP.n, mP.Ekin_1Pt, Ekin_p2, mP.Epot_1Pt, Epot_p2, tc_type, qi_m_max, qi_d_max, χm_max, χd_max, 
                                χm_AF, χm_FM, χd_0, qi_m_min, qi_d_min, χm_λ_min, χd_λ_min]

                    mode=LadderDGA.zero_freq
                    res_names = ["res_m", "res_dm", "res_m_sc", "res_dm_sc"]#, "res_m_tsc", "res_dm_tsc"]
                    for ri in res_names
                        row_i = if !haskey(f, ri) || isnothing(f[ri])
                            !(ri  in ["res_m_sc"]) && println("   -> WARNING: $ri not found!")
                            [NaN, NaN, true, true, -1, -1, repeat([NaN], 7)..., -1, -1, NaN, NaN, -1, -1, NaN, NaN, NaN]
                        else
                            res = f[ri]
                            χm_λ = 1 ./ ( 1 ./ χm .+ res.λm)
                            χd_λ = 1 ./ ( 1 ./ χd .+ res.λd)
                            λm_broken = any(χm_λ[:,nh] .< 0)
                            λd_broken = any(χd_λ[:,nh] .< 0)
                            Σ_N, ki_N, Σ_AN, ki_AN, Σ_N_lf, ki_N_lf, Σ_AN_lf, ki_AN_lf, rc = if !isnothing(res.Σ_ladder) 
                                # efmask, rc = LadderDGA.estimate_connected_ef(res.Σ_ladder, kG, mP.μ, mP.β; ν0_estimator=mode)
                                ki_N, ki_AN = find_k_N_AN(kG)
                                Σ_N = res.Σ_ladder[ki_N,0]
                                Σ_AN = res.Σ_ladder[ki_AN,0]
                                Σ_N, ki_N, Σ_AN, ki_AN, NaN, -1, NaN, -1, NaN
                            else
                                (NaN, -1, NaN, -1, NaN, -1, NaN, -1, NaN)
                            end
                            χm_λ_max, qi_λ_m_max = if isnothing(res)
                                        NaN, -1
                                    else
                                        findmax(1 ./ (1 ./ χm[:,nh] .+ res.λm))
                                    end
                            χd_λ_max, qi_λ_d_max = if isnothing(res) 
                                        NaN, -1
                                    else
                                        findmax(1 ./ (1 ./ χd[:,nh] .+ res.λd))
                                    end
                            [res.λm, res.λd, λm_broken, λd_broken, qi_λ_m_max, qi_λ_d_max, res.μ, res.EKin_p1, res.EKin_p2, res.EPot_p1, res.EPot_p2, res.PP_p1, res.PP_p2, 
                                ki_N, ki_AN, Σ_N, Σ_AN, ki_N_lf, ki_AN_lf, Σ_N_lf, Σ_AN_lf, rc]
                        end
                        row = vcat(row, row_i)
                    end
                    row
                end
                    push!(df, row)
                # catch e
                #     println("caught $e for row, skipping insertion of data!")
                # end
                end
            end
        end
    end
    println(df)
    CSV.write(out_fn, df)
end

run_gather(prefix, out_fn)
