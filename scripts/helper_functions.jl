using OffsetArrays
using ProgressMeter

function check_conditions(λGrid, μ::Float64, sc_it::Int, χm, χd, γm, γd, λ₀, h; output=true, fit_μ::Bool=true)
    Nq = size(χm,1)
    fft_νGrid = h.sP.fft_range
    ωindices, νGrid, iωn_f = LadderDGA.LambdaCorrection.gen_νω_indices(χm, χd, mP, sP)
    G_ladder::OffsetMatrix{ComplexF64, Matrix{ComplexF64}}      = OffsetArray(Matrix{ComplexF64}(undef, Nq, length(fft_νGrid)), 1:Nq, fft_νGrid) 
    Σ_ladder::OffsetMatrix{ComplexF64, Matrix{ComplexF64}}      = OffsetArray(Matrix{ComplexF64}(undef, Nq, length(νGrid)), 1:Nq, νGrid)
    Kνωq_pre = Vector{ComplexF64}(undef, Nq)
    PP_λdm_list = Matrix{Float64}(undef, size(λGrid)); PP_1_list = Matrix{Float64}(undef, size(λGrid)); EPot2_list = Matrix{Float64}(undef, size(λGrid)); EPot1_list = Matrix{Float64}(undef, size(λGrid))
    tail_factor = tail_factor(h.mP.U,h.mP.β,h.mP.n,h.Σ_loc,iν)

    @showprogress for (λi,λ) in enumerate(λGrid)
        λm, λd = λ
        χ_λ!(χm,λm)
        χ_λ!(χd,λd)
        PP_λdm    = 0.5 * (sum_kω(h.kG, χm) + 
                           sum_kω(h.kG, χd))
        PP_1      = h.mP.n/2 * (1 - h.mP.n/2)
        EPot2_λdm = h.mP.U * 0.5 * (sum_kω(h.kG, χd) - 
                                    sum_kω(h.kG, χm)) + h.mP.U * (h.mP.n/2)^2
        
        Σ_ladder = calc_Σ(χm, γm, χd, γd, λ₀, h);


        LadderDGA.calc_Σ!(Σ_ladder, Kνωq_pre, χm, γm, χd, γd, h.χloc_m_sum, λ₀, tail_factor, h.gLoc_rfft, h.kG, h.mP, h.sP; tc=true)
        μ_new = LadderDGA.G_from_Σladder!(G_ladder, Σ_ladder, h.Σ_loc, h.kG, h.mP; fix_n=fit_μ, μ=μ)
        E_kin_1, E_pot_1 = calc_E(G_ladder, Σ_ladder, μ, h.kG, h.mP)

        n_new = filling_pos(view(G_ladder, :, 0:h.sP.n_iν), h.kG, h.mP.U, μ_new, h.mP.β)
        reset!(χm)
        reset!(χd)
        if output
            println("checking conditions for λm = $λm, λd = $λd, μ = $μ, sc_it = $sc_it")
            println("PP      : ", PP_λdm, " ?=? ", PP_1)
            println("Epot    : ", E_pot_1, " ?=? ", EPot2_λdm) #, " ?=? ", run_res.EPot_p1, " ?=? ", run_res.EPot_p2
            println("filling : ", n_new, " ?=? ", h.mP.n, " // μ : ", μ_new, " ... ", μ)
        end
        PP_λdm_list[λi] = PP_λdm
        PP_1_list[λi]   = PP_1
        EPot2_list[λi] = EPot2_λdm
        EPot1_list[λi]  = E_pot_1
    end
    return PP_λdm_list, PP_1_list, EPot2_list, EPot1_list
end#

function gen_sc(λ::Tuple{Float64,Float64}; maxit::Int=100, with_tsc::Bool=false)
    χ_λ!(χm, λ[1])
    χ_λ!(χd, λ[2])
    res = run_sc(χm, γm, χd, γd, λ₀, 1.0, lDGAhelper; type=:fix, maxit=maxit, mixing=0.2, conv_abs=1e-8, trace=true, update_χ_tail=with_tsc)
    reset!(χm)
    reset!(χd)
    res.G_ladder = nothing
    res.Σ_ladder = nothing
    return res
end

function gen_sc_grid(λ_grid::Array{Tuple{Float64,Float64}}; maxit::Int=100, with_tsc::Bool=false)
    results = λ_result[]

    total = length(λm_grid)*length(λd_grid)
    println("running for grid size $total = $(length(λm_grid)) * $(length(λd_grid)) // (λm * λd)")
    @showprogress for λ in λ_grid
        λp = rpad.(lpad.(round.(λ,digits=1),4),4)
        #print("\r $(rpad(lpad(round(100.0*i/total,digits=2),5),8)) % done λ = $λp")
        res = gen_sc(λ, maxit=maxit, with_tsc=with_tsc)
        push!(results, res)
    end
    results = reshape(results, length(λm_grid), length(λd_grid))
    return results
end

