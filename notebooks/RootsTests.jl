using OffsetArrays, TimerOutputs, FiniteDiff, LinearAlgebra
using ForwardDiff
using Roots
λ₀T = LadderDGA.λ₀T
eom_ν_cutoff = LadderDGA.eom_ν_cutoff
ω2_tail= LadderDGA.ω2_tail
iν_array = LadderDGA.iν_array
tail_factor = LadderDGA.tail_factor
tail_correction_term = LadderDGA.tail_correction_term
get_λ_min = LadderDGA.LambdaCorrection.get_λ_min
newton_right = LadderDGA.LambdaCorrection.newton_right
λm_rhs = LadderDGA.LambdaCorrection.λm_rhs
λm_correction_val = LadderDGA.LambdaCorrection.λm_correction_val
calc_G_Σ! = LadderDGA.calc_G_Σ!
to = LadderDGA.to
newton_secular = LadderDGA.LambdaCorrection.newton_secular
default_Σ_tail_correction = LadderDGA.default_Σ_tail_correction
ω0_index = LadderDGA.ω0_index
KGrid = LadderDGA.KGrid
dχ_λ = LadderDGA.dχ_λ

function regula_falsi(f::Function, min::Float64, max::Float64; nsteps::Int = 500, atol::Float64=1e-8, δ::Float64=1e-4, verbose::Bool=true)::Float64
    done  = false
    a::Float64 = min
    b::Float64 = max
    c::Float64 = NaN
    i     = 1
    println("tt")
    f_a::Float64 = f(a)
    f_b::Float64 = f(b)
    sign(f_a) == sign(f_b) && error("[$min,$max] is not a bracketing interval")
    println("$f_a // $f_b")
    while !done
        lambda = f_b / (f_b - f_a)

        ϵ = √eps(Float64) / 100 # some engineering to avoid short moves; still fails on some
        ϵ ≤ lambda ≤ 1 - ϵ || (lambda = 1 / 2)
        c = b - lambda * (b - a)
        c   = (f_b*a - 0.5 * f_a * b)/(f_b - 0.5*f_a)
        f_c = f(c)
        verbose && println("[$a--$c--$b] -- [$f_a--$f_b--$f_c]")
        (norm(f_c) < atol) && (done = true)
        (i >= nsteps) && (done = true)
        
        if sign(f_c)  == sign(f_b)
            m = 1 - f_c/f_b
            f_a = (m > 0 ? m : 0.5) * f_a
        else
            a   = b
            f_a = f_b
        end
        b   = c
        f_b = f_c

        i += 1
    end
    println("nsteps = ", i-1)
    return c
end

function newton_right_test(f::Function, start::Float64, min::Float64; nsteps::Int=100, atol::Float64=1e-8, δ::Float64=1e-4, verbose::Bool=true)::Float64
    df(x) = FiniteDiff.finite_difference_derivative(f, x, typeof(Val(:forward)), Float64)
    newton_right_test(f, df, start, min; nsteps = nsteps, atol = atol, δ = δ, verbose=verbose)
end

function newton_right_test(f::Function, df::Function, start::Float64, min::Float64; nsteps::Int = 500, atol::Float64=1e-8, δ::Float64=1e-4, verbose::Bool=true)::Float64
    done  = false
    xlast = start + δ
    xi    = xlast
    fi_last = NaN
    i     = 1
    while !done
        fi = f(xi)
        dfii = 1 / df(xi)
        xi = xlast - dfii * fi
        # Found solution in the correct interval
        #verbose && println("$xlast - $dfii * $fi = $xi < $min")
        (abs(fi) < atol) && (xi > min) && (done = true)
        # only ever search to the right! bisect instead
        if xi < min
            verbose && println("reset, $xlast + $dfii * $fi = $xi < $min")
            xi = abs(xlast - (min + δ))/2 +  (min + δ)
        end
        xlast = xi
        (i >= nsteps) && (done = true)
        verbose && println("i = $i, xi = $xi, f(xi) = $fi")
        i += 1
    end
    #println("nsteps = ", i-1)
    return xi
end
Base.@assume_effects :total newton_secular_transform(x::Float64,p::Float64)::Float64 = sqrt(x)#-1/x^2 + p
Base.@assume_effects :total newton_secular_transform_df(x::Float64,p::Float64)::Float64 = 1 /(2*sqrt(x))#2 / (x^3)
function newton_secular_test(f::Function, df::Function, xp::Float64; nsteps::Int = 500, atol::Float64=1e-8)::Float64
    done::Bool  = false
    xi::Float64 = xp + 1.0
    xi_tf::Float64 = NaN
    i::Int         = 1
    while !done
        xi_tf = newton_secular_transform(xi,xp)
        fi = f(xi_tf)
        dfii = 1 / (df(xi_tf)*newton_secular_transform_df(xi, xp))
        xi = xi - dfii * fi
        # Found solution in the correct interval
        (norm(fi) < atol || i >= nsteps) && (done = true)
        i += 1
    end
    i >= nsteps && !done && @warn "Newton did not converge!"
    return xi_tf#inv_newton_secular_transform(xi,xp)
end
function λdm_correction_val_testRF(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::λ₀T, h; 
                        δλd::Float64=1e-1,
                        νmax::Int = eom_ν_cutoff(h), fix_n::Bool = true,tc::Type{<: ΣTail} = default_Σ_tail_correction(),
                        use_trivial_λmin::Bool = (tc === ΣTail_EoM),  λd_min::Float64 = NaN, 
                        validation_threshold::Float64 = 1e-8, max_steps_m::Int = 2000, 
                        max_steps_dm::Int = 2000, log_io = devnull)::Tuple{Float64,Float64}
    ωn2_tail = ω2_tail(χm)
    Nq::Int = length(h.kG.kMult)
    
    Kνωq_pre::Vector{ComplexF64} = Vector{ComplexF64}(undef, Nq)
    Σ_ladder = OffsetArray(Matrix{ComplexF64}(undef, Nq, νmax), 1:Nq, 0:νmax-1)
    G_ladder = similar(Σ_ladder)
    iν = iν_array(h.mP.β, collect(axes(Σ_ladder, 2)))
    tc_factor = tail_factor(tc, h.mP.U, h.mP.β, h.mP.n, h.Σ_loc, iν) 

    λd_min::Float64 = if !isnan(λd_min)
        λd_min
    else
        if use_trivial_λmin 
            get_λ_min(χd)
        else
            get_λd_min(χm, γm, χd, γd, λ₀, h)
        end
    end

    NF = 0
    
    function f_c2(λd_i::Float64)
        NF += 1
        rhs_c1,_ = λm_rhs(χm, χd, h; λd=λd_i)
        λm_i   = λm_correction_val(χm, rhs_c1, h.kG, ωn2_tail; max_steps=max_steps_m, eps=validation_threshold)
        tc_term  = (tc === ΣTail_EoM) ? h.χ_m_loc : tail_correction_term(sum_kω(h.kG, χm, λ=λm_i), h.χloc_m_sum, tc_factor)
        μ_new = calc_G_Σ!(G_ladder, Σ_ladder, Kνωq_pre, tc_term, χm, γm, χd, γd, λ₀, λm_i, λd_i, h, fix_n=fix_n)

        #TODO: use Epot_1
        Ekin_1, Epot_1 = calc_E(G_ladder, Σ_ladder, μ_new, h.kG, h.mP)
        Epot_2 = EPot_p2(χm, χd, λm_i, λd_i, h.mP.n, h.mP.U, h.kG)
        return  Epot_1 - Epot_2
    end
    
    λd  = regula_falsi(f_c2, λd_min +δλd,20.0; nsteps=max_steps_dm, atol=validation_threshold, verbose=false)
    println("Method: Regula Falsi. Result = ", λd, " /// NF = $NF" )
    rhs,_ = λm_rhs(χm, χd, h; λd=λd)
    λm  = λm_correction_val(χm, rhs, h; max_steps=max_steps_m, eps=validation_threshold)
    return λm, λd
end


function λdm_correction_val_testTF(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::λ₀T, h; 
                        δλd::Float64=1e-1,
                        νmax::Int = eom_ν_cutoff(h), fix_n::Bool = true,tc::Type{<: ΣTail} = default_Σ_tail_correction(),
                        use_trivial_λmin::Bool = (tc === ΣTail_EoM),  λd_min::Float64 = NaN, 
                        validation_threshold::Float64 = 1e-8, max_steps_m::Int = 2000, 
                        max_steps_dm::Int = 2000, log_io = devnull)::Tuple{Float64,Float64}
    ωn2_tail = ω2_tail(χm)
    Nq::Int = length(h.kG.kMult)
    
    Kνωq_pre::Vector{ComplexF64} = Vector{ComplexF64}(undef, Nq)
    Σ_ladder = OffsetArray(Matrix{ComplexF64}(undef, Nq, νmax), 1:Nq, 0:νmax-1)
    G_ladder = similar(Σ_ladder)
    iν = iν_array(h.mP.β, collect(axes(Σ_ladder, 2)))
    tc_factor = tail_factor(tc, h.mP.U, h.mP.β, h.mP.n, h.Σ_loc, iν) 

    λd_min::Float64 = if !isnan(λd_min)
        λd_min
    else
        if use_trivial_λmin 
            get_λ_min(χd)
        else
            get_λd_min(χm, γm, χd, γd, λ₀, h)
        end
    end

    NF = 0
    
    function f_c2(λd_i::Float64)
        NF += 1
        rhs_c1,_ = λm_rhs(χm, χd, h; λd=λd_i)
        λm_i   = λm_correction_val(χm, rhs_c1, h.kG, ωn2_tail; max_steps=max_steps_m, eps=validation_threshold)
        tc_term  = (tc === ΣTail_EoM) ? h.χ_m_loc : tail_correction_term(sum_kω(h.kG, χm, λ=λm_i), h.χloc_m_sum, tc_factor)
        μ_new = calc_G_Σ!(G_ladder, Σ_ladder, Kνωq_pre, tc_term, χm, γm, χd, γd, λ₀, λm_i, λd_i, h, fix_n=fix_n)

        #TODO: use Epot_1
        Ekin_1, Epot_1 = calc_E(G_ladder, Σ_ladder, μ_new, h.kG, h.mP)
        Epot_2 = EPot_p2(χm, χd, λm_i, λd_i, h.mP.n, h.mP.U, h.kG)
        return  Epot_1 - Epot_2
    end
    λd  = newton_secular(f_c2, λd_min; nsteps=max_steps_dm, atol=validation_threshold)
    println("Method: Secular. Result = ", λd, " /// NF = $NF" )
    rhs,_ = λm_rhs(χm, χd, h; λd=λd)
    λm  = λm_correction_val(χm, rhs, h; max_steps=max_steps_m, eps=validation_threshold)
    return λm, λd
end

function λdm_correction_val_test(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::λ₀T, h; 
                        δλd::Float64=1e-1,
                        νmax::Int = eom_ν_cutoff(h), fix_n::Bool = true,tc::Type{<: ΣTail} = default_Σ_tail_correction(),
                        use_trivial_λmin::Bool = (tc === ΣTail_EoM),  λd_min::Float64 = NaN, 
                        validation_threshold::Float64 = 1e-8, max_steps_m::Int = 2000, 
                        max_steps_dm::Int = 2000, log_io = devnull)::Tuple{Float64,Float64}
    ωn2_tail = ω2_tail(χm)
    Nq::Int = length(h.kG.kMult)
    
    Kνωq_pre::Vector{ComplexF64} = Vector{ComplexF64}(undef, Nq)
    Σ_ladder = OffsetArray(Matrix{ComplexF64}(undef, Nq, νmax), 1:Nq, 0:νmax-1)
    G_ladder = similar(Σ_ladder)
    iν = iν_array(h.mP.β, collect(axes(Σ_ladder, 2)))
    tc_factor = tail_factor(tc, h.mP.U, h.mP.β, h.mP.n, h.Σ_loc, iν) 

    λd_min::Float64 = if !isnan(λd_min)
        λd_min
    else
        if use_trivial_λmin 
            get_λ_min(χd)
        else
            get_λd_min(χm, γm, χd, γd, λ₀, h)
        end
    end
    
    NF = 0
    
    function f_c2(λd_i::Float64)
        NF += 1
        rhs_c1,_ = λm_rhs(χm, χd, h; λd=λd_i)
        λm_i   = λm_correction_val(χm, rhs_c1, h.kG, ωn2_tail; max_steps=max_steps_m, eps=validation_threshold)
        tc_term  = (tc === ΣTail_EoM) ? h.χ_m_loc : tail_correction_term(sum_kω(h.kG, χm, λ=λm_i), h.χloc_m_sum, tc_factor)
        μ_new = calc_G_Σ!(G_ladder, Σ_ladder, Kνωq_pre, tc_term, χm, γm, χd, γd, λ₀, λm_i, λd_i, h, fix_n=fix_n)

        #TODO: use Epot_1
        Ekin_1, Epot_1 = calc_E(G_ladder, Σ_ladder, μ_new, h.kG, h.mP)
        Epot_2 = EPot_p2(χm, χd, λm_i, λd_i, h.mP.n, h.mP.U, h.kG)
        return Epot_1 - Epot_2
    end
    λd  = newton_right_test(f_c2, λd_min+2.0, λd_min, verbose=false, nsteps=max_steps_dm, atol=validation_threshold, δ=δλd)
    println("Method: Reset. Result = ", λd, " /// NF = $NF" )
    rhs,_ = λm_rhs(χm, χd, h; λd=λd)
    λm  = λm_correction_val(χm, rhs, h; max_steps=max_steps_m, eps=validation_threshold)
    return λm, λd
end

function λdm_correction_val_MethodTest(method, χm::χT,γm::γT,χd::χT, γd::γT,λ₀::λ₀T, h; 
                         δλd::Float64=1e-1,
                        νmax::Int = eom_ν_cutoff(h), fix_n::Bool = true,tc::Type{<: ΣTail} = default_Σ_tail_correction(),
                        use_trivial_λmin::Bool = (tc === ΣTail_EoM),  λd_min::Float64 = NaN, 
                        validation_threshold::Float64 = 1e-8, max_steps_m::Int = 2000, 
                        max_steps_dm::Int = 2000, log_io = devnull)::Tuple{Float64,Float64}
    ωn2_tail = ω2_tail(χm)
    Nq::Int = length(h.kG.kMult)
    
    Kνωq_pre::Vector{ComplexF64} = Vector{ComplexF64}(undef, Nq)
    Σ_ladder = OffsetArray(Matrix{ComplexF64}(undef, Nq, νmax), 1:Nq, 0:νmax-1)
    G_ladder = similar(Σ_ladder)
    iν = iν_array(h.mP.β, collect(axes(Σ_ladder, 2)))
    tc_factor = tail_factor(tc, h.mP.U, h.mP.β, h.mP.n, h.Σ_loc, iν) 

    λd_min::Float64 = if !isnan(λd_min)
        λd_min
    else
        if use_trivial_λmin 
            get_λ_min(χd)
        else
            get_λd_min(χm, γm, χd, γd, λ₀, h)
        end
    end

    NF = 0

    function f_c2(λd_i::Float64; dbg::Bool=false)
        NF += 1
        rhs_c1,_ = λm_rhs(χm, χd, h; λd=λd_i)
        λm_i   = λm_correction_val(χm, rhs_c1, h.kG, ωn2_tail; max_steps=max_steps_m, eps=validation_threshold)
        tc_term  = (tc === ΣTail_EoM) ? h.χ_m_loc : tail_correction_term(sum_kω(h.kG, χm, λ=λm_i), h.χloc_m_sum, tc_factor)
        μ_new = calc_G_Σ!(G_ladder, Σ_ladder, Kνωq_pre, tc_term, χm, γm, χd, γd, λ₀, λm_i, λd_i, h, fix_n=fix_n)

        #TODO: use Epot_1
        Ekin_1, Epot_1 = calc_E(G_ladder, Σ_ladder, μ_new, h.kG, h.mP)
        Epot_2 = EPot_p2(χm, χd, λm_i, λd_i, h.mP.n, h.mP.U, h.kG)
        dbg && println("   -> λm = $λm_i // $Epot_1 - $Epot_2 // $μ_new")
        return Epot_1 - Epot_2
    end
    
    #println("dbg [a,b]: [", λd_min + 1e-3, ", ",200.0,"]")
    #println("dbg [f(a),f(b)]: (", f_c2(λd_min + 1e-3, dbg=:true), ", ",f_c2(200.0, dbg=:true),")")
    df(x) = FiniteDiff.finite_difference_derivative(f_c2, x, typeof(Val(:forward)), Float64)
    λd  = find_zero(f_c2, (λd_min + δλd, 200.0), method; atol=validation_threshold, maxiters=max_steps_dm)
    println("Method: $method. Result = ", λd, " /// NF = $NF" )
    rhs,_ = λm_rhs(χm, χd, h; λd=λd)
    λm  = λm_correction_val(χm, rhs, h; max_steps=max_steps_m, eps=validation_threshold)
    return λm, λd
end

function λm_correction_val_secular(χm::χT, rhs::Float64, kG::KGrid, ωn2_tail::Vector{Float64}; 
                           max_steps::Int=1000, eps::Float64=1e-8,δλ=1e-4)
    λm_min = get_λ_min(χm)
    χr::SubArray{Float64,2} = view(χm, :, χm.usable_ω)

    nc = 0
    function f_c1(λint::Float64)::Float64 
        nc += 1
        sum_kω(kG, χr, χm.β, χm.tail_c[3], ωn2_tail; transform = (f(x::Float64)::Float64 = χ_λ(x, λint))) - rhs
    end
    function df_c1(λint::Float64)::Float64
        nc += 1
        sum_kω(kG, χr, χm.β, χm.tail_c[3], ωn2_tail; transform = (f(x::Float64)::Float64 = dχ_λ(x, λint)))
    end
    λm = newton_secular(f_c1, df_c1, λm_min; nsteps=max_steps, atol=eps)
    println("nsteps = $nc")
    return λm
end
function λm_correction_val_reset(χm::χT, rhs::Float64, kG::KGrid, ωn2_tail::Vector{Float64}; 
                           max_steps::Int=1000, eps::Float64=1e-8,δλ=1e-4)
    λm_min = get_λ_min(χm)
    χr::SubArray{Float64,2} = view(χm, :, χm.usable_ω)
    nc = 0
    function f_c1(λint::Float64)::Float64 
        nc += 1
        sum_kω(kG, χr, χm.β, χm.tail_c[3], ωn2_tail; transform = (f(x::Float64)::Float64 = χ_λ(x, λint))) - rhs
    end
    function df_c1(λint::Float64)::Float64
        nc += 1
        sum_kω(kG, χr, χm.β, χm.tail_c[3], ωn2_tail; transform = (f(x::Float64)::Float64 = dχ_λ(x, λint)))
    end
    λm  = newton_right_test(f_c1, df_c1, λm_min+10.0, λm_min, verbose=false, nsteps=max_steps,  atol=eps, δ=δλ)
    println("nsteps = $nc")
    return λm
end
function λm_correction_val_MethodTest(method, χm::χT, rhs::Float64, kG::KGrid, ωn2_tail::Vector{Float64}; 
                           max_steps::Int=1000, eps::Float64=1e-8,δλ=1e-4)
    λm_min = get_λ_min(χm)
    χr::SubArray{Float64,2} = view(χm, :, χm.usable_ω)
    nc = 0
    function f_c1(λint::Float64)::Float64 
        nc += 1
        sum_kω(kG, χr, χm.β, χm.tail_c[3], ωn2_tail; transform = (f(x::Float64)::Float64 = χ_λ(x, λint))) - rhs
    end
    function df_c1(λint::Float64)::Float64
        nc += 1
        sum_kω(kG, χr, χm.β, χm.tail_c[3], ωn2_tail; transform = (f(x::Float64)::Float64 = dχ_λ(x, λint)))
    end
    λm  = find_zero(f_c1, (λm_min + δλ, 200.0), method; atol=eps)
    println("nsteps = $nc")
    return λm
end
function λm_correction_val_MethodTest2(method, χm::χT, rhs::Float64, kG::KGrid, ωn2_tail::Vector{Float64}; 
                           max_steps::Int=1000, eps::Float64=1e-8,δλ=1e-4)
    λm_min = get_λ_min(χm)
    χr::SubArray{Float64,2} = view(χm, :, χm.usable_ω)

    nc = 0
    function f_c1(λint::Float64)::Float64 
        nc += 1
        sum_kω(kG, χr, χm.β, χm.tail_c[3], ωn2_tail; transform = (f(x::Float64)::Float64 = χ_λ(x, λint))) - rhs
    end
    function df_c1(λint::Float64)::Float64
        nc += 1
        sum_kω(kG, χr, χm.β, χm.tail_c[3], ωn2_tail; transform = (f(x::Float64)::Float64 = dχ_λ(x, λint)))
    end
    λm  = find_zero((f_c1, df_c1), (λm_min + δλ, 200.0), method; atol=eps)
    println("nsteps = $nc")
    return λm
end