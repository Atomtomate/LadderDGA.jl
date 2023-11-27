function λ_correction(
    type::Symbol,
    χm::χT,
    χd::χT,
    h::RPAHelper;
    # common options
    verbose::Bool = false,
    validate_threshold::Float64 = 1e-8,
)
    if type == :m
        λm_correction_RPA(χm, χd, h; verbose = verbose, validate_threshold = validate_threshold)
    else
        error("RPA: λ-correction type '$type' not recognized!")
    end
end

function λm_correction_RPA(χm::χT, χd::χT, h::RPAHelper; verbose::Bool = false, validate_threshold::Float64 = 1e-8)

    kG::KGrid = h.kG
    rhs =
        h.mP.n * (1 - 0.5 * h.mP.n) -
        sum_kω(kG, χd, χd.β, 0.0, zeros(Float64, length(χd.usable_ω)); transform = nothing)

    λm_min = get_λ_min(χm)
    χr::SubArray{Float64,2} = view(χm, :, χm.usable_ω)
    iωn = (1im .* 2 .* (-h.sP.n_iω:h.sP.n_iω)[χm.usable_ω] .* π ./ h.mP.β)

    f_c1(λint::Float64)::Float64 =
        sum_kω(
            kG,
            χr,
            χm.β,
            0.0,
            zeros(Float64, length(χd.usable_ω));
            transform = (f(x::Float64)::Float64 = χ_λ(x, λint)),
        ) - rhs
    df_c1(λint::Float64)::Float64 = sum_kω(
        kG,
        χr,
        χm.β,
        0.0,
        zeros(Float64, length(χd.usable_ω));
        transform = (f(x::Float64)::Float64 = dχ_λ(x, λint)),
    )
    λm = newton_right(f_c1, df_c1, 0.0, λm_min)

    check, check2 = if isfinite(validate_threshold) || verbose
        χ_λ!(χm, λm)
        check = sum_kω(kG, χm)
        check2 = sum_ωk(kG, χm)
        reset!(χm)
        check, check2
    else
        -Inf, Inf
    end
    if verbose
        println("CHECK for rhs = $rhs  : ", check, " ?=? ", 0)
        println("sum_kω - PP = ", abs(rhs - check))
        println("sum_ωk - PP = ", abs(rhs - check2))
    end
    validation = (abs(rhs - check) <= validate_threshold) && (abs(rhs - check2) <= validate_threshold)
    return λm, validation


end
