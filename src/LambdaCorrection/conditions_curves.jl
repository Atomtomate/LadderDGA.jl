function curve_λd_of_λm(
    N_λd::Int,
    χm::χT,
    γm::γT,
    χd::χT,
    γd::γT,
    λ₀,
    h::lDΓAHelper;
    # λm related:
    λm_rhs_type::Symbol = :native,
    fit_μ::Bool = true,
    # λdm related:
    νmax::Int = -1,
    λ_min_δ::Float64 = 0.0001,
    # common options
    λ_val_only::Bool = false,
    verbose::Bool = false,
    tc::Bool = true,
)

    χm_bak = deepcopy(χm)
    χd_bak = deepcopy(χd)

    νmax = νmax < 0 ? floor(Int, size(γm, γm.axis_types[:ν]) / 2) : νmax
    rhs = λm_rhs(χm, χd, h; λ_rhs = :native)
    λm, validation = λm_correction(χm, rhs, h, verbose = verbose, validate_threshold = validate_threshold)


    χm = deepcopy(χm_bak)
    χd = deepcopy(χd_bak)
end
