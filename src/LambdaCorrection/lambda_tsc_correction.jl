# ==================================================================================================== #
#                                      lambda_dm_correction.jl                                         #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
# -------------------------------------------- TODO -------------------------------------------------- #
# ==================================================================================================== #


"""
    λdm_tsc_correction(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::Array{ComplexF64,3}, h, sP, mP;
                        validation_threshold::Float64 = 1e-8, log_io = devnull
    )


"""
function λdm_tsc_correction_clean(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::λ₀T, h;
                        νmax::Int = eom_ν_cutoff(h), tc::Type{<: ΣTail} = default_Σ_tail_correction(),
                        use_trivial_λmin::Bool = (tc === ΣTail_EoM), λd_min::Float64 = NaN, λd_max::Float64 = 200.0, λd_δ::Float64 = 1e-4,
                        validation_threshold::Float64 = 1e-8, max_steps_m::Int = 2000, 
                        max_steps_sc::Int = 100, mixing::Float64=0.3, mixing_start_it::Int=10,
                        dbg_roots_reset::Int=5,
                        max_steps_dm::Int = 2000, log_io = devnull, RF_Method=Roots.FalsePosition(), 
                        verbose::Bool=false, verbose_sc::Bool=false, trace=nothing)

   error("Not Implemented yet!")
end


function run_tsc(χm::χT, γm::γT, χd::χT, γd::γT, λ₀::λ₀T, h;
            max_steps_m::Int = 2000, 
            maxit::Int=100, mixing::Float64=0.3, mixing_start_it::Int=10,
            conv_abs::Float64=1e-8, tc::Type{<: ΣTail} = default_Σ_tail_correction(), trace::Bool=false, verbose::Bool=false)
    error("Not Implemented yet!")
end

function λdm_tsc_correction(χm::χT,γm::γT,χd::χT, γd::γT,λ₀::λ₀T, h;
                            νmax::Int = eom_ν_cutoff(h), fix_n::Bool = true, tc::Type{<: ΣTail} = default_Σ_tail_correction(),
                            use_trivial_λmin::Bool = (tc === ΣTail_EoM),  λd_min::Float64 = NaN, λd_max::Float64 = 200.0, λd_δ::Float64 = 1e-4,
                            validation_threshold::Float64 = 1e-8, max_steps_m::Int = 2000, max_steps_dm::Int = 2000, 
                            max_steps_sc::Int = 100, mixing::Float64=0.3, mixing_start_it::Int=10,
                            dbg_roots_reset::Int=5,
                            log_io = devnull, RF_Method=Roots.FalsePosition(), verbose::Bool=false, verbose_sc::Bool=false, trace=nothing)         

    error("Not Implemented yet!")
end


function run_tsc!(G_ladder_it::OffsetArray, Σ_ladder_it::OffsetArray, G_ladder_bak::OffsetArray, gLoc_rfft, 
                    Kνωq_pre, tc_factor, tc::Type{<: ΣTail}, 
                    χm::χT, γm::γT, χd::χT, γd::γT, λ₀::λ₀T, λd::Float64, h;
                    maxit::Int=500, mixing::Float64=0.3, mixing_start_it::Int=10,
                    conv_abs::Float64=1e-8, trace=nothing, verbose::Bool=false)
    error("Not Implemented yet!")

end
