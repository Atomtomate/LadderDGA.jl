include("lambdaCorrection_singleCore.jl")

function λ_correction(type::Symbol, imp_density::Float64,
            χ_m::χT, γ_m::γT, χ_d::χT, γ_d::γT,
            Gνω::GνqT, λ₀::Array{ComplexF64,3}, kG::KGrid,
            mP::ModelParameters, sP::SimulationParameters;
            workerpool::AbstractWorkerPool=default_worker_pool(),init_sp=nothing, init_spch=nothing, parallel=false, x₀::Vector{Float64}=[0.0,0.0])
    res = if type == :sp
        rhs = LambdaCorrection.λsp_rhs(imp_density, χ_m, χ_d, 0.0, kG, mP, sP)
        λm_correction(χ_m, rhs, kG, mP, sP)
    elseif type == :sp_ch
        λspch, dbg_string = if parallel
                extended_λ_par(χ_m, γ_m, χ_d, γ_d, Gνω, λ₀, x₀, kG, mP, sP, workerpool)
            else
                extended_λ(χ_m, γ_m, χ_d, γ_d, Gνω, λ₀, x₀, kG, mP, sP)
        end
        @warn "extended λ correction dbg string: " dbg_string
        λspch
    else
        error("unrecognized λ correction type: $type")
    end
    return res
end

function λ_correction!(type::Symbol, imp_density, F, Σ_loc_pos, Σ_ladderLoc,
                       χ_m::χT, γ_m::γT, χ_d::χT, γ_d::γT,
                      χ₀::χ₀T, Gνω::GνqT, kG::KGrid,
                      mP::ModelParameters, sP::SimulationParameters; init_sp=nothing, init_spch=nothing)

    λ = λ_correction(type, imp_density, F, Σ_loc_pos, Σ_ladderLoc, χ_m, γ_m, χ_d, γ_d,
                  χ₀, Gνω, kG, mP, sP; init_sp=init_sp, init_spch=init_spch)
    res = if type == :sp
        χ_λ!(χ_m, λ)
    elseif type == :sp_ch
        χ_λ!(χ_m, λ[1])
        χ_λ!(χ_d, λ[2])
    end
end
