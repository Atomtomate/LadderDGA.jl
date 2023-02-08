include("lambdaCorrection_singleCore.jl")

function λ_correction(type::Symbol, imp_density::Float64,
            χsp::χT, γsp::γT, χch::χT, γch::γT,
            Gνω::GνqT, λ₀::Array{ComplexF64,3}, kG::KGrid,
            mP::ModelParameters, sP::SimulationParameters;
            workerpool::AbstractWorkerPool=default_worker_pool(),init_sp=nothing, init_spch=nothing, parallel=false, x₀::Vector{Float64}=[0.0,0.0])
    res = if type == :sp
        rhs = LambdaCorrection.λsp_rhs(imp_density, χsp, χch, kG, mP, sP)
        λm_correction(χsp, rhs, kG, mP, sP)
    elseif type == :sp_ch
        λspch, dbg_string = if parallel
                extended_λ_par(χsp, γsp, χch, γch, Gνω, λ₀, x₀, kG, mP, sP, workerpool)
            else
                extended_λ(χsp, γsp, χch, γch, Gνω, λ₀, x₀, kG, mP, sP)
        end
        @warn "extended λ correction dbg string: " dbg_string
        λspch
    else
        error("unrecognized λ correction type: $type")
    end
    return res
end

function λ_correction!(type::Symbol, imp_density, F, Σ_loc_pos, Σ_ladderLoc,
                       χsp::χT, γsp::γT, χch::χT, γch::γT,
                      χ₀::χ₀T, Gνω::GνqT, kG::KGrid,
                      mP::ModelParameters, sP::SimulationParameters; init_sp=nothing, init_spch=nothing)

    λ = λ_correction(type, imp_density, F, Σ_loc_pos, Σ_ladderLoc, χsp, γsp, χch, γch,
                  χ₀, Gνω, kG, mP, sP; init_sp=init_sp, init_spch=init_spch)
    res = if type == :sp
        χ_λ!(χsp, λ)
    elseif type == :sp_ch
        χ_λ!(χsp, λ[1])
        χ_λ!(χch, λ[2])
    end
end
