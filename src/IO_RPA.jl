# ==================================================================================================== #
#                                           IO_RPA.jl                                                  #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Jan Frederik Weißler                                                             #
# ----------------------------------------- Description ---------------------------------------------- #
#   Functions for reading and writing RPA specific data.                                               #
# -------------------------------------------- TODO -------------------------------------------------- #
#   Unit tests                                                                                         #
# ==================================================================================================== #
"""
is_okay(χ₀qω)

check whether the given χ₀qω Array satisfies a set of expected conditions.
"""
function is_okay(χ₀qω)
    leq_zero = all(χ₀qω .≥ 0)
    χΓω0neq0 = χ₀qω[begin, begin] ≠ 0
    χΓωneq0eq0 = all(χ₀qω[begin, begin+1:end] .== 0)
    is_ok = leq_zero & χΓω0neq0 & χΓωneq0eq0
    return is_ok
end


"""
    read_RPA_input(file::String)

TBW
"""
function read_RPA_input(file::String)
    if !HDF5.ishdf5(file)
        throw(ArgumentError("The given file is not a valid hdf5 file!\ngiven path is: '$(file)'"))
    end
    println("--------------------------------------------------------")
    println("Read RPA input from file $(file)\n")
    f = h5open(file, "r")
    chi0_group = f["chi0"]

    # attributes
    attr_dict = attrs(chi0_group)
    β = attr_dict["beta"]                            # inverse temperature
    n_bz_k = attr_dict["n_cubes"]                         # number of sample points per dimension to sample the first brillouin zone
    n_bz_q = 2 * (attr_dict["n_samples"] - 1)                    # number of sample points per dimension to sample the first brillouin zone
    e_kin_q = attr_dict["tail_coeff_q_indep"]
    e_kin = attr_dict["e_kin"]

    # datasets
    ω_integers = read(chi0_group["omega_integers"])
    if ω_integers ≠ collect(0:maximum(ω_integers))
        throw(ArgumentError("ω-integers are distict from (0:$(maximum(ω_integers)))!"))
    end
    χ₀qω = read(chi0_group["values"])
    if !is_okay(χ₀qω)
        throw(ArgumentError("The given array for χ₀qω violates at least one of the expected relations."))
    end

    println("χ₀: (1:$(n_bz_q))³ x (0:$(maximum(ω_integers))) → R, indices(q,ω)↦χ₀(q,ω)")
    println("was evaluated ...")
    println("\t... for the inverse temperature $(β)")
    println("\t... using $(n_bz_k) gauss legendre sample points to perform the integration over the first brillouin zone")
    println("\nTail coefficients")
    println("\t... kinetic energy is $(e_kin)")
    println("\t... tail coeff e_kin_q is $(e_kin_q)")
    println("\nIformations about the data array")
    println("\t...Array χ₀qω has size $(size(χ₀qω))")
    close(f)
    println("--------------------------------------------------------")

    data = expand_ω(χ₀qω)
    ωnGrid = (convert(Int64, -maximum(ω_integers)):convert(Int64, maximum(ω_integers)))
    return χ₀RPA_T(data, ωnGrid, β, e_kin, e_kin_q)
end

"""
    expand_ω(χ₀qω)

    Helper function for reading RPA input. It holds χ₀(q,ω)=χ₀(q,-ω). Take an array for χ₀(q,ω) with ω-integers {0, 1, ..., m} and map onto array with ω-integers {-m, -(m-1), ..., -1, 0, 1, ..., m-1, m}.
"""
function expand_ω(χ₀qω)
    return cat(reverse(χ₀qω, dims=2)[:, begin:end-1], χ₀qω, dims=2)
end