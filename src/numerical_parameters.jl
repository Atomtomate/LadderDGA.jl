# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#   Definitions for default values of numerical values, such as cutoff parameters.                     #
# -------------------------------------------- TODO -------------------------------------------------- #
# ==================================================================================================== #


# ======================================== EoM related stuff =========================================
# ------------------------------------------- EoM ν cutoff -------------------------------------------
"""
    eom_ν_cutoff(Niν::Int, grid_shifted::Bool)::Int
    eom_ν_cutoff(sP::SimulationParameters)
    eom_ν_cutoff(h::lDΓAHelper)

Returns number of positive fermionic frequencies to be used in self-energy after calculation of EoM (as a function of available positive frequency from DMFT `Niν`).

This is especially necessary for shifted grids, since there fewer bosonic frequencies available for large ν, leading to an error in the high frequency tail of the self energy. 
"""
eom_ν_cutoff(Niν::Int, grid_shifted::Bool)::Int = trunc(Int, (2/3)* Niν)

eom_ν_cutoff(sP::SimulationParameters)::Int = eom_ν_cutoff(sP.n_iν, sP.shift)

eom_ν_cutoff(h::lDΓAHelper)::Int = eom_ν_cutoff(h.sP)

eom_ν_cutoff(h::AlDΓAHelper)::Int = eom_ν_cutoff(h.sP)

const default_ΣTail_ExpStep_δ::Float64 = 0.15
default_Σ_tail_correction()::Type{<: ΣTail} = ΣTail_ExpStep{default_ΣTail_ExpStep_δ}