# ==================================================================================================== #
#                                    parallelization_helpers.jl                                        #
# ---------------------------------------------------------------------------------------------------- #
#   Author          : Julian Stobbe                                                                    #
# ----------------------------------------- Description ---------------------------------------------- #
#   Functionality for parallel computation.                                                            #
# -------------------------------------------- TODO -------------------------------------------------- #
#   collect* functions need to be refactored                                                           #
# ==================================================================================================== #


# ============================================== Helpers =============================================

function get_workerpool()
    default_worker_pool()
end



# ============================================ Worker Cache ==========================================