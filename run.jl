using Distributed

if (myid()==1) && (nprocs()==1)
    addprocs(7)
end
println("using ", nprocs(), " workers.")

@everywhere using Pkg
@everywhere Pkg.activate(".")
@everywhere include("src/LadderDGA.jl")
bubble, nlQ_sp, nlQ_sp_λ, nlQ_ch, Σ_ladder, Σ_ladderLoc = LadderDGA.calc_Σ();
println("");
