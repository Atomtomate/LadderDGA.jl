using PackageCompiler
using Pkg
Pkg.activate(@__DIR__)

PackageCompiler.create_sysimage(["LadderDGA"]; sysimage_path="LadderDGA_Precompile.so",
                                       precompile_execution_file="run_test.jl")
