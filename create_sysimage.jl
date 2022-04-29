using PackageCompiler
using Pkg
Pkg.activate(@__DIR__)

run_test_path = ARGS[1]
 PackageCompiler.create_sysimage(["LadderDGA"]; sysimage_path="LadderDGA_Precompile.so",
                                 precompile_execution_file=run_test_path,
                                 incremental=true)
# PackageCompiler.create_sysimage(["LadderDGA"]; sysimage_path="LadderDGA_Precompile.so",
#                                 precompile_statements_file="cmpl.jl",
#                                 incremental=false)

# PackageCompiler.create_sysimage(["LadderDGA"]; sysimage_path="LadderDGA_Precompile.so",
#                                 precompile_execution_file=run_test_path,
#                                 incremental=true,
#                                 sysimage_build_args=`-O3 --check-bounds=no`)
