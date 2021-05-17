#!/usr/bin/env bash

jpath="${HOME}/julia"
mkdir -p $jpath
cd $jpath
fpath="linux/x64/1.6/julia-1.6.1-linux-x86_64.tar.gz"
url=https://julialang-s3.julialang.org/bin/$fpath
wget -c $url -O julia.tar.gz -q
tar zxf julia.tar.gz -C $jpath --strip-components 1 & status_msg "Installing Julia "
rm julia.tar.gz
echo "export PATH=$jpath/bin:\$PATH" >>~/.bash_profile



cw=$(pwd)
lDGApath="${HOME}/lDGATools"
mkdir -p $lDGApath
cd $lDGApath
for p in "EquivalenceClassesConstructor.jl" "Dispersions.jl" "SparseVertex" "LadderDGA.jl"
do
    if [ ! -d $p ]
    then
        git clone "https://github.com/Atomtomate/$p"
    fi
    cd $p
    local cwd=$(pwd)
    julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate();' >> "$lDGApath/install.log" 2>&1
    julia -e "using Pkg; Pkg.add(path=\"$cwd\")"
    cd ..
done
cd $cw
julia -e 'using Pkg; Pkg.add.(["IJulia", "Plots", "HDF5"])' >> "$lDGApath/install.log" 2>&1
dir="$lDGApath/LadderDGA.jl/notebooks"
julia -e "using IJulia; notebook(dir=\"$dir\")"
