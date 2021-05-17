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
projects=("EquivalenceClassesConstructor.jl" "Dispersions.jl" "SparseVertex" "LadderDGA.jl")
for p in $projects
do
    if [ ! -d $p ]
    then
        git clone "https://github.com/Atomtomate/$p"
    fi
done
cd $cw
dir="$lDGApath/LadderDGA.jl/notebooks"
julia -e "using IJulia; notebook(dir=\"$dir\")"
