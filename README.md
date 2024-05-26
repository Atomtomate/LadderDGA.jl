# LadderDGA


This package provides functionality for the calculation of non-local corrections on top of dynamical mean field theory (DMFT) calculations.

A tutorial for the usage of this package will hopefully follow soon. For now, examples are given in the `run*.jl` files, the `scripts`, `examples` and `notebooks` directories.

Some documentation can be accessed by clicking on the `docs` badge below.


|     Build Status    |      Coverage      |  Documentation |      Social    |
| ------------------- |:------------------:| :-------------:| :-------------:|
| [![Build Status](https://github.com/Atomtomate/LadderDGA.jl/workflows/CI/badge.svg)](https://github.com/Atomtomate/LadderDGA.jl/actions) |   [![codecov](https://codecov.io/gh/Atomtomate/LadderDGA.jl/branch/master/graph/badge.svg?token=msJVfWnlJI)](https://codecov.io/gh/Atomtomate/LadderDGA.jl) | [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://atomtomate.github.io/LadderDGA.jl/stable/) |[![Gitter](https://badges.gitter.im/JuliansBastelecke/LadderDGA.svg)](https://gitter.im/JuliansBastelecke/LadderDGA?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge) |

---

## Setup

Obtain Julia, e.g. through [juliaup](https://github.com/JuliaLang/juliaup)
Obtain the code, for example:

```
git clone git@github.com:Atomtomate/LadderDGA.jl.git
cd LadderDGA.jl
git checkout dev
```

Obtain dependencies:
```
julia
> using Pkg
> pkg"registry add https://github.com/Atomtomate/JuliaRegistry.git"
> pkg"activate ."
> pkg"instantiate"
```

## Example usage

TODO.....

```
julia
> include("examples/example02_ldm.jl")
```

