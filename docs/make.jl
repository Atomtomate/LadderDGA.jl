using Documenter
using LadderDGA

push!(LOAD_PATH, "../src")
makedocs(;
    modules=[LadderDGA],
    authors="Julian Stobbe <Atomtomate@gmx.de> and contributors",
    repo="https://github.com/Atomtomate/LadderDGA.jl/blob/{commit}{path}#L{line}",
    sitename="LadderDGA",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", nothing) == "true",
        canonical="https://Atomtomate.github.io/LadderDGA.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    branch="gh-pages",
    devbranch = "master",
    devurl = "stable",
    repo="github.com/Atomtomate/LadderDGA.jl.git",
)
