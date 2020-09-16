using Documenter, LadderDGA

makedocs(;
    modules=[LadderDGA],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/Atomtomate/LadderDGA.jl/blob/{commit}{path}#L{line}",
    sitename="LadderDGA.jl",
    authors="Julian Stobbe <Atomtomate@gmx.de>",
    assets=String[],
)

deploydocs(;
    repo="github.com/Atomtomate/LadderDGA.jl",
)
