using Documenter, ladderDGA_Julia

makedocs(;
    modules=[ladderDGA_Julia],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/Atomtomate/ladderDGA_Julia.jl/blob/{commit}{path}#L{line}",
    sitename="ladderDGA_Julia.jl",
    authors="Julian Stobbe <Atomtomate@gmx.de>",
    assets=String[],
)

deploydocs(;
    repo="github.com/Atomtomate/ladderDGA_Julia.jl",
)
