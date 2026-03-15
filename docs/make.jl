using MultiObjectiveDescent
using Documenter

DocMeta.setdocmeta!(MultiObjectiveDescent, :DocTestSetup, :(using MultiObjectiveDescent); recursive=true)

makedocs(;
    modules=[MultiObjectiveDescent],
    authors="Mohammed Classes",
    sitename="MultiObjectiveDescent.jl",
    format=Documenter.HTML(;
        canonical="https://mmogib.github.io/MultiObjectiveDescent.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/mmogib/MultiObjectiveDescent.jl",
    devbranch="main",
)
