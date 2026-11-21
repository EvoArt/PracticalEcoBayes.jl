using Documenter
using PracticalEcoBayes

DocMeta.setdocmeta!(PracticalEcoBayes, :DocTestSetup,
                   :(using PracticalEcoBayes); recursive=true)

makedocs(;
    modules=[PracticalEcoBayes],
    checkdocs=:exports,
    sitename="PracticalEcoBayes.jl",
    repo="https://github.com/EvoArt/PracticalEcoBayes.jl/blob/{commit}{path}#{line}",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://EvoArt.github.io/PracticalEcoBayes.jl",
        repolink="https://github.com/EvoArt/PracticalEcoBayes.jl",
    ),
    pages=["Home" => "index.md"],
)

deploydocs(; repo="github.com/EvoArt/PracticalEcoBayes.jl")
