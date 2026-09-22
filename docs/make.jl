using Documenter, GeometricSolvers

# What the doctests need in scope. Shared with the `doctest` job of `.github/workflows/CI.yml`,
# which includes the same file, so a build and a doctest run cannot disagree.
include(joinpath(@__DIR__, "doctestsetup.jl"))

makedocs(;
    modules = [GeometricSolvers],
    authors = "Michael Kraus",
    repo = "https://github.com/JuliaGNI/GeometricSolvers.jl/blob/{commit}{path}#{line}",
    sitename = "GeometricSolvers.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://JuliaGNI.github.io/GeometricSolvers.jl",
        edit_link = "main",
        assets = String[]
    ),
    pages = [
        "Home" => "index.md"
    ]
)

deploydocs(;
    repo = "github.com/JuliaGNI/GeometricSolvers.jl",
    devbranch = "main"
)
