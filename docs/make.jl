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
        "Home" => "index.md",
        "Manual" => [
            "Getting started" => "manual/getting_started.md",
            "The three regimes" => "manual/regimes.md",
            "Methods by regime" => "guide.md",
            "Problem types" => "manual/problems.md"
        ],
        "Algorithms" => [
            "Nonlinear methods" => "algorithms/nonlinear.md",
            "Globalization" => "algorithms/globalization.md",
            "Linear solvers" => "algorithms/linear.md",
            "Automatic differentiation" => "algorithms/ad.md",
            "Composition and preconditioning" => "algorithms/composition.md"
        ],
        "Architecture" => [
            "CPU and GPU backends" => "architecture/backends.md",
            "Mixed precision" => "architecture/mixed_precision.md",
            "Stability" => "stability.md"
        ],
        "API" => [
            "Problems and solves" => "api/problems.md",
            "Nonlinear solvers" => "api/nonlinear.md",
            "Line searches" => "api/globalization.md",
            "Linear solvers" => "api/linear.md",
            "Automatic differentiation" => "api/ad.md",
            "Combinators and preconditioning" => "api/composition.md",
            "Internals" => "api/internals.md"
        ]
    ]
)

deploydocs(;
    repo = "github.com/JuliaGNI/GeometricSolvers.jl",
    devbranch = "main"
)
