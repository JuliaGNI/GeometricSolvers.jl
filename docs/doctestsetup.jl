# What the doctests of this package need in scope.
#
# Included both by `docs/make.jl` and by the `doctest` job of `.github/workflows/CI.yml`, so that a
# documentation build and a doctest run cannot disagree. One definition, two callers: the CI
# workflow is byte-identical in every repository and cannot carry per-package knowledge, and a
# second copy of this list is exactly the thing that goes stale.

using Documenter: DocMeta

using GeometricSolvers

DocMeta.setdocmeta!(
    GeometricSolvers,
    :DocTestSetup,
    quote
        using GeometricSolvers
    end;
    recursive = true
)
