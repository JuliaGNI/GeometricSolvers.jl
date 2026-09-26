using Documenter
using GeometricSolvers

# Documenter evaluates a page's `@meta` block in `Main`, and this file runs in a module of its own.
@eval Main import GeometricSolvers

include(joinpath(@__DIR__, "..", "..", "docs", "doctestsetup.jl"))

doctest(GeometricSolvers)
