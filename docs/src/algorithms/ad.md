```@meta
CurrentModule = GeometricSolvers
```

# [Automatic differentiation](@id ad)

A [`Newton`](@ref) step needs the Jacobian, and a line search or a refinement needs
Jacobian–vector products. The `jacobian` field of [`Newton`](@ref) selects their source:

| `jacobian` | source |
|:--|:--|
| `nothing` (default) | the problem's own `J` when it has one, else `AutoForwardDiff()` |
| `AutoForwardDiff()` | ForwardDiff, in the forms below |
| another ADTypes back end | DifferentiationInterface, on a host `Array` iterate |
| `AutoSparse(...)` | DifferentiationInterface with a sparsity pattern and a colouring |
| a callable `J(x, p)` | the caller's Jacobian operator |
| [`JVPFunction`](@ref)`(jvp!)` | the caller's product, for a [`GMRES`](@ref) solve |

[`prepare_ad`](@ref) builds the preparation once, at [`init`](@ref), from the back end and the
storage of the iterate. [`jacobian!!`](@ref), [`jvp!!`](@ref) and [`gradient_hessian!!`](@ref)
then dispatch on the object it returns, so the solver makes no choice at run time.

## ForwardDiff on a static iterate

In R1 the preparation of `AutoForwardDiff()` is the back end itself. A Jacobian is one tagged
ForwardDiff evaluation of `F` on an `SVector` of duals, and a Jacobian–vector product is one
evaluation with a single partial. Everything stays `isbits`, so it runs in a kernel. A back end
other than `AutoForwardDiff()` has no R1 method.

## Chunked ForwardDiff on arrays

In R3, `AutoForwardDiff()` uses the package's own chunked forward mode, [`ChunkedForwardDiff`](@ref):
`N` Jacobian columns per residual evaluation, in preallocated dual buffers. It indexes no scalar,
so it runs on a host `Array` and on a device array alike. For a [`StationarityProblem`](@ref),
[`ChunkedHessianForwardDiff`](@ref) takes nested duals for the gradient and the Hessian block of
the action.

## DifferentiationInterface

Any other ADTypes back end goes through
[DifferentiationInterface](https://github.com/JuliaDiff/DifferentiationInterface.jl) on a host
`Array` ([`DIJacobian`](@ref), and [`DIStationarity`](@ref) for a stationarity problem). The
package of the back end must be loaded:

```jldoctest ad
julia> using GeometricSolvers, ADTypes

julia> F!(y, x, p) = (y .= x .^ 2 .- p; y);

julia> x, status = solve(NonlinearProblem(F!), ones(3), Newton(; jacobian = AutoForwardDiff()), [1.0, 4.0, 9.0]);

julia> x ≈ [1.0, 2.0, 3.0]
true
```

```julia
using FiniteDiff
solve(NonlinearProblem(F!), ones(3), Newton(; jacobian = AutoFiniteDiff()), [1.0, 4.0, 9.0])
```

On a device array only `AutoForwardDiff()` applies; another back end raises an `ArgumentError` in
[`prepare_ad`](@ref), because DifferentiationInterface indexes scalars there. `AutoEnzyme` needs
runtime activity for the parameter context, as the section
[Automatic differentiation](@ref "Automatic differentiation") of the API page explains.

## Jacobian–vector products

[`prepare_jvp`](@ref) builds the product preparation. With an analytic three-argument
`J!(J, x, p)` on the problem, [`JacobianMatrixJVP`](@ref) evaluates `J!` into a buffer of its own
and multiplies. A product is prepared only where a step reads one: a line search other than
[`Static`](@ref), a [`Refinement`](@ref) along a reused factor, and every [`GMRES`](@ref) step.

## Sparse Jacobians

For a sparse R3 Jacobian on a host `Array`, an `AutoSparse` back end takes either the
`jacobian_prototype` of the [`NonlinearProblem`](@ref) as its pattern, or, when
SparseMatrixColorings is loaded and the problem has no prototype, a sparsity detector together
with a colouring algorithm:

```julia
# Not run here: needs SparseConnectivityTracer and SparseMatrixColorings.
using GeometricSolvers, ADTypes, Sparspak
using SparseConnectivityTracer: TracerSparsityDetector
using SparseMatrixColorings: GreedyColoringAlgorithm

ad = AutoSparse(AutoForwardDiff();
    sparsity_detector = TracerSparsityDetector(),
    coloring_algorithm = GreedyColoringAlgorithm())

solve(NonlinearProblem(F!), x₀, Newton(; jacobian = ad), p)
```

The extension `GeometricSolversSparseADExt` then reads the detected pattern through
[`jacobian_pattern`](@ref), and with Sparspak loaded the solve takes the sparse LU (see
[Sparse direct solves](@ref sparse)). An explicit prototype takes precedence over a detected
pattern.

## The parameters

The parameters are a context of the preparation and not a differentiated argument. Their values,
and the parameter object itself, may change on every call, so one preparation serves a whole
time-stepping loop. Differentiation with respect to the parameters is the
[implicit differentiation](@ref "Implicit differentiation") of [`solve`](@ref).
