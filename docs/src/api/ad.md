```@meta
CurrentModule = GeometricSolvers
```

# Automatic differentiation

In R1, [`prepare_ad`](@ref) returns the `AutoForwardDiff()` back end itself for a `StaticVector`
or a `Number` iterate. [`jacobian!!`](@ref), [`gradient_hessian!!`](@ref) and [`jvp!!`](@ref) run
tagged ForwardDiff directly with that source. The source `nothing` calls the problem's own `J`; a
[`JacobianFunction`](@ref) calls a method's operator for `jacobian!!` only. A back end other than
`AutoForwardDiff` has no R1 method, and the call is a host `MethodError`.

In R3 the AD path is chosen once, by [`prepare_ad`](@ref), from the
back end and the storage of the iterate: `AutoForwardDiff()` goes through an own chunked forward
mode on every array type ([`ChunkedForwardDiff`](@ref) for [`NonlinearProblem`](@ref) and
[`ChunkedHessianForwardDiff`](@ref) for [`StationarityProblem`](@ref)), and any other back end goes
through [DifferentiationInterface](https://github.com/JuliaDiff/DifferentiationInterface.jl) on a
CPU iterate ([`DIJacobian`](@ref) and [`DIStationarity`](@ref)), so every ADTypes back end works
with no line of source here. A device array iterate takes
`AutoForwardDiff()` only, because DifferentiationInterface's `jacobian!` raises a scalar-indexing
error on an `MtlArray`, a `CuArray` and a `JLArray`. When the problem supplies an analytic
3-argument `J(Jbuf, x, p)`, [`JacobianMatrixJVP`](@ref) evaluates `prob.J` into its pre-allocated
matrix buffer and multiplies by the direction in place. Only a [`LineSearch`](@ref) other than
[`Static`](@ref) and a [`Refinement`](@ref) along a reused factor take a JVP, so under any other
configuration the analytic `J` gets no JVP preparation and no second `m × n` buffer. A
`Refinement` on a step that factorises at `x` multiplies by the regularised Jacobian that the step
holds, in a buffer of its own beside the factor, which the factorisation overwrites.
[`jacobian!!`](@ref), [`gradient_hessian!!`](@ref) and [`jvp!!`](@ref) then dispatch on the
object the preparation returned, so the solver makes no runtime choice and carries no `ad`
keyword.

For a sparse R3 Jacobian on a host `Array` iterate, an `AutoSparse` back end takes either an
explicit `jacobian_prototype` on the [`NonlinearProblem`](@ref) (which [`prepare_ad`](@ref) wraps in
`KnownJacobianSparsityDetector`) or, when `SparseMatrixColorings` is loaded and
`jacobian_prototype === nothing`, a sparsity detector such as `TracerSparsityDetector()` of
`SparseConnectivityTracer` alongside a colouring algorithm such as `GreedyColoringAlgorithm()`. The
`GeometricSolversSparseADExt` package extension then extracts the detected pattern from the prepared
[`DIJacobian`](@ref) via [`jacobian_pattern`](@ref) as a `SparseMatrixCSC{<:Any, Int}` — a
`SparseMatrixCSC{Bool, Int}` for `TracerSparsityDetector` — so
[`init_linear`](@ref) allocates the sparse factor and Jacobian buffer without a user-supplied
prototype where [`has_sparse_factor`](@ref) holds, and the dense factor elsewhere; when an explicit
`jacobian_prototype` is also given, the explicit prototype takes precedence.

In R3 the iterate is a vector; another shape raises `ArgumentError` where the preparation is
built. The parameters may be replaced on every call, which is what lets one preparation serve a
whole time-stepping loop.

`AutoEnzyme` reaches the `Array` path through DifferentiationInterface like any other back end, but
it needs runtime activity for the parameter context —
`AutoEnzyme(; mode = Enzyme.set_runtime_activity(Enzyme.Forward))`. A plain `AutoEnzyme()` raises
`EnzymeRuntimeActivityError` on a residual that broadcasts constant parameters into its result; see
`KNOWN_ISSUES.md`. The mode is the caller's choice, so the package sets none.

These names are not exported. Reach them as `GeometricSolvers.name`.

```@docs
prepare_ad
prepare_jvp
DIJacobian
DIStationarity
ChunkedForwardDiff
ChunkedHessianForwardDiff
JacobianFunction
JacobianMatrixJVP
jacobian!!
gradient_hessian!!
jvp!!
```
