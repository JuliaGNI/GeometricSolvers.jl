```@meta
CurrentModule = GeometricSolvers
```

# [The three regimes](@id regimes)

One solver loop serves three regimes. The storage of the iterate selects the regime. Most problems
and methods work in each of them, but not all: [`solve_batch!`](@ref) takes no
[`LeastSquaresProblem`](@ref), and a [`Newton`](@ref) with [`GMRES`](@ref) needs an R3 iterate.
Check the docstrings of the problem and the method for the regimes that they support.

| regime | iterate | entry point | where it runs |
|:--|:--|:--|:--|
| **R1** value semantics | `Number` or `SVector` | [`solve`](@ref), [`solve!!`](@ref) | on the host, or in one GPU thread |
| **R2** batched | an array of `Number` or `SVector`, one entry per system | [`solve_batch!`](@ref) | one work-item per system on a KernelAbstractions backend |
| **R3** in place | an `AbstractVector` that is not static | [`solve`](@ref), [`init`](@ref), [`solve!`](@ref) | on the CPU or on a device array (`CuArray`, `MtlArray`, `ROCArray`) |

## R1: value semantics

In R1 the iterate is a `Number` or an `SVector`, and every operation returns a new value. The
state of the solve holds no buffers: each buffer slot is `nothing`. The problem, the method, the
state, the [`Options`](@ref) and the [`SolverStatus`](@ref) are `isbits` when the functions of the
problem are, so a kernel can hold all of them. The static factorisations ([`StaticLU`](@ref),
[`StaticQR`](@ref), [`StaticLDLT`](@ref), [`StaticCholesky`](@ref), [`StaticSVD`](@ref)) are scalar
loops that the compiler keeps in registers.

```jldoctest regimes
julia> using GeometricSolvers, StaticArrays

julia> F(x, p) = SVector(x[1]^2 + x[2]^2 - p, x[1] - x[2]);

julia> prob = NonlinearProblem(F);

julia> x, status = solve(prob, SVector(1.0, 0.5), Newton(), 2.0);

julia> x isa SVector{2, Float64} && x ≈ SVector(1.0, 1.0)
true

julia> isbits(prob) && isbits(status)
true
```

Inside a `KernelAbstractions.@kernel`, the caller builds the state once on the host with
[`init_state`](@ref) and calls [`solve!!`](@ref) in each work-item. [`solve_batch!`](@ref) does
exactly that.

R1 suits small systems: the static factorisations unroll over the size of the system, so the
compile time and the register use grow with it.

## R2: batched solves

[`solve_batch!`](@ref) solves a batch of independent small systems in one kernel launch on
`get_backend(X)`, one work-item per system. Each entry `X[i]` is the initial guess of system `i`
and is overwritten with its solution. The kernel runs the R1 loop, so a batch takes every method
that R1 takes.

The parameters are shared across the batch unless they are wrapped:

- [`PerSystem`](@ref)`(data)` gives system `i` the value `data[i]`,
- [`BatchParams`](@ref)`(shared, per_system)` gives system `i` the tuple
  `(shared, per_system[i])`.

The form with a status vector writes one [`SolverStatus`](@ref) per system, and
[`batch_summary`](@ref) reduces them to a [`BatchSummary`](@ref):

```jldoctest regimes
julia> X = [SVector(1.0, 0.5), SVector(2.0, 1.0), SVector(0.5, 0.1)];

julia> S = Vector{SolverStatus{Float64}}(undef, length(X));

julia> solve_batch!(X, S, prob, Newton(), PerSystem([2.0, 8.0, 0.5]));

julia> X ≈ [SVector(1.0, 1.0), SVector(2.0, 2.0), SVector(0.5, 0.5)]
true

julia> batch_summary(S).all_converged
true
```

The form without a status vector returns the count of each return code:

```jldoctest regimes
julia> X = [SVector(1.0, 0.5), SVector(2.0, 1.0)];

julia> solve_batch!(X, prob, Newton(), PerSystem([2.0, 8.0])).SUCCESS
2
```

On a `Vector` the kernel runs on the KernelAbstractions `CPU()` backend. On a device, the batch,
the status vector and any `PerSystem` data are arrays of that device. The following example is
not run in this documentation, because it needs a GPU:

```julia
# Device example (not run here): one GPU thread per system.
using GeometricSolvers, StaticArrays, CUDA        # or Metal, AMDGPU

F(x, p) = SVector(x[1]^2 + x[2]^2 - p, x[1] - x[2])
prob = NonlinearProblem(F)

n = 10^6
X = CuArray(fill(SVector(1.0f0, 0.5f0), n))      # MtlArray, ROCArray
S = similar(X, SolverStatus{Float32})
p = PerSystem(CuArray(rand(Float32, n) .+ 1))

solve_batch!(X, S, prob, Newton(), p)
```

The kernel takes the problem by value, so every part of it must be `isbits`: a residual that
captures a host array, or a sparse `jacobian_prototype`, cannot enter a device kernel. On Metal
the element type is `Float32`.

## R3: in-place arrays

In R3 the iterate is an `AbstractVector` that is not static. The residual and the Jacobian write
into buffers, and [`init`](@ref) allocates every buffer once. The factorisation is LAPACK on a host
`Array` and a device factorisation on a GPU array. See [Linear solvers](@ref linear) for the
factor that each array type gets. [`CholeskyFactorization`](@ref) factorises on a host `Array`
only: it has no device factorisation, and a GPU array raises an `ArgumentError` in [`init`](@ref).

```jldoctest regimes
julia> F!(y, x, p) = (y .= x .^ 2 .- p; y);

julia> solver = init(NonlinearProblem(F!), ones(4), Newton(), [1.0, 4.0, 9.0, 16.0]);

julia> x, status = solve!(solver);

julia> x ≈ [1.0, 2.0, 3.0, 4.0]
true
```

The same code runs on a device array. The residual must not index scalars, so broadcast
operations are the usual form. This example is not run here:

```julia
# Device example (not run here).
using GeometricSolvers, CUDA        # Metal: MtlArray and Float32 only

F!(y, x, p) = (y .= x .^ 2 .- p; y)
x₀ = CUDA.ones(Float64, 1000)
p = CuArray(collect(1.0:1000.0))

x, status = solve(NonlinearProblem(F!), x₀, Newton(), p)
```

On a device array, the default Jacobian is the own chunked forward mode of ForwardDiff, and the
factor is [`DeviceLU`](@ref), [`DeviceQR`](@ref), [`DeviceLDLT`](@ref) or [`DeviceSVD`](@ref). A
matrix-free [`GMRES`](@ref) solve needs no factor at all.

## How to choose a regime

The regime follows the size and the number of the systems.

- **One small system** (a few unknowns, such as the stage equations of an implicit integrator
  for a low-dimensional system): R1 with an `SVector`. It allocates nothing and the static
  factorisations run in registers.
- **Many independent small systems of the same form** (one per particle, per cell or per sample):
  R2 with [`solve_batch!`](@ref). One launch solves all of them, on the CPU or on a GPU.
- **One large system**: R3 with a `Vector` or a device array, through [`init`](@ref) and
  [`solve!`](@ref) when it is solved more than once. For a large system whose Jacobian no array
  can hold, use the matrix-free [`GMRES`](@ref). For a large sparse system on the host, use a
  sparse `jacobian_prototype` with Sparspak (see [Sparse direct solves](@ref sparse)).

The size where R3 on a device starts to pay depends on the device, the element type and the cost
of the residual; this documentation records no measurement of it.
