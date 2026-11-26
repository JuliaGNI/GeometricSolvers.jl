```@meta
CurrentModule = GeometricSolvers
```

# [CPU and GPU backends](@id backends)

GeometricSolvers runs the same solvers on the CPU and on CUDA, ROCm and Metal devices. This page
describes how the code is built for that: the four kinds of object, the storage seam, the
reduction seam, dispatch in place of checks, and the vendor libraries that replace some of the
own kernels.

## Four kinds of object

| object | holds | example |
|:--|:--|:--|
| **problem** | functions | [`NonlinearProblem`](@ref) |
| **method** | numbers, types and sub-methods, and no buffer of the solve | [`Newton`](@ref), [`LUFactorization`](@ref), [`Backtracking`](@ref) |
| **state** | everything that is allocated or that changes during a solve | the result of [`init_state`](@ref) |
| **status** | the result | [`SolverStatus`](@ref) |

A problem holds no parameters and no iterate, and a method holds no buffer of the solve. When every
field of the problem is `isbits` (the residual, the Jacobian function and the prototype, if any),
the problem and every method that runs in R1 are `isbits`, so a kernel takes them by value. A
`Matrix` prototype, for example, makes the problem non-`isbits`, and [`solve_batch!`](@ref) rejects
it. The state of an R1 solve is `isbits` too: every buffer slot is `nothing`.

## KernelAbstractions backends

The device code is written once with
[KernelAbstractions](https://github.com/JuliaGPU/KernelAbstractions.jl). `get_backend(x)` of the
iterate selects the backend: `CPU()` for a host `Array`, and the backend of `CUDA`, `AMDGPU`,
`Metal` or any other GPU array package. The package depends on no vendor package; a vendor package
loads its own backend, and the package extensions for CUDA and Metal load with it.

- [`solve_batch!`](@ref) launches one kernel on `get_backend(X)` with one work-item per system.
- The device factorisations ([`DeviceLU`](@ref), [`DeviceQR`](@ref), [`DeviceLDLT`](@ref),
  [`DeviceSVD`](@ref)) are sequences of KernelAbstractions kernels on an `AbstractGPUArray`.
- The chunked forward mode of ForwardDiff and the reductions run on any array type, with
  broadcasting and `sum` and no scalar indexing.

## The storage seam: one algorithm over two kinds of storage

R1 returns new values and R3 writes into buffers. That difference is at the leaves of an
algorithm, not in its logic. Every algorithm — the loop, [`Newton`](@ref), [`Picard`](@ref), the
line searches, [`DogLeg`](@ref), the combinators and [`Anderson`](@ref) — is written once over a
small set of functions with a `!!` suffix. Each has a method for a value and a method for a buffer:

```julia
update_iterate!!(::Nothing, x, α, d) = x + α * d                   # R1: a new value
update_iterate!!(y::AbstractArray, x, α, d) = (y .= x .+ α .* d)     # R3: into the buffer
```

An algorithm always rebinds the result, `xt = update_iterate!!(state.xt, x, α, d)`, so it reads
the same in both regimes. The seam consists of [`buffer`](@ref), [`residual!!`](@ref),
[`update_iterate!!`](@ref), [`map!!`](@ref), [`mul!!`](@ref), [`copy!!`](@ref), [`scale!!`](@ref),
[`axpby!!`](@ref) and [`regularize!!`](@ref), together with [`jacobian!!`](@ref), [`jvp!!`](@ref),
[`factorize!!`](@ref) and [`ldiv!!`](@ref). A backend or a library adds methods to these
functions, and a vendor library to the back substitution `backsolve!` of its factor, never a new
function.

The `!!` convention is that of BangBang.jl; the package defines its own methods and does not
depend on BangBang.

## The reduction seam

Every norm, dot product and merit value goes through three functions:
[`norm2`](@ref) (the squared norm), [`rnorm`](@ref) (the norm, scaled against overflow and
underflow) and [`rdot`](@ref) (the real part of the inner product). Each returns
`R = real(eltype(x))`, and none of them indexes scalars, so they run on an `SVector` inside a
kernel, on an `Array` and on a GPU array. A widened or compensated accumulator would be a change
in these three functions and nowhere else.

## Dispatch in place of checks

A pairing that the package does not support has no method. The package writes no flag that the
storage could contradict and no runtime test of a type:

| what is selected or excluded | how |
|:--|:--|
| the call form `F(x, p)` or `F!(y, x, p)` | the storage of the iterate, through [`residual!!`](@ref) |
| value code or in-place code | `nothing` or an array in the buffer slots of the state |
| the factor | [`init_linear`](@ref) dispatches on the method and the iterate type |
| a vendor fast path | a more specific [`factorize!!`](@ref) method on the factor type, such as `DeviceLU{<:CuMatrix}` |
| `TF ≠ T` in the Newton step | a second [`ldiv!!`](@ref) method for a factor whose element type differs |
| one parameter set or one per system | the [`PerSystem`](@ref) wrapper |
| a `Float64` literal in a method reaching a `Float32` kernel | the [`ToReal`](@ref) adaptor, which converts every float of a method to `real(T)` |
| an AD back end other than ForwardDiff in R1 | no [`jacobian!!`](@ref) method for it |

In R1 and R3 a missing method is a `MethodError` on the host. The value checks that a type cannot
express, such as a negative tolerance, raise an `ArgumentError` in a constructor, on the host.
Inside a kernel nothing throws: a numerical failure is a [`ReturnCode`](@ref).

## Kernel-safe types

[`ReturnCode`](@ref) is one byte, and the counters of [`SolverStatus`](@ref), [`StepInfo`](@ref)
and [`Options`](@ref) are `Int32` or `Int16`, because 64-bit integers cost registers on a GPU.
Every real field is in `R = real(T)`, so these types bring no `Float64` into a `Float32` kernel;
the parameters and the factorisation precision are the caller's (see
[Device precision](@ref "Device precision")). A
factorisation precision is a type parameter of the method (`LUFactorization{Float32}`) and not a
field, because a type stored in a field is not `isbits`. Verbosity, traces and warnings do not
exist in the core; a caller reads the status.

## What runs where

The factor that a dense R3 solve uses:

| factorisation | host `Array` | `CuArray` (CUDA) | `MtlArray` (Metal) | `ROCArray` (ROCm), other GPU arrays |
|:--|:--|:--|:--|:--|
| [`LUFactorization`](@ref) | LAPACK `getrf`/`getrs`; an own recursive LU and `getrs` in a band of `n` (below) | cuSOLVER `getrf`/`getrs` for BLAS element types | MPS `lu!` for `Float32`; own kernel otherwise | rocSOLVER `getrf`/`getrs` on `ROCArray` for BLAS element types; own kernel otherwise |
| [`QRFactorization`](@ref) | LAPACK `geqrf`/`ormqr` | cuSOLVER `geqrf`/`ormqr` and cuBLAS `trsv` for BLAS element types | own kernel | rocSOLVER `geqrf`/`ormqr` and rocBLAS `trsv` on `ROCArray` for BLAS element types; own kernel otherwise |
| [`SVDFactorization`](@ref) | LAPACK `gesvd` | cuSOLVER `gesvdj`, own kernel where it does not converge or where the `rtol` cutoff lies below `√m · eps(R)` (`m` rows, `R` the factor's real type) | own kernel | own kernel |
| [`LDLTFactorization`](@ref) | LAPACK `sytrf`/`hetrf` | own kernel | own kernel | own kernel |
| [`CholeskyFactorization`](@ref) | LAPACK `potrf`/`potrs` for BLAS element types; generic loop otherwise | no device factor: `ArgumentError` | no device factor: `ArgumentError` | no device factor: `ArgumentError` |
| [`GMRES`](@ref) | no factor | no factor | no factor | no factor |

- On the **host**, `LUFactorization()` factorises an `n × n` `Float32`, `Float64` or `ComplexF64`
  matrix with `8 ≤ n ≤ 32` by an own recursive LU with partial pivoting: the columns split in
  halves down to panels of at most 16 columns, which an unblocked kernel factorises, and BLAS
  `trsm` and `gemm` update the rest. It picks the pivot of largest `|Re| + |Im|`, the first one
  on a tie, as LAPACK does, and writes LAPACK's `ipiv`, so `getrs` solves with its factor. The
  band is where `scripts/hardware/rl_host_lu.jl` measured the recursion at 0.90 or less of the
  time of `getrf`, with BLAS on one thread and on all threads: Apple M4 Max, OpenBLAS 0.3.30,
  Julia 1.13.4. Every other `n`, and `ComplexF32`, call `getrf`.
- The **CUDA** extension `GeometricSolversCUDAExt` adds methods for `DeviceLU`, `DeviceQR` and
  `DeviceSVD` on `CuMatrix{T}` with `T <: LinearAlgebra.BlasFloat`. Other element types stay on
  the own kernels.
- The **Metal** extension `GeometricSolversMetalExt` adds the Metal Performance Shaders LU and
  triangular solve for `MtlMatrix{Float32}`. Every MPS failure status gives a singular factor.
  `ComplexF32` and the other factorisations stay on the own kernels.
- The **ROCm** extension `GeometricSolversAMDGPUExt` adds the rocSOLVER and rocBLAS methods for
  `DeviceLU` and `DeviceQR` on `ROCMatrix{T}` with `T <: LinearAlgebra.BlasFloat`. rocSOLVER
  `geqrf` and `ormqr` and rocBLAS `trsv` serve every shape of the QR factor: the solve with the
  `m × n` factor of a least-squares problem, `m > n`, runs `trsv` on its leading `n × n` block.
  Other element types and the other factorisations stay on the own kernels.

A vendor method replaces only `factorize!!` and the back substitution of its factor type. The
scaling by a power of two, the mixed-precision solve and the status handling are the same on
every backend.

The factor that a sparse R3 solve uses, on a sparse `jacobian_prototype`:

| factorisation | host `Vector`, `SparseMatrixCSC` | `CuVector`, `CuSparseMatrixCSR` (CUDA) | other arrays |
|:--|:--|:--|:--|
| [`LUFactorization`](@ref) | Sparspak for a real element type, with `Sparspak` loaded | cuDSS for BLAS element types, with `CUDSS` loaded | no sparse factor |
| [`UMFPACKFactorization`](@ref) | UMFPACK for `Float64` and `ComplexF64` | no sparse factor | no sparse factor |

- The **cuDSS** extension `GeometricSolversCUDSSExt`, loaded with `CUDA` and `CUDSS`, adds the
  sparse LU of `LUFactorization()` on a `CuSparseMatrixCSR{T}` prototype and a `CuVector{T}`
  iterate, `T <: LinearAlgebra.BlasFloat`. The cuDSS analysis runs once, at the first
  factorisation, and every later factorisation is a refactorisation. Its Jacobian comes from an
  analytic `J`, because an `AutoSparse` back end needs a host array.

A sparse factor has its own `init_linear`, `factorize!!` and `ldiv!!`, and factorises in the
element type of the iterate only: it takes no scaling by a power of two and no mixed precision.

R1 and R2 use the static factorisations on every backend, because they run in one thread. The
sparse LU of Sparspak and that of UMFPACK run on a host `Vector` only. The stacked
[`LevenbergMarquardt`](@ref) path, that of every factorisation other than SVD and of
`scaling = :diagonal`, indexes scalars and runs on a host array only, and an AD back end other
than `AutoForwardDiff()` needs a host array.

## Device precision

| backend | `Float64` | `Float32` | complex |
|:--|:--|:--|:--|
| CPU | yes | yes | yes |
| CUDA | yes | yes | yes |
| ROCm | yes | yes | yes |
| Metal | no | yes | `ComplexF32` |

Metal has no `Float64`, so a Metal solve runs in `Float32` throughout: the iterate, the residual,
the tolerances, the factorisation and the status. The keywords build the [`Options`](@ref) in
`real(eltype(x))`, and [`ToReal`](@ref) converts every float value of the method to that type.
`ToReal` does not change a type parameter or the parameters `p`: a
factorisation precision such as `LUFactorization(Float64)` and a `Float64` parameter keep their
types, so the caller chooses them in `Float32`. With `T = Float32`, the pair
`T = Float64`, `TF = Float32` of [Mixed precision](@ref mixed) does not apply on Metal; it applies
on the CPU, CUDA and ROCm.

The markers [`TF32`](@ref), [`FP16`](@ref) and [`BF16`](@ref) name tensor-core precisions as a
type only. No factorisation method takes a marker, so a marker selects no code path.

## Device tests

The device tests run by hand on the machines that have the devices; the repository's
`test/gpu/README.md` describes the runs. The `JLArrays` reference GPU array runs the device code
paths on the CPU in the regular test suite, with scalar indexing disallowed.
