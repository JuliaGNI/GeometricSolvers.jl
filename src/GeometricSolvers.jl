@doc raw"""
Solvers with the problem/method/solver interface of `SimpleSolvers`, built so that the same
solvers run on the CPU and on CUDA, ROCm and Metal devices through KernelAbstractions.
"""
module GeometricSolvers

using ADTypes: AbstractADType, AutoFiniteDiff, AutoForwardDiff, AutoSparse,
               KnownJacobianSparsityDetector, coloring_algorithm, dense_ad
using Adapt: Adapt
import CommonSolve: init, solve, solve!
using DifferentiationInterface: DifferentiationInterface as DI
using ForwardDiff: ForwardDiff
using GPUArraysCore: AbstractGPUArray
using KernelAbstractions: CPU, KernelAbstractions, @Const, @index, @kernel, allocate,
                          get_backend, synchronize
using LinearAlgebra: BLAS, BlasFloat, BlasInt, LAPACK, LU, QR, RowMaximum, diagind, givens,
                     ldiv!, lu!, mul!, qr!
using SparseArrays: SparseArrays
using StaticArrays: MMatrix, MVector, SMatrix, SizedMatrix, SizedVector, StaticArray,
                    StaticMatrix, StaticVector, SVector,
                    setindex, similar_type

# KernelAbstractions 0.10 has no `isgpu`. Every backend subtypes `Backend` and `CPU` aliases
# `POCLBackend`, the host backend, so a backend is a device exactly when it is not a `CPU`.
# A `CPU` launch completes before it returns and `synchronize(::CPU)` does nothing, so
# `isgpu(backend) && synchronize(backend)` skips nothing.
isgpu(backend) = !(backend isa CPU)

export ReturnCode, SUCCESS, STALLED, MAXITERS, SINGULAR, NONFINITE, LINESEARCH_FAILED
export SolverStatus, StepInfo, Options, stopping_test
export NonlinearProblem, StationarityProblem, LeastSquaresProblem, Reparametrized,
       BlockEliminated, VariableProjection
export LUFactorization, LDLTFactorization, QRFactorization, SVDFactorization
export TF32, FP16, BF16
export jacobi_eigen
export Static, Backtracking, Bisection, StrongWolfe, DogLeg, LevenbergMarquardt
export Newton, GaussNewton, QuasiNewton, NoRefinement, Refinement, PerSystem, Picard,
       NonlinearSolver, ReparametrizedSolver, init, solve, solve!
export OptimalWeights, FixedWeights, Additive, Multiplicative, Repeat, Switch, Anderson
export solve_batch!

public solve!!, sciml_retcode

include("base/returncodes.jl")
include("base/status.jl")
include("base/options.jl")
include("base/reductions.jl")
include("base/adapt.jl")
include("base/bang.jl")

include("linear/methods.jl")
include("linear/factors.jl")
include("linear/static/lu.jl")
include("linear/static/ldlt.jl")
include("linear/static/qr.jl")
include("linear/static/qr_rect.jl")
include("linear/static/svd.jl")
include("linear/static/svd_rect.jl")
include("linear/static/eigen.jl")
include("linear/dense.jl")
include("linear/ka/lu.jl")
include("linear/ka/ldlt.jl")
include("linear/ka/qr.jl")
include("linear/ka/svd.jl")
include("linear/mixed.jl")

include("globalization/linesearch.jl")
include("globalization/static.jl")
include("globalization/backtracking.jl")
include("globalization/bisection.jl")
include("globalization/strongwolfe.jl")

include("nonlinear/problem.jl")
include("nonlinear/leastsquares.jl")

include("ad/forwarddiff_static.jl")
include("ad/di.jl")
include("ad/chunked.jl")
include("ad/jvp.jl")

include("globalization/linefunction.jl")
include("globalization/dogleg.jl")
include("globalization/levenbergmarquardt.jl")
include("nonlinear/solver.jl")
include("nonlinear/refinement.jl")
include("nonlinear/newton.jl")
include("nonlinear/picard.jl")
include("nonlinear/implicit.jl")
include("nonlinear/batched.jl")
include("nonlinear/stationarity.jl")
include("ad/hessian.jl")
include("nonlinear/reparametrized.jl")
include("nonlinear/eliminate.jl")

include("composition/weights.jl")
include("composition/additive.jl")
include("composition/multiplicative.jl")
include("composition/repeat.jl")
include("composition/switch.jl")
include("composition/anderson.jl")
include("composition/implicit.jl")

# `solve!` asks the type of the problem which path it takes. The predicate is defined below the
# `eliminate.jl` include, and not beside `solve!`, because its signature names `BlockEliminated`,
# which `eliminate.jl` defines after `solver.jl` has loaded. A `<:` test on the problem type would
# build a `UnionAll` at run time and allocate, so the test is a dispatch on the type instead.

@inline function is_elim_stationarity_problem(
        ::Type{<:BlockEliminated{<:StationarityProblem}})
    return true
end
@inline is_elim_stationarity_problem(::Type) = false

end
