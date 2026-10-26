@doc raw"""
Solvers with the problem/method/solver interface of `SimpleSolvers`, built so that the same
solvers run on the CPU and on CUDA, ROCm and Metal devices through KernelAbstractions.
"""
module GeometricSolvers

using ADTypes: AbstractADType, AutoForwardDiff
using Adapt: Adapt
using DifferentiationInterface: DifferentiationInterface as DI
using ForwardDiff: ForwardDiff
using GPUArraysCore: AbstractGPUArray
using LinearAlgebra: BLAS, BlasFloat, BlasInt, LAPACK, LU, QR, RowMaximum, ldiv!, lu!,
                     mul!, qr!
using StaticArrays: MMatrix, MVector, SMatrix, StaticArray, StaticMatrix, StaticVector,
                    SVector,
                    setindex

export ReturnCode, SUCCESS, STALLED, MAXITERS, SINGULAR, NONFINITE, LINESEARCH_FAILED
export SolverStatus, StepInfo, Options
export NonlinearProblem
export LUFactorization, QRFactorization, SVDFactorization, TF32, FP16, BF16
export Static, Backtracking, Bisection, StrongWolfe, DogLeg
export Newton, NoRefinement, Picard, solve

public solve!!

include("base/returncodes.jl")
include("base/status.jl")
include("base/options.jl")
include("base/reductions.jl")
include("base/adapt.jl")
include("base/bang.jl")

include("linear/methods.jl")
include("linear/factors.jl")
include("linear/static/lu.jl")
include("linear/static/qr.jl")
include("linear/dense.jl")
include("linear/mixed.jl")

include("globalization/linesearch.jl")
include("globalization/static.jl")
include("globalization/backtracking.jl")
include("globalization/bisection.jl")
include("globalization/strongwolfe.jl")

include("nonlinear/problem.jl")

include("ad/forwarddiff_static.jl")
include("ad/di.jl")
include("ad/chunked.jl")
include("ad/jvp.jl")

include("globalization/linefunction.jl")
include("globalization/dogleg.jl")
include("nonlinear/solver.jl")
include("nonlinear/newton.jl")
include("nonlinear/picard.jl")

end
