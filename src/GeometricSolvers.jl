@doc raw"""
Solvers with the problem/method/solver interface of `SimpleSolvers`, built so that the same
solvers run on the CPU and on CUDA, ROCm and Metal devices through KernelAbstractions.
"""
module GeometricSolvers

using Adapt: Adapt

export ReturnCode, SUCCESS, STALLED, MAXITERS, SINGULAR, NONFINITE, LINESEARCH_FAILED
export SolverStatus, StepInfo, Options
export LUFactorization, QRFactorization, SVDFactorization, TF32, FP16, BF16

include("base/returncodes.jl")
include("base/status.jl")
include("base/options.jl")
include("base/reductions.jl")
include("base/adapt.jl")

include("linear/methods.jl")

end
