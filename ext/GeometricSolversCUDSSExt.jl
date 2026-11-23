# The cuDSS sparse LU factorisation and solve of `LUFactorization()` on a `CuSparseMatrixCSR{T}`
# Jacobian prototype and a `CuVector{T}` iterate, `T <: LinearAlgebra.BlasFloat`, through
# CUDSS.jl, as the CUDA extension serves a dense `CuMatrix` with cuSOLVER.

module GeometricSolversCUDSSExt

using CUDA: CUDA, CuVector
using CUDSS: CUDSS, CudssMatrix, CudssSolver, cudss, cudss_get
using CUDSS.cuSPARSE: CuSparseMatrixCSR
using GeometricSolvers: GeometricSolvers, LUFactorization, SINGULAR, SUCCESS
using LinearAlgebra: BlasFloat

# The cuDSS factor of a `CuSparseMatrixCSR{T}` Jacobian: `factors` is the device buffer with the
# pattern of `jacobian_prototype`, into which `factorize!!` writes the Jacobian; `solver` is the
# `CudssSolver` of the general (`"G"`, LDU) structure on that buffer; `info` is `1` for a Jacobian
# with a non-finite entry or a nonzero cuDSS `info` after the factorisation, and `0` otherwise;
# `analyses` holds the count of the symbolic analyses that `solver` ran. `init_linear` runs none;
# the first `factorize!!` of finite values runs the analysis and the factorisation, and every later
# one the refactorisation alone, so a solver runs one analysis. Every factor that `factorize!!`
# returns from one `init_linear` shares `factors`, `solver` and the `analyses` cell, because a
# Newton step with no reuse factorises from the factor that `init` stored in its state.
struct CUDSSLU{T <: BlasFloat, A <: CuSparseMatrixCSR{T}, S <: CudssSolver{T}}
    factors::A
    solver::S
    info::Int
    analyses::Base.RefValue{Int}
end

GeometricSolvers.returncode(fac::CUDSSLU) = iszero(fac.info) ? SUCCESS : SINGULAR

@inline function GeometricSolvers.factor_diagnostics(
        d, ::CUDSSLU, r, fnorm::R) where {R <: Real}
    return Int32(-1), R(NaN)
end

function GeometricSolvers.init_linear(
        m::LUFactorization,
        P::CuSparseMatrixCSR{<:Any, Ti},
        x::CuVector{T}
) where {T <: BlasFloat, Ti <: Union{Int32, Int64}}
    TF = GeometricSolvers.factor_eltype(m, T)
    TF === T || throw(
        ArgumentError(
        "sparse LUFactorization with a CuSparseMatrixCSR jacobian_prototype does not " *
        "support mixed precision ($TF vs $T)"
    )
    )
    n = length(x)
    size(P) == (n, n) || throw(
        ArgumentError(
        "jacobian_prototype has size $(size(P)), expected ($n, $n)"
    )
    )
    nzval = fill!(similar(P.nzVal, T), zero(T))
    factors = CuSparseMatrixCSR{T, Ti}(copy(P.rowPtr), copy(P.colVal), nzval, size(P))
    return CUDSSLU(factors, CudssSolver(factors, "G", 'F'), 0, Ref(0))
end

function GeometricSolvers.factorize!!(
        fac::CUDSSLU{T},
        ::LUFactorization,
        J::CuSparseMatrixCSR{T}
) where {T <: BlasFloat}
    A = fac.factors
    if size(J) != size(A) ||
       (J !== A && (J.rowPtr != A.rowPtr || J.colVal != A.colVal))
        throw(
            ArgumentError(
            "sparse Jacobian sparsity pattern differs from jacobian_prototype"
        )
        )
    end
    J === A || copyto!(A.nzVal, J.nzVal)
    all(isfinite, A.nzVal) || return CUDSSLU(A, fac.solver, 1, fac.analyses)
    n = size(A, 1)
    x = CudssMatrix(T, n)
    b = CudssMatrix(T, n)
    if iszero(fac.analyses[])
        cudss("analysis", fac.solver, x, b)
        cudss("factorization", fac.solver, x, b; asynchronous = false)
        fac.analyses[] += 1
    else
        cudss("refactorization", fac.solver, x, b; asynchronous = false)
    end
    info = cudss_get(fac.solver, "info")
    return CUDSSLU(A, fac.solver, iszero(info) ? 0 : 1, fac.analyses)
end

function GeometricSolvers.ldiv!!(
        d::CuVector{T},
        fac::CUDSSLU{T},
        r::CuVector{T}
) where {T <: BlasFloat}
    d === r || copyto!(d, r)
    cudss("solve", fac.solver, d, d; asynchronous = false)
    return d
end

end
