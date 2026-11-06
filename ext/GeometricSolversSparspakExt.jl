module GeometricSolversSparspakExt

using GeometricSolvers: GeometricSolvers, LUFactorization, SINGULAR, SUCCESS
using LinearAlgebra: LinearAlgebra
using SparseArrays: SparseArrays, SparseMatrixCSC
using Sparspak: Sparspak
using Sparspak.SparseCSCInterface: sparspaklu, sparspaklu!

# Sparse LU factorisation workspace wrapping a `Sparspak` solver `lu` over a preallocated
# `SparseMatrixCSC` buffer `factors` whose column pointers `colptr` and row indices `rowval` match
# `jacobian_prototype`. `init_linear` builds the unfactorised `Sparspak.SparseSolver`, and the first
# `factorize!!` call runs the symbolic ordering and factorisation, which subsequent `factorize!!`
# calls reuse with `allow_pattern_change = false`, mapping a zero pivot or a non-finite entry to
# `info = 1` (`SINGULAR`).
struct SparspakLU{A <: SparseMatrixCSC, L}
    factors::A
    colptr::Vector{Int}
    rowval::Vector{Int}
    lu::L
    info::Int
    scale_exponent::Int
end

function GeometricSolvers.returncode(fac::SparspakLU)
    return iszero(fac.info) ? SUCCESS : SINGULAR
end

@inline function GeometricSolvers.factor_diagnostics(
        d, ::SparspakLU, r, fnorm::R) where {R <: Real}
    return Int32(-1), R(NaN)
end

function GeometricSolvers.init_linear(
        m::LUFactorization,
        P::SparseMatrixCSC{<:Any, Int},
        x::Vector{T}
) where {T <: AbstractFloat}
    TF = GeometricSolvers.factor_eltype(m, T)
    TF === T || throw(
        ArgumentError(
        "sparse LUFactorization with jacobian_prototype does not support mixed precision ($TF vs $T)"
    )
    )
    n = length(x)
    size(P) == (n, n) || throw(
        ArgumentError(
        "jacobian_prototype has size $(size(P)), expected ($n, $n)"
    )
    )
    factors = similar(P, T)
    fill!(SparseArrays.nonzeros(factors), zero(T))
    colptr = copy(SparseArrays.getcolptr(P))
    rowval = copy(SparseArrays.rowvals(P))
    lu = sparspaklu(factors; factorize = false)
    return SparspakLU(factors, colptr, rowval, lu, 0, 0)
end

function GeometricSolvers.factorize!!(
        fac::SparspakLU,
        ::LUFactorization{Nothing},
        J::SparseMatrixCSC{T, Int}
) where {T <: AbstractFloat}
    if size(J) != size(fac.factors) ||
       SparseArrays.getcolptr(J) != fac.colptr ||
       SparseArrays.rowvals(J) != fac.rowval
        throw(
            ArgumentError(
            "sparse Jacobian sparsity pattern differs from jacobian_prototype"
        )
        )
    end
    J === fac.factors ||
        copyto!(SparseArrays.nonzeros(fac.factors), SparseArrays.nonzeros(J))
    all(isfinite, SparseArrays.nonzeros(fac.factors)) ||
        return SparspakLU(fac.factors, fac.colptr, fac.rowval, fac.lu, 1, 0)
    try
        sparspaklu!(fac.lu, fac.factors; allow_pattern_change = false)
    catch e
        # only the numerical factorisation sets `errflag` (a zero pivot); every other error of
        # Sparspak propagates
        e isa ErrorException && !iszero(fac.lu.slvr.errflag) || rethrow()
        return SparspakLU(fac.factors, fac.colptr, fac.rowval, fac.lu, 1, 0)
    end
    ok = iszero(fac.lu.slvr.errflag) &&
         all(isfinite, fac.lu.slvr.lnz) &&
         all(isfinite, fac.lu.slvr.unz)
    return SparspakLU(fac.factors, fac.colptr, fac.rowval, fac.lu, ok ? 0 : 1, 0)
end

function GeometricSolvers.ldiv!!(
        d::Vector{T},
        fac::SparspakLU{SparseMatrixCSC{T, Int}},
        r::Vector{T}
) where {T <: AbstractFloat}
    d === r || copyto!(d, r)
    LinearAlgebra.ldiv!(fac.lu, d)
    return d
end

end
