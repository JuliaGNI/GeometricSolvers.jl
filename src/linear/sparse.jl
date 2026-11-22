# The sparse LU factor of `UMFPACKFactorization`: UMFPACK, from `SparseArrays`, on a
# `SparseMatrixCSC` Jacobian in `Float64` or `ComplexF64`.

using LinearAlgebra: issuccess
using SparseArrays: SparseMatrixCSC, getcolptr, nonzeros, rowvals
using SparseArrays.UMFPACK: UmfpackLU

# The element types that UMFPACK factorises.
const UMFPACKEltype = Union{Float64, ComplexF64}

"""
    UMFPACKLU{T}

The sparse LU factor of [`UMFPACKFactorization`](@ref) in element type `T`, `Float64` or
`ComplexF64`. `factors` is the `SparseMatrixCSC{T, Int}` buffer with the pattern of the Jacobian
prototype, into which [`factorize!!`](@ref) writes the Jacobian; `lu` is the UMFPACK factor
`SparseArrays.UMFPACK.UmfpackLU{T, Int}` of that pattern; `info` is `1` for a singular Jacobian or
one with a non-finite entry, and `0` otherwise.

[`init_linear`](@ref) builds `lu` with no analysis and no factorisation. The first `factorize!!`
runs UMFPACK's symbolic analysis and its numeric factorisation, and every later `factorize!!` the
numeric factorisation alone, on the symbolic object `lu.symbolic` of the first. Every factor that
`factorize!!` returns from one `init_linear` shares `factors` and `lu`.
"""
struct UMFPACKLU{T <: UMFPACKEltype}
    factors::SparseMatrixCSC{T, Int}
    lu::UmfpackLU{T, Int}
    info::Int
end

returncode(fac::UMFPACKLU) = iszero(fac.info) ? SUCCESS : SINGULAR

@inline function factor_diagnostics(d, ::UMFPACKLU, r, fnorm::R) where {R <: Real}
    return Int32(-1), R(NaN)
end

# The configuration that `init_linear` below builds a factor for, which is the one where a
# detected Jacobian pattern replaces the dense factor.
@inline has_sparse_factor(::UMFPACKFactorization, ::Vector{<:UMFPACKEltype}) = true

function init_linear(
        ::UMFPACKFactorization, P::SparseMatrixCSC{<:Any, Int},
        x::Vector{T}) where {T <: UMFPACKEltype}
    n = length(x)
    size(P) == (n, n) || throw(
        ArgumentError(
        "jacobian_prototype has size $(size(P)), expected ($n, $n)"
    )
    )
    factors = similar(P, T)
    fill!(nonzeros(factors), zero(T))
    return UMFPACKLU(factors, UmfpackLU(factors), 0)
end

function init_linear(::UMFPACKFactorization, pattern, x::AbstractVector)
    throw(
        ArgumentError(
        "UMFPACKFactorization factorises a SparseMatrixCSC{<:Any, Int} Jacobian pattern on a " *
        "Vector{Float64} or Vector{ComplexF64} iterate; got the pattern $(typeof(pattern)) " *
        "and the iterate $(typeof(x))"
    )
    )
end

# The `m_rows × n` factor of a `LeastSquaresProblem`, which carries no sparse pattern, takes the
# refusal above.
function init_linear(m::UMFPACKFactorization, pattern, x::AbstractVector, ::Int)
    init_linear(m, pattern, x)
end

function factorize!!(
        fac::UMFPACKLU{T}, ::UMFPACKFactorization,
        J::SparseMatrixCSC{T, Int}) where {T <: UMFPACKEltype}
    if size(J) != size(fac.factors) ||
       getcolptr(J) != getcolptr(fac.factors) ||
       rowvals(J) != rowvals(fac.factors)
        throw(
            ArgumentError(
            "sparse Jacobian sparsity pattern differs from jacobian_prototype"
        )
        )
    end
    J === fac.factors || copyto!(nonzeros(fac.factors), nonzeros(J))
    # UMFPACK factorises some matrices with a `NaN` entry without a warning and throws
    # `SingularException` on others, so a non-finite entry is refused first
    all(isfinite, nonzeros(fac.factors)) || return UMFPACKLU(fac.factors, fac.lu, 1)
    # `check = false` leaves a singular matrix to `issuccess` instead of a `SingularException`
    lu!(fac.lu, fac.factors; check = false)
    return UMFPACKLU(fac.factors, fac.lu, issuccess(fac.lu) ? 0 : 1)
end

function ldiv!!(d::Vector{T}, fac::UMFPACKLU{T}, r::Vector{T}) where {T <: UMFPACKEltype}
    ldiv!(d, fac.lu, r)
    return d
end
