# The public linear-solve object: a matrix `A` factorised once by `init` with one of the dense
# factor types, and solved for many right-hand sides by `solve!`. It holds the factor of
# `init_linear` and `factorize!!` and solves through `ldiv!!` where `m ≥ n`. Where `m < n` it
# factorises `Aᴴ` and forms the minimum-norm solution from that factor itself: `x = U Σ⁺ Vᴴ b`
# from the SVD `Aᴴ = U Σ Vᴴ`, and `x = Q R⁻ᴴ b` from the QR `Aᴴ = Q R`.

"""
    LinearProblem(A)

The linear system `A x = b` of a dense `m × n` matrix `A`, held in the field `A`, for the
right-hand sides `b` that `solve!` of a [`LinearSolver`](@ref) is given.
`A` is a static matrix (R1) or any other dense `AbstractMatrix`, on the host or on a device (R3).
A matrix with a zero dimension and a sparse matrix raise an `ArgumentError`. The element type of
`A` is a real or complex floating-point type that the factors of the method take; `A` is not
converted, so that an integer matrix raises an error in the factorisation or in the solve.

`init(prob, method)` factorises `A` with `method` and returns a [`LinearSolver`](@ref); `method` is
one of [`LUFactorization`](@ref), [`QRFactorization`](@ref), [`SVDFactorization`](@ref),
[`LDLTFactorization`](@ref) and [`CholeskyFactorization`](@ref), with a factorisation precision of
`nothing` or a float type. Another method, such as [`GMRES`](@ref), and a precision marker such as
[`TF32`](@ref) raise an `ArgumentError`.

- `m == n`: every factor type; [`CholeskyFactorization`](@ref) on a device array raises an
  `ArgumentError`, because it has no device factor.
- `m > n`: the least-squares solution through `QRFactorization` and `SVDFactorization`, the
  minimum-norm one where the rank of the SVD is below `n`. `LUFactorization`,
  `LDLTFactorization` and `CholeskyFactorization` raise an `ArgumentError` for `m ≠ n`.
- `m < n`: the minimum-norm solution, through the SVD of `Aᴴ`, truncated at its rank, on every
  route, or through a QR of `Aᴴ` on a static matrix and a host array: `x = Q R⁻ᴴ b` with
  `Aᴴ = Q R`. A device array with `QRFactorization` raises an `ArgumentError`. The QR is not
  rank-revealing: an exact zero on the diagonal of `R` gives `SINGULAR`, and a numerically
  rank-deficient `A` without one is solved with no truncation, dividing by the small diagonal
  entries of `R`.

The factors take the element type `factor_eltype(method, eltype(A))`: a method with the precision
`Float32` factorises a `Float64` matrix in `Float32`, and the solve divides `b` by its largest
absolute entry before it converts it, as [`ldiv!!`](@ref) does. The rank cutoff of an SVD is that
of [`SVDFactorization`](@ref) for the factored matrix, which has `max(m, n)` rows: by default
`√max(m, n) · eps(R) · σ₁`.

# Examples

```jldoctest
julia> solver = init(LinearProblem([2.0 1.0; 1.0 3.0]), LUFactorization());

julia> x, status = solve!(solver, [3.0, 5.0]);

julia> x ≈ [0.8, 1.4]
true

julia> status.code
SUCCESS::ReturnCode = 0x00

julia> x, status = solve!(init(LinearProblem([1.0 1.0]), SVDFactorization()), [2.0]);

julia> x ≈ [1.0, 1.0], status.rank
(true, 1)
```
"""
struct LinearProblem{M <: AbstractMatrix}
    A::M
    function LinearProblem(A::AbstractMatrix)
        A isa SparseArrays.AbstractSparseMatrix && throw(ArgumentError(
            "LinearProblem takes a dense matrix, got the type $(nameof(typeof(A)))"))
        isempty(A) && throw(ArgumentError(
            "LinearProblem needs a matrix of at least one row and one column, got the size " *
            "$(size(A))"))
        Base.require_one_based_indexing(A)
        return new{typeof(A)}(A)
    end
end

"""
    LinearStatus

The result of `solve!` of a [`LinearSolver`](@ref) beside the solution. It is `isbits`.

# Fields

- `code::ReturnCode`: `SUCCESS`; `SINGULAR` for a factor whose factorisation failed (`info ≠ 0`,
  as [`factorize!!`](@ref) states), with every entry of `x` `NaN`; or `NONFINITE` for a matrix
  with a non-finite entry, which is not factorised and gives every entry of `x` `NaN`, and for a
  right-hand side or a solution with a non-finite entry. A non-finite right-hand side gives every
  entry of `x` `NaN`.
- `rank::Int32`: the numerical rank of an SVD factor, the number of its singular values above
  the cutoff of [`SVDFactorization`](@ref), and `Int32(-1)` for every other factor, for a failed
  factorisation and for a non-finite matrix, as `rank` of [`SolverStatus`](@ref).
"""
struct LinearStatus
    code::ReturnCode
    rank::Int32
end

"""
    LinearSolver

The factorised [`LinearProblem`](@ref) that `init(prob, method)` returns.

    solve!(solver::LinearSolver, b) -> (x, status)

solves `A x = b` with its factors and returns the solution `x` and a [`LinearStatus`](@ref). `b`
has `m` entries, the element type of `A` and its storage: a static vector for a solver of a static
matrix, and an array of the same `get_backend` for a dense or device solver. Another length raises
a `DimensionMismatch`, and another element type or storage an `ArgumentError`. `x` has `n` entries
in the element type of `A`: a new `SVector` for a static matrix, and the buffer `solver.x` for a
dense or device matrix, which the next `solve!` overwrites. It is the solution, the least-squares
solution or the minimum-norm solution that [`LinearProblem`](@ref) states for the shape of `A` and
the method. Every entry of `x` is `NaN` where the status is `SINGULAR`, where `A` is not finite,
and where `b` is not finite. [`refactorize!!`](@ref) factorises a new matrix with the solver.

On a static matrix the solver is a value, `isbits` for an `isbits` element type; on a dense or
device array it holds the buffers that `solve!` and `refactorize!!` reuse. Its fields are
`method`, `factor`, the factor of the method, `x`, the solution buffer, or `nothing` on a static
matrix, `y`, a buffer of `m` entries in the factor element type for an `m < n` system whose factor
element type is not that of `A`, and `nothing` otherwise, `m` and `n`, the size of `A`, `finite`,
whether every entry of the factorised `A` is finite, and `scale_exponent`, the exponent `e` of the
power of two `2^e` that `Aᴴ` was divided by when it was loaded into the factor of an `m < n`
system on a dense or device array, and 0 otherwise.
"""
struct LinearSolver{T, W, M <: LinearMethod, F, X, Y}
    method::M
    factor::F
    x::X
    y::Y
    m::Int
    n::Int
    finite::Bool
    scale_exponent::Int
end

# `W` is `true` for an `m < n` system, whose factor is that of `Aᴴ`.
function LinearSolver{T, W}(method, factor, x, y, m, n, finite, e) where {T, W}
    return LinearSolver{T, W, typeof(method), typeof(factor), typeof(x), typeof(y)}(
        method, factor, x, y, m, n, finite, e)
end

const StaticLinearSolver{T, W} = LinearSolver{T, W, <:LinearMethod, <:Any, Nothing}
const ArrayLinearSolver{T, W} = LinearSolver{T, W, <:LinearMethod, <:Any, <:AbstractVector}

const DenseFactorMethod = Union{LUFactorization, QRFactorization, SVDFactorization,
    LDLTFactorization, CholeskyFactorization}
const SquareFactorMethod = Union{LUFactorization, LDLTFactorization, CholeskyFactorization}

function init(prob::LinearProblem, method)
    check_linear_method(method)
    A = prob.A
    m, n = size(A)
    m != n && method isa SquareFactorMethod &&
        throw(ArgumentError(
            "$(nameof(typeof(method))) factorises a square matrix alone, got the size ($m, $n); " *
            "take QRFactorization() or SVDFactorization()"))
    # the device factor follows the template `similar(A, T, m)` of `linear_solver`, which is a
    # device array for a wrapped device array too, such as its adjoint
    m < n && method isa QRFactorization && similar(A, m) isa AbstractGPUArray &&
        throw(ArgumentError(
            "QRFactorization has no minimum-norm solve of an m × n system with m < n on a device " *
            "array; take SVDFactorization()"))
    return linear_solver(A, method)
end

# One body for every refusal of `init`, so that no two methods of it overlap. The precision is
# asked by dispatch: an `isa` test on `LinearMethod{<:PrecisionMarker}` builds the `UnionAll` at
# run time and allocates.
function check_linear_method(method)
    method isa DenseFactorMethod || throw(ArgumentError(
        "a LinearProblem is solved by a dense factorisation, got $(nameof(typeof(method)))"))
    has_precision_marker(method) && throw(ArgumentError(
        "a precision marker has no factorisation; give the precision as a float type"))
    return nothing
end
has_precision_marker(::LinearMethod{<:PrecisionMarker}) = true
has_precision_marker(::LinearMethod) = false

# --- R1: the factor of a static matrix is a value ---

# A non-finite matrix is not factorised: the factor of the zero matrix of its type stands in, so
# that the solver has one type, and `solve!` reads `finite` before the factor.
function linear_solver(A::StaticMatrix{M, N, T}, method) where {M, N, T}
    finite = all(isfinite, A)
    S = SMatrix{M, N, T}(A)
    fac = static_factor(method, finite ? S : zero(S), Val(M < N))
    return LinearSolver{T, M < N}(method, fac, nothing, nothing, M, N, finite, 0)
end

@inline static_factor(method, S, ::Val{false}) = factorize!!(nothing, method, S)
@inline static_factor(method, S, ::Val{true}) = factorize!!(nothing, method, adjoint(S))

# --- R3: the factor of a dense or device matrix holds buffers ---

function linear_solver(A::AbstractMatrix{T}, method) where {T}
    m, n = size(A)
    x = similar(A, T, n)
    if m < n
        fac = init_linear(method, nothing, similar(A, T, m), n)
        F = factor_eltype(method, T)
        y = F === T ? nothing : similar(A, F, m)
        return factorize_into(LinearSolver{T, true}(method, fac, x, y, m, n, false, 0), A)
    end
    fac = init_linear(method, nothing, x, m == n ? nothing : m)
    return factorize_into(LinearSolver{T, false}(method, fac, x, nothing, m, n, false, 0), A)
end

# Factorise `A` into the buffers of `s`. An `m < n` system loads `Aᴴ / 2^e` into the factor with a
# broadcast, which reads a device array with no scalar index, and `e`, the exponent of the largest
# `|re|` or `|im|` of `A`, keeps it in the range of a narrower factor element type.
function factorize_into(s::LinearSolver{T, W}, A::AbstractMatrix) where {T, W}
    fac = s.factor
    all(isfinite, A) ||
        return LinearSolver{T, W}(s.method, fac, s.x, s.y, s.m, s.n, false, 0)
    e = 0
    if W
        e = matrix_exponent(A)
        fac.factors .= ldexp2.(adjoint(A), -e)
        fac = factorize!!(fac, s.method, fac.factors)
    else
        fac = factorize!!(fac, s.method, A)
    end
    return LinearSolver{T, W}(s.method, fac, s.x, s.y, s.m, s.n, true, e)
end

"""
    refactorize!!(solver::LinearSolver, A)

Factorise the matrix `A` with the method of `solver`, and return the solver of `A`. `A` has the
size, the element type and the storage of the matrix that `solver` was built for: a static matrix
of the same size for a solver of a static matrix, and an array of the same `get_backend` for a
dense or device solver. Another size raises a `DimensionMismatch`, and another element type or
storage an `ArgumentError`.

The `!!` follows the convention of the package: a solver of a static matrix is a value, and the
result is a new solver, while `solver` still solves its own matrix; a dense or device solver
writes the factors of `A` into its buffers, and the result holds them, the solution buffer `x`
included, so that `solver` is not to be used again. A non-finite `A` is not factorised, and the
solve of the result gives `NONFINITE` until a finite matrix is factorised.
"""
function refactorize!!(s::StaticLinearSolver{T}, A::StaticMatrix) where {T}
    check_matrix(s, A)
    return linear_solver(A, s.method)
end

function refactorize!!(s::ArrayLinearSolver, A::AbstractMatrix)
    A isa StaticArray && throw(ArgumentError(
        "a solver of a dense or device matrix takes no static matrix"))
    A isa SparseArrays.AbstractSparseMatrix && throw(ArgumentError(
        "a solver of a dense or device matrix takes no sparse matrix"))
    check_matrix(s, A)
    get_backend(A) == get_backend(s.x) || throw(ArgumentError(
        "the matrix is on the backend $(nameof(typeof(get_backend(A)))), the solver on " *
        "$(nameof(typeof(get_backend(s.x))))"))
    Base.require_one_based_indexing(A)
    return factorize_into(s, A)
end

function refactorize!!(::StaticLinearSolver, A::AbstractMatrix)
    throw(ArgumentError(
        "a solver of a static matrix takes a static matrix, got the type $(nameof(typeof(A)))"))
end

function check_matrix(s::LinearSolver{T}, A::AbstractMatrix) where {T}
    size(A) == (s.m, s.n) || throw(DimensionMismatch(
        "the solver factorises a matrix of the size $((s.m, s.n)), got $(size(A))"))
    eltype(A) === T || throw(ArgumentError(
        "the solver factorises a matrix of the element type $T, got $(eltype(A))"))
    return nothing
end

function check_rhs(s::LinearSolver{T}, b::AbstractVector) where {T}
    length(b) == s.m || throw(DimensionMismatch(
        "the right-hand side has $(length(b)) entries, the matrix $(s.m) rows"))
    eltype(b) === T || throw(ArgumentError(
        "the right-hand side has the element type $(eltype(b)), the matrix $T"))
    return nothing
end

function solve!(s::StaticLinearSolver{T, W}, b::StaticVector) where {T, W}
    check_rhs(s, b)
    fac = s.factor
    nan = static_nan(fac, Val(W), T)
    s.finite || return nan, LinearStatus(NONFINITE, Int32(-1))
    returncode(fac) == SUCCESS || return nan, LinearStatus(SINGULAR, Int32(-1))
    x = static_solve(fac, b, Val(W), T)
    all(isfinite, b) || (x = nan)
    return x, LinearStatus(all(isfinite, x) ? SUCCESS : NONFINITE, factor_rank(fac))
end

function solve!(::StaticLinearSolver, b::AbstractVector)
    throw(ArgumentError(
        "a solver of a static matrix takes a static right-hand side, got the type " *
        "$(nameof(typeof(b)))"))
end

function solve!(s::ArrayLinearSolver{T, W}, b::AbstractVector) where {T, W}
    b isa StaticArray && throw(ArgumentError(
        "a solver of a dense or device matrix takes no static right-hand side"))
    check_rhs(s, b)
    x = s.x
    get_backend(b) == get_backend(x) || throw(ArgumentError(
        "the right-hand side is on the backend $(nameof(typeof(get_backend(b)))), the solver " *
        "on $(nameof(typeof(get_backend(x))))"))
    s.finite || return fill!(x, T(NaN)), LinearStatus(NONFINITE, Int32(-1))
    fac = s.factor
    returncode(fac) == SUCCESS || return fill!(x, T(NaN)), LinearStatus(SINGULAR, Int32(-1))
    if W
        wide_solve!(x, fac, b, s.y, s.scale_exponent + fac.scale_exponent)
    else
        ldiv!!(x, fac, b)
    end
    all(isfinite, b) || fill!(x, T(NaN))
    return x, LinearStatus(all(isfinite, x) ? SUCCESS : NONFINITE, factor_rank(fac))
end

factor_rank(fac) = Int32(-1)
function factor_rank(fac::Union{StaticSVD, StaticRectSVD, DenseSVD, DeviceSVD})
    return Int32(getfield(fac, :rank))
end

# --- the solves of a static matrix ---

# The `SVector` of `n` entries `NaN` in `T`: `n` is the order of a square factor, the column count
# of the factor of a tall matrix and the row count of the factor of `Aᴴ`.
@inline function static_nan(fac, w::Val, ::Type{T}) where {T}
    n = static_columns(fac, w)
    return SVector{n, T}(ntuple(_ -> T(NaN), Val(n)))
end
function static_columns(
        ::Union{StaticLU{N}, StaticQR{N}, StaticSVD{N}, StaticLDLT{N},
            StaticCholesky{N}},
        ::Val{false}) where {N}
    N
end
function static_columns(::Union{StaticRectQR{M, N}, StaticRectSVD{M, N}}, ::Val{false}) where {
        M, N}
    N
end
function static_columns(::Union{StaticRectQR{M, N}, StaticRectSVD{M, N}}, ::Val{true}) where {
        M, N}
    M
end

@inline static_solve(fac, b, ::Val{false}, ::Type{T}) where {T} = ldiv!!(nothing, fac, b)

# `x = z · 2^-e` of the solution `z` in `TF` of `Aᴴ / 2^e`, or, where `b` was divided by its largest
# absolute entry `s = frac · 2^ex` before it was converted to `TF`, `z · frac · 2^(ex - e)` in one
# rounding, converted to `T`.
@inline function static_solve(fac::Union{StaticRectQR{N, M, TF}, StaticRectSVD{N, M, TF}},
        b::StaticVector{M}, ::Val{true}, ::Type{T}) where {N, M, TF, T}
    e = Int(getfield(fac, :scale_exponent))
    if TF === T
        z = static_wide_solve(fac, SVector{M, TF}(b))
        return iszero(e) ? z : map(a -> ldexp2(a, -e), z)
    end
    s = maximum(abs, b)
    iszero(s) && return zero(SVector{N, T})
    z = static_wide_solve(fac, map(a -> TF(a / s), SVector{M, T}(b)))
    frac, ex = frexp(s)
    return map(a -> T(ldexp2(a * frac, ex - e)), z)
end

# `U Σ⁺ Vᴴ r` of the SVD `U Σ Vᴴ` of `Aᴴ / 2^e`, truncated at its rank.
@inline function static_wide_solve(
        fac::StaticRectSVD{N, M, TF}, r::SVector{M, TF}) where {N, M, TF}
    σ̂ = getfield(fac, :σ̂)
    rk = getfield(fac, :rank)
    c = getfield(fac, :V)' * r
    d = SVector{M, TF}(ntuple(j -> j <= rk ? c[j] / σ̂[j] : zero(TF), Val(M)))
    return getfield(fac, :U) * d
end

# `Q R⁻ᴴ r` of the QR `Q R` of `Aᴴ / 2^e`.
@inline function static_wide_solve(fac::StaticRectQR{N, M, TF}, r::SVector{
        M, TF}) where {N, M, TF}
    z = local_vector(TF, Val(N))
    wide_qr_solve!(z, fac.factors, fac.τ, r, nothing)
    return SVector{N, TF}(z)
end

# The minimum-norm solution `z = Q R⁻ᴴ (r / s)` of `Rᴴ Qᴴ z = r / s` from the Householder QR
# `Q R` of the `n × m` matrix `Aᴴ`, `n > m`, in `factors` and `τ`: the forward substitution
# `Rᴴ y = r / s`, then `z = Q [y; 0] = H₁ ⋯ Hₘ [y; 0]` with `Hₖ = I - τₖ w wᴴ`, `w[k] = 1`, `Hₘ`
# applied first. `s = nothing` takes `r` as it is. Scalar loops, for a static factor and a host
# array.
@inline function wide_qr_solve!(z::AbstractVector{F}, factors::AbstractMatrix{F},
        τ::AbstractVector, r::AbstractVector, s) where {F}
    n, m = size(factors)
    @inbounds begin
        for k in 1:m
            acc = F(scaled_entry(r[k], s))
            for j in 1:(k - 1)
                acc -= conj(factors[j, k]) * z[j]
            end
            z[k] = acc / conj(factors[k, k])
        end
        for i in (m + 1):n
            z[i] = zero(F)
        end
        for k in m:-1:1
            c = z[k]
            for i in (k + 1):n
                c += conj(factors[i, k]) * z[i]
            end
            c *= τ[k]
            z[k] -= c
            for i in (k + 1):n
                z[i] -= factors[i, k] * c
            end
        end
    end
    return z
end

@inline scaled_entry(a, ::Nothing) = a
@inline scaled_entry(a, s) = a / s

# --- the m < n solves of a dense or device matrix ---

# `x = U Σ⁺ Vᴴ b · 2^-e` from the SVD `U Σ Vᴴ` of `Aᴴ / 2^e`, truncated at its rank: the device SVD
# zeroes the columns of `U` past the rank and the host SVD does not, so the coordinates past it
# are zeroed here. `mul!` and broadcasts alone, so that a device array is read with no scalar index.
function wide_solve!(x::AbstractVector{T}, fac::Union{DenseSVD, DeviceSVD},
        b::AbstractVector{T}, y, e::Int) where {T}
    F = eltype(fac.U)
    m = size(fac.U, 2)
    rk = fac.rank
    c = fac.tmp
    σ̂ = svd_divisors(fac, m)
    if F === T
        mul!(c, fac.VT, b)
        wide_coordinates!(c, σ̂, rk, m)
        mul!(x, fac.U, c)
        iszero(e) || (x .= ldexp2.(x, -e))
        return x
    end
    s = maximum(abs, b)
    iszero(s) && return fill!(x, zero(T))
    y .= F.(b ./ s)
    mul!(c, fac.VT, y)
    wide_coordinates!(c, σ̂, rk, m)
    z = fac.rhs::AbstractVector
    mul!(z, fac.U, c)
    frac, ex = frexp(s)
    x .= T.(ldexp2.(z .* frac, ex - e))
    return x
end

# The coordinates `Vᴴ b` divided by the singular values `σ̂` of the first `rk` directions, and zero
# past them.
function wide_coordinates!(c::AbstractVector, σ̂::AbstractVector, rk::Int, m::Int)
    view(c, 1:rk) ./= view(σ̂, 1:rk)
    fill!(view(c, (rk + 1):m), zero(eltype(c)))
    return c
end

# The singular values `σ̂` of `Aᴴ / 2^e`: the first `m` entries of `rwork` on the host and `σ̂` on a
# device.
svd_divisors(fac::DenseSVD, m::Int) = view(fac.rwork, 1:m)
svd_divisors(fac::DeviceSVD, m::Int) = fac.σ̂

# `x = Q R⁻ᴴ b · 2^-e` from the QR `Q R` of `Aᴴ / 2^e`, on the host: scalar loops in `x`, or in the
# factor's buffer `rhs` of `n` entries where the factor element type is not that of `A`.
function wide_solve!(x::AbstractVector{T}, fac::DenseQR, b::AbstractVector{T}, y,
        e::Int) where {T}
    F = eltype(fac.factors)
    if F === T
        wide_qr_solve!(x, fac.factors, fac.τ, b, nothing)
        iszero(e) || (x .= ldexp2.(x, -e))
        return x
    end
    s = maximum(abs, b)
    iszero(s) && return fill!(x, zero(T))
    z = fac.rhs::AbstractVector
    wide_qr_solve!(z, fac.factors, fac.τ, b, s)
    frac, ex = frexp(s)
    x .= T.(ldexp2.(z .* frac, ex - e))
    return x
end
