# The factors of the linear-solver methods, and the two functions of the linear-solver seam.
#
# `factorize!!` and `ldiv!!` follow the `!!` convention of `bang.jl`: the first argument is a
# buffer where the storage has one and `nothing` where it does not, and the caller always uses
# the returned value. R1 has no buffer: its factors are `isbits` values, returned new. R3 holds
# its Jacobian and workspace buffers in `DenseLU`, `DenseQR`, `DenseSVD` and `DenseLDLT` on the
# host and in `DeviceLU`, `DeviceQR`, `DeviceSVD` and `DeviceLDLT` on a device.

"""
    factorize!!(fac, method, J)

Factorise the matrix `J` with the linear-solver method `method` and return the factors.

For a static matrix (R1), `fac` is `nothing` or an earlier factor of the same type, whose value
is not used, and the result is a new [`StaticLU`](@ref), [`StaticQR`](@ref), [`StaticSVD`](@ref)
or [`StaticLDLT`](@ref) for a square `J`, and a [`StaticRectQR`](@ref) or [`StaticRectSVD`](@ref)
for the QR or SVD of an `M × N` matrix with `M > N` (the LU and the LDLᵀ have no rectangular
method). `J` is converted to the element type of the factors,
[`factor_eltype`](@ref). Where that is not the element type of `J`, `J` is first divided by the
power of two `2^e` that brings its largest `|re|` or `|im|` into `[1, 2)`, or as close as the
exponent range allows, and `e` is kept in the factor, so that the overall scale of `J` neither
overflows nor underflows in the factor type. A zero or non-finite `J` gives `e = 0`. A singular
`J` does not throw. LU
records the first zero pivot in `info`, QR the first zero diagonal entry of `R`, LDLᵀ the first
zero pivot block (or `1` for a non-finite entry), and
[`returncode`](@ref) then gives `SINGULAR`. QR is not rank-revealing: a singular `J` whose `R` has
no exact zero on the diagonal gives `SUCCESS`, and two equal rows, as a rule, give no exact zero.
The SVD records its numerical rank in `rank`, the number of singular values above the cutoff
`√m · eps(TF) · σ₁`, `m` the number of rows of `J`, and a rank deficiency gives `info == 0` and
`SUCCESS`; its `info` is nonzero
only where the sweep did not converge or a singular value is not finite, and `rank` is then 0. The
SVD is prescaled at the working precision too, because the sweep squares the entries.

For a dense or device factor (R3), `fac` is a [`DenseLU`](@ref), [`DenseQR`](@ref),
[`DenseSVD`](@ref), [`DenseLDLT`](@ref), [`DeviceLU`](@ref), [`DeviceQR`](@ref),
[`DeviceSVD`](@ref) or [`DeviceLDLT`](@ref)
built by [`init_linear`](@ref). `J` is written into `fac.factors`, divided by `2^e` as for a static
matrix where its element type is not that of `fac.factors`, and `fac.factors` is factorised in
place; `J === fac.factors` is factorised as it is. A [`DenseSVD`](@ref) or [`DeviceSVD`](@ref)
divides `J` by `2^e` unconditionally, `J === fac.factors` too: into `fac.factors` on the host before
`gesvd`, and into `fac.U` on a device before the one-sided Jacobi sweep. For [`DenseSVD`](@ref),
`info` is `1` where `J` has a non-finite entry, which is refused before `gesvd` and
leaves `U`, `S` and `VT` unwritten, `gesvd`'s non-convergence code where it fails, or `1` where a
singular value of `J / 2^e` is not finite; for [`DeviceSVD`](@ref) it is `1` where the Jacobi
sweep reaches its sweep cap without converging or meets a non-finite entry; `rank` is the
numerical rank as for a static matrix. An `Inf` in a dense `J` does not throw; the SVD and LDLᵀ
then report `SINGULAR`, and the step that the LU or QR factors give is not specified.

For a sparse factor (R3), which an extension builds in [`init_linear`](@ref) from a
`SparseMatrixCSC` `jacobian_prototype`, `J` must have the sparsity pattern of that prototype, or
`ArgumentError` is raised. A zero pivot, a non-finite entry of `J` and a non-finite factor give
`info = 1` and `SINGULAR` without throwing.

A precision marker such as [`TF32`](@ref) has no method for a static matrix: the call raises a
`MethodError`.
"""
function factorize!! end

"""
    ldiv!!(d, fac, r)

Solve `A d = r` with the factors `fac` of `A`, and return `d` in the element type of `r`. For a
rectangular `A` of size `m × n` with `m > n` (the QR and SVD factors of a
[`LeastSquaresProblem`](@ref)), `r` has `m` entries, `d` has `n`, and `d` is the least-squares
solution of `A d = r`, the minimum-norm one where an SVD factor has numerical rank below `n`.

For a static factor (R1), `d` is `nothing` and the result is a new `SVector`. For a dense factor
(R3), `d` is the vector buffer written in place and returned. Where the element
type of `r` is the element type of the factors, the solve runs directly in it. There a finite,
nonzero `r` whose largest `|re|` or `|im|` is within a few bits of `floatmax` (`2 + ⌈log2 N⌉`
bits for `N` entries) is divided by the smallest power of two `2^k` that leaves that headroom; any
other `r` is solved as it is. This keeps `Qᴴ r` and `Uᴴ r`, which the QR and SVD solves form
first, finite. The SVD, whose factors are those of `J / 2^e`, then divides the coordinates
`Uᴴ r` by its singular values times the smallest power of two `2^t` that keeps the coordinates
the same headroom below `floatmax`; `t` is 0 wherever they are already below it, and for LU and
QR. The step is multiplied by `2^(k + t - e)`, in one rounding, with `e` the exponent of
[`factorize!!`](@ref), so that the SVD step does not overflow where it is representable. The LU
and QR solves can still overflow in the triangular solve for such an `r`. An entry of `r` within
`k` bits of the subnormal range can lose those bits to the division, and a coordinate within `t`
bits of it to the SVD's. The step still overflows on every route where the solution itself
exceeds `floatmax`.
Where the element types differ,
a finite, nonzero `r` is scaled by `s = maximum(abs, r)`, converted to the factor type, solved,
converted back and multiplied by `s` (Kelley 2023, eqs. 2.17–2.18). Normalisation prevents
overflow from the overall right-hand-side scale in a lower factor precision; entries small
relative to the largest entry can still underflow. The scale takes no squares and overflows
only where an entry's absolute value does. In the mixed-type path, a zero `r` returns a zero
step without solving, and an `Inf` or `NaN` in `r` gives a non-finite step. In both paths the
step is multiplied by `2^-e`, the power of two that [`factorize!!`](@ref) divided `J` by, in the
element type of `r`. The SVD's solve leaves out the directions that its rank cutoff counted as
zero, so that a rank-deficient system gives a finite, truncated step rather than an enormous one;
an SVD factor with `info ≠ 0` has rank 0 and gives the zero step.
"""
function ldiv!! end

"""
    factor_eltype(method, T)

The element type of the factors of a working element type `T`: `T` when the method's
factorisation precision is `Nothing`, the precision `TF` for a real `T`, and `complex(TF)` for a
complex `T`.
"""
factor_eltype(::LinearMethod{Nothing}, ::Type{T}) where {T <: Number} = T
factor_eltype(::LinearMethod{F}, ::Type{<:Real}) where {F <: AbstractFloat} = F
factor_eltype(::LinearMethod{F}, ::Type{<:Complex}) where {F <: AbstractFloat} = complex(F)

"""
    StaticLU{N, TF, L}

The LU factors with partial pivoting of an `N × N` static matrix, in element type `TF`:
`factors` holds `U` on and above the diagonal and the multipliers of the unit lower `L` below it,
`ipiv` the row exchanged with row `k` at step `k`, `info` the index of the first zero pivot,
or 0, and `scale_exponent` the exponent `e` of the power of two `2^e` that the matrix was divided
by. `L = N * N` makes the `SMatrix` field concrete, so the factor is `isbits`.
"""
struct StaticLU{N, TF, L}
    factors::SMatrix{N, N, TF, L}
    ipiv::SVector{N, Int32}
    info::Int32
    scale_exponent::Int32
end

"""
    StaticQR{N, TF, L}

The Householder QR factors of an `N × N` static matrix, in element type `TF`: `factors` holds `R`
on and above the diagonal and below it the reflector vectors `w`, scaled to `w[k] = 1`; the
reflector of step `k` is `I - τ[k] w wᴴ`. `info` is the index of the first zero diagonal entry of
`R`, or 0, and `scale_exponent` the exponent `e` of the power of two `2^e` that the matrix was
divided by.
"""
struct StaticQR{N, TF, L}
    factors::SMatrix{N, N, TF, L}
    τ::SVector{N, TF}
    info::Int32
    scale_exponent::Int32
end

"""
    StaticSVD{N, TF, R, L}

The singular value decomposition of an `N × N` static matrix, in element type `TF`, computed by
the one-sided Jacobi iteration of `src/linear/static/svd.jl`. `factors` holds `U`, `V` holds the
right singular vectors as columns, and `σ̂` holds the `N` singular values of the matrix divided by
`2^e`, the ones the sweep computed and the solve divides by, in non-increasing order in
`R = real(TF)`; [`singular_values`](@ref) gives those of the matrix as it was passed. `rank` is
the number of singular values above the rank cutoff `√N · eps(TF) · σ̂₁`; the directions past it
are zeroed in `U`, and the solve of [`ldiv!!`](@ref) leaves their coordinates out. `info` is 1
where the sweep reached its cap of 30 sweeps before the columns were orthogonal or a singular
value is not finite, and `rank` is then 0; it is 0 otherwise, also for a rank-deficient matrix,
so that [`returncode`](@ref) reports a failed factorisation alone. `scale_exponent` is the
exponent `e` of the power of two `2^e` that the matrix was divided by.

`L = N * N` makes the `SMatrix` fields concrete, so the factor is `isbits`.
"""
struct StaticSVD{N, TF, R, L}
    factors::SMatrix{N, N, TF, L}
    V::SMatrix{N, N, TF, L}
    σ̂::SVector{N, R}
    info::Int32
    rank::Int32
    scale_exponent::Int32
end

"""
    DenseLU{A, P, V}

The LU factors with partial pivoting of a dense `n × n` host matrix, with `factors` the `n × n`
buffer in the factor element type, `ipiv` the `BlasInt` pivot vector of length `n`, `rhs` the
right-hand-side buffer of the factor element type when that differs from the working element type
and `nothing` otherwise, `info` the index of the first zero pivot, or 0, and `scale_exponent`
the exponent `e` of the power of two `2^e` that the matrix was divided by.
"""
struct DenseLU{
    A <: AbstractMatrix, P <: AbstractVector, V <: Union{Nothing, AbstractVector}}
    factors::A
    ipiv::P
    rhs::V
    info::Int
    scale_exponent::Int
end

"""
    DeviceLU{A, P, V, S}

The LU factors with partial pivoting of a dense `n × n` device matrix, with `factors` the `n × n`
device buffer in the factor element type, `ipiv` the `Int32` pivot vector of length `n`, `rhs`
the right-hand-side buffer of the factor element type when that differs from the working element
type and `nothing` otherwise, `info` the index of the first zero pivot, or 0, `scale_exponent`
the exponent `e` of the power of two `2^e` that the matrix was divided by, and `status` the
1-element `Int32` device status buffer.
"""
struct DeviceLU{A <: AbstractMatrix, P <: AbstractVector{Int32},
    V <: Union{Nothing, AbstractVector}, S <: AbstractVector{Int32}}
    factors::A
    ipiv::P
    rhs::V
    info::Int
    scale_exponent::Int
    status::S
end

"""
    DenseQR{A, T, W, V}

The Householder QR factors of a dense `m × n` host matrix (`m ≥ n`), with `factors` the `m × n`
buffer in the factor element type (`R` on and above the diagonal of its leading `n × n` block,
reflectors below it), `τ` the reflector scale vector of length `n`, `work` the LAPACK workspace
buffer, `rhs` the right-hand-side buffer of the factor element type, which is `nothing` only
where the working element type is the factor's and the problem is not a `LeastSquaresProblem`,
`info` the index of the first zero diagonal entry of `R`, or 0, and `scale_exponent` the
exponent `e` of the power of two `2^e` that the matrix was divided by.
"""
struct DenseQR{A <: AbstractMatrix, T <: AbstractVector, W <: AbstractVector,
    V <: Union{Nothing, AbstractVector}}
    factors::A
    τ::T
    work::W
    rhs::V
    info::Int
    scale_exponent::Int
end

"""
    DeviceQR{A, T, V, S}

The Householder QR factors of a dense `m × n` device matrix (`m ≥ n`), with `factors` the `m × n`
device buffer in the factor element type (`R` on and above the diagonal of its leading `n × n`
block, reflectors below it), `τ` the reflector scale vector of length `n`, `rhs` the
right-hand-side buffer of the factor element type, which is `nothing` only where the working
element type is the factor's and the problem is not a `LeastSquaresProblem`, `info` the index of
the first zero diagonal entry of `R`, or 0, `scale_exponent` the exponent `e` of the power of two
`2^e` that the matrix was divided by, and `status` the 1-element `Int32` device status buffer.
"""
struct DeviceQR{A <: AbstractMatrix, T <: AbstractVector,
    V <: Union{Nothing, AbstractVector}, S <: AbstractVector{Int32}}
    factors::A
    τ::T
    rhs::V
    info::Int
    scale_exponent::Int
    status::S
end

"""
    DenseSVD{A, S, W, RW, T, V}

The singular value decomposition factors `U * Diagonal(S) * VT` of a dense `m × n` host matrix
(`m ≥ n`), with `factors` the `m × n` Jacobian buffer in the factor element type, `U` the thin
`m × n` and `VT` the `n × n` unitary factors, `S` the vector of the singular values `σ̂ · 2^e` of
the matrix in the real factor type, which overflow to `Inf` where they exceed `floatmax`, `work`
the LAPACK workspace buffer, `rwork` the real LAPACK workspace during `gesvd` and the singular
values `σ̂` of the matrix divided by `2^e` in its first `n` entries after `gesvd`, `tmp` the
vector workspace buffer, `rhs` the right-hand-side buffer of the factor element type, which is
`nothing` only where the working element type is the factor's and the problem is not a
`LeastSquaresProblem`, `info` `1` where the matrix has a non-finite entry, which is refused before
`gesvd`, `gesvd`'s non-convergence code, `1` where a `σ̂` is not finite, or `0`, `rank` the number
of singular values above the cutoff `√m · eps(real(TF)) · σ̂₁`, and 0 where `info ≠ 0`, and
`scale_exponent` the exponent `e` of the power of two `2^e` that the matrix was divided by.
"""
struct DenseSVD{A <: AbstractMatrix, S <: AbstractVector, W <: AbstractVector,
    RW <: AbstractVector, T <: AbstractVector,
    V <: Union{Nothing, AbstractVector}}
    factors::A
    U::A
    S::S
    VT::A
    work::W
    rwork::RW
    tmp::T
    rhs::V
    info::Int
    rank::Int
    scale_exponent::Int
end

"""
    DeviceSVD{A, S, F, T, V}

The singular value decomposition factors `U * Diagonal(S) * VT` of a dense `m × n` device matrix
(`m ≥ n`), computed by the parallel one-sided Jacobi iteration of `src/linear/ka/svd.jl`, with
`factors` the `m × n` Jacobian buffer in the factor element type, `U` the thin `m × n` and `VT`
the `n × n` unitary factors, `S` the vector of the singular values `σ̂ · 2^e` of the matrix in the
real factor type, `V` the `n × n` right-singular-vector workspace during the Jacobi sweep, `σ̂`
the vector of the singular values of the matrix divided by `2^e` (and the squared column norms
during the sweep), `flag` the 1-element `Int32` rotation-flag buffer, `tmp` the vector workspace
buffer, `rhs` the right-hand-side buffer of the factor element type, which is `nothing` only
where the working element type is the factor's and the problem is not a `LeastSquaresProblem`,
`info` `1` where the Jacobi sweep reaches its sweep cap without converging or meets a non-finite
entry, or `0`, `rank` the number of singular values above the cutoff `√m · eps(real(TF)) · σ̂₁`,
and 0 where `info ≠ 0`, and `scale_exponent` the exponent `e` of the power of two `2^e` that the
matrix was divided by.
"""
struct DeviceSVD{A <: AbstractMatrix, S <: AbstractVector, F <: AbstractVector{Int32},
    T <: AbstractVector, V <: Union{Nothing, AbstractVector}}
    factors::A
    U::A
    S::S
    VT::A
    V::A
    σ̂::S
    flag::F
    tmp::T
    rhs::V
    info::Int
    rank::Int
    scale_exponent::Int
end

"""
    StaticLDLT{N, TF, L}

The Bunch–Kaufman `L D Lᵀ` / `L D Lᴴ` factors of an `N × N` static matrix, in element type `TF`:
`factors` holds the `1 × 1` and `2 × 2` diagonal blocks of `D` on the diagonal and first
subdiagonal and the unit lower-triangular multipliers of `L` below them, `ipiv` encodes the
symmetric row/column swaps (`ipiv[k] = p > 0` for a `1 × 1` pivot swapping `k ↔ p`, and
`ipiv[k] = ipiv[k + 1] = -p < 0` for a `2 × 2` pivot swapping `k + 1 ↔ p`), `info` is the index
of the first zero pivot block or `1` for a non-finite matrix, or `0`, `scale_exponent` is the
exponent `e` of the power of two `2^e` that the matrix was divided by, and `hermitian` is `true`
for a real symmetric or complex Hermitian factorisation (`L D Lᴴ`) and `false` for a complex
symmetric factorisation (`L D Lᵀ`). `L = N * N` makes the `SMatrix` field concrete, so the factor
is `isbits`.
"""
struct StaticLDLT{N, TF, L}
    factors::SMatrix{N, N, TF, L}
    ipiv::SVector{N, Int32}
    info::Int32
    scale_exponent::Int32
    hermitian::Bool
end

"""
    DenseLDLT{A, P, W, V}

The Bunch–Kaufman `L D Lᵀ` / `L D Lᴴ` factors of a dense `n × n` host matrix, with `factors` the
`n × n` buffer in the factor element type, `ipiv` the `BlasInt` pivot vector of length `n`,
`work` the LAPACK workspace buffer, `rhs` the right-hand-side buffer of the factor element type
when that differs from the working element type and `nothing` otherwise, `info` the index of the
first zero pivot block or `1` for a non-finite entry, or `0`, `scale_exponent` the exponent `e`
of the power of two `2^e` that the matrix was divided by, and `hermitian` `true` for a real
symmetric or complex Hermitian factorisation (`L D Lᴴ`) and `false` for a complex symmetric
factorisation (`L D Lᵀ`). A `BlasFloat` matrix is factorised by `hetrf!` where `hermitian` and the
element type is complex, and by `sytrf!` otherwise; another element type by `generic_ldlt!`.
"""
struct DenseLDLT{A <: AbstractMatrix, P <: AbstractVector, W <: AbstractVector,
    V <: Union{Nothing, AbstractVector}}
    factors::A
    ipiv::P
    work::W
    rhs::V
    info::Int
    scale_exponent::Int
    hermitian::Bool
end

"""
    DeviceLDLT{A, P, V, S}

The Bunch–Kaufman `L D Lᵀ` / `L D Lᴴ` factors of a dense `n × n` device matrix, with `factors`
the `n × n` device buffer in the factor element type, `ipiv` the `Int32` pivot vector of length
`n`, `rhs` the right-hand-side buffer of the factor element type when that differs from the
working element type and `nothing` otherwise, `info` the index of the first zero pivot block or
`1` for a non-finite entry, or `0`, `scale_exponent` the exponent `e` of the power of two `2^e`
that the matrix was divided by, `status` the 1-element `Int32` device status buffer, and
`hermitian` `true` for a real symmetric or complex Hermitian (`L D Lᴴ`) factorisation and `false`
for a complex symmetric (`L D Lᵀ`) factorisation.
"""
struct DeviceLDLT{A <: AbstractMatrix, P <: AbstractVector{Int32},
    V <: Union{Nothing, AbstractVector}, S <: AbstractVector{Int32}}
    factors::A
    ipiv::P
    rhs::V
    info::Int
    scale_exponent::Int
    status::S
    hermitian::Bool
end

"""
    dense_factor(method::LinearMethod, prob, x::AbstractVector, p)

Build the linear factor for `method` on `x` using `jacobian_prototype(prob)`, with the rows of
the Jacobian of `prob`: `m × n` for a [`LeastSquaresProblem`](@ref) of residual length `m`, and
`n × n` otherwise.
"""
function dense_factor(m::LinearMethod, prob, x::AbstractVector, p)
    init_linear(m, jacobian_prototype(prob), x, factor_rows(prob, x))
end

# The row count of the Jacobian whose factor `init_linear` builds, where it is not the length of
# the iterate `x`: the residual length of a `LeastSquaresProblem`, and `nothing` for every other
# problem, whose Jacobian is square.
@inline factor_rows(prob, x) = nothing

"""
    init_linear(method::LinearMethod, pattern, x::AbstractVector)

Build the linear factor for `method` on `x`: the dense host factor [`DenseLU`](@ref),
[`DenseQR`](@ref), [`DenseSVD`](@ref) or [`DenseLDLT`](@ref) or device factor [`DeviceLU`](@ref),
[`DeviceQR`](@ref), [`DeviceSVD`](@ref) or [`DeviceLDLT`](@ref) when `pattern === nothing`, or an
extension's sparse factor when `pattern` is a sparse matrix prototype.
"""
function init_linear(m::LUFactorization, ::Nothing, x::AbstractVector{T}) where {T <:
                                                                                 Number}
    F = factor_eltype(m, T)
    n = length(x)
    factors = similar(x, F, n, n)
    ipiv = Vector{BlasInt}(undef, n)
    rhs = F === T ? nothing : similar(x, F, n)
    return DenseLU(factors, ipiv, rhs, 0, 0)
end

function init_linear(
        m::LUFactorization, ::Nothing, x::AbstractGPUArray{T, 1}) where {T <: Number}
    F = factor_eltype(m, T)
    n = length(x)
    factors = similar(x, F, n, n)
    ipiv = similar(x, Int32, n)
    rhs = F === T ? nothing : similar(x, F, n)
    status = similar(x, Int32, 1)
    return DeviceLU(factors, ipiv, rhs, 0, 0, status)
end

function init_linear(m::QRFactorization, ::Nothing, x::AbstractVector{T}) where {T <:
                                                                                 Number}
    F = factor_eltype(m, T)
    n = length(x)
    factors = similar(x, F, n, n)
    τ = similar(x, F, n)
    work = similar(x, F, qr_work_length(factors, τ))
    rhs = F === T ? nothing : similar(x, F, n)
    return DenseQR(factors, τ, work, rhs, 0, 0)
end

function init_linear(
        m::QRFactorization, ::Nothing, x::AbstractGPUArray{T, 1}) where {T <: Number}
    F = factor_eltype(m, T)
    n = length(x)
    factors = similar(x, F, n, n)
    τ = similar(x, F, n)
    rhs = F === T ? nothing : similar(x, F, n)
    status = similar(x, Int32, 1)
    return DeviceQR(factors, τ, rhs, 0, 0, status)
end

function init_linear(
        m::SVDFactorization, ::Nothing, x::AbstractVector{T}) where {T <: Number}
    F = factor_eltype(m, T)
    R = real(F)
    n = length(x)
    factors = similar(x, F, n, n)
    U = similar(x, F, n, n)
    S = similar(x, R, n)
    VT = similar(x, F, n, n)
    rwork = similar(x, R, F <: Complex ? max(1, 5 * n) : n)
    work = similar(x, F, svd_work_length(factors, S, U, VT, rwork))
    tmp = similar(x, F, n)
    rhs = F === T ? nothing : similar(x, F, n)
    return DenseSVD(factors, U, S, VT, work, rwork, tmp, rhs, 0, 0, 0)
end

function init_linear(
        m::SVDFactorization, ::Nothing, x::AbstractGPUArray{T, 1}) where {T <: Number}
    F = factor_eltype(m, T)
    R = real(F)
    n = length(x)
    factors = similar(x, F, n, n)
    U = similar(x, F, n, n)
    S = similar(x, R, n)
    VT = similar(x, F, n, n)
    V = similar(x, F, n, n)
    σ̂ = similar(x, R, n)
    flag = similar(x, Int32, 1)
    tmp = similar(x, F, n)
    rhs = F === T ? nothing : similar(x, F, n)
    return DeviceSVD(factors, U, S, VT, V, σ̂, flag, tmp, rhs, 0, 0, 0)
end

function init_linear(
        m::LDLTFactorization, ::Nothing, x::AbstractVector{T}) where {T <: Number}
    F = factor_eltype(m, T)
    n = length(x)
    factors = similar(x, F, n, n)
    ipiv = Vector{BlasInt}(undef, n)
    work = similar(x, F, ldlt_work_length(factors, ipiv))
    rhs = F === T ? nothing : similar(x, F, n)
    return DenseLDLT(factors, ipiv, work, rhs, 0, 0, true)
end

function init_linear(
        m::LDLTFactorization, ::Nothing, x::AbstractGPUArray{T, 1}) where {T <: Number}
    F = factor_eltype(m, T)
    n = length(x)
    factors = similar(x, F, n, n)
    ipiv = similar(x, Int32, n)
    rhs = F === T ? nothing : similar(x, F, n)
    status = similar(x, Int32, 1)
    return DeviceLDLT(factors, ipiv, rhs, 0, 0, status, true)
end

const AnyStaticFactor = Union{StaticLU, StaticQR, StaticSVD, StaticLDLT}
const AnyDenseFactor = Union{DenseLU, DenseQR, DenseSVD, DenseLDLT}
const AnyDeviceFactor = Union{DeviceLU, DeviceQR, DeviceSVD, DeviceLDLT}
const AnyFactor = Union{AnyStaticFactor, AnyDenseFactor, AnyDeviceFactor}

const ExactStaticFactor = Union{StaticLU, StaticQR, StaticLDLT}
const ExactDenseFactor = Union{DenseLU, DenseQR, DenseLDLT}
const ExactDeviceFactor = Union{DeviceLU, DeviceQR, DeviceLDLT}
const ExactFactor = Union{ExactStaticFactor, ExactDenseFactor, ExactDeviceFactor}

"""
    init_linear(method::LinearMethod, pattern, x::AbstractVector, m_rows)

Build the linear factor for `method` on `x` for a Jacobian of `m_rows` rows: the square factor of
the three-argument method for `m_rows === nothing`, and for an `Int` the rectangular
`m_rows × length(x)` factor of a [`LeastSquaresProblem`](@ref), a [`DenseQR`](@ref),
[`DenseSVD`](@ref), [`DeviceQR`](@ref) or [`DeviceSVD`](@ref) with a right-hand side buffer of
`m_rows` entries, because its solve reads `m_rows` residual entries and writes `length(x)` step
entries. Whether `m_rows` is an `Int` decides the buffer, so a factor's type is inferred from its
call. `LUFactorization` and `LDLTFactorization` take the square factor whatever `m_rows` is: they
factorise no rectangular matrix, and `LevenbergMarquardt`, the one globalization that takes them
on a least-squares problem, solves its own stacked system and never factorises that factor.
"""
function init_linear(m::LinearMethod, pattern, x::AbstractVector, ::Nothing)
    init_linear(m, pattern, x)
end
function init_linear(
        m::Union{LUFactorization, LDLTFactorization}, pattern, x::AbstractVector, ::Int)
    init_linear(m, pattern, x)
end

function init_linear(
        m::QRFactorization, ::Nothing, x::AbstractVector{T}, m_rows::Int) where {T <:
                                                                                 Number}
    F = factor_eltype(m, T)
    n = length(x)
    factors = similar(x, F, m_rows, n)
    τ = similar(x, F, n)
    work = similar(x, F, qr_work_length(factors, τ))
    return DenseQR(factors, τ, work, similar(x, F, m_rows), 0, 0)
end

function init_linear(m::QRFactorization, ::Nothing, x::AbstractGPUArray{T, 1},
        m_rows::Int) where {T <: Number}
    F = factor_eltype(m, T)
    n = length(x)
    factors = similar(x, F, m_rows, n)
    τ = similar(x, F, n)
    status = similar(x, Int32, 1)
    return DeviceQR(factors, τ, similar(x, F, m_rows), 0, 0, status)
end

function init_linear(
        m::SVDFactorization, ::Nothing, x::AbstractVector{T}, m_rows::Int) where {T <:
                                                                                  Number}
    F = factor_eltype(m, T)
    R = real(F)
    n = length(x)
    factors = similar(x, F, m_rows, n)
    U = similar(x, F, m_rows, n)
    S = similar(x, R, n)
    VT = similar(x, F, n, n)
    rwork = similar(x, R, F <: Complex ? max(1, 5 * n) : n)
    work = similar(x, F, svd_work_length(factors, S, U, VT, rwork))
    tmp = similar(x, F, n)
    return DenseSVD(factors, U, S, VT, work, rwork, tmp, similar(x, F, m_rows), 0, 0, 0)
end

function init_linear(m::SVDFactorization, ::Nothing, x::AbstractGPUArray{T, 1},
        m_rows::Int) where {T <: Number}
    F = factor_eltype(m, T)
    R = real(F)
    n = length(x)
    factors = similar(x, F, m_rows, n)
    U = similar(x, F, m_rows, n)
    S = similar(x, R, n)
    VT = similar(x, F, n, n)
    V = similar(x, F, n, n)
    σ̂ = similar(x, R, n)
    flag = similar(x, Int32, 1)
    tmp = similar(x, F, n)
    return DeviceSVD(factors, U, S, VT, V, σ̂, flag, tmp, similar(x, F, m_rows), 0, 0, 0)
end

"""
    returncode(fac)

`SUCCESS` for a factor with `info == 0`, and `SINGULAR` otherwise.
"""
returncode(fac::AnyFactor) = iszero(fac.info) ? SUCCESS : SINGULAR

"""
    exact_solve(fac)

`true` where the solve of [`ldiv!!`](@ref) with the factor `fac` is the exact solve of the
factored matrix, and `false` for an SVD factor of numerical rank below `n`, whose solve is the
truncated minimum-norm step. `Newton` measures the slope of a truncated step, because `−2φ(0)`
is the slope of an exact Newton step alone.
"""
exact_solve(fac) = true
exact_solve(fac::StaticSVD{N}) where {N} = fac.rank == N
exact_solve(fac::Union{DenseSVD, DeviceSVD}) = fac.rank == length(fac.S)

@inline function factor_diagnostics(d, ::ExactFactor, r, fnorm::R) where {R <: Real}
    return Int32(-1), R(NaN)
end

"""
    singular_values(fac::Union{StaticSVD, StaticRectSVD})

The singular values of the matrix that [`factorize!!`](@ref) factorised, in non-increasing order,
as an `SVector` in `real(TF)`: the stored `σ̂` of the matrix divided by `2^e`, multiplied by `2^e`.
A singular value above `floatmax(real(TF))` overflows to `Inf`, although every entry of the matrix
is finite; the solve divides by `σ̂` and stays finite.
"""
singular_values(fac::StaticSVD) = map(a -> ldexp2(a, fac.scale_exponent), fac.σ̂)

# `a · 2^e`, exact unless the result leaves the floating-point range; by component for a complex `a`.
# Bitwise on `IEEEFloat` so that a device with flush-to-zero enabled (Metal) preserves subnormals.
@inline function ldexp2(x::T, e::Integer) where {T <: Base.IEEEFloat}
    xu = reinterpret(Unsigned, x)
    U = typeof(xu)
    sign = xu & Base.sign_mask(T)
    xs = xu & ~Base.sign_mask(T)
    xs >= Base.exponent_mask(T) && return x
    xs == zero(U) && return x
    sig_bits = Base.significand_bits(T)
    k = Int(xs >> sig_bits)
    if k == 0
        m = leading_zeros(xs) - Base.exponent_bits(T)
        xs <<= unsigned(m)
        k = 1 - m
    end
    sig = (xs & Base.significand_mask(T)) | (one(U) << sig_bits)
    k += Int(e)
    if k >= Base.exponent_raw_max(T)
        return reinterpret(T, sign | Base.exponent_mask(T))
    elseif k > 0
        return reinterpret(T, sign | (U(k) << sig_bits) | (sig & Base.significand_mask(T)))
    elseif k <= -sig_bits - 1
        return reinterpret(T, sign)
    else
        shift = 1 - k
        lower = sig & ((one(U) << shift) - one(U))
        half = one(U) << (shift - 1)
        sub = sig >> shift
        if lower > half || (lower == half && isodd(sub))
            sub += one(U)
        end
        return reinterpret(T, sign | sub)
    end
end
@inline ldexp2(a::AbstractFloat, e::Integer) = ldexp(a, e)
@inline ldexp2(a::Real, e::Integer) = ldexp2(float(a), e)
@inline ldexp2(a::Complex, e::Integer) = complex(ldexp2(real(a), e), ldexp2(imag(a), e))

@inline abs_bits(x::T) where {T <: Base.IEEEFloat} = reinterpret(Unsigned, x) &
                                                     ~Base.sign_mask(T)

# `J` in the factor element type `F`, as an `SMatrix`, which keeps the conversion off the heap, and
# the exponent `e` of the power of two that `J` was divided by: 0 where `F` is the element type of
# `J`, else the exponent of the largest |re| or |im| of `J`, which takes no `abs` that can overflow.
# `e` is clamped to the exponent range, so that `2^-e` is a float and the product with it is exact.
function factor_matrix(::Type{F}, J::StaticMatrix{M, N, T}) where {F, M, N, T}
    F === T && return SMatrix{M, N, F}(J), Int32(0)
    e = matrix_exponent(J)
    return SMatrix{M, N, F}(SMatrix(J) * ldexp(one(float(real(T))), -e)), Int32(e)
end

# A mutable local copy of a static array for the scalar loops of the R1 kernels: an `MMatrix` or
# `MVector` for an `isbits` element type, which the compiler keeps off the heap, and otherwise a
# `SizedMatrix` or `SizedVector` on a heap array, because a mutable static array of a non-`isbits`
# element type such as `BigFloat` has no `setindex!`. `isbitstype(F)` is a constant of the type,
# so each method compiles to one of its two branches.
@inline function local_array(A::SMatrix{N, M, F}) where {N, M, F}
    return isbitstype(F) ? MMatrix{N, M, F, N * M}(A) : SizedMatrix{N, M, F}(Matrix(A))
end
@inline function local_array(x::SVector{N, F}) where {N, F}
    return isbitstype(F) ? MVector{N, F}(x) : SizedVector{N, F}(Vector(x))
end

# An uninitialised local vector of `N` entries of type `F`, as `local_array` chooses its type.
@inline function local_vector(::Type{F}, ::Val{N}) where {F, N}
    return isbitstype(F) ? MVector{N, F}(undef) : SizedVector{N, F}(undef)
end

# The exponent of the largest |re| or |im| of a nonempty array `J`, a matrix or a right-hand side,
# clamped to the exponent range, and 0 for a zero or non-finite `J`.
function matrix_exponent(J::AbstractArray)
    R = real(eltype(J))
    m = if R <: Base.IEEEFloat
        reinterpret(R, maximum(a -> max(abs_bits(real(a)), abs_bits(imag(a))), J))
    else
        float(maximum(a -> max(abs(real(a)), abs(imag(a))), J))
    end
    emax = exponent(floatmax(m))
    return isfinite(m) && !iszero(m) ? clamp(exponent(m), -emax, emax) : 0
end

# The exponent `k ≥ 0` of the power of two `2^k` that the same-type solve divides the right-hand
# side `r` by, the smallest one with which this bound keeps `Qᴴ r` and `Uᴴ r` of the QR and SVD
# solves finite: their entries sum `N` products of an entry of `r` with a factor of modulus at most
# 1, a complex one up to twice the largest |re| or |im| of `r`, so the largest |re| or |im| of
# `r / 2^k` is held below `2^(emax - 1 - ⌈log2 N⌉)`, with `emax` the exponent of `floatmax`, and
# each |re| and |im| of `Qᴴ r` and `Uᴴ r` below `2^emax`. An `r` below the first bound is not
# scaled, and an `r` above it is divided by at most `2 + ⌈log2 N⌉` bits, so that only a small
# entry of such an `r`, within that many bits of the subnormal range, can lose bits to the scaling,
# and the large entries keep a scaled solution far above the subnormal range. 0 for a zero,
# non-finite or empty `r`.
function rhs_exponent(r::AbstractVector)
    isempty(r) && return 0
    emax = exponent(floatmax(float(real(eltype(r)))))
    return max(0, matrix_exponent(r) - (emax - headroom(length(r))))
end

# The bits `2 + ⌈log2 n⌉` that `rhs_exponent` and `divisor_exponent` keep below `floatmax`, for
# `n ≥ 1`: a sum of `n` products of a factor of modulus at most 1 with an entry whose |re| and |im|
# lie below `2^(emax + 1 - headroom(n))`, a complex one of modulus up to twice that, stays finite.
headroom(n::Integer) = 2 + 8sizeof(Int) - leading_zeros(n - 1)

# The exponent `t ≥ 0` of the power of two `2^t` that the same-type SVD solve multiplies its
# divisors `σ̂` by, from the coordinates `y = Uᴴ r` of the first `rank` directions: the smallest one
# with which every |re| and |im| of `y / (σ̂ 2^t)` lies below `2^c`, `c = emax + 1 - headroom(n)`,
# so that `V (y / (σ̂ 2^t))` stays finite. `q` is the largest `max(|re y|, |im y|) · 2^-c / σ̂`,
# which does not overflow where `y / σ̂` does; `t` is 0 where `q < 1`, and also where `q` is not
# finite, which only a non-finite `y` makes, whose step is not finite either way. Where `t` is 0
# the solve divides by `σ̂` itself, as it does without this exponent.
function divisor_exponent(y::AbstractVector, σ̂::AbstractVector, rank::Integer)
    c = coordinate_bound(eltype(y), length(y))
    q = zero(real(eltype(y)))
    @inbounds for j in 1:rank
        q = max(q, coordinate_ratio(y[j], σ̂[j], c))
    end
    return shift_exponent(q)
end

function coordinate_bound(::Type{T}, n::Integer) where {T}
    exponent(floatmax(real(T))) + 1 - headroom(n)
end
@inline coordinate_ratio(y, σ̂, c) = ldexp2(max(abs(real(y)), abs(imag(y))), -c) / σ̂
shift_exponent(q::Real) = isfinite(q) && q >= 1 ? exponent(q) + 1 : 0

# The step for the matrix `J` and the right-hand side `r`, from the step `d` for `J / 2^e` and
# `r / 2^k` that the stored factors solve: `d · 2^(k - e)`, in one rounding.
function unscale(fac, d, k)
    e = k - fac.scale_exponent
    return iszero(e) ? d : map(a -> ldexp2(a, e), d)
end

# The solve of the same-type `ldiv!!` with the stored factors, and the exponent `t` of the power of
# two its result is to be multiplied by besides `2^(k - e)`: 0 for LU and QR, and for the SVD the
# `divisor_exponent` of its coordinates.
backsolve_shifted(fac, r) = (backsolve(fac, r), 0)

const StaticLUQR{N, TF} = Union{StaticLU{N, TF}, StaticQR{N, TF}}
const StaticSVDLDLT{N, TF} = Union{StaticSVD{N, TF}, StaticLDLT{N, TF}}
const StaticFac{N, TF} = Union{StaticLUQR{N, TF}, StaticSVDLDLT{N, TF}}

function ldiv!!(::Nothing, fac::StaticFac{N, TF}, r::StaticVector{N, TF}) where {N, TF}
    k = rhs_exponent(r)
    r̂ = iszero(k) ? SVector{N, TF}(r) : map(a -> ldexp2(a, -k), SVector{N, TF}(r))
    d, t = backsolve_shifted(fac, r̂)
    return unscale(fac, d, k + t)
end

# The `T ≠ TF` method: the right-hand side has another element type than the factors. With
# `s = frac · 2^ex` and `frac` in `[0.5, 1)`, the step is multiplied by `frac` and then by
# `2^(ex - e)` in one step, with `e` the scale exponent of the factors, both in the wider of `TF`
# and `T`, and only then converted to `T`, so that it overflows or underflows only where the
# solution itself does.
function ldiv!!(::Nothing, fac::StaticFac{N, TF}, r::StaticVector{N, T}) where {N, TF, T}
    s = maximum(abs, r)
    iszero(s) && return zero(SVector{N, T})
    d = backsolve(fac, SVector{N, TF}(SVector{N, T}(r) / s))
    frac, ex = frexp(s)
    return map(a -> T(ldexp2(a * frac, ex - fac.scale_exponent)), d)
end
