# The R1 linear systems of the static factorisation tests, named once: the sizes, the precision
# pairs, the test matrices with a stated condition number, the `BigFloat` reference, the error
# bound, and the KernelAbstractions kernel that solves one system per thread.
#
# The bound of check 1: partial-pivot LU and Householder QR have a backward
# error of order `n · eps(TF)` when the growth factor is near 1, and `κ∞(A)` carries it to `x`, so
#
#     ‖x − x_ref‖∞ / ‖x_ref‖∞ ≤ 4n · eps(TF) · κ∞(A),
#
# with `x_ref` the `BigFloat` solve of `A` and `b` as stored in `T`, and `κ∞` measured in `BigFloat`.
# The test files include `matrix.jl` before this file.

using GeometricSolvers: factorize!!, ldiv!!
using KernelAbstractions: KernelAbstractions, @kernel, @index, @Const
using LinearAlgebra: Diagonal, opnorm, qr
using Random: Xoshiro, randn
using StaticArrays: SMatrix, SVector

# n = 1, 4, 16 and 24: 24 is the largest size of the R1 band on Metal and on CUDA, and 16 the top
# of its general estimate n ≤ 8…16.
const STATIC_SIZES = (1, 4, 16, 24)

# (T, TF) for a method `M{TF}()`: every element type with `TF = Nothing`, the working type, and the
# lower pairs, `Float64` and `ComplexF64` factorised with `TF = Float32`.
const PRECISION_PAIRS = (map(T -> (T, Nothing), ELTYPES)...,
    map(T -> (T, Float32), filter(T -> real(T) === Float64, ELTYPES))...)

"The element type of the factors: `T`, or `TF` made complex for a complex `T`."
factortype(::Type{T}, ::Type{Nothing}) where {T} = T
factortype(::Type{T}, ::Type{TF}) where {T, TF} = T <: Complex ? complex(TF) : TF

"The unit roundoff of the factorisation, `eps(real(factortype(T, TF)))`."
epsf(::Type{T}, ::Type{TF}) where {T, TF} = eps(real(factortype(T, TF)))

"The largest `κ∞` a test draws: `0.1 / (n · eps(TF))` (check 1)."
κcap(n, ::Type{T}, ::Type{TF}) where {T, TF} = 0.1 / (n * epsf(T, TF))

"""
    extreme_scales(R)

The scales for which an unscaled square overflows or underflows in the real factor type `R`: `|p|²`
of a complex LU pivot and the squares of a QR column norm.
"""
extreme_scales(::Type{Float32}) = (1e30, 1e-30)
extreme_scales(::Type{Float64}) = (1e200, 1e-200)

"`κ∞(A)`, in `BigFloat`."
function κinf(A)
    Ab = big.(Matrix(A))
    return opnorm(Ab, Inf) * opnorm(inv(Ab), Inf)
end

"The `BigFloat` solve of `A` and `b` as stored."
reference(A, b) = big.(Matrix(A)) \ big.(Vector(b))

"`‖x − x_ref‖∞ / ‖x_ref‖∞`, in `BigFloat`."
relerr(x, xref) = maximum(abs, big.(Vector(x)) .- xref) / maximum(abs, xref)

"The bound of check 1 for `A`."
bound(A, ::Type{T}, ::Type{TF}) where {T, TF} = 4 * size(A, 1) * epsf(T, TF) * κinf(A)

"""
    random_matrix(rng, T, n, κ)

`U Σ Vᴴ` in `T`, with `U` and `V` unitary and a geometric `Σ` from 1 down to `1/κ`, so that
`κ₂ = κ` before the rounding to `T`.
"""
function random_matrix(rng, ::Type{T}, n, κ) where {T}
    U = Matrix(qr(randn(rng, T, n, n)).Q)
    V = Matrix(qr(randn(rng, T, n, n)).Q)
    σ = n == 1 ? [1.0] : [κ^(-(i - 1) / (n - 1)) for i in 1:n]
    return SMatrix{n, n, T}(U * Diagonal(σ) * V')
end

"""
    conditioned_systems(seed, T, TF, n)

Systems `(A, b)` with `κ₂` geometric from 10 to `κcap`; a draw whose `κ∞` exceeds the cap is
discarded, except the top draw: its `κ₂` halves, with the same `U` and `V`, until `κ∞` is within
the cap, so that the systems reach the top of check 1's domain. At `n = 1` the one system has
`κ = 1`.
"""
function conditioned_systems(seed, ::Type{T}, ::Type{TF}, n) where {T, TF}
    rng = Xoshiro(seed)
    cap = κcap(n, T, TF)
    κs = n == 1 ? [1.0] : exp10.(range(1, log10(cap); length = 5))
    systems = Tuple{SMatrix{n, n, T, n * n}, SVector{n, T}}[]
    for κ in κs
        top = copy(rng)
        A = random_matrix(rng, T, n, κ)
        b = SVector{n, T}(randn(rng, T, n))
        if κ == last(κs)
            κ₂ = κ
            while κinf(A) > cap
                κ₂ /= 2
                A = random_matrix(copy(top), T, n, κ₂)
            end
        end
        κinf(A) <= cap && push!(systems, (A, b))
    end
    return systems
end

"""
    pivoting_matrices(seed, T, TF, n)

The matrices of check 2: a well-conditioned `A` with `A[1,1] = 0`, the same with
`A[1,1] = eps(TF)^2`, and, for a complex `T`, `im * B` for a real `B` with the same two edits,
whose real parts are all zero.
"""
function pivoting_matrices(seed, ::Type{T}, ::Type{TF}, n) where {T, TF}
    edits(A) = (Base.setindex(A, zero(eltype(A)), 1, 1),
        Base.setindex(A, eltype(A)(epsf(T, TF)^2), 1, 1))
    matrices = edits(random_matrix(Xoshiro(seed), T, n, 10.0))
    T <: Complex || return matrices
    imaginary = map(B -> im * T.(B), edits(random_matrix(Xoshiro(seed), real(T), n, 10.0)))
    return (matrices..., imaginary...)
end

"""
    singular_matrices(seed, T, n)

The matrices of check 3, by name: a zero column (at each position `j`), two equal rows, and the zero
matrix.
"""
function singular_matrices(seed, ::Type{T}, n) where {T}
    rng = Xoshiro(seed)
    A = random_matrix(rng, T, n, 10.0)
    zero_columns = [SMatrix{n, n, T}([j == c ? zero(T) : A[i, j] for i in 1:n, j in 1:n])
                    for c in 1:n]
    r = n == 1 ? 1 : 2
    equal_rows = SMatrix{n, n, T}([(i == r ? A[1, j] : A[i, j]) for i in 1:n, j in 1:n])
    return (; zero_columns, equal_rows, zero_matrix = zero(SMatrix{n, n, T}))
end

"""
    rank_deficient(seed, T, n, r)

`U Σ Vᴴ` in `T` with `U` and `V` unitary and `Σ` holding the `r` values `1, 1/2, …` and zeros,
so that the stored matrix has the numerical rank `r` at `√n · eps` for every element type. The
SVD's check 2 and check 3 and the Metal device file share this one factory.
"""
function rank_deficient(seed, ::Type{T}, n, r) where {T}
    rng = Xoshiro(seed)
    U = SMatrix{n, n, T}(Matrix(qr(randn(rng, T, n, n)).Q))
    V = SMatrix{n, n, T}(Matrix(qr(randn(rng, T, n, n)).Q))
    Σ = real(T)[i ≤ r ? real(T)(0.5)^(i - 1) : zero(real(T)) for i in 1:n]
    return SMatrix{n, n, T}(U * Diagonal(Σ) * V')
end

"Factorise and solve on the host: the factors and the step."
function host_solve(m, A, b)
    fac = factorize!!(nothing, m, A)
    return fac, ldiv!!(nothing, fac, b)
end

@kernel function static_solve_kernel!(x, info, m, @Const(A), @Const(b))
    i = @index(Global)
    fac = factorize!!(nothing, m, A[i])
    x[i] = ldiv!!(nothing, fac, b[i])
    info[i] = fac.info
end

"""
    kernel_solve(AT, m, As, bs)

Solve `As[i] \\ bs[i]` with method `m`, one system per thread of a KernelAbstractions kernel on the
backend of the array type `AT`, and return the steps and the `info` codes on the host.
"""
function kernel_solve(AT, m, As, bs)
    A = AT(As)
    b = AT(bs)
    x = similar(b)
    info = similar(b, Int32)
    backend = KernelAbstractions.get_backend(A)
    static_solve_kernel!(backend)(x, info, m, A, b; ndrange = length(As))
    # `JLBackend` runs a kernel to completion at its launch, and has no `synchronize` method
    applicable(KernelAbstractions.synchronize, backend) &&
        KernelAbstractions.synchronize(backend)
    return Array(x), Array(info)
end
