using GeometricSolvers: norm2, rnorm, rdot
using ForwardDiff: ForwardDiff, Dual
using JET: JET, @test_opt
using JLArrays: JLArray
using KernelAbstractions: KernelAbstractions, @kernel, @index
using LinearAlgebra: norm
using Random: Xoshiro
using StaticArrays: SVector
using Test

include("../helpers/matrix.jl")

# The error bound of a sum of n products in any order, ``γ_{n+2} Σ |aᵢ bᵢ|`` with
# ``γ_k = k u / (1 - k u)`` (Higham, *Accuracy and Stability*, §3.1): n - 1 additions, and at
# most one product and one addition per term for a complex `abs2` or `conj(x) * y`.
γ(k, R) = k * eps(R) / (1 - k * eps(R))

@testset "exact against BigFloat within γ(n+2): $T, n = $n, $(AT)" for T in ELTYPES,
    n in (1, 7, 1000), AT in ARRAY_BACKENDS
    R = real(T)
    rng = Xoshiro(n)
    a, b = randn(rng, T, n), randn(rng, T, n)
    A, B = AT(a), AT(b)

    ref2 = sum(abs2, big.(a))
    @test norm2(A) isa R
    @test abs(norm2(A) - ref2) <= γ(n + 2, R) * ref2

    # sqrt halves the relative error of its argument and adds one rounding of its own
    @test rnorm(A) isa R
    @test abs(rnorm(A) - sqrt(ref2)) <= (γ(n + 2, R) / 2 + eps(R)) * sqrt(ref2)

    refd = real(sum(conj.(big.(a)) .* big.(b)))
    @test rdot(A, B) isa R
    @test abs(rdot(A, B) - refd) <= γ(n + 2, R) * sum(abs.(big.(a)) .* abs.(big.(b)))
end

# The pairwise sum has an error that grows like log2(n) u, a sequential fold one that grows like
# n u. On this positive uniform data, a sequential `Float32` fold misses the bound below by more
# than 50 times. All-ones data does not separate the two: its sum is exact below 2^24.
@testset "pairwise reduction: Float32, n = 10^6, within ceil(log2(n)) eps" begin
    n = 10^6
    a, b = rand(Xoshiro(1), Float32, n), rand(Xoshiro(2), Float32, n)
    bound = ceil(log2(n)) * eps(Float32)
    a64, b64 = Float64.(a), Float64.(b)
    ref2 = sum(abs2, a64)
    refd = sum(a64 .* b64)
    @test abs(norm2(a) - ref2) <= bound * ref2
    @test abs(rdot(a, b) - refd) <= bound * refd
end

@testset "a JLArray refuses scalar indexing" begin
    @test_throws ErrorException JLArray(zeros(Float32, 2))[1]
end

@testset "empty arrays reduce to zero(R)" for T in ELTYPES
    @test norm2(T[]) === zero(real(T))
    @test rdot(T[], T[]) === zero(real(T))
end

# A vector whose largest |re| or |im| is `top`, with smaller entries, and complex entries for a
# complex `T`. `LinearAlgebra.norm` scales its sum, so it is finite and positive at every `top`.
function range_vector(::Type{T}, top) where {T}
    T <: Complex && return T[top, im * top / 3, 0, (1 + im) * top / 7]
    return T[top, top / 3, 0, top / 7]
end

range_tops(R) = (floatmax(R) / 4, one(R), floatmin(R), 4nextfloat(zero(R)))

@testset "rnorm agrees with norm across the range: $T, $(AT)" for T in ELTYPES,
    AT in (ARRAY_BACKENDS..., SVector)

    R = real(T)
    for top in range_tops(R)
        v = range_vector(T, top)
        ref = norm(v)
        a = AT === SVector ? SVector{4}(v) : AT(v)
        r = rnorm(a)
        @test r isa R
        @test abs(r - ref) <= 2eps(R) * ref
    end
end

@testset "rnorm of a scalar and of an SVector above √floatmax" begin
    @test rnorm(SVector(1.0f20, 0.0f0)) === 1.0f20
    for R in REAL_ELTYPES
        @test rnorm(R(-3)) === R(3)
        @test rnorm(floatmin(R) / 8) === floatmin(R) / 8
        @test rnorm(Complex{R}(3, -4)) === R(5)
    end
end

# The scale takes the larger of the real and the imaginary part of each entry: a scale from one
# part alone overflows on the other where the two are `floatmax / floatmin` apart.
@testset "rnorm scales by the larger part of a complex entry: $R" for R in REAL_ELTYPES
    hi, lo = floatmax(R) / 4, floatmin(R)
    for v in (Complex{R}[complex(lo, hi)], Complex{R}[complex(hi, lo)])
        @test rnorm(v) ≈ norm(v) rtol = 2eps(R)
    end
end

# Where the unscaled sum is finite and at least `floatmin / eps`, `rnorm` returns its square root
# and takes no scaled path, for every length.
@testset "rnorm is sqrt(norm2) where the squares stay in range: $R" for R in REAL_ELTYPES
    rng = Xoshiro(29)
    for k in -8:8, n in (5, 1000), _ in 1:20
        v = R.((0.5 .+ 1.5 .* rand(rng, n)) .* 10.0^k)
        @test rnorm(v) === sqrt(norm2(v))
    end
    for k in -8:8, _ in 1:20

        s = SVector{5}(R.((0.5 .+ 1.5 .* rand(rng, 5)) .* 10.0^k))
        @test rnorm(s) === sqrt(norm2(s))
    end
end

# 128 is a power of two, so each scaled square is 1, and a `Float16` sum of 10^5 of them overflows,
# while the norm of this vector is finite: 128 √(10^5) ≈ 40480. The unscaled sum overflows as well.
@testset "rnorm of a long Float16 vector whose sum of squares overflows" begin
    v = fill(Float16(128), 10^5)
    @test norm2(v) === Float16(Inf)
    @test rnorm(v) ≈ Float16(128 * sqrt(1e5)) rtol = 2eps(Float16)
end

@testset "rnorm of empty, zero and non-finite vectors: $T" for T in ELTYPES
    R = real(T)
    @test rnorm(T[]) === zero(R)
    @test rnorm(zeros(T, 3)) === zero(R)
    @test rnorm(SVector{3, T}(0, 0, 0)) === zero(R)
    @test isnan(rnorm(T[1, NaN]))
    @test rnorm(T[1, Inf]) === R(Inf)
    @test rnorm(T[Inf, 1]) === R(Inf)
    @test isnan(rnorm(T[Inf, NaN]))
    @test isnan(rnorm(T[NaN, Inf]))
end

@testset "rnorm of a subnormal vector: $T" for T in ELTYPES
    R = real(T)
    t = nextfloat(zero(R))
    for v in (T[t, 3t], fill(T(floatmin(R)), 3))
        r = rnorm(v)
        @test r > 0
        @test abs(r - norm(v)) <= 2eps(R) * norm(v)
    end
end

# The squared norm stays unscaled: finite where its true value is below `floatmax(R)`.
@testset "norm2 is finite below floatmax and Inf above it: $R" for R in REAL_ELTYPES
    h = sqrt(floatmax(R) / 3)
    @test isfinite(norm2(R[h, h]))
    @test norm2(R[2h, 2h]) === R(Inf)
end

@testset "SVector: inferred and allocation-free, $T" for T in ELTYPES
    a = SVector{4, T}(1, 2, 3, 4)
    b = SVector{4, T}(4, 3, 2, 1)
    @test @inferred(norm2(a)) === real(T)(30)
    @test @inferred(rnorm(a)) === sqrt(real(T)(30))
    @test @inferred(rdot(a, b)) === real(T)(20)
    @test (@allocated collect(a)) > 0        # the control: `@allocated` sees an allocation
    @test (@allocated norm2(a)) == 0
    @test (@allocated rnorm(a)) == 0
    @test (@allocated rdot(a, b)) == 0
end

@testset "Array: allocation-free, $T" for T in ELTYPES
    a, b = randn(Xoshiro(1), T, 100), randn(Xoshiro(2), T, 100)
    @test (@allocated norm2(a)) == 0
    @test (@allocated rnorm(a)) == 0
    @test (@allocated rdot(a, b)) == 0
end

@testset "arguments with different axes raise a DimensionMismatch" begin
    @test_throws DimensionMismatch rdot([1.0], [1.0, 2.0, 3.0])
    @test_throws DimensionMismatch rdot([1.0, 2.0, 3.0], [1.0 2.0 3.0])
end

# On a Julia it does not support, JET loads empty stubs that throw. JET 0.12 exports
# `JET_AVAILABLE` to say so; JET 0.9 and 0.10 name the same flag `JET_LOADABLE`.
const JET_WORKS = isdefined(JET, :JET_AVAILABLE) ? JET.JET_AVAILABLE : JET.JET_LOADABLE

@testset "JET: no runtime dispatch, $T" for T in ELTYPES
    a = SVector{3, T}(1, 2, 3)
    v = T[1, 2, 3]
    if JET_WORKS
        @test_opt norm2(a)
        @test_opt rnorm(a)
        @test_opt rdot(a, a)
        @test_opt norm2(v)
        @test_opt rnorm(v)
        @test_opt rdot(v, v)
    else
        @test_skip JET_WORKS  # issue #8
    end
end

@kernel function reduce_kernel!(n2, d, xs, ys)
    i = @index(Global)
    @inbounds begin
        n2[i] = norm2(xs[i])
        d[i] = rdot(xs[i], ys[i])
    end
end

@testset "SVector inside a KernelAbstractions kernel, $T" for T in ELTYPES
    R = real(T)
    rng = Xoshiro(1)
    xs = [randn(rng, SVector{3, T}) for _ in 1:16]
    ys = [randn(rng, SVector{3, T}) for _ in 1:16]
    n2 = zeros(R, 16)
    d = zeros(R, 16)
    backend = KA_BACKEND
    reduce_kernel!(backend)(n2, d, xs, ys; ndrange = 16)
    KernelAbstractions.synchronize(backend)
    @test n2 == norm2.(xs)
    @test d == rdot.(xs, ys)
end

@kernel function rnorm_kernel!(r, xs)
    i = @index(Global)
    @inbounds r[i] = rnorm(xs[i])
end

allocations(f::F, a::A) where {F, A} = (f(a); @allocated f(a))

@testset "rnorm of an SVector in a kernel and behind a barrier, $T" for T in ELTYPES
    R = real(T)
    xs = [SVector{4}(range_vector(T, top)) for top in range_tops(R)]
    r = zeros(R, length(xs))
    rnorm_kernel!(KA_BACKEND)(r, xs; ndrange = length(xs))
    KernelAbstractions.synchronize(KA_BACKEND)
    @test all(abs.(r .- norm.(xs)) .<= 2eps(R) .* norm.(xs))
    @test allocations(collect, xs[1]) > 0    # the control
    for x in xs
        @test @inferred(rnorm(x)) isa R
        @test allocations(rnorm, x) == 0
    end
    @test allocations(rnorm, randn(Xoshiro(1), T, 100)) == 0
end

# The scaled path needs a floating-point `R`: it converts its scaled sum back to an integer `R`,
# which raises an `InexactError`, and its `ldexp` recurses without end on a `Dual`. For these
# element types `rnorm` is `sqrt(norm2(a))`. The `Dual` cases come last.
@testset "rnorm of a non-float Number gives the base's value" begin
    @test rnorm(3) === 3.0
    @test rnorm([1, 2, 2]) === 3.0
    @test rnorm(Int32[3, 4]) === 5.0
    @test rnorm(3 + 4im) === 5.0
    @test rnorm(Int[]) === 0.0
    @test rnorm(Dual(3.0, 1.0)) === Dual(3.0, 1.0)
    r = rnorm(SVector(Dual(3.0, 1.0), Dual(4.0, 0.0)))
    @test ForwardDiff.value(r) === 5.0
    @test abs(ForwardDiff.partials(r)[1] - 0.6) <= 8eps()
end
