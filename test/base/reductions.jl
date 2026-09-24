using GeometricSolvers: norm2, rnorm, rdot
using JET: JET, @test_opt
using JLArrays: JLArray
using KernelAbstractions: KernelAbstractions, @kernel, @index
using Random: Xoshiro
using StaticArrays: SVector
using Test

const ELTYPES = (Float32, Float64, ComplexF32, ComplexF64)

# The error bound of a sum of n products in any order, ``γ_{n+2} Σ |aᵢ bᵢ|`` with
# ``γ_k = k u / (1 - k u)`` (Higham, *Accuracy and Stability*, §3.1): n - 1 additions, and at
# most one product and one addition per term for a complex `abs2` or `conj(x) * y`.
γ(k, R) = k * eps(R) / (1 - k * eps(R))

@testset "exact against BigFloat within γ(n+2): $T, n = $n, $(AT)" for T in ELTYPES,
    n in (1, 7, 1000), AT in (Array, JLArray)
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

@testset "a JLArray refuses scalar indexing" begin
    @test_throws ErrorException JLArray(zeros(Float32, 2))[1]
end

@testset "empty arrays reduce to zero(R)" for T in ELTYPES
    @test norm2(T[]) === zero(real(T))
    @test rdot(T[], T[]) === zero(real(T))
end

# Inside a function with concrete argument types, so that the measurement does not count the
# boxing of a result or an argument: 16 to 96 bytes on Julia 1.11 at testset scope or through
# an unspecialised varargs method.
allocations(f::F, a::A) where {F, A} = (f(a); @allocated f(a))
allocations(f::F, a::A, b::B) where {F, A, B} = (f(a, b); @allocated f(a, b))

@testset "SVector: inferred and allocation-free, $T" for T in ELTYPES
    a = SVector{4, T}(1, 2, 3, 4)
    b = SVector{4, T}(4, 3, 2, 1)
    @test @inferred(norm2(a)) === real(T)(30)
    @test @inferred(rnorm(a)) === sqrt(real(T)(30))
    @test @inferred(rdot(a, b)) === real(T)(20)
    @test allocations(collect, a) > 0        # the control: the helper sees an allocation
    @test allocations(norm2, a) == 0
    @test allocations(rnorm, a) == 0
    @test allocations(rdot, a, b) == 0
end

@testset "Array: allocation-free, $T" for T in ELTYPES
    a, b = randn(Xoshiro(1), T, 100), randn(Xoshiro(2), T, 100)
    @test allocations(norm2, a) == 0
    @test allocations(rnorm, a) == 0
    @test allocations(rdot, a, b) == 0
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
        @test_skip JET_WORKS
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
    backend = KernelAbstractions.CPU()
    reduce_kernel!(backend)(n2, d, xs, ys; ndrange = 16)
    KernelAbstractions.synchronize(backend)
    @test n2 == norm2.(xs)
    @test d == rdot.(xs, ys)
end
