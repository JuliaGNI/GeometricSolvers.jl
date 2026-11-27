using ForwardDiff: ForwardDiff
using GeometricSolvers
using GeometricSolvers: dense_factor, factorize!!, ldiv!!, returncode, StaticLU, StaticQR
using LinearAlgebra: cond, norm
using StaticArrays: SMatrix, SVector
using Test

include("../helpers/matrix.jl")
include("../helpers/staticsystems.jl")

const STATIC_METHODS = (LUFactorization, QRFactorization)

# the lower pairs: (T, Float32) for T = Float64 and ComplexF64
const LOWER_PAIRS = filter(p -> p[2] !== Nothing, PRECISION_PAIRS)

@testset "check 4: the T ≠ TF solve scales the right-hand side to unit size" begin
    @testset "$M, $T with TF = $TF, n = $n" for M in STATIC_METHODS, (T, TF) in LOWER_PAIRS,
        n in STATIC_SIZES

        m = M(TF)
        F = factortype(T, TF)
        A, b = first(conditioned_systems(4000n + 1, T, TF, n))
        fac = factorize!!(nothing, m, A)
        @test eltype(fac.factors) === F
        # 1e-40 is below floatmin(Float32), 1e40 above floatmax(Float32); the unscaled
        # sqrt(norm2(r)) gives 0 for fill(1e-163, n) and Inf for fill(1e155, n) in Float64, so
        # the scale is no unscaled 2-norm; a purely imaginary r has no real part to scale by
        rs = SVector{n, T}[T.(s * b / maximum(abs, b)) for s in (1e-40, 1e40)]
        append!(rs, [fill(T(1e-163), n), fill(T(1e155), n)])
        T <: Complex &&
            append!(rs, [T.(im * s * real.(b) / maximum(abs ∘ real, b))
                         for s in (1e-40, 1e40)])
        for r in rs
            x = ldiv!!(nothing, fac, r)
            @test eltype(x) === T
            @test all(isfinite, x)
            @test relerr(x, reference(A, r)) <= bound(A, T, TF)
        end
        # r = 0 gives a step of exactly 0
        @test ldiv!!(nothing, fac, zero(SVector{n, T})) === zero(SVector{n, T})
        # Inf or NaN in r gives a non-finite step, and no throw
        for v in (Inf, -Inf, NaN)
            x = ldiv!!(nothing, fac, Base.setindex(b, T(v), 1))
            @test !all(isfinite, x)
        end
    end

    # r near floatmax(T) and a step entry above 1 in the scaled system: the step times s overflows
    # T unless the power of two of s and the 2^-e of J are applied together
    @testset "r near floatmax, $M, $T with TF = $TF" for M in (STATIC_METHODS...,
            SVDFactorization),
        (T, TF) in LOWER_PAIRS

        s = 0.6 * floatmax(real(T))
        A0 = SMatrix{2, 2, T}(1, 0.1, 0.1, 0.25)
        A = T(s) * A0
        r = SVector{2, T}(s, s)
        x = ldiv!!(nothing, factorize!!(nothing, M(TF), A), r)
        @test eltype(x) === T
        @test all(isfinite, x)
        @test relerr(x, reference(A, r)) <= bound(A0, T, TF)
    end

    # T narrower than TF: the step entry 2^-24 of the scaled system is the smallest subnormal
    # Float16, and half of it rounds to 0 unless frac and the power of two of s are applied before
    # the conversion to T. The identity returns r.
    @testset "T narrower than TF, $M" for M in (STATIC_METHODS..., SVDFactorization)
        A = SMatrix{2, 2, Float16}(1, 0, 0, 1)
        r = SVector{2, Float16}(floatmin(Float16), 1024)
        x = ldiv!!(nothing, factorize!!(nothing, M(Float32), A), r)
        @test eltype(x) === Float16
        @test all(isapprox.(x, r; rtol = 8 * eps(Float16)))
    end
end

@testset "the factor element type is TF, made complex for a complex T" begin
    for M in STATIC_METHODS, T in ELTYPES

        A = SMatrix{2, 2, T}(1, 2, 3, 5)
        @test eltype(factorize!!(nothing, M(), A).factors) === T
        @test eltype(factorize!!(nothing, M(Float32), A).factors) === factortype(T, Float32)
    end
    @test eltype(factorize!!(nothing, LUFactorization(Float32),
        SMatrix{1, 1, ComplexF64}(1)).factors) === ComplexF32
end

@testset "a precision marker on an SMatrix has no method" begin
    for M in STATIC_METHODS, marker in (TF32(), FP16(), BF16()), T in ELTYPES
        @test_throws MethodError factorize!!(nothing, M(marker), SMatrix{2, 2, T}(1, 2, 3, 5))
    end
end

@testset "returncode maps info to SUCCESS or SINGULAR" begin
    A = SMatrix{2, 2, Float64}(1, 2, 3, 4)
    for M in STATIC_METHODS
        @test returncode(factorize!!(nothing, M(), A)) === SUCCESS
        @test returncode(factorize!!(nothing, M(), zero(A))) === SINGULAR
    end
    @test returncode(StaticLU(A, SVector{2, Int32}(1, 2), Int32(0), Int32(0))) === SUCCESS
    @test returncode(StaticLU(A, SVector{2, Int32}(1, 2), Int32(2), Int32(0))) === SINGULAR
    @test returncode(StaticQR(A, SVector(0.0, 0.0), Int32(1), Int32(0))) === SINGULAR
end

@testset "the T ≠ TF factorisation divides J by a power of two, $M, $T with TF = $TF" for M in STATIC_METHODS,
    (T, TF) in LOWER_PAIRS

    m = M(TF)
    F = factortype(T, TF)
    A, b = first(conditioned_systems(5001, T, TF, 4))
    # 1e-50 and 1e-39 are below floatmin(Float32), 1e40 and 1e50 above floatmax(Float32): unscaled,
    # the conversion of s·A to Float32 flushes it to zero or overflows
    for s in (1e-50, 1e-39, 1e40, 1e50)
        sA = real(T)(s) * A
        fac, x = host_solve(m, sA, b)
        @test returncode(fac) === SUCCESS
        @test all(isfinite, x)
        @test relerr(x, reference(sA, b)) <= bound(sA, T, TF)
    end
    # the division is exact: 2^100 A has the factors of A, and an exponent larger by 100
    fac = factorize!!(nothing, m, A)
    fac100 = factorize!!(nothing, m, ldexp(one(real(T)), 100) * A)
    @test fac100.factors == fac.factors
    @test fac100.scale_exponent == fac.scale_exponent + 100
    # a right-hand side in the factor type takes the direct solve, which applies 2^-e too
    A20 = ldexp(one(real(T)), 20) * A
    r = SVector{4, F}(b)
    @test relerr(ldiv!!(nothing, factorize!!(nothing, m, A20), r), reference(A20, r)) <=
          bound(A20, T, TF)
    # without a conversion there is no division
    @test factorize!!(nothing, M(), real(T)(1e40) * A).scale_exponent == 0
end

# The matrix needs a `BigFloat` exponent outside `Int32`. MPFR's largest exponent,
# `get_emax_max()`, exceeds `typemax(Int32)` only where its exponent type `long` has 64 bits; on
# Windows `long` has 32 bits, and the CI job there gives `SINGULAR` and `NaN` for these matrices.
if Base.MPFR.get_emax_max() > typemax(Int32)
    @testset "a BigFloat matrix whose exponent lies outside Int32 is factorised unscaled, $M" for M in (
        LUFactorization, QRFactorization, CholeskyFactorization, LDLTFactorization)

        # A factor in the matrix's own element type divides it by `2^e` only for `e < 0`, and keeps
        # `J` as it is where `e` does not fit the `Int32` field of the static factor: for
        # `e = ±(2^31 + 7)`, `factorize!!` returns a factor with `scale_exponent = 0` and solves.
        for k in (2^31 + 5, -(2^31 + 9))
            s = ldexp(big(1.0), k)
            A = SMatrix{2, 2, BigFloat}(s * big.([4 1; 1 5]))
            fac = factorize!!(nothing, M(), A)
            @test returncode(fac) === SUCCESS
            @test fac.scale_exponent == 0
            x = ldiv!!(nothing, fac, SVector{2, BigFloat}(A * SVector(big(1.0), big(-2.0))))
            @test isapprox(x, [1, -2]; rtol = 64 * eps(BigFloat))
        end
    end
end

@testset "a 0 × 0 matrix sets no scale, $M($TF)" for M in (
        LUFactorization, QRFactorization, CholeskyFactorization, LDLTFactorization),
    TF in (nothing, Float32)

    # The static and the dense host factor of the empty matrix: `e = 0`, and no reduction runs
    # over its entries, in the matrix's element type and in another.
    m = TF === nothing ? M() : M(TF)
    fs = factorize!!(nothing, m, SMatrix{0, 0, Float64}())
    @test returncode(fs) === SUCCESS
    @test fs.scale_exponent == 0
    fd = factorize!!(dense_factor(m, nothing, Float64[], nothing), m, zeros(0, 0))
    @test returncode(fd) === SUCCESS
    @test fd.scale_exponent == 0
end

@testset "the static Cholesky exponent of a matrix of subnormal entries, $T" for T in filter(
    T -> real(T) === Float64, ELTYPES)

    # `cholesky_exponent` clamps to `-emax = -1023`: a factor of another element type keeps it,
    # and one of the matrix's own element type rounds it down to the even `-1024`
    A = SMatrix{2, 2, T}(T[3 1; 1 3] * ldexp(1.0, -1060))
    emax = exponent(floatmax(Float64))
    fac = factorize!!(nothing, CholeskyFactorization(Float32), A)
    @test returncode(fac) === SUCCESS
    @test fac.scale_exponent == -emax
    fac = factorize!!(nothing, CholeskyFactorization(), A)
    @test returncode(fac) === SUCCESS
    @test fac.scale_exponent == -emax - 1
end

@testset "a ForwardDiff.Dual matrix below 1 in its own element type, $M, $R" for M in (
        LUFactorization, QRFactorization, CholeskyFactorization, LDLTFactorization),
    R in REAL_ELTYPES

    # `J(t) = A₀ + t·1·1ᵀ` at `t = 0`, `A₀ = 2^-3 [4 1; 1 5]`: its largest entry is below 1, so
    # the own-type factor divides it by `2^e`, `e = -1` (`-2` for Cholesky, whose exponent is
    # even), which `ldexp2` does for a `Dual` too.
    # The reference is the BigFloat solve: `x = A₀⁻¹ b`, `ẋ = -A₀⁻¹ (1·1ᵀ) x`. The bound is
    # `8 κ² eps(R)`, `κ = cond(A₀) ≈ 1.66`: the value has an error of order `κ eps(R)`, and the
    # partials pass two solves with `A₀`, of order `κ² eps(R)`.
    A₀ = [4 1; 1 5] .// 8
    b₀ = [1, -2]
    x = big.(A₀) \ big.(b₀)
    ẋ = -(big.(A₀) \ (ones(BigFloat, 2, 2) * x))
    rtol = 8 * (BigFloat(cond(Float64.(A₀))))^2 * eps(R)
    J = SMatrix{2, 2}(ForwardDiff.Dual.(R.(A₀), one(R)))
    b = SVector{2}(ForwardDiff.Dual.(R.(b₀), zero(R)))
    fs = factorize!!(nothing, M(), J)
    xs = ldiv!!(nothing, fs, b)
    fd = factorize!!(dense_factor(M(), nothing, Vector(b), nothing), M(), Matrix(J))
    xd = ldiv!!(similar(Vector(b)), fd, Vector(b))
    for (fac, y) in ((fs, xs), (fd, xd))
        @test returncode(fac) === SUCCESS
        @test fac.scale_exponent == (M === CholeskyFactorization ? -2 : -1)
        @test norm(ForwardDiff.value.(y) - x) <= rtol * norm(x)
        @test norm(map(a -> ForwardDiff.partials(a, 1), y) - ẋ) <= rtol * norm(ẋ)
    end
end

@testset "ldexp2 of a ForwardDiff.Dual is exact at e = emax + 1, $R" for R in REAL_ELTYPES
    # ForwardDiff scales the partial of `ldexp(x, e)` by `exp2(e)`, which is `Inf` here
    emax = exponent(floatmax(R))
    s = ldexp(one(R), -emax - 10)
    for p in (s, zero(R), one(R) / 4)
        d = GeometricSolvers.ldexp2(ForwardDiff.Dual(s, p), emax + 1)
        @test isequal(ForwardDiff.value(d), GeometricSolvers.ldexp2(s, emax + 1))
        @test isequal(ForwardDiff.partials(d, 1), GeometricSolvers.ldexp2(p, emax + 1))
    end
end

@testset "Cholesky of a ForwardDiff.Dual matrix of subnormal entries, $R, $kind" for R in REAL_ELTYPES,
    kind in (:static, :dense)

    # `J(t) = s (A₀ + t·1·1ᵀ)` at `t = 0` and `b = s b₀`, `s = 2^(-emax - 10)`: the entries are
    # exact subnormals below `2^-emax`, so the own-type Cholesky exponent is `-emax - 1`, and the
    # factor multiplies `J` by `2^(emax + 1)`. The solution is that of `A₀`: `x = A₀⁻¹ b₀`,
    # `ẋ = -A₀⁻¹ (1·1ᵀ) x`. The solve lifts the right-hand side with the matrix, by
    # `2^(emax + 1)`, so it runs in the normal range. Each of the two solves with `A₀` that the
    # partials pass multiplies the rounding error by at most `κ = cond(A₀) ≈ 1.66`, so the bound
    # is `8 κ² eps(R)`.
    emax = exponent(floatmax(R))
    s = ldexp(one(R), -emax - 10)
    A₀ = [4 1; 1 5] .// 8
    b₀ = [1, -2]
    x = big.(A₀) \ big.(b₀)
    ẋ = -(big.(A₀) \ (ones(BigFloat, 2, 2) * x))
    rtol = 8 * (BigFloat(cond(Float64.(A₀))))^2 * eps(R)
    J = SMatrix{2, 2}(ForwardDiff.Dual.(R.(s .* A₀), s))
    b = SVector{2}(ForwardDiff.Dual.(R.(s .* b₀), zero(R)))
    M = CholeskyFactorization
    if kind === :static
        fac = factorize!!(nothing, M(), J)
        y = ldiv!!(nothing, fac, b)
    else
        fac = factorize!!(dense_factor(M(), nothing, Vector(b), nothing), M(), Matrix(J))
        y = ldiv!!(similar(Vector(b)), fac, Vector(b))
    end
    v = ForwardDiff.value.(y)
    ẏ = map(a -> ForwardDiff.partials(a, 1), y)
    @test returncode(fac) === SUCCESS
    @test fac.scale_exponent == -emax - 1
    @test all(isfinite, v) && all(isfinite, ẏ)
    @test norm(v - x) <= rtol * norm(x)
    @test norm(ẏ - ẋ) <= rtol * norm(ẋ)
end

@testset "DeviceCholesky of a ForwardDiff.Dual JLArray, $R" for R in REAL_ELTYPES
    # The load kernel takes the largest entry and its exponent from the bits of the values of the
    # `Dual`s. `J(t) = s (A₀ + t·1·1ᵀ)` at `t = 0` and `b = s b₀`, as above,
    # at three scales: `s = 8`, whose exponent 2 the own type does not take (`e = 0`); `s = 1`,
    # exponent -1, rounded down to the even `e = -2`; and the subnormal `s = 2^(-emax - 10)`,
    # `e = -emax - 1`. The CPU does not flush subnormals, so the device and the host
    # `DenseCholesky` take the same exponent. The solution and the bound are those above.
    emax = exponent(floatmax(R))
    A₀ = [4 1; 1 5] .// 8
    b₀ = [1, -2]
    x = big.(A₀) \ big.(b₀)
    ẋ = -(big.(A₀) \ (ones(BigFloat, 2, 2) * x))
    M = CholeskyFactorization
    for (s, e) in ((R(8), 0), (one(R), -2), (ldexp(one(R), -emax - 10), -emax - 1))
        rtol = 8 * (BigFloat(cond(Float64.(A₀))))^2 * eps(R)
        J = ForwardDiff.Dual.(R.(s .* A₀), s)
        b = ForwardDiff.Dual.(R.(s .* b₀), zero(R))
        fh = factorize!!(dense_factor(M(), nothing, b, nothing), M(), J)
        bd = JLArray(b)
        fd = factorize!!(dense_factor(M(), nothing, bd, nothing), M(), JLArray(J))
        y = Array(ldiv!!(similar(bd), fd, bd))
        v = ForwardDiff.value.(y)
        ẏ = map(a -> ForwardDiff.partials(a, 1), y)
        @test returncode(fd) === returncode(fh) === SUCCESS
        @test fd.scale_exponent == fh.scale_exponent == e
        @test all(isfinite, v) && all(isfinite, ẏ)
        @test norm(v - x) <= rtol * norm(x)
        @test norm(ẏ - ẋ) <= rtol * norm(ẋ)
    end
end

@testset "a ForwardDiff.Dual matrix of zero values sets no scale, $R" for R in REAL_ELTYPES
    # Every value is zero and every partial is one, for a `Dual` and for a `Dual` of `Dual`s.
    # ForwardDiff's `iszero`, `==` and `<` of a `Dual` also read the partials, so the scale is
    # taken from the innermost values alone: the zero matrix keeps `e = 0`, and the factor buffer
    # takes `J` as it is.
    Z = ForwardDiff.Dual.(zeros(R, 3, 3), one(R))
    for J in (Z, ForwardDiff.Dual.(Z, one(R)))
        b = map(a -> oftype(a, 1), J[:, 1])
        bd = JLArray(b)
        m = CholeskyFactorization()
        fd = factorize!!(dense_factor(m, nothing, bd, nothing), m, JLArray(J))
        J === Z && @test returncode(fd) === SINGULAR
        @test fd.scale_exponent == 0
        for M in (LUFactorization, QRFactorization, LDLTFactorization, CholeskyFactorization),
            (A, r) in ((J, b), (JLArray(J), bd))

            M === CholeskyFactorization && A isa JLArray && continue
            fac = factorize!!(dense_factor(M(), nothing, r, nothing), M(), A)
            @test fac.scale_exponent == 0
        end
        me, ce = GeometricSolvers.matrix_exponent, GeometricSolvers.cholesky_exponent
        for (A, se) in ((J, me), (J, ce), (JLArray(J), me))
            B = similar(A)
            @test GeometricSolvers.load_factors!(B, A, se) == 0
            @test all(map(===, Array(B), J))
        end
    end
end

@testset "the Cholesky scale of a complex Dual matrix reads the lower imaginary parts, $R" for R in REAL_ELTYPES
    # `J = [2^-20 0; 3·2^-10 im 2^-20]`: the largest entry that a Cholesky factorisation reads is
    # the imaginary part `3·2^-10` of `J[2, 1]`, of exponent -9, which the own type rounds down to
    # the even `e = -10`; without that imaginary part the exponent is -20. Only `e` is asserted.
    D(x) = ForwardDiff.Dual(R(x), one(R))
    s = ldexp(one(R), -20)
    t = 3 * ldexp(s, 10)
    J = [complex(D(s), D(0)) complex(D(0), D(0)); complex(D(0), D(t)) complex(D(s), D(0))]
    b = [complex(D(1), D(0)), complex(D(2), D(0))]
    m = CholeskyFactorization()
    fh = factorize!!(dense_factor(m, nothing, b, nothing), m, J)
    bd = JLArray(b)
    fd = factorize!!(dense_factor(m, nothing, bd, nothing), m, JLArray(J))
    @test fh.scale_exponent == -10
    @test fd.scale_exponent == -10
end

@testset "the LU and QR scale of a complex matrix reads the imaginary parts, $M, $R, dual = $dual" for M in (
        LUFactorization, QRFactorization),
    R in REAL_ELTYPES, dual in (false, true)

    # `J = [2^-20 0; 3·2^-10 im 2^-20]`, in `Complex{R}` and in `Complex{Dual{R}}`: the largest
    # |re| or |im| is the imaginary part `3·2^-10` of `J[2, 1]`, of exponent -9, which the own
    # type divides by; without the imaginary parts the exponent is -20. Only `e` is asserted.
    D(x) = dual ? ForwardDiff.Dual(R(x), one(R)) : R(x)
    s = ldexp(one(R), -20)
    t = 3 * ldexp(s, 10)
    J = [complex(D(s), D(0)) complex(D(0), D(0)); complex(D(0), D(t)) complex(D(s), D(0))]
    b = [complex(D(1), D(0)), complex(D(2), D(0))]
    fh = factorize!!(dense_factor(M(), nothing, b, nothing), M(), J)
    bd = JLArray(b)
    fd = factorize!!(dense_factor(M(), nothing, bd, nothing), M(), JLArray(J))
    fs = factorize!!(nothing, M(), SMatrix{2, 2}(J))
    @test fh.scale_exponent == -9
    @test fd.scale_exponent == -9
    @test fs.scale_exponent == -9
end

@testset "ldexp2 of a ForwardDiff.Dual scales its value and partials in their own type, $R" for R in REAL_ELTYPES
    # The value and each partial are scaled by `ldexp2` of `R`, and a `Dual` of `Dual`s by
    # component down to its innermost values, so that the result is exact and has the type of the
    # argument, its tag included; no `exp2(e)` of a `Float64` enters, which the Apple GPU cannot
    # compile.
    emax = exponent(floatmax(R))
    s = ldexp(one(R), -emax - 10)
    d = ForwardDiff.Dual{typeof(ForwardDiff.Tag(sin, R))}(s, R(3), one(R) / 4)
    for e in (emax + 1, -3, 7)
        y = @inferred GeometricSolvers.ldexp2(d, e)
        @test typeof(y) === typeof(d)
        @test isequal(ForwardDiff.value(y), GeometricSolvers.ldexp2(s, e))
        @test isequal(ForwardDiff.partials(y, 1), GeometricSolvers.ldexp2(R(3), e))
        @test isequal(ForwardDiff.partials(y, 2), GeometricSolvers.ldexp2(one(R) / 4, e))
        dd = ForwardDiff.Dual{typeof(ForwardDiff.Tag(cos, typeof(d)))}(d, d)
        yy = @inferred GeometricSolvers.ldexp2(dd, e)
        @test typeof(yy) === typeof(dd)
        @test ForwardDiff.value(yy) === y
        @test ForwardDiff.partials(yy, 1) === y
        # A `Dual` with no partials, as `Dual(x)` builds: its empty partials keep the value type.
        d0 = ForwardDiff.Dual(s)
        y0 = @inferred GeometricSolvers.ldexp2(d0, e)
        @test typeof(y0) === typeof(d0)
        @test isequal(ForwardDiff.value(y0), GeometricSolvers.ldexp2(s, e))
    end
    # The factorisations of a matrix of `Dual`s with no partials, `J = 2^k A₀` and `b = 2^k b₀`:
    # at `k = -20` the largest entry `2^-18` makes the own type divide by `2^-18` (an even
    # exponent for Cholesky too), and at `k = 0` it keeps `e = 0`. The solution is `A₀⁻¹ b₀`,
    # within `8 κ eps(R)`, `κ = cond(A₀) ≈ 2.1`; the device row is the JLArray Cholesky load kernel.
    A₀ = R[4 1 0; 1 4 1; 0 1 4]
    b₀ = R[1, 2, 3]
    x = big.(A₀) \ big.(b₀)
    rtol = 8 * BigFloat(cond(Float64.(A₀))) * eps(R)
    for (k, e) in ((0, 0), (-20, -18)),
        M in (LUFactorization, QRFactorization, CholeskyFactorization)

        J = ForwardDiff.Dual.(ldexp.(A₀, k))
        b = ForwardDiff.Dual.(ldexp.(b₀, k))
        bd = JLArray(b)
        runs = [(factorize!!(dense_factor(M(), nothing, b, nothing), M(), copy(J)), b),
            (factorize!!(nothing, M(), SMatrix{3, 3}(J)), SVector{3}(b))]
        if M === CholeskyFactorization
            fd = factorize!!(dense_factor(M(), nothing, bd, nothing), M(), JLArray(J))
            push!(runs, (fd, bd))
        end
        for (fac, r) in runs
            y = Array(ldiv!!(r isa SVector ? nothing : similar(r), fac, r))
            @test returncode(fac) === SUCCESS
            @test fac.scale_exponent == e
            @test norm(ForwardDiff.value.(y) - x) <= rtol * norm(x)
        end
    end
end

@testset "ldexp2 of a real whose float is no IEEEFloat scales it by ldexp" begin
    # `float` of a `BigInt` or a `Rational{BigInt}` is a `BigFloat`, whose exponent range holds
    # `2^±2000`, so the scaled value is exact.
    for a in (big(3), big(3) // 1), e in (-2, 2000, -2000)

        y = GeometricSolvers.ldexp2(a, e)
        @test y isa BigFloat
        @test y == 3 * big(2.0)^e
    end
end

@testset "the Cholesky load kernel of a ForwardDiff.Dual{Float32} JLArray holds no Float64" begin
    # The device LLVM IR of the load kernel, for the matrix `2^-130 [4 1 0; 1 4 1; 0 1 4]`, whose
    # scale the own type divides by: a `double` in it is a Float64 operation, which the Apple GPU
    # refuses to compile ("unsupported use of double value").
    D(x, p) = ForwardDiff.Dual(x, p)
    m = CholeskyFactorization()
    A = JLArray(D.(Float32[4 1 0; 1 4 1; 0 1 4] .* ldexp(1.0f0, -130), 1.0f0))
    b = JLArray(D.(Float32[1, 2, 3], 0.0f0))
    io = IOBuffer()
    KernelAbstractions.GPUCompiler.@device_code_llvm io=io debuginfo=:none factorize!!(
        dense_factor(m, nothing, b, nothing), m, A)
    ir = String[]
    inload = false
    for l in split(String(take!(io)), '\n')
        if startswith(l, "; ") && occursin("CompilerJob", l)
            inload = occursin("load_kernel", l)
        end
        inload && !startswith(lstrip(l), ";") && push!(ir, l)
    end
    @test !isempty(ir)
    @test count(l -> occursin("double", l), ir) == 0
end

@testset "a 0 × 0 matrix sets no scale in the static SVD" begin
    # the exponent of the empty matrix runs no reduction over its entries; the static SVD of it
    # throws later, at `σ[1]` (K127 of `KNOWN_ISSUES.md`)
    @test GeometricSolvers.svd_exponent(SMatrix{0, 0, Float64}()) == 0
    @test GeometricSolvers.svd_exponent(SMatrix{0, 0, ComplexF32}()) == 0
end

@testset "the same-type solve scales r by a power of two, $T" for T in REAL_ELTYPES
    @testset "$label, $M" for (label, J, b, xref, with_qr, atol) in extreme_rhs_systems(T),
        M in (SVDFactorization, QRFactorization, LUFactorization)

        M === QRFactorization && !with_qr && continue
        M === LUFactorization && !with_qr && continue
        fac = factorize!!(nothing, M(), SMatrix{4, 4, T}(J))
        x = ldiv!!(nothing, fac, SVector{4, T}(b))
        @test eltype(x) === T
        @test all(isfinite, x)
        @test relerr(x, xref) <= 100 * eps(T)
        @test all(isapprox.(big.(x), xref; rtol = 100 * eps(T), atol))
    end

    # a solution above floatmax overflows, by design: 2^-(emax + 3) I has the solution 2^(emax + 3)
    @testset "a solution above floatmax overflows, $M" for M in (LUFactorization,
        QRFactorization, SVDFactorization)
        J = SMatrix{4, 4, T}(ldexp(one(T), -(exponent(floatmax(T)) + 3)) *
                             Matrix{T}(I, 4, 4))
        x = ldiv!!(nothing, factorize!!(nothing, M(), J), ones(SVector{4, T}))
        @test !any(isfinite, x)
    end

    # a zero or non-finite r takes the path of an r that is not scaled
    @testset "a zero or non-finite r, $M" for M in (QRFactorization, SVDFactorization)
        fac = factorize!!(nothing, M(), SMatrix{4, 4, T}(2 * Matrix{T}(I, 4, 4)))
        @test ldiv!!(nothing, fac, zeros(SVector{4, T})) == zeros(SVector{4, T})
        for v in (T(Inf), T(NaN))
            @test !all(isfinite, ldiv!!(nothing, fac, SVector{4, T}(v, 1, 1, 1)))
        end
    end
end
