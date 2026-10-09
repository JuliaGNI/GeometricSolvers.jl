# The R3 Jacobian: the own chunked forward mode of `AutoForwardDiff`, on an `Array` and on a device
# array, and the ways it can be wrong; and DifferentiationInterface for the other back ends.
#
# Every comparison of the chunked mode is exact (`==`), not approximate: the residuals of
# `test/helpers/adproblems.jl` are broadcasts with no reduction, so the arithmetic per element is
# the same on every backend and the same as `ForwardDiff.jacobian`'s on an `Array`. A tolerance
# there would hide exactly the faults the file is for.

using ADTypes: AutoFiniteDiff, AutoForwardDiff
using FiniteDiff: FiniteDiff
using ForwardDiff: ForwardDiff
using GPUArraysCore: allowscalar
using JET: JET
using JLArrays: JLArray
using LinearAlgebra: Diagonal
using Test

using GeometricSolvers: ChunkedForwardDiff, DIJacobian, jacobian!!, jvp!!, prepare_ad

include("../helpers/matrix.jl")
include("../helpers/adproblems.jl")

# The tag a preparation carries: a type parameter of the chunked mode.
ad_tag(::ChunkedForwardDiff{N, Tg}) where {N, Tg} = Tg

# A chunk size of 3 does not divide 7, so the last chunk is partial.
const N_AD = 7
const CHUNK = 3

# Scalar indexing on a device array is an error for every test in this file, so a path that falls
# back to a scalar loop — DI's `jacobian!` on a `JLArray`, an `MtlArray` or a `CuArray` — fails
# here rather than passing slowly. A non-interactive session disallows it already; this states it,
# and the testset below shows that the setting is in force rather than assumed.
allowscalar(false)

@testset "scalar indexing on a device array is an error in this file" begin
    @test_throws ErrorException JLArray(zeros(Float32, 2))[1]
end

# The allocation assertion goes through a function barrier whose arguments all have concrete
# types, and which calls the function once before it measures: at testset scope Julia 1.11 boxes
# the captured values and the assertion fails although the code allocates nothing.
function jacobian_allocations(J::A, prep::P, prob::Q, x::X, p::R) where {A, P, Q, X, R}
    jacobian!!(J, prep, prob, x, p)
    return @allocated jacobian!!(J, prep, prob, x, p)
end

@testset "the AD path is selected by the back end and the iterate, once" begin
    @testset "$AT, $T" for AT in ARRAY_BACKENDS, T in REAL_ELTYPES

        prob = StubProblem(Coupled(cyclic_perm(AT, N_AD)))
        x, p, r = ad_inputs(AT, T, N_AD)
        @test prepare_ad(AutoForwardDiff(), prob, r, x, p) isa ChunkedForwardDiff
    end

    # A back end other than ForwardDiff has no device path, rather than a fallback that would
    # fail deep inside DI with a scalar-indexing error. The iterate decides, not the residual
    # buffer: a host buffer beside a device iterate is refused as well.
    prob = StubProblem(Scaled())
    x, p, r = ad_inputs(JLArray, Float64, N_AD)
    @test_throws ArgumentError prepare_ad(AutoFiniteDiff(), prob, r, x, p)
    @test_throws ArgumentError prepare_ad(AutoFiniteDiff(), prob, Array(r), x, p)
end

@testset "any other back end goes through DifferentiationInterface on an Array" begin
    # `AutoFiniteDiff()` stands for every back end but ForwardDiff. A forward difference with step
    # `√eps` is accurate to about `√eps`, so these comparisons carry that tolerance, with a factor
    # ten; a wrong column is off by `O(1)`.
    @testset "$T" for T in REAL_ELTYPES
        tol = 10 * sqrt(eps(T))
        prob = StubProblem(Coupled(cyclic_perm(Array, N_AD)))
        x, p, r = ad_inputs(Array, T, N_AD)
        prep = prepare_ad(AutoFiniteDiff(), prob, r, x, p)
        @test prep isa DIJacobian
        J = similar(x, N_AD, N_AD)
        @test jacobian!!(J, prep, prob, x, p) === J
        @test J ≈ forwarddiff_jacobian(prob.F, x, p) rtol = tol
        Jv = similar(x)
        v = T[(i + 2) / (N_AD + 1) for i in 1:N_AD]
        @test jvp!!(Jv, prep, prob, x, v, p) === Jv
        @test Jv ≈ forwarddiff_jacobian(prob.F, x, p) * v rtol = tol

        # the parameters are a constant, and a replacement object takes effect
        scaled = StubProblem(Scaled())
        sprep = prepare_ad(AutoFiniteDiff(), scaled, r, x, p)
        q = 3 .* p
        jacobian!!(J, sprep, scaled, x, q)
        @test J ≈ Diagonal(q) rtol = tol

        # and an empty iterate writes nothing and returns its output, where DI with
        # `AutoFiniteDiff()` raises on its own
        x0, p0, r0 = ad_inputs(Array, T, 0)
        prep0 = prepare_ad(AutoFiniteDiff(), scaled, r0, x0, p0)
        J0 = similar(x0, 0, 0)
        @test jacobian!!(J0, prep0, scaled, x0, p0) === J0
        Jv0 = similar(x0)
        @test jvp!!(Jv0, prep0, scaled, x0, similar(x0), p0) === Jv0
    end
end

@testset "the chunk size: the caller's where it is positive, otherwise ForwardDiff's, clamped" begin
    # The chunk size is a type parameter of the preparation, so each case is read off the type and
    # none of them needs a Jacobian.
    prob = StubProblem(Coupled(cyclic_perm(JLArray, N_AD)))
    @testset "$T" for T in REAL_ELTYPES
        x, p, r = ad_inputs(JLArray, T, N_AD)
        prepared(c) = prepare_ad(AutoForwardDiff(; chunksize = c), prob, r, x, p)

        # the caller's, where it is positive — one included, which `pickchunksize` would not give
        @test prepared(CHUNK) isa ChunkedForwardDiff{CHUNK}
        @test prepared(1) isa ChunkedForwardDiff{1}

        # wider than the iterate is clamped to `n`: a wider chunk would seed columns that do not
        # exist
        @test prepared(N_AD + 5) isa ChunkedForwardDiff{N_AD}

        # not positive is no chunk size at all, so ForwardDiff's own choice stands, as it does
        # when the caller names none
        @test prepared(0) isa ChunkedForwardDiff{ForwardDiff.pickchunksize(N_AD)}
        @test prepared(-2) isa ChunkedForwardDiff{ForwardDiff.pickchunksize(N_AD)}
        @test prepare_ad(AutoForwardDiff(), prob, r, x, p) isa
              ChunkedForwardDiff{ForwardDiff.pickchunksize(N_AD)}

        # and an empty iterate still has a chunk of at least one: `pickchunksize(0)` is `0`, and a
        # step of zero is not a range. There is no column, so the residual is never called.
        x0, p0, r0 = ad_inputs(JLArray, T, 0)
        prep0 = prepare_ad(AutoForwardDiff(), prob, r0, x0, p0)
        @test prep0 isa ChunkedForwardDiff{1}
        J0 = similar(x0, 0, 0)
        @test jacobian!!(J0, prep0, prob, x0, p0) === J0
    end
end

@testset "a Jacobian costs ceil(n / N) residual evaluations" begin
    # `N` columns come out of one evaluation. This also says that the chunk loop stops at the
    # last column: a loop one step too long evaluates the residual once more for a chunk with no
    # column in it.
    @testset "$AT, $T, chunk $N" for AT in ARRAY_BACKENDS, T in REAL_ELTYPES,
        N in (CHUNK, N_AD)
        counting = Counting(Coupled(cyclic_perm(AT, N_AD)))
        prob = StubProblem(counting)
        x, p, r = ad_inputs(AT, T, N_AD)
        prep = prepare_ad(AutoForwardDiff(; chunksize = N), prob, r, x, p)
        J = similar(x, N_AD, N_AD)
        # the preparation evaluates the residual as often as it needs; the count starts here
        counting.calls[] = 0
        jacobian!!(J, prep, prob, x, p)
        @test counting.calls[] == cld(N_AD, N)
    end
end

@testset "the caller's tag is kept where the caller named one" begin
    # The solver's own tag is for a back end that carries none. A caller who nests this solver
    # inside another differentiation needs its own tag to survive.
    @testset "$AT, $T" for AT in ARRAY_BACKENDS, T in REAL_ELTYPES

        host = Coupled(cyclic_perm(Array, N_AD))
        prob = StubProblem(Coupled(cyclic_perm(AT, N_AD)))
        x, p, r = ad_inputs(AT, T, N_AD)
        mine = ForwardDiff.Tag(CallerTagged(), T)
        prep = prepare_ad(AutoForwardDiff(; tag = mine), prob, r, x, p)
        @test ad_tag(prep) === typeof(mine)
        @test ad_tag(prep) !== typeof(ForwardDiff.Tag(prob.F, T))
        J = similar(x, N_AD, N_AD)
        jacobian!!(J, prep, prob, x, p)
        @test Array(J) == forwarddiff_jacobian(host, x, p)
    end
end

@testset "a holomorphic residual in a complex element type" begin
    # A complex iterate gives `Complex{Dual}` buffers, not `Dual{Complex}`: the derivative is the
    # complex one, `d r / d z`, which is what a holomorphic residual has. The seed therefore
    # perturbs the real component by one and the imaginary component by zero.
    @testset "$AT, $T, chunk $N" for AT in ARRAY_BACKENDS,
        T in filter(T -> T <: Complex, ELTYPES), N in (CHUNK, N_AD)
        prob = StubProblem(Holomorphic())
        x, p, r = ad_inputs(AT, T, N_AD)
        prep = prepare_ad(AutoForwardDiff(; chunksize = N), prob, r, x, p)
        J = similar(x, N_AD, N_AD)
        jacobian!!(J, prep, prob, x, p)
        # `r_i = x_i² + p_i x_i`, so `dr_i/dx_i = 2 x_i + p_i` — exactly, in the arithmetic the
        # dual numbers do: the partial of `x*x` is `x + x` and that of `p*x` is `p`.
        @test Array(J) == Diagonal(Array(x) .+ Array(x) .+ Array(p))
        # the imaginary component of the seed is a zero, not a one: with a one the result would
        # carry the perturbation of the other component too
        @test Array(J) != Diagonal((Array(x) .+ Array(x) .+ Array(p)) .* (1 + im))
    end
end

@testset "the Jacobian agrees with ForwardDiff on an Array" begin
    @testset "$AT, $T" for AT in ARRAY_BACKENDS, T in REAL_ELTYPES

        host = Coupled(cyclic_perm(Array, N_AD))
        prob = StubProblem(Coupled(cyclic_perm(AT, N_AD)))
        x, p, r = ad_inputs(AT, T, N_AD)
        prep = prepare_ad(AutoForwardDiff(), prob, r, x, p)
        J = similar(x, N_AD, N_AD)
        @test @inferred(jacobian!!(J, prep, prob, x, p)) === J
        @test Array(J) == forwarddiff_jacobian(host, x, p)
        # the comparison is a test only because the Jacobian is neither symmetric nor diagonal
        @test Array(J) != transpose(Array(J))
    end
end

@testset "the chunk size does not divide n: n = $N_AD, chunk = $CHUNK" begin
    # The partial last chunk is where a column gets no seed, or the same seed twice. Scalar
    # indexing is disallowed for the whole file, so a scalar fallback fails here.
    @testset "$AT, $T" for AT in ARRAY_BACKENDS, T in REAL_ELTYPES

        host = Coupled(cyclic_perm(Array, N_AD))
        prob = StubProblem(Coupled(cyclic_perm(AT, N_AD)))
        x, p, r = ad_inputs(AT, T, N_AD)
        prep = prepare_ad(AutoForwardDiff(; chunksize = CHUNK), prob, r, x, p)
        @test prep isa ChunkedForwardDiff{CHUNK}
        J = similar(x, N_AD, N_AD)
        jacobian!!(J, prep, prob, x, p)
        @test Array(J) == forwarddiff_jacobian(host, x, p)
    end
end

@testset "a residual that differentiates: the tag is this solver's own" begin
    # `Nested` calls `ForwardDiff.derivative` in its body. Without an owned tag the inner and the
    # outer perturbation are confused, and the result is wrong rather than an error.
    @testset "$AT, $T" for AT in ARRAY_BACKENDS, T in REAL_ELTYPES

        prob = StubProblem(Nested())
        x, p, r = ad_inputs(AT, T, N_AD)
        prep = prepare_ad(AutoForwardDiff(), prob, r, x, p)
        J = similar(x, N_AD, N_AD)
        jacobian!!(J, prep, prob, x, p)
        @test Array(J) == forwarddiff_jacobian(Nested(), x, p)
        # and the reference is itself right: r_i = 2 p_i x_i², so J = Diagonal(4 p_i x_i). The two
        # expressions round differently, so this one carries a tolerance, named here: four ulps of
        # the element type.
        @test Array(J) ≈ Diagonal(4 .* Array(p) .* Array(x)) rtol = 4 * eps(T)

        # The value comparison above does not by itself show that the tag is owned: the
        # `ForwardDiff.derivative` inside the residual builds a tag of its own, which differs from
        # `Nothing` as much as from ours, so the nesting still comes out right with no tag at all.
        # The tag is therefore asserted directly: this solver's own, named for the residual and
        # the element type, and never `Nothing`.
        @test ad_tag(prep) === typeof(ForwardDiff.Tag(prob.F, T))
        @test ad_tag(prep) <: ForwardDiff.Tag
    end
end

@testset "the parameters are a constant, and a replacement takes effect" begin
    @testset "$AT, $T" for AT in ARRAY_BACKENDS, T in REAL_ELTYPES

        prob = StubProblem(Scaled())
        x, p, r = ad_inputs(AT, T, N_AD)
        prep = prepare_ad(AutoForwardDiff(), prob, r, x, p)
        J = similar(x, N_AD, N_AD)
        jacobian!!(J, prep, prob, x, p)
        # `r = p .* x`, so the Jacobian is `Diagonal(p)` exactly. A parameter differentiated by
        # mistake puts `x` into the diagonal instead.
        @test Array(J) == Diagonal(Array(p))

        # the same array, new values: no new solver, and no stale preparation
        p .= 2 .* p
        jacobian!!(J, prep, prob, x, p)
        @test Array(J) == Diagonal(Array(p))

        # a different array object, as a caller that rebuilt its parameters passes
        q = AT(3 .* Array(p))
        jacobian!!(J, prep, prob, x, q)
        @test Array(J) == Diagonal(Array(q))
        @test Array(J) != Diagonal(Array(p))

        # and an object of another type: the chunked mode holds no parameter, so it needs no new
        # preparation either
        jacobian!!(J, prep, prob, x, Tuple(Array(p)))
        @test Array(J) == Diagonal(Array(p))
    end
end

@testset "a non-vector iterate is refused, on both paths" begin
    # A chunk is a range of Jacobian columns, and a matrix unknown has no such column numbering.
    # Both paths refuse it where the preparation is built, and the message carries the shape — a
    # `DimensionMismatch` out of a broadcast names no size the caller can place. The DI path is
    # reached on an `Array` only, and refuses the shape before DI is called.
    @testset "$AT, $T" for AT in ARRAY_BACKENDS, T in REAL_ELTYPES

        prob = StubProblem(Scaled())
        x = AT(reshape(T[(i + 1) / 8 for i in 1:6], 3, 2))
        p = AT(reshape(T[(2i + 3) / 11 for i in 1:6], 3, 2))
        backends = AT === Array ? (AutoForwardDiff(), AutoFiniteDiff()) :
                   (AutoForwardDiff(),)
        for backend in backends
            err = @test_throws ArgumentError prepare_ad(backend, prob, similar(x), x, p)
            @test occursin("(3, 2)", err.value.msg)
        end
        # a vector of the same length is prepared, so it is the shape and not the size that is
        # refused
        xv, pv, rv = ad_inputs(AT, T, 6)
        @test prepare_ad(AutoForwardDiff(; chunksize = CHUNK), prob, rv, xv, pv) isa
              ChunkedForwardDiff
    end
end

@testset "an empty iterate writes nothing and returns its output" begin
    # The chunked mode runs no chunk at `n = 0`. An `n = 0` solve is what a problem whose unknowns
    # are all eliminated gives, and it must not throw.
    @testset "$AT, $T" for AT in ARRAY_BACKENDS, T in REAL_ELTYPES

        prob = StubProblem(Scaled())
        x, p, r = ad_inputs(AT, T, 0)
        prep = prepare_ad(AutoForwardDiff(), prob, r, x, p)
        J = similar(x, 0, 0)
        @test jacobian!!(J, prep, prob, x, p) === J
        Jv = similar(x)
        @test jvp!!(Jv, prep, prob, x, similar(x), p) === Jv
        @test length(Jv) == 0
    end
end

@testset "the dual buffers are allocated once, in prepare_ad" begin
    # On an `Array` no call allocates: not with one chunk, and not with several, at `n` and at
    # `4n`. A buffer allocated per call, or per chunk, would show here.
    @testset "an Array allocates nothing, $T, n = $n, chunk $c" for T in REAL_ELTYPES,
        n in (N_AD, 4 * N_AD), c in (0, CHUNK)
        prob = StubProblem(Scaled())
        x, p, r = ad_inputs(Array, T, n)
        prep = prepare_ad(AutoForwardDiff(; chunksize = c), prob, r, x, p)
        @test jacobian_allocations(similar(x, n, n), prep, prob, x, p) == 0
    end

    # On a device array the dual buffers are the state slots, and they stay the arrays that
    # `prepare_ad` made. A `JLArray` broadcast allocates host bookkeeping per call, so an exact
    # zero is not reachable there; what is required is that no call allocates a dual buffer —
    # which identity shows — and that the host allocation does not grow from call to call.
    @testset "the state slots keep their identity on a JLArray, $T" for T in REAL_ELTYPES
        prob = StubProblem(Coupled(cyclic_perm(JLArray, N_AD)))
        x, p, r = ad_inputs(JLArray, T, N_AD)
        prep = prepare_ad(AutoForwardDiff(; chunksize = CHUNK), prob, r, x, p)
        slots = (prep.xdual, prep.rdual, prep.xdual1, prep.rdual1)
        J = similar(x, N_AD, N_AD)
        jacobian!!(J, prep, prob, x, p)
        jacobian!!(J, prep, prob, x, p)
        @test prep.xdual === slots[1]
        @test prep.rdual === slots[2]
        @test prep.xdual1 === slots[3]
        @test prep.rdual1 === slots[4]
        # stable host allocation: two later calls allocate the same, up to the 16-byte steps in
        # which Windows reports allocations, in either direction
        a2 = jacobian_allocations(J, prep, prob, x, p)
        a3 = jacobian_allocations(J, prep, prob, x, p)
        @test abs(a3 - a2) <= 32

        # and the slots are not merely kept, they are the buffers the call writes: zeroed here, so
        # that a call working on a fresh `similar` copy instead would leave them zero.
        fill!(prep.xdual, zero(eltype(prep.xdual)))
        fill!(prep.rdual, zero(eltype(prep.rdual)))
        jacobian!!(J, prep, prob, x, p)
        rx = similar(r)
        prob.F(rx, x, p)
        @test Array(ForwardDiff.value.(prep.xdual)) == Array(x)
        @test Array(ForwardDiff.value.(prep.rdual)) == Array(rx)
    end
end

@testset "the chunked mode is optimisable and error-free on concrete arguments" begin
    # `test/quality/jet.jl` runs `report_package`, which analyses every method at its declared
    # signature: `jacobian!!(J, prep, prob, x, p)` takes `J` as `Any`, so `J[:, cols]` is analysed
    # over every `AbstractArray` there is, `CartesianIndices` included, and reports follow from
    # that alone. What a solve does is this: concrete arguments. The flag is the one of
    # `test/base/reductions.jl`, for a Julia whose JET loads only stubs.
    JET_WORKS = isdefined(JET, :JET_AVAILABLE) ? JET.JET_AVAILABLE : JET.JET_LOADABLE
    @testset "$T, chunk $N" for T in REAL_ELTYPES, N in (CHUNK, N_AD)

        prob = StubProblem(Scaled())
        x, p, r = ad_inputs(Array, T, N_AD)
        J = similar(x, N_AD, N_AD)
        v = similar(x)
        v .= one(T)
        Jv = similar(x)
        if JET_WORKS
            prep = prepare_ad(AutoForwardDiff(; chunksize = N), prob, r, x, p)
            JET.test_opt(jacobian!!, typeof.((J, prep, prob, x, p)))
            JET.test_call(jacobian!!, typeof.((J, prep, prob, x, p)))
            JET.test_opt(jvp!!, typeof.((Jv, prep, prob, x, v, p)))
            JET.test_call(jvp!!, typeof.((Jv, prep, prob, x, v, p)))
        else
            @test_skip "JET does not load on Julia $(VERSION)"  # issue #8
        end
    end
end
