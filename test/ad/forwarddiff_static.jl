# The R1 Jacobian of the AD seam — checks 3, 4 and 6 of §13.L-a, and the AD half of check 7.
#
# The value of the Jacobian of every problem of the R1 set, the tag the differentiation carries,
# the sources that have no R1 method, and the kernel safety of the preparation. Every comparison
# is exact (`==`) where the analytic Jacobian is the expression the duals evaluate, and the one
# approximate comparison carries a stated tolerance, as `test/ad/jacobian.jl` does for the chunked
# mode.

using ADTypes: AutoFiniteDiff, AutoForwardDiff
using ForwardDiff: ForwardDiff
using JET: JET, @test_opt
using KernelAbstractions: KernelAbstractions, @Const, @index, @kernel
using StaticArrays: MMatrix, MVector, SMatrix, SVector
using Test

using GeometricSolvers
using GeometricSolvers: JacobianFunction, NonlinearProblem, dual_partial, dual_type,
                        jacobian!!,
                        jvp!!, mul!!, prepare_ad, seed, tag_of

include("../helpers/matrix.jl")
include("../helpers/r1problems.jl")

const N_STATIC = 7

"An operator that is not the derivative of `R1Scaled`: the `jacobian` argument of a method."
r1_member_operator(x, p) = 2 .* r1_scaled_jacobian(x, p)

"""
`R1Scaled`'s Jacobian scaled by `scale`, as the operator of a method: an `isbits` closure over an
`isbits` value, so that `JacobianFunction` holds a field and is no singleton of an empty struct.
"""
r1_scaled_operator(scale::R) where {R} = (z, q) -> scale .* r1_scaled_jacobian(z, q)

"the `R1Coupled` problem of the element type `T`, with its analytic Jacobian"
function r1_coupled_problem(::Type{T}) where {T}
    perm = r1_shift(T, N_STATIC)
    return NonlinearProblem(R1Coupled(perm), (z, q) -> r1_coupled_jacobian(z, q, perm))
end

r1_scaled_problem() = NonlinearProblem(R1Scaled(), r1_scaled_jacobian)

# The allocation assertion goes through a function barrier whose arguments all have concrete
# types, and which calls the function once before it measures: at testset scope Julia 1.11 boxes
# the captured values and the assertion fails although the code allocates nothing.
function jacobian_allocations(ad::A, prob::P, x::X, p::Q) where {A, P, X, Q}
    (jacobian!!(nothing, ad, prob, x, p); @allocated jacobian!!(nothing, ad, prob, x, p))
end

# On a Julia it does not support, JET loads empty stubs that throw. JET 0.12 exports
# `JET_AVAILABLE` to say so; JET 0.9 and 0.10 name the same flag `JET_LOADABLE`.
const JET_WORKS = isdefined(JET, :JET_AVAILABLE) ? JET.JET_AVAILABLE : JET.JET_LOADABLE

@testset "the R1 Jacobian is the problem's own, exactly, $T" for T in ELTYPES
    x, p = r1_inputs(T, N_STATIC)
    perm = r1_shift(T, N_STATIC)
    scaled = r1_scaled_problem()
    holo = NonlinearProblem(R1Holomorphic(), r1_holomorphic_jacobian)
    coupled = r1_coupled_problem(T)

    @test jacobian!!(nothing, AutoForwardDiff(), scaled, x, p) == r1_scaled_jacobian(x, p)
    @test jacobian!!(nothing, AutoForwardDiff(), holo, x, p) ==
          r1_holomorphic_jacobian(x, p)

    J = @inferred jacobian!!(nothing, AutoForwardDiff(), coupled, x, p)
    @test J == r1_coupled_jacobian(x, p, perm)
    # the comparison is a test only because the Jacobian is neither symmetric nor diagonal: a
    # transposed or a shifted column would pass on a diagonal one
    @test J != transpose(J)

    # the source `nothing` is the problem's `J`, whatever the iterate, and for a `Number` iterate
    # — `n = 1` — the Jacobian itself is a `Number` and not a `1 × 1` matrix
    @test jacobian!!(nothing, nothing, scaled, x, p) == scaled.J(x, p)
    scalar = NonlinearProblem(R1ScalarScaled(), r1_scalar_scaled_jacobian)
    @test jacobian!!(nothing, AutoForwardDiff(), scalar, T(2), T(3)) == T(3)
    @test jacobian!!(nothing, nothing, scalar, T(2), T(3)) == T(3)

    # an `MVector` counts as a value (§4.3), so the value methods take it like an `SVector`. The
    # general direction goes through the dual pass, whose complex products round in a different
    # order from the dense matrix–vector product, so its comparison carries the tolerance of the
    # R3 tests: eight ulps of the real part.
    mv = MVector(x)
    @test jacobian!!(nothing, AutoForwardDiff(), scaled, mv, p) == r1_scaled_jacobian(x, p)
    @test isapprox(jvp!!(nothing, AutoForwardDiff(), scaled, mv, mv, p),
        r1_scaled_jacobian(x, p) * x; rtol = 8 * eps(real(T)))
end

@testset "the problem's J and the AD Jacobian are different values, $T" for T in ELTYPES
    # `R1Cube`'s analytic Jacobian and the one the duals build round differently, so at these
    # entries they differ in the last bit. That is what makes the two calls below distinguishable,
    # and it is the assertion the `nothing` preparation of a `jacobian!!` or a `jvp!!` is read
    # against: a call that reached ForwardDiff where the problem's own `J` was asked for gives the
    # other value.
    x = r1_cube_input(T, N_STATIC)
    p = SVector{N_STATIC, T}(ones(T, N_STATIC))
    cube = NonlinearProblem(R1Cube(), r1_cube_jacobian)
    Jad = jacobian!!(nothing, AutoForwardDiff(), cube, x, p)
    @test Jad != r1_cube_jacobian(x, p)
    @test jacobian!!(nothing, nothing, cube, x, p) == r1_cube_jacobian(x, p)
    @test jacobian!!(nothing, nothing, cube, x, p) != Jad
end

@testset "an empty iterate is a 0 × 0 Jacobian, $T" for T in ELTYPES
    # `n = 0` is a decided edge of §13.L-a: the Jacobian is a `0 × 0` `SMatrix`, the JVP an empty
    # vector, and neither call throws. `n = 1` is the `Number` iterate, tested above.
    x = SVector{0, T}()
    p = SVector{0, T}()
    prob = r1_scaled_problem()
    @test jacobian!!(nothing, AutoForwardDiff(), prob, x, p) == SMatrix{0, 0, T}()
    @test jacobian!!(nothing, nothing, prob, x, p) == r1_scaled_jacobian(x, p)
    @test jvp!!(nothing, nothing, prob, x, x, p) == SVector{0, T}()
    @test jvp!!(nothing, AutoForwardDiff(), prob, x, x, p) == SVector{0, T}()
end

@testset "the Jacobian agrees with ForwardDiff on a real element type, $T" for T in REAL_ELTYPES
    x, p = r1_inputs(T, N_STATIC)
    coupled = r1_coupled_problem(T)
    @test Array(jacobian!!(nothing, AutoForwardDiff(), coupled, x, p)) ==
          r1_forwarddiff_jacobian(R1Coupled(r1_shift(T, N_STATIC)), x, p)
end

# `jacobian!!`'s own seeding with the tag dropped — the named mutant N1, in process. The fault
# lives in the seeding, so nothing in the package can be asked for an untagged pass; a residual
# that differentiates in its own body is what this pass must throw on.
function r1_untagged_jacobian(F, x::SVector{N, T}, p) where {N, T}
    D = dual_type(T, Nothing, N)
    lanes(j) = ntuple(k -> k == j, Val(N))
    xdual = SVector{N, D}(ntuple(j -> seed(zero(D), x[j], lanes(j)), Val(N))...)
    rdual = F(xdual, p)
    return SMatrix{N, N, T}(
        ntuple(k -> dual_partial(rdual[(k - 1) % N + 1], (k - 1) ÷ N + 1), Val(N * N)))
end

# Check 4: the tag. A residual that differentiates in its own body carries the outer dual into its
# inner derivative, so the two perturbations meet, and only a distinct tag keeps them apart.
@testset "a residual that differentiates in its own body needs the tag, $T" for T in ELTYPES
    x, p = r1_inputs(T, N_STATIC)
    closing = NonlinearProblem(R1Closing(), r1_closing_jacobian)

    # the tagged pass is the holomorphic `J = Diagonal(2x)`, and exactly: the inner variable is a
    # plain real in every element type, so no `DimensionMismatch` intervenes, and the duals add
    # `fl(1 ⋅ x_i) + fl(x_i ⋅ 1)`
    @test jacobian!!(nothing, AutoForwardDiff(), closing, x, p) == r1_closing_jacobian(x, p)

    # and the untagged pass throws: with the outer tag missing the inner perturbation and the
    # outer one are confused, which is the fault the tag rule exists to prevent
    @test_throws ForwardDiff.DualMismatchError r1_untagged_jacobian(R1Closing(), x, p)
end

@testset "the tag is the caller's where it named one, $T" for T in ELTYPES
    x, p = r1_inputs(T, N_STATIC)
    prob = NonlinearProblem(R1Tagged())

    # The rule itself, shared with the R3 chunked mode: the caller's tag where the back end
    # carries one, and otherwise this solver's own, named for the residual and the element type —
    # never `Nothing`.
    caller = AutoForwardDiff(; tag = ForwardDiff.Tag(CallerTag(), T))
    @test typeof(tag_of(caller, prob, x)) === typeof(ForwardDiff.Tag(CallerTag(), T))
    @test typeof(tag_of(AutoForwardDiff(), prob, x)) === typeof(ForwardDiff.Tag(prob.F, T))

    # this solver's own tag, which names the residual: the residual reads `1`
    @test jacobian!!(nothing, AutoForwardDiff(), prob, x, p) == identity_matrix(T, N_STATIC)
    @test jvp!!(nothing, AutoForwardDiff(), prob, x, x, p) == x

    # A caller who differentiates this solve from outside names its own tag, and it must survive
    # the seeding: the residual reads `2`, so the Jacobian is `2I` and the JVP `2v`. A tag built
    # here instead of taken from the back end gives `I` and `v`.
    tagged = AutoForwardDiff(; tag = ForwardDiff.Tag(CallerTag(), T))
    @test jacobian!!(nothing, tagged, prob, x, p) == T(2) * identity_matrix(T, N_STATIC)
    @test jvp!!(nothing, tagged, prob, x, x, p) == T(2) * x
end

@testset "a method's own operator is not the problem's J, $T" for T in ELTYPES
    x, p = r1_inputs(T, N_STATIC)
    prob = r1_scaled_problem()
    A = JacobianFunction(r1_member_operator)

    @test jacobian!!(nothing, A, prob, x, p) == r1_member_operator(x, p)
    # the operator is not the derivative of the residual, so this is a real distinction
    @test r1_member_operator(x, p) != prob.J(x, p)
    @test @inferred(jacobian!!(nothing, A, prob, x, p)) == r1_member_operator(x, p)
    # an `isbits` closure over an `isbits` value is a source like any other — the value it holds
    # is what it scales by, so it is no singleton — and it is what check 7's assertion names
    scaled = JacobianFunction(r1_scaled_operator(T(2)))
    @test @inferred(jacobian!!(nothing, scaled, prob, x, p)) == T(2) .* prob.J(x, p)
    # and no source serves a JVP: the product must come from the problem, and an operator of a
    # step is not the derivative of the residual (§4.3)
    @test_throws MethodError jvp!!(nothing, A, prob, x, x, p)
    @test_throws MethodError jvp!!(nothing, scaled, prob, x, x, p)
end

# The exception a call raises, for the assertions that must see it: `@test_throws MethodError`
# accepts a `MethodError` raised anywhere inside the call, including one raised by a method that
# should not have been reached at all (the named mutant N10).
function r1_raises_missing_method(f::F, g) where {F}
    err = try
        f()
        nothing
    catch e
        e
    end
    return err isa MethodError && err.f === g
end

@testset "a back end other than ForwardDiff has no R1 method, $T" for T in ELTYPES
    x, p = r1_inputs(T, N_STATIC)
    prob = r1_scaled_problem()
    # the failure is the call's own, at the dispatch boundary — a back end reached at all would
    # fail inside itself instead
    @test r1_raises_missing_method(
        () -> jacobian!!(nothing, AutoFiniteDiff(), prob, x, p), jacobian!!)
    @test r1_raises_missing_method(
        () -> jvp!!(nothing, AutoFiniteDiff(), prob, x, x, p), jvp!!)
    # the R1 preparation of another back end does not exist either, and the call that would build
    # it names the missing method rather than failing deep inside a back end
    @test r1_raises_missing_method(
        () -> prepare_ad(AutoFiniteDiff(), prob, nothing, x, p), prepare_ad)
end

@testset "a static matrix is not an iterate, $T" for T in ELTYPES
    _, p = r1_inputs(T, N_STATIC)
    prob = r1_scaled_problem()
    # A static matrix enters the seam as a Jacobian, a factor or an operand of `mul!!`, never as an
    # iterate, so the R1 AD refuses it on the host — in each of its calls, and for a mutable matrix
    # as much as for an immutable one. Each failure is the call's own, so that a method widened to
    # take a matrix and failing inside cannot pass as this refusal.
    for A in (diagonal_matrix(p), MMatrix(diagonal_matrix(p)))
        @test r1_raises_missing_method(
            () -> prepare_ad(AutoForwardDiff(), prob, nothing, A, p), prepare_ad)
        @test r1_raises_missing_method(
            () -> jacobian!!(nothing, AutoForwardDiff(), prob, A, p), jacobian!!)
        @test r1_raises_missing_method(
            () -> jvp!!(nothing, AutoForwardDiff(), prob, A, A, p), jvp!!)
        @test r1_raises_missing_method(
            () -> jacobian!!(nothing, nothing, prob, A, p), jacobian!!)
    end
end

"The other precision of the same kind: `Float32` ↔ `Float64`, `ComplexF32` ↔ `ComplexF64`."
r1_other_precision(::Type{Float32}) = Float64
r1_other_precision(::Type{Float64}) = Float32
r1_other_precision(::Type{ComplexF32}) = ComplexF64
r1_other_precision(::Type{ComplexF64}) = ComplexF32

@testset "the R1 direction has the iterate's element type, $T" for T in ELTYPES
    W = r1_other_precision(T)
    x, p = r1_inputs(T, N_STATIC)
    prob = r1_scaled_problem()
    scalar = NonlinearProblem(R1ScalarScaled(), r1_scalar_scaled_jacobian)
    # The direction has the element type of the iterate, as the line-search slope does, so a
    # direction of the other precision is a host `MethodError` raised by `jvp!!` itself — through
    # both R1 preparations — rather than a value a widened method rounds.
    for prep in (nothing, AutoForwardDiff())
        @test r1_raises_missing_method(() -> jvp!!(nothing, prep, prob, x, W.(x), p), jvp!!)
        @test r1_raises_missing_method(
            () -> jvp!!(nothing, prep, scalar, T(2), W(5), T(3)), jvp!!)
    end
end

@testset "every R1 preparation value is isbits, $T" for T in ELTYPES
    x, p = r1_inputs(T, N_STATIC)
    prob = r1_scaled_problem()
    @test isbits(nothing)
    @test isbits(AutoForwardDiff())
    @test isbits(AutoForwardDiff(; tag = ForwardDiff.Tag(CallerTag(), T)))
    @test isbits(JacobianFunction(r1_member_operator))
    @test isbits(JacobianFunction(r1_scaled_operator(T(2))))

    ad = prepare_ad(AutoForwardDiff(), prob, nothing, x, p)
    @test ad === AutoForwardDiff()
    @test @inferred(prepare_ad(AutoForwardDiff(), prob, nothing, x, p)) ===
          AutoForwardDiff()
    @test isbits(ad)
end

@testset "the R1 Jacobian allocates nothing on an SVector, $T" for T in ELTYPES
    x, p = r1_inputs(T, N_STATIC)
    prob = r1_coupled_problem(T)
    @test jacobian_allocations(AutoForwardDiff(), prob, x, p) == 0
    @test jacobian_allocations(nothing, prob, x, p) == 0
end

# The `Number` iterate — `n = 1` — has a Jacobian that is a `Number` and not a `1 × 1` matrix, so
# it carries its own value, tag, inference and cost assertions: a call that ignored the tag or
# returned a matrix would pass every vector test.
function r1_scalar_jacobian_allocations(ad::A, prob::P, x::X, p::Q) where {A, P, X, Q}
    (jacobian!!(nothing, ad, prob, x, p); @allocated jacobian!!(nothing, ad, prob, x, p))
end

@testset "the Number iterate: value, tag, inference and cost, $T" for T in ELTYPES
    prob = NonlinearProblem(R1ScalarScaled(), r1_scalar_scaled_jacobian)
    x, p = T(2), T(3)
    @test @inferred(jacobian!!(nothing, AutoForwardDiff(), prob, x, p)) == T(3)
    @test @inferred(jacobian!!(nothing, nothing, prob, x, p)) == T(3)
    @test r1_scalar_jacobian_allocations(AutoForwardDiff(), prob, x, p) == 0
    @test r1_scalar_jacobian_allocations(nothing, prob, x, p) == 0

    # the tag rule is the vector one, on a residual that reads it: `R1Tagged` scales by `2` under
    # the caller's tag and by `1` under this solver's own
    tagged = NonlinearProblem(R1Tagged())
    @test jacobian!!(nothing, AutoForwardDiff(), tagged, x, p) == T(1)
    @test jacobian!!(nothing,
        AutoForwardDiff(; tag = ForwardDiff.Tag(CallerTag(), T)), tagged, x, p) == T(2)
end

@testset "JET: no runtime dispatch in the R1 Jacobian, $T" for T in ELTYPES
    x, p = r1_inputs(T, N_STATIC)
    prob = r1_coupled_problem(T)
    scalar = NonlinearProblem(R1ScalarScaled(), r1_scalar_scaled_jacobian)
    if JET_WORKS
        @test_opt jacobian!!(nothing, AutoForwardDiff(), prob, x, p)
        @test_opt jacobian!!(nothing, nothing, prob, x, p)
        @test_opt jacobian!!(nothing, JacobianFunction(r1_member_operator), prob, x, p)
        @test_opt jacobian!!(nothing, AutoForwardDiff(), scalar, T(2), T(3))
        @test_opt jacobian!!(nothing, nothing, scalar, T(2), T(3))
    else
        @test_skip JET_WORKS  # issue #8
    end
end

# Check 7, the AD half: checks 3, 4 and 5 run here inside a kernel, on an `isbits` preparation and
# on values only — the tagged ForwardDiff Jacobian and JVP, the `nothing` source of both, the
# member's `JacobianFunction`, a residual with a nested derivative under a caller's tag, the tag
# rule itself, and `mul!!` on two value results.
const R1KernelResult{T, N, L} = Tuple{
    SMatrix{N, N, T, L}, SVector{N, T}, SMatrix{N, N, T, L}, SVector{N, T},
    SMatrix{N, N, T, L}, SMatrix{N, N, T, L}, SMatrix{N, N, T, L}, SMatrix{N, N, T, L}}

@kernel function r1_ad_kernel!(out, prob, tagged, nested, ad, caller, fun, noth, @Const(X),
        @Const(P))
    i = @index(Global)
    x = X[i]
    p = P[i]
    J = jacobian!!(nothing, ad, prob, x, p)
    Jn = jacobian!!(nothing, noth, prob, x, p)
    out[i] = (
        J, jvp!!(nothing, ad, prob, x, p, p), Jn, jvp!!(nothing, noth, prob, x, p, p),
        jacobian!!(nothing, fun, prob, x, p), jacobian!!(nothing, caller, nested, x, p),
        jacobian!!(nothing, caller, tagged, x, p), mul!!(nothing, J, Jn))
end

@testset "the R1 Jacobian and JVP inside a KA kernel on CPU(), $T" for T in ELTYPES
    backend = KA_BACKEND
    x0, p0 = r1_inputs(T, N_STATIC)
    prob = r1_scaled_problem()
    tagged = NonlinearProblem(R1Tagged())
    nested = NonlinearProblem(R1Closing(), r1_closing_jacobian)
    caller = AutoForwardDiff(; tag = ForwardDiff.Tag(CallerTag(), T))
    X = [x0 * T(i) for i in 1:3]
    P = [p0 for _ in 1:3]
    out = Vector{R1KernelResult{T, N_STATIC, N_STATIC * N_STATIC}}(undef, 3)
    r1_ad_kernel!(backend)(out, prob, tagged, nested, AutoForwardDiff(), caller,
        JacobianFunction(r1_member_operator), nothing, X, P; ndrange = 3)
    KernelAbstractions.synchronize(backend)

    for i in 1:3
        J = r1_scaled_jacobian(X[i], P[i])
        J1, Jv, J2, Jvn, Jf, Jg, Jt, C = out[i]
        @test J1 == J
        @test J2 == J
        @test Jf == r1_member_operator(X[i], P[i])
        @test Jg == r1_closing_jacobian(X[i], P[i])
        @test Jt == T(2) * identity_matrix(T, N_STATIC)
        @test C == J * J
        # the Jacobians are exact in every element type; the product of a general direction is
        # not, in a complex one, so it carries the eight-ulp tolerance named above
        @test isapprox(Jv, J * P[i]; rtol = 8 * eps(real(T)))
        @test isapprox(Jvn, J * P[i]; rtol = 8 * eps(real(T)))
    end
end

# ... and the `Number` iterate, `n = 1`, in a kernel: the value methods of both AD functions, the
# tagged one and the problem's own `J`, all of them `isbits`.
@kernel function r1_number_kernel!(out, prob, tagged, ad, caller, @Const(X), @Const(P))
    i = @index(Global)
    x = X[i]
    p = P[i]
    out[i] = (jacobian!!(nothing, ad, prob, x, p), jvp!!(nothing, ad, prob, x, x, p),
        jvp!!(nothing, nothing, prob, x, x, p), jacobian!!(nothing, caller, tagged, x, p))
end

@testset "the Number iterate of the AD seam inside a KA kernel on CPU(), $T" for T in ELTYPES
    backend = KA_BACKEND
    prob = NonlinearProblem(R1ScalarScaled(), r1_scalar_scaled_jacobian)
    tagged = NonlinearProblem(R1Tagged())
    caller = AutoForwardDiff(; tag = ForwardDiff.Tag(CallerTag(), T))
    X = [T(i) for i in 1:3]
    P = [T(3) for _ in 1:3]
    out = Vector{NTuple{4, T}}(undef, 3)
    r1_number_kernel!(backend)(
        out, prob, tagged, AutoForwardDiff(), caller, X, P; ndrange = 3)
    KernelAbstractions.synchronize(backend)
    @test out == [(T(3), T(3) * X[i], T(3) * X[i], T(2)) for i in 1:3]
end
