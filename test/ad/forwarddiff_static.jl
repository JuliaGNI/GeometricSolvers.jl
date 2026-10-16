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
using StaticArrays: MVector, SMatrix, SVector
using Test

using GeometricSolvers
using GeometricSolvers: JacobianFunction, NonlinearProblem, jacobian!!, jvp!!, prepare_ad,
                        tag_of

include("../helpers/matrix.jl")
include("../helpers/r1problems.jl")

const N_STATIC = 7

"An operator that is not the derivative of `R1Scaled`: the `jacobian` argument of a method."
r1_member_operator(x, p) = 2 .* r1_scaled_jacobian(x, p)

"an `isbits` closure — it captures nothing, so its struct has no field"
const r1_operator_closure = (z, q) -> r1_scaled_jacobian(z, q)

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

    # a residual that differentiates in its own body: the inner perturbation and the outer one
    # must carry different tags, or the result is wrong rather than an error
    nested = NonlinearProblem(R1Nested(), r1_nested_jacobian)
    Jn = jacobian!!(nothing, AutoForwardDiff(), nested, x, p)
    @test Array(Jn) == r1_forwarddiff_jacobian(R1Nested(), x, p)
    # and the reference is itself right: `r_i = 2 p_i x_i²`, so `J = Diagonal(4 p_i x_i)`. The
    # two expressions round differently, so this one carries a tolerance, named here: four ulps.
    @test isapprox(Array(Jn), r1_nested_jacobian(x, p); rtol = 4 * eps(T))
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
    # and it serves no JVP: the product must come from the problem, and an operator of a step is
    # not the derivative of the residual (§4.3)
    @test_throws MethodError jvp!!(nothing, A, prob, x, x, p)
end

@testset "a back end other than ForwardDiff has no R1 method, $T" for T in ELTYPES
    x, p = r1_inputs(T, N_STATIC)
    prob = r1_scaled_problem()
    @test_throws MethodError jacobian!!(nothing, AutoFiniteDiff(), prob, x, p)
    @test_throws MethodError jvp!!(nothing, AutoFiniteDiff(), prob, x, x, p)
    # the R1 preparation of another back end does not exist either, and the call that would build
    # it names the missing method rather than failing deep inside a back end
    @test_throws MethodError prepare_ad(AutoFiniteDiff(), prob, nothing, x, p)
end

@testset "every R1 preparation value is isbits, $T" for T in ELTYPES
    x, p = r1_inputs(T, N_STATIC)
    prob = r1_scaled_problem()
    @test isbits(nothing)
    @test isbits(AutoForwardDiff())
    @test isbits(AutoForwardDiff(; tag = ForwardDiff.Tag(CallerTag(), T)))
    @test isbits(JacobianFunction(r1_member_operator))
    @test isbits(JacobianFunction(r1_operator_closure))

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

@testset "JET: no runtime dispatch in the R1 Jacobian, $T" for T in ELTYPES
    x, p = r1_inputs(T, N_STATIC)
    prob = r1_coupled_problem(T)
    if JET_WORKS
        @test_opt jacobian!!(nothing, AutoForwardDiff(), prob, x, p)
        @test_opt jacobian!!(nothing, nothing, prob, x, p)
        @test_opt jacobian!!(nothing, JacobianFunction(r1_member_operator), prob, x, p)
    else
        @test_skip JET_WORKS  # issue #8
    end
end

# Check 7, the AD half: the same calls inside a kernel, on an `isbits` preparation.
@kernel function r1_ad_kernel!(Js, Jvs, prob, ad, @Const(X), @Const(P))
    i = @index(Global)
    Js[i] = jacobian!!(nothing, ad, prob, X[i], P[i])
    Jvs[i] = jvp!!(nothing, ad, prob, X[i], P[i], P[i])
end

@testset "the R1 Jacobian and JVP inside a KA kernel on CPU(), $T" for T in ELTYPES
    backend = KA_BACKEND
    prob = r1_scaled_problem()
    X = [r1_inputs(T, N_STATIC)[1] * T(i) for i in 1:3]
    P = [r1_inputs(T, N_STATIC)[2] for i in 1:3]
    Js = Vector{SMatrix{N_STATIC, N_STATIC, T, N_STATIC * N_STATIC}}(undef, 3)
    Jvs = Vector{SVector{N_STATIC, T}}(undef, 3)
    r1_ad_kernel!(backend)(Js, Jvs, prob, AutoForwardDiff(), X, P; ndrange = 3)
    KernelAbstractions.synchronize(backend)
    @test Js == [r1_scaled_jacobian(X[i], P[i]) for i in 1:3]
    # the Jacobian is exact in every element type; the product of a general direction is not, in a
    # complex one, so it carries the eight-ulp tolerance named above
    @test all(isapprox(Jvs[i], r1_scaled_jacobian(X[i], P[i]) * P[i];
                  rtol = 8 * eps(real(T))) for i in 1:3)
end
