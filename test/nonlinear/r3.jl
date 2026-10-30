# R3 nonlinear solvers (`NonlinearSolver`, `init`, `solve!`, `solve`), the array methods of the
# storage seam (`src/base/bang.jl`), the analytic and member-operator AD/JVP seam (`src/ad/`),
# `StrongWolfe` trial-residual refresh, R1/R3 iterate and status parity, and zero steady-state
# heap allocations on `Array`.

using Adapt: Adapt
using ADTypes: AutoFiniteDiff, AutoForwardDiff
using FiniteDiff: FiniteDiff
using ForwardDiff: ForwardDiff
using GPUArraysCore: allowscalar
using JLArrays: JLArray
using LinearAlgebra: BLAS, Diagonal, I, diagind, mul!
using StaticArrays: MVector, SMatrix, SVector
using Test

using GeometricSolvers
using GeometricSolvers: Backtracking, Bisection, ChunkedForwardDiff, DIJacobian, DenseLU,
                        DenseQR, DogLeg, ExactStep, JacobianFunction, JacobianMatrixJVP,
                        LINESEARCH_FAILED, LUFactorization, LineFunction, MAXITERS,
                        MeasuredSlope, NONFINITE, Newton, NonlinearProblem, NonlinearSolver,
                        Options, Picard, QRFactorization, SINGULAR, STALLED, SUCCESS,
                        SolverStatus, Static, StrongWolfe, ToReal, axpby!!, buffer, copy!!,
                        globalize!!, init, jacobian!!, jvp!!, linesearch, map!!, mul!!,
                        norm2, prepare_ad, rdot, regularize!!, residual!!, rnorm, roundoff,
                        scale!!, solve, solve!, solve!!, step!!, update_iterate!!, φ, φ′

include("../helpers/matrix.jl")
include("../helpers/adproblems.jl")
include("../helpers/r1problems.jl")

const N_R3 = 4

allowscalar(false)

# Coupled in-place residual `r_i = x_i^2 + p_i x_i + 2 x_{perm_i} - q_i` and its analytic
# Jacobian, plus a void-returning wrapper to verify that `residual!!` and `jacobian!!` return the
# destination buffer even when the user callback returns `nothing`.
struct R3Coupled{P}
    perm::P
end

function (f::R3Coupled)(r, x, p)
    r .= x .* x .+ p.a .* x .+ 2 .* x[f.perm] .- p.b
    return r
end

struct R3CoupledVoid{P}
    perm::P
end

function (f::R3CoupledVoid)(r, x, p)
    r .= x .* x .+ p.a .* x .+ 2 .* x[f.perm] .- p.b
    return nothing
end

function r3_coupled_jacobian!(J, x, p, perm)
    n = length(x)
    T = eltype(x)
    Jh = zeros(T, n, n)
    xh = Array(x)
    ah = Array(p.a)
    permh = Array(perm)
    for i in 1:n
        d = xh[i] + xh[i] + ah[i]
        Jh[i, i] = d
        Jh[i, permh[i]] += T(2)
    end
    copyto!(J, Jh)
    return nothing
end

struct R3CoupledJ{P}
    perm::P
end

function (j::R3CoupledJ)(J, x, p)
    r3_coupled_jacobian!(J, x, p, j.perm)
    return J
end

struct R1CoupledR3{P}
    perm::P
end

function (f::R1CoupledR3)(y, q)
    return y .* y .+ q.a .* y .+ 2 .* y[f.perm] .- q.b
end

struct R1CoupledR3J{P}
    perm::P
end

function (j::R1CoupledR3J)(y, q)
    return r1_coupled_jacobian(y, q.a, j.perm)
end

struct R3PicardF{R}
    α::R
end

function (f::R3PicardF)(r, y, q)
    r .= f.α .* (y .- q)
    return r
end

struct R3PicardJ{T}
    α::T
end

function (j::R3PicardJ)(J, y, q)
    fill!(J, zero(eltype(J)))
    regularize!!(J, j.α)
    return J
end

struct R3SqF end

function (::R3SqF)(r, y, q)
    r .= y .* y .- q
    return r
end

struct R3SqJ end

function (::R3SqJ)(J, y, q)
    fill!(J, zero(eltype(J)))
    if J isa Matrix
        @inbounds for i in axes(J, 1)
            J[i, i] = 2 * y[i]
        end
    else
        view(J, diagind(J)) .= 2 .* y
    end
    return J
end

struct R3LinearF end

function (::R3LinearF)(r, y, q)
    mul!(r, q.A, y)
    r .-= q.b
    return r
end

struct R3LinearJ end

function (::R3LinearJ)(J, y, q)
    copyto!(J, q.A)
    return J
end

struct R1LinearF end

(::R1LinearF)(y, q) = q.A * y - q.b

struct R1LinearJ end

(::R1LinearJ)(y, q) = q.A

struct R3VoidScaled end

(::R3VoidScaled)(r, z, q) = (r .= q .* z; nothing)

struct CountingCoupledJ{P}
    calls::Base.RefValue{Int}
    perm::P
end

function (j::CountingCoupledJ)(J, z, q)
    j.calls[] += 1
    r3_coupled_jacobian!(J, z, q, j.perm)
    return nothing
end

struct CountingDiagJ
    calls::Base.RefValue{Int}
end

function (j::CountingDiagJ)(J, z, q)
    j.calls[] += 1
    T = eltype(J)
    fill!(J, zero(T))
    regularize!!(J, T(3))
    return nothing
end

struct R3ZoomF end

function (::R3ZoomF)(r, z, q)
    T = eltype(z)
    R = real(T)
    zh = Array(z)
    if all(==(T(2)), zh)
        copyto!(r, z .- q)
    elseif all(==(one(T)), zh)
        copyto!(r, T(R(0.5)) .* q)
    else
        copyto!(r, T(4) .* (z .- q))
    end
    return r
end

struct R3ZoomJ end

function (::R3ZoomJ)(J, z, q)
    T = eltype(J)
    n = size(J, 1)
    fill!(J, zero(T))
    zh = Array(z)
    diag_val = all(==(T(2)), zh) ? one(T) : -T(4)
    Jh = Matrix(diag_val * I(n))
    copyto!(J, Jh)
    return J
end

struct R1PicardF{R}
    α::R
end

(f::R1PicardF)(y, q) = f.α .* (y .- q)

struct R1PicardJ{T}
    α::T
end

(j::R1PicardJ{T})(y, q) where {T} = diagonal_matrix(SVector{2, T}(j.α, j.α))

struct CountingSqJ
    calls::Base.RefValue{Int}
end

function (j::CountingSqJ)(J, y, q)
    j.calls[] += 1
    fill!(J, zero(eltype(J)))
    for i in axes(J, 1)
        @inbounds J[i, i] = 2 * y[i]
    end
    return J
end

struct CountingSqF
    real_calls::Base.RefValue{Int}
    dual1_calls::Base.RefValue{Int}
    dual2_calls::Base.RefValue{Int}
end

function (f::CountingSqF)(r, y, q)
    E = real(eltype(y))
    if E <: ForwardDiff.Dual
        if ForwardDiff.npartials(E) == 1
            f.dual1_calls[] += 1
        else
            f.dual2_calls[] += 1
        end
    else
        f.real_calls[] += 1
    end
    r .= y .* y .- q
    return r
end

function r3_coupled_inputs(AT, ::Type{T}, n::Int) where {T}
    x₀_s, a_s = r1_inputs(T, n)
    perm_s = r1_shift(T, n)
    x_star_s = x₀_s .+ SVector{n, T}(ntuple(i -> T(i) / T(2n), Val(n)))
    b_s = x_star_s .* x_star_s .+ a_s .* x_star_s .+ 2 .* x_star_s[perm_s]
    x₀ = AT(collect(x₀_s))
    a = AT(collect(a_s))
    b = AT(collect(b_s))
    perm = AT(collect(perm_s))
    return (; x₀, p = (; a, b), perm, x₀_s, p_s = (; a = a_s, b = b_s), perm_s, x_star_s)
end

# Function barriers for allocation assertions on `Array`.
function bang_residual_alloc(y::Y, prob::P, x::X, p::Q) where {Y, P, X, Q}
    residual!!(y, prob, x, p)
    return @allocated residual!!(y, prob, x, p)
end

function bang_update_alloc(y::Y, x::X, α::A, d::D) where {Y, X, A, D}
    update_iterate!!(y, x, α, d)
    return @allocated update_iterate!!(y, x, α, d)
end

function bang_map1_alloc(f::F, y::Y, x::X) where {F, Y, X}
    map!!(f, y, x)
    return @allocated map!!(f, y, x)
end

function bang_map2_alloc(f::F, y::Y, x::X, z::Z) where {F, Y, X, Z}
    map!!(f, y, x, z)
    return @allocated map!!(f, y, x, z)
end

function bang_mul_alloc(C::Y, A::M, B::X) where {Y, M, X}
    mul!!(C, A, B)
    return @allocated mul!!(C, A, B)
end

function bang_copy_alloc(dst::Y, src::X) where {Y, X}
    copy!!(dst, src)
    return @allocated copy!!(dst, src)
end

function bang_scale_alloc(dst::Y, α::A, x::X) where {Y, A, X}
    scale!!(dst, α, x)
    return @allocated scale!!(dst, α, x)
end

function bang_axpby_alloc(dst::Y, α::A, x::X, β::B, y::Z) where {Y, A, X, B, Z}
    axpby!!(dst, α, x, β, y)
    return @allocated axpby!!(dst, α, x, β, y)
end

function bang_reg_alloc(J::M, μ::R) where {M, R}
    regularize!!(J, μ)
    return @allocated regularize!!(J, μ)
end

function seam_jac_alloc(J::M, prep::S, prob::P, x::X, p::Q) where {M, S, P, X, Q}
    jacobian!!(J, prep, prob, x, p)
    return @allocated jacobian!!(J, prep, prob, x, p)
end

function seam_jvp_alloc(Jv::Y, prep::S, prob::P, x::X, v::V, p::Q) where {Y, S, P, X, V, Q}
    jvp!!(Jv, prep, prob, x, v, p)
    return @allocated jvp!!(Jv, prep, prob, x, v, p)
end

function solve!_allocations(x::X, x₀::X, solver::S, p::Q) where {X, S, Q}
    copyto!(x, x₀)
    solve!(x, solver, p)
    copyto!(x, x₀)
    return @allocated solve!(x, solver, p)
end

@testset "scalar indexing on a device array is disallowed" begin
    @test_throws ErrorException JLArray(zeros(Float32, 2))[1]
end

@testset "Scenario 1: array methods of the storage seam, $AT, $T" for AT in ARRAY_BACKENDS,
    T in ELTYPES

    R = real(T)
    x, p, y = ad_inputs(AT, T, N_R3)
    d = AT(T[(i + 3) / (N_R3 + 4) for i in 1:N_R3])
    α = T(2)
    β = T(3)

    # `buffer` allocates a fresh mutable array on the same backend
    b1 = @inferred buffer(x)
    @test b1 isa AT{T, 1}
    @test size(b1) == (N_R3,)
    @test b1 !== x
    b2 = @inferred buffer(x, Float32, N_R3, N_R3)
    @test b2 isa AT{Float32, 2}
    @test size(b2) == (N_R3, N_R3)

    # `residual!!` returns `y` itself even when `prob.F` returns `nothing`, and calling
    # `residual!!(nothing, ...)` on a 3-argument-only `F!(y, x, p)` throws `MethodError`
    prob_ret = NonlinearProblem(Scaled())
    prob_void = NonlinearProblem(R3VoidScaled())
    @test @inferred(residual!!(y, prob_ret, x, p)) === y
    @test Array(y) == Array(p) .* Array(x)
    fill!(y, zero(T))
    @test @inferred(residual!!(y, prob_void, x, p)) === y
    @test Array(y) == Array(p) .* Array(x)
    @test_throws MethodError residual!!(nothing, prob_void, x, p)

    # `update_iterate!!` updates `y` in place and returns `y`
    @test @inferred(update_iterate!!(y, x, α, d)) === y
    @test Array(y) == Array(x) .+ α .* Array(d)

    # `map!!` updates `y` in place and returns `y`
    @test @inferred(map!!(-, y, x)) === y
    @test Array(y) == -Array(x)
    @test @inferred(map!!(+, y, x, d)) === y
    @test Array(y) == Array(x) .+ Array(d)

    # `mul!!` computes matrix-vector and matrix-matrix products in place and returns `C`
    A_host = reshape(T[(i + 2j) / (N_R3 + 3) for i in 1:N_R3 for j in 1:N_R3], N_R3, N_R3)
    B_host = reshape(T[(2i + j) / (N_R3 + 5) for i in 1:N_R3 for j in 1:N_R3], N_R3, N_R3)
    A = AT(A_host)
    B = AT(B_host)
    C = similar(A)
    @test @inferred(mul!!(y, A, x)) === y
    @test isapprox(Array(y), A_host * Array(x); rtol = 8 * eps(R))
    @test @inferred(mul!!(C, A, B)) === C
    @test isapprox(Array(C), A_host * B_host; rtol = 8 * eps(R))

    # `copy!!`, `scale!!`, `axpby!!`: value (`nothing`, `StaticArray`, `Number`) and array methods
    x_s, p_s = r1_inputs(T, N_R3)
    @test @inferred(copy!!(nothing, x_s)) === x_s
    @test @inferred(copy!!(x_s, p_s)) === p_s
    @test @inferred(copy!!(one(T), T(4))) === T(4)
    fill!(y, zero(T))
    @test @inferred(copy!!(y, x)) === y
    @test Array(y) == Array(x)

    @test @inferred(scale!!(nothing, α, x_s)) === α * x_s
    @test @inferred(scale!!(nothing, α, T(4))) === α * T(4)
    @test @inferred(scale!!(y, α, x)) === y
    @test Array(y) == α .* Array(x)

    @test @inferred(axpby!!(nothing, α, x_s, β, p_s)) === α * x_s + β * p_s
    @test @inferred(axpby!!(nothing, α, T(2), β, T(4))) === α * T(2) + β * T(4)
    @test @inferred(axpby!!(y, α, x, β, d)) === y
    @test Array(y) == α .* Array(x) .+ β .* Array(d)

    # `regularize!!` on `SMatrix`, `Number`, and `AT` (`Array` and `JLArray`):
    # `μ == 0` leaves every entry untouched; `μ != 0` shifts only the diagonal by `T(μ)` (MU1)
    s_mat = SMatrix{N_R3, N_R3, T}(A_host)
    @test @inferred(regularize!!(s_mat, zero(R))) === s_mat
    s_reg = @inferred regularize!!(s_mat, R(0.5))
    @test Array(s_reg) == A_host + T(R(0.5)) * I
    @test @inferred(regularize!!(T(3), zero(R))) === T(3)
    @test @inferred(regularize!!(T(3), R(0.5))) === T(3) + T(R(0.5))

    J = AT(copy(A_host))
    @test @inferred(regularize!!(J, zero(R))) === J
    @test Array(J) == A_host
    @test @inferred(regularize!!(J, R(0.5))) === J
    Jh = Array(J)
    for j in 1:N_R3, i in 1:N_R3

        if i == j
            @test Jh[i, j] == A_host[i, j] + T(R(0.5))
        else
            @test Jh[i, j] == A_host[i, j]
        end
    end

    # Zero heap allocations on `Array`
    if AT === Array
        @test bang_residual_alloc(y, prob_ret, x, p) == 0
        @test bang_update_alloc(y, x, α, d) == 0
        @test bang_map1_alloc(-, y, x) == 0
        @test bang_map2_alloc(+, y, x, d) == 0
        @test bang_mul_alloc(y, A, x) == 0
        @test bang_mul_alloc(C, A, B) == 0
        @test bang_copy_alloc(y, x) == 0
        @test bang_scale_alloc(y, α, x) == 0
        @test bang_axpby_alloc(y, α, x, β, d) == 0
        @test bang_reg_alloc(J, R(0.5)) == 0
    end

    # Hostile inputs: non-finite values propagate without throwing
    for a in r1_hostile_values(T)
        xh = AT(fill(a, N_R3))
        yh = similar(xh)
        @test residual!!(yh, prob_ret, xh, p) === yh
        @test isequal(Array(yh), Array(p) .* fill(a, N_R3))
        @test update_iterate!!(yh, xh, α, d) === yh
        @test isequal(Array(yh), fill(a, N_R3) .+ α .* Array(d))
        @test copy!!(yh, xh) === yh
        @test isequal(Array(yh), fill(a, N_R3))
        @test scale!!(yh, α, xh) === yh
        @test isequal(Array(yh), α .* fill(a, N_R3))
        @test axpby!!(yh, α, xh, β, d) === yh
        @test isequal(Array(yh), α .* fill(a, N_R3) .+ β .* Array(d))
        Jh_in = fill(a, N_R3, N_R3)
        J_bad = AT(copy(Jh_in))
        @test regularize!!(J_bad, R(0.5)) === J_bad
        @test isequal(Array(J_bad)[1, 1], a + T(R(0.5)))
        @test isequal(Array(J_bad)[1, 2], a)
    end
end

@testset "Scenario 2: R3 AD and JVP seam for analytic and member operators, $AT, $T" for AT in ARRAY_BACKENDS,
    T in ELTYPES

    R = real(T)
    inp = r3_coupled_inputs(AT, T, N_R3)
    x, p, perm = inp.x₀, inp.p, inp.perm

    j_calls = Ref(0)
    op_calls = Ref(0)
    J_analytic! = CountingCoupledJ(j_calls, perm)
    A_member! = CountingDiagJ(op_calls)

    prob_j = NonlinearProblem(R3Coupled(perm), J_analytic!)
    prob_ad = NonlinearProblem(R3Coupled(perm))

    # `jacobian!!` with `::Nothing` calls `prob.J(J, x, p)` and returns `J` (MU2)
    J = buffer(x, T, N_R3, N_R3)
    fill!(J, zero(T))
    j_calls[] = 0
    @test @inferred(jacobian!!(J, nothing, prob_j, x, p)) === J
    @test j_calls[] == 1
    @test Array(J) == Array(r1_coupled_jacobian(inp.x₀_s, inp.p_s.a, inp.perm_s))

    # `jacobian!!` with `JacobianFunction` calls `s.f(J, x, p)`, returns `J`, and never calls `prob.J`
    jf = JacobianFunction(A_member!)
    j_calls[] = 0
    op_calls[] = 0
    @test @inferred(jacobian!!(J, jf, prob_j, x, p)) === J
    @test op_calls[] == 1
    @test j_calls[] == 0
    @test Array(J) == Matrix(T(3) * I(N_R3))

    # `jvp!!` with `JacobianMatrixJVP` evaluates `prob.J(prep.J, x, p)` and multiplies into `Jv` (MU3)
    jvp_prep = JacobianMatrixJVP(buffer(x, T, N_R3, N_R3))
    v = AT(T[(i + 1) / (N_R3 + 2) for i in 1:N_R3])
    Jv = buffer(x)
    j_calls[] = 0
    @test @inferred(jvp!!(Jv, jvp_prep, prob_j, x, v, p)) === Jv
    @test j_calls[] == 1
    J_exact = Matrix(r1_coupled_jacobian(inp.x₀_s, inp.p_s.a, inp.perm_s))
    @test isapprox(Array(Jv), J_exact * Array(v); rtol = 8 * eps(R))

    if AT === Array
        # Allocation-free analytic J! for allocation checks on `Array`
        prob_scaled_j = NonlinearProblem(Scaled(), R3PicardJ(T(R(0.5))))
        xs, ps, _ = ad_inputs(Array, T, N_R3)
        Js = zeros(T, N_R3, N_R3)
        Jvs = zeros(T, N_R3)
        vs = ones(T, N_R3)
        jfs = JacobianFunction(prob_scaled_j.J)
        jvps = JacobianMatrixJVP(zeros(T, N_R3, N_R3))
        @test seam_jac_alloc(Js, nothing, prob_scaled_j, xs, ps) == 0
        @test seam_jac_alloc(Js, jfs, prob_scaled_j, xs, ps) == 0
        @test seam_jvp_alloc(Jvs, jvps, prob_scaled_j, xs, vs, ps) == 0
    end

    # `init(::Newton, ...)` shares one `ChunkedForwardDiff` when both `jacprep` and `jvpprep` use AD,
    # and builds `JacobianMatrixJVP` when `prob.J !== nothing` under a line search, and no JVP
    # preparation under any other globalization
    if AT === Array
        st_ad = init(Newton(), prob_ad, x, p)
        @test st_ad.jacprep isa ChunkedForwardDiff
        @test st_ad.jacprep === st_ad.jvpprep

        st_ad_explicit = init(Newton(; jacobian = AutoForwardDiff()), prob_ad, x, p)
        @test st_ad_explicit.jacprep isa ChunkedForwardDiff
        @test st_ad_explicit.jacprep === st_ad_explicit.jvpprep

        st_j = init(Newton(; globalization = Backtracking()), prob_j, x, p)
        @test st_j.jacprep === nothing
        @test st_j.jvpprep isa JacobianMatrixJVP
        @test size(st_j.jvpprep.J) == (N_R3, N_R3)
        @test init(Newton(), prob_j, x, p).jvpprep === nothing
        @test init(Newton(; globalization = DogLeg()), prob_j, x, p).jvpprep === nothing

        st_op_j = init(
            Newton(; jacobian = A_member!, globalization = Backtracking()), prob_j, x, p)
        @test st_op_j.jacprep === JacobianFunction(A_member!)
        @test st_op_j.jvpprep isa JacobianMatrixJVP
        @test init(Newton(; jacobian = A_member!), prob_j, x, p).jvpprep === nothing

        st_op_ad = init(Newton(; jacobian = A_member!), prob_ad, x, p)
        @test st_op_ad.jacprep === JacobianFunction(A_member!)
        @test st_op_ad.jvpprep isa ChunkedForwardDiff
    end

    st_pic_j = init(Picard(; globalization = Bisection()), prob_j, x, p)
    @test st_pic_j.jvpprep isa JacobianMatrixJVP
    @test size(st_pic_j.jvpprep.J) == (N_R3, N_R3)
    @test init(Picard(), prob_j, x, p).jvpprep === nothing
    st_pic_ad = init(Picard(), prob_ad, x, p)
    @test st_pic_ad.jvpprep isa ChunkedForwardDiff
end

@testset "Scenario 3a: StrongWolfe refreshes trial residual when returning earlier αlo (MU8), $AT, $T" for AT in ARRAY_BACKENDS,
    T in ELTYPES

    R = real(T)
    x₀ = AT(fill(T(2), N_R3))
    d = AT(fill(-one(T), N_R3))
    p = AT(fill(one(T), N_R3))
    prob_zw = NonlinearProblem(R3ZoomF(), R3ZoomJ())
    Fx = buffer(x₀)
    residual!!(Fx, prob_zw, x₀, p)
    xt = buffer(x₀)
    Ft = buffer(x₀)
    Jv = buffer(x₀)
    jvpprep = JacobianMatrixJVP(buffer(x₀, T, N_R3, N_R3))
    lf = LineFunction(prob_zw, x₀, Fx, d, p, jvpprep, xt, Ft, Jv)
    φ₀ = norm2(Fx)
    sw = Adapt.adapt(ToReal{R}(), StrongWolfe(; c₁ = 1e-4, c₂ = 0.1, maxiter = 1))
    res = linesearch(sw, lf, ExactStep(), φ₀, one(R))
    @test res.α == one(R)
    @test res.residual === Ft
    expected_F1 = fill(T(R(0.5)), N_R3)
    @test Array(res.residual) == expected_F1

    # Also check `globalize!!` with `StrongWolfe` on the same problem: `Fx` after `globalize!!`
    # must equal `F(x + res.α * d, p)`, not the rejected zoom trial `F(x + 0.5 * d, p)`!
    x_work = copy(x₀)
    Fx_work = copy(Fx)
    g_sw = (; xt = buffer(x₀), Ft = buffer(x₀), Jv = buffer(x₀))
    x_out, Fx_out, _, info_sw = globalize!!(
        g_sw, sw, prob_zw, x_work, Fx_work, d, p, jvpprep, ExactStep())
    @test x_out === x_work
    @test Fx_out === Fx_work
    @test Array(x_work) == fill(one(T), N_R3)
    @test Array(Fx_work) == expected_F1
    @test info_sw.fnorm == rnorm(Fx_work)
end

@testset "Scenario 3b: Picard R1/R3 parity on Array and JLArray, $T" for T in ELTYPES
    R = real(T)
    x₀_s = SVector{2, T}(T(3), T(5))
    p_s = SVector{2, T}(one(T), T(2))

    prob_r1 = NonlinearProblem(R1PicardF(R(0.5)), R1PicardJ(T(R(0.5))))
    prob_r1_ad = NonlinearProblem(prob_r1.F)
    prob_r3_ad = NonlinearProblem(R3PicardF(R(0.5)))
    prob_r3_j = NonlinearProblem(R3PicardF(R(0.5)), R3PicardJ(T(R(0.5))))

    opts1 = Options(
        T; max_iterations = 1, f_abstol = zero(R), f_reltol = zero(R),
        x_abstol = zero(R), x_reltol = zero(R))
    opts_full = Options(T)

    # 1-step and full-solve parity on Array and JLArray (catches MU6 / Picard step sign and JVP)
    m_sw = Picard(; damping = 0.8, globalization = StrongWolfe())
    x1_r1, st1_r1 = solve(prob_r1_ad, x₀_s, m_sw, p_s; options = opts1)
    xf_r1, stf_r1 = solve(prob_r1_ad, x₀_s, m_sw, p_s; options = opts_full)
    for AT in ARRAY_BACKENDS
        x₀ = AT(collect(x₀_s))
        p = AT(collect(p_s))
        x1_r3, st1_r3 = solve(prob_r3_ad, x₀, m_sw, p; options = opts1)
        @test x1_r3 isa AT{T, 1}
        @test st1_r3.code == st1_r1.code
        @test st1_r3.iterations == st1_r1.iterations == 1
        @test isapprox(Array(x1_r3), Array(x1_r1); rtol = 8 * eps(R))
        @test isapprox(st1_r3.fnorm, st1_r1.fnorm; rtol = 8 * eps(R))
        @test isapprox(st1_r3.stepnorm, st1_r1.stepnorm; rtol = 8 * eps(R))

        x_in = copy(x₀)
        solver_r3 = init(prob_r3_ad, x_in, m_sw, p; options = opts_full)
        st_in = solve!(x_in, solver_r3, p)
        @test st_in.code == stf_r1.code == SUCCESS
        @test st_in.iterations == stf_r1.iterations
        @test isapprox(Array(x_in), Array(xf_r1); rtol = 8 * eps(R))
        @test isapprox(st_in.fnorm, stf_r1.fnorm; rtol = 8 * eps(R))
    end

    # Full solve parity across all globalizations on Array
    globs_3b = (
        (Static(), prob_r3_ad, prob_r1_ad),
        (Backtracking(), prob_r3_ad, prob_r1_ad),
        (Bisection(), prob_r3_ad, prob_r1_ad),
        (StrongWolfe(), prob_r3_ad, prob_r1_ad),
        (StrongWolfe(), prob_r3_j, prob_r1)
    )
    x₀_arr = collect(x₀_s)
    p_arr = collect(p_s)
    for (glob, prob_r3, prob_ref) in globs_3b
        m = Picard(; damping = 0.8, globalization = glob)
        x_r1, st_r1 = solve(prob_ref, x₀_s, m, p_s; options = opts_full)
        x_r3, st_r3 = solve(prob_r3, x₀_arr, m, p_arr; options = opts_full)
        @test st_r3.code == st_r1.code == SUCCESS
        @test st_r3.iterations == st_r1.iterations
        @test isapprox(x_r3, Array(x_r1); rtol = 8 * eps(R))
        @test isapprox(st_r3.fnorm, st_r1.fnorm; rtol = 8 * eps(R))
    end
end

@testset "Scenario 3c: Newton R1/R3 parity across globalizations, reuse, regularization, and DogLeg, $T" for T in ELTYPES
    R = real(T)
    inp = r3_coupled_inputs(Array, T, N_R3)
    x₀, p, perm = inp.x₀, inp.p, inp.perm
    x₀_s, p_s, perm_s = inp.x₀_s, inp.p_s, inp.perm_s

    prob_r1_ad = NonlinearProblem(R1CoupledR3(perm_s))
    prob_r1_j = NonlinearProblem(R1CoupledR3(perm_s), R1CoupledR3J(perm_s))

    prob_r3_ad = NonlinearProblem(R3Coupled(perm))
    prob_r3_j = NonlinearProblem(R3Coupled(perm), R3CoupledJ(perm))

    opts1 = Options(
        T; max_iterations = 1, f_abstol = zero(R), f_reltol = zero(R),
        x_abstol = zero(R), x_reltol = zero(R))
    opts_full = Options(T)

    globs_3c = (
        (LUFactorization(), Static(), prob_r3_ad, prob_r1_ad),
        (LUFactorization(), StrongWolfe(), prob_r3_ad, prob_r1_ad),
        (LUFactorization(), Static(), prob_r3_j, prob_r1_j),
        (LUFactorization(), Backtracking(), prob_r3_j, prob_r1_j),
        (LUFactorization(), Bisection(), prob_r3_j, prob_r1_j),
        (LUFactorization(), StrongWolfe(), prob_r3_j, prob_r1_j),
        (LUFactorization(), DogLeg(), prob_r3_j, prob_r1_j),
        (LUFactorization(), DogLeg(; radius = 0.5), prob_r3_j, prob_r1_j)
    )
    for (ls, glob, prob_r3, prob_r1) in globs_3c
        m = Newton(; linsolve = ls, globalization = glob)
        x_r1, st_r1 = solve(prob_r1, x₀_s, m, p_s; options = opts_full)
        x_r3 = copy(x₀)
        solver_r3 = init(prob_r3, x_r3, m, p; options = opts_full)
        st_r3 = solve!(x_r3, solver_r3, p)
        @test st_r3.code == st_r1.code == SUCCESS
        @test st_r3.iterations == st_r1.iterations
        @test isapprox(x_r3, Array(x_r1); rtol = 10 * sqrt(eps(R)))
        @test isapprox(x_r3, Array(inp.x_star_s); rtol = 10 * sqrt(eps(R)))
    end

    for glob in (Static(), DogLeg())
        m_qr = Newton(; linsolve = QRFactorization(), globalization = glob)
        x_r1, st_r1 = solve(prob_r1_j, x₀_s, m_qr, p_s; options = opts_full)
        x_r3, st_r3 = solve(prob_r3_j, x₀, m_qr, p; options = opts_full)
        @test st_r3.code == st_r1.code == SUCCESS
        @test st_r3.iterations == st_r1.iterations
        @test isapprox(x_r3, Array(x_r1); rtol = 10 * sqrt(eps(R)))
    end

    # `Newton(; reuse = 2)` parity across line-search globalizations
    reuse_cases = (
        (LUFactorization(), Static(), prob_r3_ad, prob_r1_ad),
        (LUFactorization(), Backtracking(), prob_r3_j, prob_r1_j),
        (LUFactorization(), StrongWolfe(), prob_r3_j, prob_r1_j)
    )
    for (ls, glob, prob_r3, prob_r1) in reuse_cases
        m_reuse = Newton(; linsolve = ls, globalization = glob, reuse = 2)
        xr_r1, str_r1 = solve(prob_r1, x₀_s, m_reuse, p_s; options = opts_full)
        xr_r3 = copy(x₀)
        solver_reuse = init(prob_r3, xr_r3, m_reuse, p; options = opts_full)
        str_r3 = solve!(xr_r3, solver_reuse, p)
        @test str_r3.code == str_r1.code == SUCCESS
        @test str_r3.iterations == str_r1.iterations
        @test isapprox(xr_r3, Array(xr_r1); rtol = 10 * sqrt(eps(R)))
    end

    # Singular start and diagonal regularization in R3
    x_zero = zeros(T, 2)
    p_one = ones(T, 2)
    prob_sq_r3 = NonlinearProblem(R3SqF(), R3SqJ())
    x_sing, st_sing = solve(prob_sq_r3, x_zero, Newton(), p_one; options = opts_full)
    @test st_sing.code == SINGULAR
    @test x_sing == x_zero
    @test st_sing.iterations == 1

    x_reg1, st_reg1 = solve(
        prob_sq_r3, x_zero, Newton(; regularization = 0.5), p_one; options = opts1)
    @test x_reg1 == T[2, 2]
    @test st_reg1.iterations == 1

    x_reg_sol, st_reg_sol = solve(
        prob_sq_r3, x_zero, Newton(; regularization = 0.5), p_one; options = opts_full)
    @test st_reg_sol.code == SUCCESS
    @test isapprox(x_reg_sol, p_one; rtol = 10 * sqrt(eps(R)))

    # DogLeg branches in R3 (`:cauchy`, `:segment`, `:newton`) and singular Cauchy fallback (MU7)
    c = T <: Complex ? T(im) : one(T)
    A_dl = T[3 c; 0 2]
    b_dl = T[3, 2]
    prob_dl = NonlinearProblem(R3LinearF(), R3LinearJ())
    A_dl_s = SMatrix{2, 2, T}(3, 0, c, 2)
    b_dl_s = SVector{2, T}(3, 2)
    prob_dl_s = NonlinearProblem(R1LinearF(), R1LinearJ())
    for δ in (R(0.2), R(0.85), R(3.0))
        m_dl = Newton(; globalization = DogLeg(; radius = δ))
        x_dl_r3, st_dl_r3 = solve(
            prob_dl, zeros(T, 2), m_dl, (; A = A_dl, b = b_dl); options = opts1)
        x_dl_r1, st_dl_r1 = solve(
            prob_dl_s, zero(SVector{2, T}), m_dl, (; A = A_dl_s, b = b_dl_s); options = opts1)
        @test st_dl_r3.code == st_dl_r1.code
        @test st_dl_r3.iterations == st_dl_r1.iterations == 1
        @test isapprox(x_dl_r3, Array(x_dl_r1); rtol = 8 * eps(R))
        @test isapprox(st_dl_r3.stepnorm, st_dl_r1.stepnorm; rtol = 8 * eps(R))
    end

    # Singular Jacobian under DogLeg in R3 takes the Cauchy step instead of returning SINGULAR
    J_sing_dl = T[1 1; 1 1]
    b_sing_dl = T[0, 1]
    x_cauchy, st_cauchy = solve(
        prob_dl, zeros(T, 2), Newton(; globalization = DogLeg()),
        (; A = J_sing_dl, b = b_sing_dl); options = opts1)
    @test st_cauchy.code != SINGULAR
    @test st_cauchy.iterations == 1
    @test isapprox(x_cauchy, T[R(0.25), R(0.25)]; rtol = 8 * eps(R))
end

@testset "Scenario 4: NonlinearSolver, init, solve!, re-solve state reset (MU4), and solve non-mutation (MU5), $T" for T in ELTYPES
    R = real(T)
    @test Base.isexported(GeometricSolvers, :NonlinearSolver)
    @test Base.isexported(GeometricSolvers, :init)
    @test Base.isexported(GeometricSolvers, :solve!)

    x₀ = T[3, 5]
    p₁ = T[4, 9]
    p₂ = T[16, 25]
    j_calls = Ref(0)
    prob = NonlinearProblem(R3SqF(), CountingSqJ(j_calls))

    # `init(prob, x, method, p)` builds `NonlinearSolver` without mutating `x`
    x = copy(x₀)
    solver = init(prob, x, Newton(; reuse = 4), p₁)
    @test solver isa NonlinearSolver{T}
    @test solver.n == 2
    @test x == x₀
    @test solver.prob === prob
    @test solver.method === Adapt.adapt(ToReal{R}(), Newton(; reuse = 4))
    @test solver.options isa Options{R}

    # `init` rejects a `StaticArray` iterate, which `solve!` cannot update in place
    @test_throws ArgumentError init(prob, SVector{2, T}(x₀), Newton(), p₁)
    @test_throws ArgumentError init(prob, MVector{2, T}(x₀), Newton(), p₁)

    # `solve!` rejects an iterate whose element type or length differs from the template
    T_other = T <: Complex ? real(T) : complex(T)
    x_other = T_other.(x₀)
    @test_throws ArgumentError solve!(x_other, solver, p₁)
    @test x_other == T_other.(x₀)
    @test_throws ArgumentError solve!(T[3, 5, 7], solver, p₁)
    # the check reads the template from the solver, not from the state, which may have no `Fx`
    solver_nofx = NonlinearSolver(prob, x₀, Newton(), solver.options, (;))
    @test_throws ArgumentError solve!(x_other, solver_nofx, p₁)
    @test_throws ArgumentError solve!(T[3, 5, 7], solver_nofx, p₁)

    # First `solve!(x, solver, p₁)` converges in place to `[2, 3]` and leaves `age > 0`
    st₁ = solve!(x, solver, p₁)
    @test st₁ isa SolverStatus{R}
    @test st₁.code == SUCCESS
    @test isapprox(x, T[2, 3]; rtol = 10 * sqrt(eps(R)))
    @test solver.state.age > 0

    # Second `solve!(x, solver, p₂)` on a new initial guess and new parameter resets `age = 0`
    # and `stalled = false`, so step 1 refactorises at the new `(x, p₂)` (catches MU4!)
    solver_1step = init(
        prob, copy(x₀), Newton(; reuse = 4), p₁;
        min_iterations = 1, max_iterations = 1,
        f_abstol = zero(R), f_reltol = zero(R), x_abstol = zero(R), x_reltol = zero(R))
    x_s1 = copy(x₀)
    solve!(x_s1, solver_1step, p₁)
    @test solver_1step.state.age == Int32(1)
    j_calls[] = 0
    x_s2 = T[5, 6]
    st_s2 = solve!(x_s2, solver_1step, p₂)
    @test st_s2.iterations == 1
    @test j_calls[] == 1
    # Exact 1-step Newton iterate from `[5, 6]` on `y^2 - [16, 25]` is `[5 - 9/10, 6 - 11/12] = [4.1, 61/12]`;
    # a stale factor from `x₀ = [3, 5]` (`J = diag(6, 10)`) would give `[5 - 9/6, 6 - 11/10] = [3.5, 4.9]`!
    @test isapprox(x_s2, T[T(41) / T(10), T(61) / T(12)]; rtol = 8 * eps(R))

    # Full second `solve!(x, solver, p₂)` converges to `[4, 5]`
    x_re = T[5, 6]
    st₂ = solve!(x_re, solver, p₂)
    @test st₂.code == SUCCESS
    @test isapprox(x_re, T[4, 5]; rtol = 10 * sqrt(eps(R)))

    # Re-solve with `DogLeg()` resets `solver.state.radius` at the start of `solve!`
    solver_dl = init(
        prob, copy(x₀), Newton(; globalization = DogLeg()), p₁;
        min_iterations = 1, max_iterations = 1,
        f_abstol = zero(R), f_reltol = zero(R), x_abstol = zero(R), x_reltol = zero(R))
    x_dl1 = copy(x₀)
    solve!(x_dl1, solver_dl, p₁)
    @test solver_dl.state.radius > zero(R)
    x_dl2 = T[5, 6]
    solve!(x_dl2, solver_dl, p₂)
    x_dl2_fresh, _ = solve(
        prob, T[5, 6], Newton(; globalization = DogLeg()), p₂;
        min_iterations = 1, max_iterations = 1,
        f_abstol = zero(R), f_reltol = zero(R), x_abstol = zero(R), x_reltol = zero(R))
    @test x_dl2 == x_dl2_fresh

    # `solve(prob, x₀, method, p)` does not mutate `x₀` (catches MU5!) and matches `solve!`
    for AT in ARRAY_BACKENDS
        x₀_at = AT(copy(x₀))
        p₁_at = AT(copy(p₁))
        m_test = AT === Array ? Newton() : Picard(; damping = 0.8)
        prob_test = AT === Array ? prob : NonlinearProblem(R3PicardF(R(0.5)))
        x_out, st_out = solve(prob_test, x₀_at, m_test, p₁_at)
        @test Array(x₀_at) == x₀
        @test x_out !== x₀_at
        @test x_out isa AT{T, 1}
        @test st_out.code == SUCCESS

        x_in = copy(x₀_at)
        solver_in = init(prob_test, x_in, m_test, p₁_at)
        st_in = solve!(x_in, solver_in, p₁_at)
        @test Array(x_out) == Array(x_in)
        @test st_out === st_in
    end

    # R3 `Newton` on a non-`Array` backend, with LU and QR, a line search and DogLeg
    prob_ad = NonlinearProblem(R3SqF())
    for m in (Newton(), Newton(; globalization = StrongWolfe()),
        Newton(; globalization = DogLeg()), Newton(; linsolve = QRFactorization()))
        x_jl = JLArray(copy(x₀))
        p_jl = JLArray(copy(p₁))
        solver_jl = init(prob_ad, x_jl, m, p_jl)
        st_jl = solve!(x_jl, solver_jl, p_jl)
        @test st_jl.code == SUCCESS
        @test isapprox(Array(x_jl), T[2, 3]; rtol = 10 * sqrt(eps(R)))
    end
end

function check_picard_allocations(pr, x, x₀, glob, p)
    solver_p = init(pr, x, Picard(; damping = 0.8, globalization = glob), p)
    st_pre_p = solver_p.state
    @test solve!_allocations(x, x₀, solver_p, p) == 0
    @test solver_p.state.Fx === st_pre_p.Fx
    @test solver_p.state.d === st_pre_p.d
    @test solver_p.state.g === st_pre_p.g
    return nothing
end

function check_newton_allocations(pr, x, x₀, ls, glob, p; jacobian = AutoForwardDiff())
    solver_n = init(
        pr, x, Newton(; linsolve = ls, globalization = glob, jacobian = jacobian), p)
    @test solver_n.state.fac isa (ls isa LUFactorization ? DenseLU : DenseQR)
    if glob isa DogLeg
        @test solver_n.state.J !== solver_n.state.fac.factors
    else
        @test solver_n.state.J === solver_n.state.fac.factors
    end
    st_pre_n = solver_n.state
    @test solve!_allocations(x, x₀, solver_n, p) == 0
    @test solver_n.state.Fx === st_pre_n.Fx
    @test solver_n.state.d === st_pre_n.d
    @test solver_n.state.fac === st_pre_n.fac
    @test solver_n.state.fac.factors === st_pre_n.fac.factors
    @test solver_n.state.J === st_pre_n.J
    @test solver_n.state.g === st_pre_n.g
    return nothing
end

@testset "Scenario 5: zero steady-state heap allocations, slot identity, and JVP call count on Array, $T" for T in ELTYPES
    old_threads = BLAS.get_num_threads()
    try
        BLAS.set_num_threads(1)
        R = real(T)
        prob_pic_ad = NonlinearProblem(R3PicardF(R(0.5)))
        prob_pic_j = NonlinearProblem(R3PicardF(R(0.5)), R3PicardJ(T(R(0.5))))
        prob_newt_ad = NonlinearProblem(R3SqF())
        prob_newt_j = NonlinearProblem(R3SqF(), R3SqJ())

        n = 2
        x₀ = T[(i + 2) / (n + 1) + 2 for i in 1:n]
        p = T[(i + 1) / (n + 2) + 1 for i in 1:n]
        x = similar(x₀)

        # Control: `collect` allocates
        @test (@allocated collect(x₀)) > 0

        # Picard allocations and slot identity across all globalizations
        for glob in (Static(), Backtracking(), Bisection(), StrongWolfe())
            check_picard_allocations(prob_pic_ad, x, x₀, glob, p)
        end
        check_picard_allocations(prob_pic_j, x, x₀, StrongWolfe(), p)

        # Newton with LU and QR, analytic J! and AutoForwardDiff(), and all globalizations + reuse
        for glob in (Static(), StrongWolfe())
            check_newton_allocations(prob_newt_ad, x, x₀, LUFactorization(), glob, p)
        end
        for glob in (Static(), Backtracking(), Bisection(), StrongWolfe(), DogLeg())
            check_newton_allocations(
                prob_newt_j, x, x₀, LUFactorization(), glob, p; jacobian = nothing)
        end
        for glob in (Static(), DogLeg())
            check_newton_allocations(
                prob_newt_j, x, x₀, QRFactorization(), glob, p; jacobian = nothing)
        end
        for pr in (prob_newt_ad, prob_newt_j)
            solver_reuse = init(
                pr, x,
                Newton(;
                    reuse = 2, regularization = 0.1,
                    globalization = Backtracking()),
                p)
            st_pre_r = solver_reuse.state
            @test solve!_allocations(x, x₀, solver_reuse, p) == 0
            @test solver_reuse.state.Fx === st_pre_r.Fx
            @test solver_reuse.state.d === st_pre_r.d
            @test solver_reuse.state.fac === st_pre_r.fac
            @test solver_reuse.state.J === st_pre_r.J
            @test solver_reuse.state.g === st_pre_r.g
        end

        if real(T) === Float64
            solver_mp = init(
                prob_newt_j, x, Newton(; linsolve = LUFactorization(Float32)), p)
            @test solver_mp.state.J !== solver_mp.state.fac.factors
            @test eltype(solver_mp.state.J) === T
            @test eltype(solver_mp.state.fac.factors) ===
                  (T <: Complex ? ComplexF32 : Float32)
            @test solve!_allocations(x, x₀, solver_mp, p) == 0
        end

        # Cross-size checks at `n = 10`, `n = 100`, and `n = 1000` (`abs(b(100) - b(10)) <= 64`)
        x₀_10 = T[(i + 2) / 11 + 2 for i in 1:10]
        p_10 = T[(i + 1) / 12 + 1 for i in 1:10]
        x_10 = similar(x₀_10)
        b10 = solve!_allocations(x_10, x₀_10, init(prob_newt_j, x_10, Newton(), p_10), p_10)
        @test b10 == 0

        x₀_100 = T[(i + 2) / 101 + 2 for i in 1:100]
        p_100 = T[(i + 1) / 102 + 1 for i in 1:100]
        x_100 = similar(x₀_100)
        b100 = solve!_allocations(
            x_100, x₀_100, init(prob_newt_j, x_100, Newton(), p_100), p_100)
        @test b100 == 0
        @test abs(b100 - b10) <= 64
        check_newton_allocations(
            prob_newt_ad, x_100, x₀_100, LUFactorization(), Static(), p_100;
            jacobian = AutoForwardDiff(; chunksize = 2))

        n_large = 1000
        x₀_1000 = T[(i + 2) / (n_large + 1) + 2 for i in 1:n_large]
        p_1000 = T[(i + 1) / (n_large + 2) + 1 for i in 1:n_large]
        x_1000 = similar(x₀_1000)
        solver_pic_1000 = init(prob_pic_ad, x_1000, Picard(; damping = 0.8), p_1000)
        @test solve!_allocations(x_1000, x₀_1000, solver_pic_1000, p_1000) == 0

        solver_newt_1000 = init(prob_newt_j, x_1000, Newton(), p_1000)
        @test solve!_allocations(x_1000, x₀_1000, solver_newt_1000, p_1000) == 0

        # Under `StrongWolfe` and `Bisection` at `n = 8` with `AutoForwardDiff(chunksize = 2)`,
        # slope evaluations use the 1-dual JVP pushforward (1 `F!` call per slope evaluation)
        # rather than an `n / 2 = 4`-call Jacobian sweep (MU6).
        f_real_calls = Ref(0)
        f_dual1_calls = Ref(0)
        f_dual2_calls = Ref(0)
        n8 = 8
        prob_cnt = NonlinearProblem(
            CountingSqF(f_real_calls, f_dual1_calls, f_dual2_calls))
        x₀_8 = T[(i + 2) / 9 + 2 for i in 1:n8]
        p_8 = T[(i + 1) / 10 + 1 for i in 1:n8]
        for glob in (StrongWolfe(), Bisection())
            x_8 = copy(x₀_8)
            solver_cnt = init(
                prob_cnt, x_8,
                Newton(;
                    jacobian = AutoForwardDiff(; chunksize = 2), globalization = glob),
                p_8)
            f_real_calls[] = 0
            f_dual1_calls[] = 0
            f_dual2_calls[] = 0
            st_cnt = solve!(x_8, solver_cnt, p_8)
            @test st_cnt.code == SUCCESS
            # Each Newton step does `ceil(Int, 8 / 2) = 4` chunk-2 dual calls for `jacobian!!`,
            # and each line-search slope evaluation does 1 chunk-1 dual call for `jvp!!`.
            @test f_dual2_calls[] == st_cnt.iterations * 4
            @test f_dual1_calls[] >= st_cnt.iterations
        end

        # Non-ForwardDiff AD backend (`AutoFiniteDiff()`) works end to end through `DIJacobian` on `Array`
        x_fd = copy(x₀)
        solver_fd = init(
            prob_newt_ad, x_fd,
            Newton(; jacobian = AutoFiniteDiff(), globalization = StrongWolfe()), p)
        @test solver_fd.state.jacprep isa DIJacobian
        st_fd = solve!(x_fd, solver_fd, p)
        @test st_fd.code == SUCCESS
        @test isapprox(x_fd, sqrt.(p); rtol = 10 * sqrt(eps(R)))
    finally
        BLAS.set_num_threads(old_threads)
    end
end
