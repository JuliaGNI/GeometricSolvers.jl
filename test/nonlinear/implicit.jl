# Implicit differentiation of `solve` for `ForwardDiff.Dual` inputs (`StripDual`, primal solve
# delegation, single tangent solve `J(x*, p) ẋ = -(∂F/∂p)(x*, p) ṗ` at `x*`, non-`SUCCESS` and
# singular tangent propagation, and `ArgumentError` refusals on `init`, `solve!`, `solve_batch!`,
# nested duals, mixed tags, unadapted parameter structs, and captured duals).

using Adapt: Adapt
using ADTypes: AutoForwardDiff
using ForwardDiff: ForwardDiff
using GPUArraysCore: allowscalar
using JET: JET
using JLArrays: JLArray
using LinearAlgebra: norm
using StaticArrays: SMatrix, SVector, StaticVector
using Test

using GeometricSolvers
using GeometricSolvers: Backtracking, DogLeg, LinearMethod, LUFactorization, MAXITERS,
                        NONFINITE, Newton, NonlinearProblem, Options, PerSystem, Picard,
                        QRFactorization, ReturnCode, SINGULAR, STALLED, SUCCESS,
                        SVDFactorization, SolverStatus, StripDual, empty_static_factor,
                        factorize!!, init, init_factor, ldiv!!, returncode, solve, solve!,
                        solve_batch!

include("../helpers/matrix.jl")

allowscalar(false)

struct ImplicitTestTagA end
struct ImplicitTestTagB end

struct AdaptedParam{A, B}
    a::A
    b::B
end
Adapt.@adapt_structure AdaptedParam

struct UnadaptedParam{A}
    a::A
end

# Counting linear solver wrapper to record factorisations, their input matrices, and
# back-substitutions per factor.
struct CountingLinear{L <: LinearMethod} <: LinearMethod{Nothing}
    inner::L
    fac_calls::Base.RefValue{Int}
    ldiv_calls::Base.RefValue{Int}
    fac_eltypes::Vector{DataType}
    fac_matrices::Vector{Any}
    ldiv_fac_ids::Vector{Int}
end

function CountingLinear(inner::LinearMethod = LUFactorization())
    return CountingLinear(
        inner, Ref(0), Ref(0), DataType[], Any[], Int[])
end

struct CountingFactor{F}
    inner::F
    id::Int
    ldiv_calls::Base.RefValue{Int}
    ldiv_fac_ids::Vector{Int}
end

function GeometricSolvers.empty_static_factor(
        m::CountingLinear, x::Union{StaticVector, Number})
    return CountingFactor(
        empty_static_factor(m.inner, x), 0, m.ldiv_calls, m.ldiv_fac_ids)
end

function GeometricSolvers.init_factor(
        m::CountingLinear, prob, x::AbstractVector, p, jacprep)
    return CountingFactor(
        GeometricSolvers.dense_factor(m.inner, prob, x, p), 0, m.ldiv_calls, m.ldiv_fac_ids)
end

function GeometricSolvers.factorize!!(
        fac::Union{Nothing, CountingFactor}, m::CountingLinear, A)
    m.fac_calls[] += 1
    id = m.fac_calls[]
    push!(m.fac_eltypes, eltype(A))
    push!(m.fac_matrices, A isa Number ? A : Array(A))
    inner_slot = fac === nothing ? nothing : fac.inner
    inner_fac = factorize!!(inner_slot, m.inner, A)
    return CountingFactor(inner_fac, id, m.ldiv_calls, m.ldiv_fac_ids)
end

GeometricSolvers.returncode(fac::CountingFactor) = returncode(fac.inner)
GeometricSolvers.exact_solve(fac::CountingFactor) = GeometricSolvers.exact_solve(fac.inner)
function GeometricSolvers.factor_diagnostics(buf, fac::CountingFactor, Fx, fnorm)
    GeometricSolvers.factor_diagnostics(buf, fac.inner, Fx, fnorm)
end

function Base.getproperty(fac::CountingFactor, s::Symbol)
    s === :info && return getfield(fac, :inner).info
    s === :rank && return getfield(fac, :inner).rank
    s === :factors && return getfield(fac, :inner).factors
    return getfield(fac, s)
end

function GeometricSolvers.ldiv!!(d, fac::CountingFactor, b)
    fac.ldiv_calls[] += 1
    push!(fac.ldiv_fac_ids, fac.id)
    return ldiv!!(d, fac.inner, b)
end

# Reference dual-iteration Newton loop in dual arithmetic until `x₊ == x` (cap 50).
function dual_newton_reference(x0::SVector{2}, p_dual::SVector{2}; max_iter::Int = 50)
    D = promote_type(eltype(x0), eltype(p_dual))
    x = SVector{2, D}(D(x0[1]), D(x0[2]))
    q = SVector{2, D}(D(p_dual[1]), D(p_dual[2]))
    for _ in 1:max_iter
        r = x .^ 2 .- q
        dx = SVector{2, D}(r[1] / (2 * x[1]), r[2] / (2 * x[2]))
        x_next = x .- dx
        x_next == x && return x_next
        x = x_next
    end
    return x
end

const PROB_VAL = NonlinearProblem(
    (x, p) -> x .^ 2 .- p,
    (x, p) -> SMatrix{2, 2, eltype(x)}(
        2 * x[1], zero(eltype(x)), zero(eltype(x)), 2 * x[2]))

const PROB_VAL_NOJ = NonlinearProblem((x, p) -> x .^ 2 .- p)

const PROB_SCALAR = NonlinearProblem((x, p) -> x^2 - p, (x, p) -> 2 * x)

const PROB_R3 = NonlinearProblem((r, x, p) -> (r .= x .^ 2 .- p; r))

function r3_analytic_j!(J, x, p)
    fill!(J, zero(eltype(J)))
    J[1, 1] = 2 * x[1]
    J[2, 2] = 2 * x[2]
    return J
end

const PROB_R3_J = NonlinearProblem(
    (r, x, p) -> (r .= x .^ 2 .- p; r),
    r3_analytic_j!)

const PROB_CPLX = NonlinearProblem(
    (x, p) -> begin
    R = real(eltype(x))
    return SVector(
        x[1]^2 - Complex{R}(1, 1) * p[1],
        x[2]^2 - Complex{R}(1, -1) * p[2])
end)

const PROB_STALL = NonlinearProblem(
    (x, p) -> SVector(one(eltype(x)), one(eltype(x))),
    (x, p) -> SMatrix{2, 2, eltype(x)}(
        eltype(x)(1e16), zero(eltype(x)), zero(eltype(x)), eltype(x)(1e16)))

const PROB_ZERO_J = NonlinearProblem(
    PROB_VAL.F,
    (x, p) -> SMatrix{2, 2, eltype(x)}(
        zero(eltype(x)), zero(eltype(x)), zero(eltype(x)), zero(eltype(x))))

r3_zero_j!(J, x, p) = fill!(J, zero(eltype(J)))

const PROB_R3_ZERO_J = NonlinearProblem(PROB_R3.F, r3_zero_j!)

const PROB_INF_J = NonlinearProblem(
    PROB_VAL.F,
    (x, p) -> SMatrix{2, 2, eltype(x)}(
        eltype(x)(Inf), zero(eltype(x)), zero(eltype(x)), one(eltype(x))))

const PROB_RD_R1 = NonlinearProblem(
    (x, p) -> SVector(x[1] + x[2] - p, 2 * (x[1] + x[2] - p)))

function rd_r3_f!(r, x, p)
    s = x[1] + x[2] - p
    r[1] = s
    r[2] = 2 * s
    return r
end

const PROB_RD_R3 = NonlinearProblem(rd_r3_f!)

function scaled_j_r1(x, p)
    return SMatrix{2, 2, eltype(x)}(
        4 * x[1], zero(eltype(x)), zero(eltype(x)), 4 * x[2])
end

function scaled_j_r3!(J, x, p)
    fill!(J, zero(eltype(J)))
    J[1, 1] = 4 * x[1]
    J[2, 2] = 4 * x[2]
    return J
end

const PROB_PIC_R1 = NonlinearProblem((x, p) -> (x .^ 2 .- p) ./ eltype(x)(10))

const PROB_PIC_R1_J = NonlinearProblem(
    (x, p) -> (x .^ 2 .- p) ./ eltype(x)(10),
    (x, p) -> SMatrix{2, 2, eltype(x)}(
        x[1] / eltype(x)(5), zero(eltype(x)), zero(eltype(x)), x[2] / eltype(x)(5)))

const PROB_PIC_R3 = NonlinearProblem((r, x, p) -> (r .= (x .^ 2 .- p) ./ eltype(x)(10); r))

function dual_r1_allocated(prob, x, m, p)
    solve(prob, x, m, p)
    return @allocated solve(prob, x, m, p)
end

@testset "StripDual on scalars, Complex, arrays, tuples, and @adapt_structure" begin
    for R in REAL_ELTYPES
        d1 = ForwardDiff.Dual{ImplicitTestTagA}(R(3.5), R(1.0), R(-2.0))
        d2 = ForwardDiff.Dual{ImplicitTestTagA}(R(-1.25), R(0.5), R(4.0))
        @test Adapt.adapt(StripDual(), d1) === R(3.5)
        @test Adapt.adapt(StripDual(), R(2.0)) === R(2.0)

        cd = Complex(d1, d2)
        @test Adapt.adapt(StripDual(), cd) === Complex(R(3.5), R(-1.25))

        sv = SVector(d1, d2)
        @test Adapt.adapt(StripDual(), sv) === SVector(R(3.5), R(-1.25))

        svc = SVector(cd, cd)
        @test Adapt.adapt(StripDual(), svc) ===
              SVector(Complex(R(3.5), R(-1.25)), Complex(R(3.5), R(-1.25)))

        av = [d1, d2]
        av_s = Adapt.adapt(StripDual(), av)
        @test av_s isa Vector{R}
        @test av_s == [R(3.5), R(-1.25)]

        avc = [cd, cd]
        avc_s = Adapt.adapt(StripDual(), avc)
        @test avc_s isa Vector{Complex{R}}
        @test avc_s == [Complex(R(3.5), R(-1.25)), Complex(R(3.5), R(-1.25))]

        jv = JLArray([d1, d2])
        jv_s = Adapt.adapt(StripDual(), jv)
        @test jv_s isa JLArray{R, 1}
        @test Array(jv_s) == [R(3.5), R(-1.25)]

        jvc = JLArray([cd, cd])
        jvc_s = Adapt.adapt(StripDual(), jvc)
        @test jvc_s isa JLArray{Complex{R}, 1}
        @test Array(jvc_s) == [Complex(R(3.5), R(-1.25)), Complex(R(3.5), R(-1.25))]

        tup = (d1, (sv, R(7)))
        @test Adapt.adapt(StripDual(), tup) === (R(3.5), (SVector(R(3.5), R(-1.25)), R(7)))

        nt = (x = d1, y = sv, z = cd)
        @test Adapt.adapt(StripDual(), nt) ===
              (x = R(3.5), y = SVector(R(3.5), R(-1.25)), z = Complex(R(3.5), R(-1.25)))

        ap = AdaptedParam(d1, sv)
        ap_s = Adapt.adapt(StripDual(), ap)
        @test ap_s === AdaptedParam(R(3.5), SVector(R(3.5), R(-1.25)))
    end
end

@testset "Closed-form and dual-iteration agreement across types and containers" begin
    for R in REAL_ELTYPES
        rtol = 1000 * eps(R)
        opt = Options(R; f_abstol = 10 * eps(R), f_reltol = zero(R))
        p1 = SVector{2, R}(R(4), R(9))
        x0 = SVector{2, R}(R(1.5), R(2.5))
        expected_d = inv.(2 .* sqrt.(p1))
        expected_J = SMatrix{2, 2, R}(
            expected_d[1], zero(R), zero(R), expected_d[2])

        # SVector: N = 1 and N = 2
        dx_dt = ForwardDiff.derivative(
            t -> solve(PROB_VAL, x0, Newton(), p1 .+ t; options = opt)[1], zero(R))
        @test isapprox(dx_dt, expected_d; rtol = rtol)

        J_ad = ForwardDiff.jacobian(
            q -> solve(PROB_VAL, x0, Newton(), q; options = opt)[1], p1)
        @test isapprox(J_ad, expected_J; rtol = rtol)

        # Dual-iteration reference and primal/status identity
        p_d2 = SVector(
            ForwardDiff.Dual{ImplicitTestTagA}(p1[1], one(R), zero(R)),
            ForwardDiff.Dual{ImplicitTestTagA}(p1[2], zero(R), one(R)))
        x_ref = dual_newton_reference(x0, p_d2)
        x_p, st_p = solve(PROB_VAL, x0, Newton(), p1; options = opt)
        x_d, st_d = solve(PROB_VAL, x0, Newton(), p_d2; options = opt)
        @test Adapt.adapt(StripDual(), x_d) == x_p
        @test st_d === st_p
        for i in 1:2, k in 1:2

            @test isapprox(
                ForwardDiff.partials(x_d[i], k),
                ForwardDiff.partials(x_ref[i], k); rtol = rtol)
        end

        # Scalar (Number): N = 1 and N = 2
        dx_scalar = ForwardDiff.derivative(
            t -> solve(PROB_SCALAR, R(1.5), Newton(), R(4) + t; options = opt)[1], zero(R))
        @test isapprox(dx_scalar, inv(2 * sqrt(R(4))); rtol = rtol)
        ps_d2 = ForwardDiff.Dual{ImplicitTestTagA}(R(4), one(R), R(3))
        xs_p, sts_p = solve(PROB_SCALAR, R(1.5), Newton(), R(4); options = opt)
        xs_d, sts_d = solve(PROB_SCALAR, R(1.5), Newton(), ps_d2; options = opt)
        @test ForwardDiff.value(xs_d) == xs_p
        @test sts_d === sts_p
        @test isapprox(
            ForwardDiff.partials(xs_d, 1), inv(2 * sqrt(R(4))); rtol = rtol)
        @test isapprox(
            ForwardDiff.partials(xs_d, 2), R(3) * inv(2 * sqrt(R(4))); rtol = rtol)

        # Array (R3): N = 1 and N = 2
        x0_arr = Vector{R}(x0)
        p1_arr = Vector{R}(p1)
        dx_arr = ForwardDiff.derivative(
            t -> solve(PROB_R3, x0_arr, Newton(), p1_arr .+ t; options = opt)[1], zero(R))
        @test isapprox(dx_arr, Vector(expected_d); rtol = rtol)
        J_arr = ForwardDiff.jacobian(
            q -> solve(PROB_R3, x0_arr, Newton(), q; options = opt)[1], p1_arr)
        @test isapprox(J_arr, Matrix(expected_J); rtol = rtol)
        xa_p, sta_p = solve(PROB_R3, x0_arr, Newton(), p1_arr; options = opt)
        xa_d, sta_d = solve(PROB_R3, x0_arr, Newton(), collect(p_d2); options = opt)
        @test Adapt.adapt(StripDual(), xa_d) == xa_p
        @test sta_d === sta_p

        # JLArray (R3 GPU-style, allowscalar(false)): N = 1 and N = 2
        x0_jl = JLArray(x0_arr)
        p_jl_1 = JLArray([
            ForwardDiff.Dual{ImplicitTestTagA}(p1[1], one(R)),
            ForwardDiff.Dual{ImplicitTestTagA}(p1[2], one(R))])
        xj_p, stj_p = solve(PROB_R3, x0_jl, Newton(), JLArray(p1_arr); options = opt)
        xj_1, stj_1 = solve(PROB_R3, x0_jl, Newton(), p_jl_1; options = opt)
        @test xj_1 isa JLArray
        @test Array(Adapt.adapt(StripDual(), xj_1)) == Array(xj_p)
        @test stj_1 === stj_p
        @test isapprox(
            ForwardDiff.partials.(Array(xj_1), 1), Vector(expected_d); rtol = rtol)

        p_jl_2 = JLArray(collect(p_d2))
        xj_2, stj_2 = solve(PROB_R3, x0_jl, Newton(), p_jl_2; options = opt)
        @test xj_2 isa JLArray
        @test Array(Adapt.adapt(StripDual(), xj_2)) == Array(xj_p)
        @test stj_2 === stj_p
        @test isapprox(
            [ForwardDiff.partials(Array(xj_2)[i], k) for i in 1:2, k in 1:2],
            Matrix(expected_J); rtol = rtol)
    end

    # Complex state with real dual p across SVector, Array, and scalar Number
    prob_cplx_scalar = NonlinearProblem(
        (x, p) -> x^2 - Complex{real(typeof(x))}(1, 1) * p,
        (x, p) -> 2 * x)
    for T in filter(T -> T <: Complex, ELTYPES)
        R = real(T)
        rtol = 1000 * eps(R)
        opt = Options(R; f_abstol = 10 * eps(R), f_reltol = zero(R))
        p1 = SVector{2, R}(R(4), R(9))
        x0_c = SVector{2, T}(T(R(1.5), R(0.1)), T(R(2.5), R(-0.1)))
        expected_d = T.(inv.(2 .* sqrt.(p1)))
        expected_J = SMatrix{2, 2, T}(
            expected_d[1], zero(T), zero(T), expected_d[2])

        dx_c = ForwardDiff.derivative(
            t -> solve(PROB_VAL, x0_c, Newton(), p1 .+ t; options = opt)[1], zero(R))
        @test isapprox(dx_c, expected_d; rtol = rtol)

        p_d2 = SVector(
            ForwardDiff.Dual{ImplicitTestTagA}(p1[1], one(R), zero(R)),
            ForwardDiff.Dual{ImplicitTestTagA}(p1[2], zero(R), one(R)))
        xc_p, stc_p = solve(PROB_VAL, x0_c, Newton(), p1; options = opt)
        xc_d, stc_d = solve(PROB_VAL, x0_c, Newton(), p_d2; options = opt)
        @test Adapt.adapt(StripDual(), xc_d) == xc_p
        @test stc_d === stc_p
        for i in 1:2, k in 1:2

            pk = complex(
                ForwardDiff.partials(real(xc_d[i]), k),
                ForwardDiff.partials(imag(xc_d[i]), k))
            @test isapprox(pk, expected_J[i, k]; rtol = rtol, atol = rtol)
        end

        # Complex Array (R3)
        x0_ca = Vector{T}(x0_c)
        p1_a = Vector{R}(p1)
        dx_ca = ForwardDiff.derivative(
            t -> solve(PROB_R3, x0_ca, Newton(), p1_a .+ t; options = opt)[1], zero(R))
        @test isapprox(dx_ca, Vector(expected_d); rtol = rtol)
        xca_p, stca_p = solve(PROB_R3, x0_ca, Newton(), p1_a; options = opt)
        xca_d, stca_d = solve(PROB_R3, x0_ca, Newton(), collect(p_d2); options = opt)
        @test Adapt.adapt(StripDual(), xca_d) == xca_p
        @test stca_d === stca_p
        for i in 1:2, k in 1:2

            pk = complex(
                ForwardDiff.partials(real(xca_d[i]), k),
                ForwardDiff.partials(imag(xca_d[i]), k))
            @test isapprox(pk, expected_J[i, k]; rtol = rtol, atol = rtol)
        end

        # Complex problem with genuinely complex root x* = sqrt((1 +- im) * p)
        p_d1 = SVector(
            ForwardDiff.Dual{ImplicitTestTagA}(p1[1], one(R)),
            ForwardDiff.Dual{ImplicitTestTagA}(p1[2], one(R)))
        xc_star, _ = solve(PROB_CPLX, x0_c, Newton(), p1; options = opt)
        xc_d1, _ = solve(PROB_CPLX, x0_c, Newton(), p_d1; options = opt)
        dx_cplx = SVector(
            complex(
                ForwardDiff.partials(real(xc_d1[1]), 1),
                ForwardDiff.partials(imag(xc_d1[1]), 1)),
            complex(
                ForwardDiff.partials(real(xc_d1[2]), 1),
                ForwardDiff.partials(imag(xc_d1[2]), 1)))
        @test isapprox(
            dx_cplx,
            SVector(T(1, 1) / (2 * xc_star[1]), T(1, -1) / (2 * xc_star[2]));
            rtol = rtol)

        # Complex scalar Number (both real-root and genuinely complex-root tangents)
        x0_cs = T(R(1.5), R(0.1))
        p_s_d2 = ForwardDiff.Dual{ImplicitTestTagA}(p1[1], one(R), R(3))
        xcs_p, stcs_p = solve(PROB_SCALAR, x0_cs, Newton(), p1[1]; options = opt)
        xcs_d, stcs_d = solve(PROB_SCALAR, x0_cs, Newton(), p_s_d2; options = opt)
        @test Adapt.adapt(StripDual(), xcs_d) == xcs_p
        @test stcs_d === stcs_p
        for (k, scale) in ((1, one(R)), (2, R(3)))
            pk = complex(
                ForwardDiff.partials(real(xcs_d), k),
                ForwardDiff.partials(imag(xcs_d), k))
            @test isapprox(pk, scale * expected_d[1]; rtol = rtol, atol = rtol)
        end

        x0_cs2 = T(R(1.5), R(0.5))
        xcs2_star, stcs2_p = solve(
            prob_cplx_scalar, x0_cs2, Newton(), p1[1]; options = opt)
        xcs2_d, stcs2_d = solve(
            prob_cplx_scalar, x0_cs2, Newton(), p_s_d2; options = opt)
        @test Adapt.adapt(StripDual(), xcs2_d) == xcs2_star
        @test stcs2_d === stcs2_p
        expected_cs2 = T(1, 1) / (2 * xcs2_star)
        for (k, scale) in ((1, one(R)), (2, R(3)))
            pk = complex(
                ForwardDiff.partials(real(xcs2_d), k),
                ForwardDiff.partials(imag(xcs2_d), k))
            @test isapprox(pk, scale * expected_cs2; rtol = rtol)
        end
    end
end

@testset "Counting wrappers verify primal iterations, +1 factorisation at x*, +N ldivs, and 1 dual F call" begin
    let R = Float64
        p1 = SVector{2, R}(R(4), R(9))
        x0 = SVector{2, R}(R(1.5), R(2.5))
        p_d2 = SVector(
            ForwardDiff.Dual{ImplicitTestTagA}(p1[1], one(R), zero(R)),
            ForwardDiff.Dual{ImplicitTestTagA}(p1[2], zero(R), one(R)))

        # R1 counting
        f_dual_calls = Ref(0)
        f_dual_x = Any[]
        prob_cnt = NonlinearProblem(
            (x, p) -> begin
                if eltype(x) <: ForwardDiff.Dual || eltype(p) <: ForwardDiff.Dual
                    f_dual_calls[] += 1
                    push!(f_dual_x, x)
                end
                return x .^ 2 .- p
            end,
            PROB_VAL.J)

        ls_p = CountingLinear()
        x_p, st_p = solve(prob_cnt, x0, Newton(; linsolve = ls_p), p1)
        @test f_dual_calls[] == 0

        ls_d = CountingLinear()
        _, st_d = solve(prob_cnt, x0, Newton(; linsolve = ls_d), p_d2)
        @test st_d.iterations == st_p.iterations
        @test ls_d.fac_calls[] == ls_p.fac_calls[] + 1
        @test ls_d.ldiv_calls[] == ls_p.ldiv_calls[] + 2
        @test all(T -> T === R, ls_d.fac_eltypes)
        @test ls_d.fac_matrices[end] == Array(PROB_VAL.J(x_p, p1))
        @test ls_d.ldiv_fac_ids[(end - 1):end] == [ls_d.fac_calls[], ls_d.fac_calls[]]
        @test f_dual_calls[] == 1
        @test Adapt.adapt(StripDual(), only(f_dual_x)) == x_p
        @test all(d -> all(iszero, ForwardDiff.partials(d)), only(f_dual_x))

        # R3 counting
        f_r3_dual_calls = Ref(0)
        prob_r3_cnt = NonlinearProblem(
            (r, x, p) -> begin
                if eltype(x) <: ForwardDiff.Dual || eltype(p) <: ForwardDiff.Dual
                    f_r3_dual_calls[] += 1
                end
                r .= x .^ 2 .- p
                return r
            end,
            r3_analytic_j!)

        ls_r3_p = CountingLinear()
        xr_p, str_p = solve(
            prob_r3_cnt, Vector(x0), Newton(; linsolve = ls_r3_p), Vector(p1))
        @test f_r3_dual_calls[] == 0

        ls_r3_d = CountingLinear()
        _, str_d = solve(
            prob_r3_cnt, Vector(x0), Newton(; linsolve = ls_r3_d), collect(p_d2))
        @test str_d.iterations == str_p.iterations
        @test ls_r3_d.fac_calls[] == ls_r3_p.fac_calls[] + 1
        @test ls_r3_d.ldiv_calls[] == ls_r3_p.ldiv_calls[] + 2
        @test all(T -> T === R, ls_r3_d.fac_eltypes)
        @test ls_r3_d.fac_matrices[end] == [2 * xr_p[1] zero(R); zero(R) 2 * xr_p[2]]
        @test ls_r3_d.ldiv_fac_ids[(end - 1):end] ==
              [ls_r3_d.fac_calls[], ls_r3_d.fac_calls[]]
        @test f_r3_dual_calls[] == 1
    end
end

@testset "Dual x0 with primal p has zero partials; dual x0 + dual p ignores x0 partials" begin
    for R in REAL_ELTYPES
        rtol = 1000 * eps(R)
        p1 = SVector{2, R}(R(4), R(9))
        x0 = SVector{2, R}(R(1.5), R(2.5))
        x0_d = SVector(
            ForwardDiff.Dual{ImplicitTestTagA}(x0[1], R(7), R(-3)),
            ForwardDiff.Dual{ImplicitTestTagA}(x0[2], R(11), R(5)))

        x_p, st_p = solve(PROB_VAL, x0, Newton(), p1)
        x_d, st_d = solve(PROB_VAL, x0_d, Newton(), p1)
        @test st_d === st_p
        @test Adapt.adapt(StripDual(), x_d) == x_p
        for i in 1:2, k in 1:2

            @test ForwardDiff.partials(x_d[i], k) == zero(R)
        end

        # Scalar dual x0 with primal p
        xs_p, sts_p = solve(PROB_SCALAR, R(1.5), Newton(), R(4))
        xs_d, sts_d = solve(
            PROB_SCALAR, ForwardDiff.Dual{ImplicitTestTagA}(R(1.5), R(9)), Newton(), R(4))
        @test sts_d === sts_p
        @test ForwardDiff.value(xs_d) == xs_p
        @test ForwardDiff.partials(xs_d, 1) == zero(R)

        # Array dual x0 with primal p
        xa_p, sta_p = solve(PROB_R3, Vector(x0), Newton(), Vector(p1))
        xa_d, sta_d = solve(PROB_R3, collect(x0_d), Newton(), Vector(p1))
        @test sta_d === sta_p
        @test Adapt.adapt(StripDual(), xa_d) == xa_p
        for i in 1:2, k in 1:2

            @test ForwardDiff.partials(xa_d[i], k) == zero(R)
        end

        # Both x0 and p dual with the same tag: x0 partials do not affect the tangent
        p_d = SVector(
            ForwardDiff.Dual{ImplicitTestTagA}(p1[1], one(R), zero(R)),
            ForwardDiff.Dual{ImplicitTestTagA}(p1[2], zero(R), one(R)))
        x_both, st_both = solve(PROB_VAL, x0_d, Newton(), p_d)
        @test st_both === st_p
        @test Adapt.adapt(StripDual(), x_both) == x_p
        @test isapprox(
            ForwardDiff.partials(x_both[1], 1), inv(2 * sqrt(p1[1])); rtol = rtol)
        @test isapprox(ForwardDiff.partials(x_both[1], 2), zero(R); atol = rtol)
        @test isapprox(ForwardDiff.partials(x_both[2], 1), zero(R); atol = rtol)
        @test isapprox(
            ForwardDiff.partials(x_both[2], 2), inv(2 * sqrt(p1[2])); rtol = rtol)

        xa_both, sta_both = solve(PROB_R3, collect(x0_d), Newton(), collect(p_d))
        @test sta_both === sta_p
        @test Adapt.adapt(StripDual(), xa_both) == xa_p
        @test isapprox(
            ForwardDiff.partials(xa_both[1], 1), inv(2 * sqrt(p1[1])); rtol = rtol)
        @test isapprox(
            ForwardDiff.partials(xa_both[2], 2), inv(2 * sqrt(p1[2])); rtol = rtol)
    end
end

@testset "Outer ForwardDiff over solve with AutoForwardDiff inside has no tag collision" begin
    let R = Float64
        rtol = 1000 * eps(R)
        p1 = SVector{2, R}(R(4), R(9))
        x0 = SVector{2, R}(R(1.5), R(2.5))
        expected_d = inv.(2 .* sqrt.(p1))

        dx_r1 = ForwardDiff.derivative(
            t -> solve(
                PROB_VAL_NOJ, x0, Newton(; jacobian = AutoForwardDiff()), p1 .+ t)[1],
            zero(R))
        @test isapprox(dx_r1, expected_d; rtol = rtol)

        dx_r3 = ForwardDiff.derivative(
            t -> solve(
                PROB_R3, Vector(x0), Newton(; jacobian = AutoForwardDiff()),
                Vector(p1) .+ t)[1],
            zero(R))
        @test isapprox(dx_r3, Vector(expected_d); rtol = rtol)
    end
end

@testset "Non-SUCCESS primal solve returns primal iterate, NaN partials, primal status, and 0 extra factorisations" begin
    let R = Float64
        p1 = SVector{2, R}(R(4), R(9))
        x0 = SVector{2, R}(R(100), R(200))
        p_d2 = SVector(
            ForwardDiff.Dual{ImplicitTestTagA}(p1[1], one(R), zero(R)),
            ForwardDiff.Dual{ImplicitTestTagA}(p1[2], zero(R), one(R)))
        opt1 = Options(R; max_iterations = 1)

        # MAXITERS in R1
        ls_p = CountingLinear()
        x_p, st_p = solve(PROB_VAL, x0, Newton(; linsolve = ls_p), p1; options = opt1)
        @test st_p.code === MAXITERS

        ls_d = CountingLinear()
        x_d, st_d = solve(PROB_VAL, x0, Newton(; linsolve = ls_d), p_d2; options = opt1)
        @test st_d === st_p
        @test ls_d.fac_calls[] == ls_p.fac_calls[]
        @test ls_d.ldiv_calls[] == ls_p.ldiv_calls[]
        @test Adapt.adapt(StripDual(), x_d) == x_p
        for i in 1:2, k in 1:2

            @test isnan(ForwardDiff.partials(x_d[i], k))
        end

        # MAXITERS in R3
        ls_r3_p = CountingLinear()
        xr_p, str_p = solve(
            PROB_R3, Vector(x0), Newton(; linsolve = ls_r3_p), Vector(p1); options = opt1)
        @test str_p.code === MAXITERS

        ls_r3_d = CountingLinear()
        xr_d, str_d = solve(
            PROB_R3, Vector(x0), Newton(; linsolve = ls_r3_d), collect(p_d2);
            options = opt1)
        @test str_d === str_p
        @test ls_r3_d.fac_calls[] == ls_r3_p.fac_calls[]
        @test Adapt.adapt(StripDual(), xr_d) == xr_p
        for i in 1:2, k in 1:2

            @test isnan(ForwardDiff.partials(xr_d[i], k))
        end

        # STALLED in R1
        ls_sp = CountingLinear()
        xs_p, sts_p = solve(PROB_STALL, x0, Newton(; linsolve = ls_sp), p1)
        @test sts_p.code === STALLED

        ls_sd = CountingLinear()
        xs_d, sts_d = solve(PROB_STALL, x0, Newton(; linsolve = ls_sd), p_d2)
        @test sts_d === sts_p
        @test ls_sd.fac_calls[] == ls_sp.fac_calls[]
        @test Adapt.adapt(StripDual(), xs_d) == xs_p
        for i in 1:2, k in 1:2

            @test isnan(ForwardDiff.partials(xs_d[i], k))
        end
    end
end

@testset "Singular tangent factorisation at x* performs 0 back-substitutions and returns SINGULAR with NaN partials" begin
    let R = Float64
        p1 = SVector{2, R}(R(4), R(9))
        x0 = SVector{2, R}(R(1.5), R(2.5))
        p_d2 = SVector(
            ForwardDiff.Dual{ImplicitTestTagA}(p1[1], one(R), zero(R)),
            ForwardDiff.Dual{ImplicitTestTagA}(p1[2], zero(R), one(R)))
        good_J = PROB_VAL.J

        # Case A: LUFactorization with zero prob.J at x* (R1 and R3)
        ls_lu_p = CountingLinear(LUFactorization())
        x_p, st_p = solve(
            PROB_ZERO_J, x0, Newton(; jacobian = good_J, linsolve = ls_lu_p), p1)
        @test st_p.code === SUCCESS

        ls_lu_d = CountingLinear(LUFactorization())
        x_d, st_d = solve(
            PROB_ZERO_J, x0, Newton(; jacobian = good_J, linsolve = ls_lu_d), p_d2)
        @test st_d.code === SINGULAR
        @test st_d.iterations == st_p.iterations
        @test ls_lu_d.fac_calls[] == ls_lu_p.fac_calls[] + 1
        @test ls_lu_d.ldiv_calls[] == ls_lu_p.ldiv_calls[]
        @test Adapt.adapt(StripDual(), x_d) == x_p
        for i in 1:2, k in 1:2

            @test isnan(ForwardDiff.partials(x_d[i], k))
        end

        ls_r3_p = CountingLinear(LUFactorization())
        xr_p, str_p = solve(
            PROB_R3_ZERO_J, Vector(x0),
            Newton(; jacobian = r3_analytic_j!, linsolve = ls_r3_p), Vector(p1))
        @test str_p.code === SUCCESS

        ls_r3_d = CountingLinear(LUFactorization())
        xr_d, str_d = solve(
            PROB_R3_ZERO_J, Vector(x0),
            Newton(; jacobian = r3_analytic_j!, linsolve = ls_r3_d), collect(p_d2))
        @test str_d.code === SINGULAR
        @test ls_r3_d.fac_calls[] == ls_r3_p.fac_calls[] + 1
        @test ls_r3_d.ldiv_calls[] == ls_r3_p.ldiv_calls[]
        @test Adapt.adapt(StripDual(), xr_d) == xr_p
        for i in 1:2, k in 1:2

            @test isnan(ForwardDiff.partials(xr_d[i], k))
        end

        # Case B: SVDFactorization with Inf entry in prob.J at x* (R1)
        ls_svd_p = CountingLinear(SVDFactorization())
        x_sp, st_sp = solve(
            PROB_INF_J, x0, Newton(; jacobian = good_J, linsolve = ls_svd_p), p1)
        @test st_sp.code === SUCCESS

        ls_svd_d = CountingLinear(SVDFactorization())
        x_sd, st_sd = solve(
            PROB_INF_J, x0, Newton(; jacobian = good_J, linsolve = ls_svd_d), p_d2)
        @test st_sd.code === SINGULAR
        @test ls_svd_d.fac_calls[] == ls_svd_p.fac_calls[] + 1
        @test ls_svd_d.ldiv_calls[] == ls_svd_p.ldiv_calls[]
        @test Adapt.adapt(StripDual(), x_sd) == x_sp
        for i in 1:2, k in 1:2

            @test isnan(ForwardDiff.partials(x_sd[i], k))
        end
    end
end

@testset "Consistent rank-deficient problem with SVDFactorization returns minimum-norm tangent" begin
    let R = Float64
        rtol = 1000 * eps(R)
        x0 = SVector{2, R}(R(1), R(2))
        p_d1 = ForwardDiff.Dual{ImplicitTestTagA}(R(4), one(R))
        x_r1, st_r1 = solve(
            PROB_RD_R1, x0, Newton(; linsolve = SVDFactorization()), p_d1)
        @test st_r1.code === SUCCESS
        dx_r1 = SVector{2, R}(
            ForwardDiff.partials(x_r1[1], 1), ForwardDiff.partials(x_r1[2], 1))
        @test isapprox(dx_r1, SVector{2, R}(R(1) / R(2), R(1) / R(2)); rtol = rtol)

        x_r3, st_r3 = solve(
            PROB_RD_R3, Vector(x0), Newton(; linsolve = SVDFactorization()), p_d1)
        @test st_r3.code === SUCCESS
        dx_r3 = [ForwardDiff.partials(x_r3[1], 1), ForwardDiff.partials(x_r3[2], 1)]
        @test isapprox(dx_r3, R[1 // 2, 1 // 2]; rtol = rtol)
    end
end

@testset "Regularisation, custom m.jacobian, reuse, DogLeg, Picard, and mixed-precision linsolve" begin
    let R = Float64
        rtol = 1000 * eps(R)
        p1 = SVector{2, R}(R(4), R(9))
        x0 = SVector{2, R}(R(1.5), R(2.5))
        expected_d = inv.(2 .* sqrt.(p1))
        opt = Options(R; f_abstol = 100 * eps(R), f_reltol = zero(R))
        p_d1 = SVector(
            ForwardDiff.Dual{ImplicitTestTagA}(p1[1], one(R)),
            ForwardDiff.Dual{ImplicitTestTagA}(p1[2], one(R)))
        p_d1_vec = collect(p_d1)

        for m_r1 in (
            Newton(; regularization = 1 // 10),
            Newton(; jacobian = scaled_j_r1),
            Newton(; reuse = 3),
            Newton(; globalization = DogLeg())
        )
            x_d, st_d = solve(PROB_VAL_NOJ, x0, m_r1, p_d1; options = opt)
            @test st_d.code === SUCCESS
            dx = SVector{2, R}(
                ForwardDiff.partials(x_d[1], 1), ForwardDiff.partials(x_d[2], 1))
            @test isapprox(dx, expected_d; rtol = rtol)
        end

        for m_r3 in (
            Newton(; regularization = 1 // 10),
            Newton(; jacobian = scaled_j_r3!),
            Newton(; reuse = 3),
            Newton(; globalization = DogLeg())
        )
            x_d, st_d = solve(PROB_R3, Vector(x0), m_r3, p_d1_vec; options = opt)
            @test st_d.code === SUCCESS
            dx = [ForwardDiff.partials(x_d[1], 1), ForwardDiff.partials(x_d[2], 1)]
            @test isapprox(dx, Vector(expected_d); rtol = rtol)
        end

        # Picard(; damping = 1//2) in R1 and R3 (with and without prob.J)
        opt_pic = Options(
            R; max_iterations = 500, f_abstol = 10 * eps(R), f_reltol = zero(R))
        for pr in (PROB_PIC_R1, PROB_PIC_R1_J)
            x_pic, st_pic = solve(
                pr, x0, Picard(; damping = 1 // 2), p_d1; options = opt_pic)
            @test st_pic.code === SUCCESS
            dx_pic = SVector{2, R}(
                ForwardDiff.partials(x_pic[1], 1), ForwardDiff.partials(x_pic[2], 1))
            @test isapprox(dx_pic, expected_d; rtol = rtol)
        end

        x_pic_r3, st_pic_r3 = solve(
            PROB_PIC_R3, Vector(x0), Picard(; damping = 1 // 2), p_d1_vec;
            options = opt_pic)
        @test st_pic_r3.code === SUCCESS
        dx_pic_r3 = [
            ForwardDiff.partials(x_pic_r3[1], 1), ForwardDiff.partials(x_pic_r3[2], 1)]
        @test isapprox(dx_pic_r3, Vector(expected_d); rtol = rtol)
    end

    # Mixed-precision LUFactorization(Float32) on Float64 inputs
    p1_64 = SVector{2, Float64}(4.0, 9.0)
    x0_64 = SVector{2, Float64}(1.5, 2.5)
    p_d1_64 = SVector(
        ForwardDiff.Dual{ImplicitTestTagA}(p1_64[1], 1.0),
        ForwardDiff.Dual{ImplicitTestTagA}(p1_64[2], 1.0))
    x_mp, st_mp = solve(
        PROB_VAL, x0_64, Newton(; linsolve = LUFactorization(Float32)), p_d1_64)
    @test st_mp.code === SUCCESS
    dx_mp = SVector(ForwardDiff.partials(x_mp[1], 1), ForwardDiff.partials(x_mp[2], 1))
    @test isapprox(dx_mp, inv.(2 .* sqrt.(p1_64)); rtol = 1000 * eps(Float32))
end

@testset "ArgumentError refusals on nested duals, mixed tags, unadapted structs, captured duals, init, solve!, and solve_batch!" begin
    dA = ForwardDiff.Dual{ImplicitTestTagA}(4.0, 1.0)
    dB = ForwardDiff.Dual{ImplicitTestTagB}(9.0, 1.0)
    d_nested = ForwardDiff.Dual{ImplicitTestTagB}(
        dA, ForwardDiff.Dual{ImplicitTestTagA}(1.0, 0.0))
    x0 = SVector(1.5, 2.5)
    p0 = SVector(4.0, 9.0)
    p_dual = SVector(dA, dA)
    x_dual = SVector(
        ForwardDiff.Dual{ImplicitTestTagA}(1.5, 1.0),
        ForwardDiff.Dual{ImplicitTestTagA}(2.5, 1.0))

    # Nested dual in p or x0
    @test_throws ArgumentError solve(
        PROB_VAL, x0, Newton(), SVector(d_nested, d_nested))
    @test_throws "nested ForwardDiff.Dual" solve(
        PROB_VAL, x0, Newton(), SVector(d_nested, d_nested))
    @test_throws ArgumentError solve(
        PROB_VAL, SVector(d_nested, d_nested), Newton(), p0)
    @test_throws "nested ForwardDiff.Dual" solve(
        PROB_VAL, SVector(d_nested, d_nested), Newton(), p0)
    @test_throws ArgumentError solve(
        PROB_R3, Vector(x0), Newton(), [d_nested, d_nested])
    @test_throws "nested ForwardDiff.Dual" solve(
        PROB_R3, Vector(x0), Newton(), [d_nested, d_nested])

    # Two distinct tags between x0 and p, or within p
    @test_throws ArgumentError solve(
        PROB_VAL, x_dual, Newton(), SVector(dB, dB))
    @test_throws "single ForwardDiff.Dual tag" solve(
        PROB_VAL, x_dual, Newton(), SVector(dB, dB))
    prob_tup = NonlinearProblem((x, p) -> SVector(x[1]^2 - p[1], x[2]^2 - p[2]))
    @test_throws ArgumentError solve(prob_tup, x0, Newton(), (dA, dB))
    @test_throws "single ForwardDiff.Dual tag" solve(prob_tup, x0, Newton(), (dA, dB))

    # Dual inside a parameter struct without Adapt.@adapt_structure
    prob_unadapt = NonlinearProblem((x, p) -> SVector(x[1]^2 - p.a, x[2]^2 - p.a))
    @test_throws ArgumentError solve(
        prob_unadapt, x0, Newton(), UnadaptedParam(dA))
    @test_throws "Adapt.adapt(StripDual()" solve(
        prob_unadapt, x0, Newton(), UnadaptedParam(dA))
    prob_unadapt_primal_ok = NonlinearProblem(
        (x, p) -> SVector(x[1]^2 - ForwardDiff.value(p.a), x[2]^2 - 2.0),
        (x, p) -> SMatrix{2, 2, eltype(x)}(
            2 * x[1], zero(eltype(x)), zero(eltype(x)), 2 * x[2]))
    @test_throws ArgumentError solve(
        prob_unadapt_primal_ok, x0, Newton(), UnadaptedParam(dA))
    @test_throws "Adapt.adapt(StripDual()" solve(
        prob_unadapt_primal_ok, x0, Newton(), UnadaptedParam(dA))

    # options together with keyword tolerances, refused as on primal inputs
    opt_both = Options(Float64; f_abstol = 1e-14)
    @test_throws "not both" solve(
        PROB_VAL, x0, Newton(), p_dual; options = opt_both, f_reltol = 0.0)
    @test_throws "not both" solve(
        PROB_R3, Vector(x0), Newton(), collect(p_dual); options = opt_both, f_reltol = 0.0)

    # Dual captured inside prob.F (typed closure and untyped container), refused by solve!! when
    # it reaches the residual; an R3 residual fails on the write into its primal buffer
    prob_cap = NonlinearProblem((x, p) -> SVector(x[1]^2 - dA, x[2]^2 - dA))
    @test_throws ArgumentError solve(prob_cap, x0, Newton(), 2.0)
    @test_throws "solve!! residual contains a ForwardDiff.Dual" solve(
        prob_cap, x0, Newton(), 2.0)
    @test_throws ArgumentError solve(prob_cap, x0, Newton(), dA)
    @test_throws "solve!! residual contains a ForwardDiff.Dual" solve(
        prob_cap, x0, Newton(), dA)
    prob_cap_r3 = NonlinearProblem((r, x, p) -> (r .= x .^ 2 .- dA; r))
    @test_throws MethodError solve(prob_cap_r3, Vector(x0), Newton(), [2.0, 2.0])

    r_box = Ref{Any}(dA)
    prob_box = NonlinearProblem((x, p) -> SVector(x[1]^2 - r_box[], x[2]^2 - 2.0))
    @test_throws ArgumentError solve(prob_box, x0, Newton(), 2.0)
    @test_throws "solve!! residual contains a ForwardDiff.Dual" solve(
        prob_box, x0, Newton(), 2.0)
    @test_throws ArgumentError solve(prob_box, x0, Newton(), dA)
    @test_throws "solve!! residual contains a ForwardDiff.Dual" solve(
        prob_box, x0, Newton(), dA)

    # a residual that is primal at x0 and contains a captured Dual once x[1] ≥ 1.9, which the
    # first step reaches: refused at the residual that the step evaluates, under every
    # globalization of Newton and Picard
    prob_late = NonlinearProblem(
        (x, p) -> x[1] < 1.9 ? SVector(x[1]^2 - p[1], x[2]^2 - p[2]) :
                  SVector(x[1]^2 - dA, x[2]^2 - p[2]))
    prob_late_sc = NonlinearProblem((x, p) -> x < 1.9 ? x^2 - p : x^2 - dA)
    @test prob_late.F(x0, p0) isa SVector{2, Float64}
    for m in (Newton(), Newton(; globalization = Backtracking()),
        Newton(; globalization = StrongWolfe()), Newton(; globalization = DogLeg()),
        Newton(; globalization = LevenbergMarquardt()),
        Newton(; linsolve = SVDFactorization(), globalization = LevenbergMarquardt()),
        Picard(), Picard(; globalization = Backtracking()))
        @test_throws "solve!! residual contains a ForwardDiff.Dual" solve(
            prob_late, x0, m, p0)
        @test_throws "solve!! residual contains a ForwardDiff.Dual" solve(
            prob_late, x0, m, p_dual)
        @test_throws "solve!! residual contains a ForwardDiff.Dual" solve(
            prob_late_sc, 1.5, m, 4.0)
        @test_throws "solve!! residual contains a ForwardDiff.Dual" solve!(
            init(prob_late, x0, m, p0))
    end

    # init and solve! refuse dual x0, x, or p
    @test_throws ArgumentError init(PROB_R3, collect(x_dual), Newton(), collect(p0))
    @test_throws "use solve(" init(PROB_R3, collect(x_dual), Newton(), collect(p0))
    @test_throws ArgumentError init(PROB_R3, collect(x0), Newton(), collect(p_dual))
    @test_throws "use solve(" init(PROB_R3, collect(x0), Newton(), collect(p_dual))
    solver_cap = init(prob_cap, x0, Newton(), 2.0)
    @test_throws ArgumentError solve!(solver_cap)
    @test_throws "solve!! residual contains a ForwardDiff.Dual" solve!(solver_cap)
    solver = init(PROB_R3, collect(x0), Newton(), collect(p0))
    @test_throws ArgumentError solve!(solver, collect(x_dual), collect(p0))
    @test_throws "use solve(" solve!(solver, collect(x_dual), collect(p0))
    @test_throws ArgumentError solve!(solver, collect(x0), collect(p_dual))
    @test_throws "use solve(" solve!(solver, collect(x0), collect(p_dual))
    solver_r1 = init(PROB_VAL, x0, Newton(), p0)
    @test_throws ArgumentError solve!(solver_r1, x_dual, p0)
    @test_throws "use solve(" solve!(solver_r1, x_dual, p0)
    @test_throws ArgumentError solve!(solver_r1, x0, p_dual)
    @test_throws "use solve(" solve!(solver_r1, x0, p_dual)

    # solve_batch! refuses dual X or p in both 4-arg and 5-arg forms, including PerSystem
    statuses = Vector{SolverStatus{Float64}}(undef, 1)
    @test_throws ArgumentError solve_batch!(
        [x_dual], statuses, PROB_VAL, Newton(), p0)
    @test_throws "use solve(" solve_batch!(
        [x_dual], statuses, PROB_VAL, Newton(), p0)
    @test_throws ArgumentError solve_batch!(
        [x0], statuses, PROB_VAL, Newton(), p_dual)
    @test_throws "use solve(" solve_batch!(
        [x0], statuses, PROB_VAL, Newton(), p_dual)
    @test_throws ArgumentError solve_batch!(
        [x0], statuses, PROB_VAL, Newton(), PerSystem([p_dual]))
    @test_throws "use solve(" solve_batch!(
        [x0], statuses, PROB_VAL, Newton(), PerSystem([p_dual]))
    @test_throws ArgumentError solve_batch!(
        [x_dual], PROB_VAL, Newton(), p0)
    @test_throws "use solve(" solve_batch!(
        [x_dual], PROB_VAL, Newton(), p0)
    @test_throws ArgumentError solve_batch!(
        [x0], PROB_VAL, Newton(), p_dual)
    @test_throws "use solve(" solve_batch!(
        [x0], PROB_VAL, Newton(), p_dual)
    @test_throws ArgumentError solve_batch!(
        [x0], PROB_VAL, Newton(), PerSystem([p_dual]))
    @test_throws "use solve(" solve_batch!(
        [x0], PROB_VAL, Newton(), PerSystem([p_dual]))
end

@testset "Primal solve, init, solve! and solve_batch! accept a prob that captures a Dual-typed object" begin
    x0 = SVector(1.5, 2.5)
    p0 = SVector(4.0, 9.0)
    # the closed form holds to the default stopping tolerances; the solve of the same residual
    # without the capture holds to a few ULPs
    x_ref = sqrt.(p0)
    rtol_ref = sqrt(eps(Float64))
    rtol = 8 * eps(Float64)

    # an R1 F that reads only the primal value of a captured Dual work buffer
    buf = zeros(ForwardDiff.Dual{ImplicitTestTagA, Float64, 1}, 2)
    prob_buf = NonlinearProblem((x, p) -> x .^ 2 .- p .+ ForwardDiff.value(buf[1]))
    x_base, st_base = solve(PROB_VAL_NOJ, x0, Newton(), p0)
    x, st = solve(prob_buf, x0, Newton(), p0)
    @test st.code === SUCCESS
    @test st.iterations == st_base.iterations
    @test isapprox(x, x_ref; rtol = rtol_ref)
    @test isapprox(x, x_base; rtol)
    solver = init(prob_buf, x0, Newton(), p0)
    x, st = solve!(solver)
    @test st.code === SUCCESS
    @test isapprox(x, x_base; rtol)
    X = [x0, x0]
    statuses = Vector{SolverStatus{Float64}}(undef, 2)
    solve_batch!(X, statuses, prob_buf, Newton(), p0)
    @test all(s -> s.code === SUCCESS, statuses)
    @test all(xi -> isapprox(xi, x_base; rtol), X)
    X = [x0, x0]
    @test solve_batch!(X, prob_buf, Newton(), p0).SUCCESS == 2
    @test all(xi -> isapprox(xi, x_base; rtol), X)

    # an R3 F that calls ForwardDiff.gradient with a captured GradientConfig
    g = x -> sum(abs2, x)
    cfg = ForwardDiff.GradientConfig(g, zeros(2))
    prob_cfg = NonlinearProblem(
        (r, x, p) -> (r .= ForwardDiff.gradient(g, x, cfg) .* x ./ 2 .- p; r),
        r3_analytic_j!)
    x_base, st_base = solve(PROB_R3_J, Vector(x0), Newton(), Vector(p0))
    x, st = solve(prob_cfg, Vector(x0), Newton(), Vector(p0))
    @test st.code === SUCCESS
    @test st.iterations == st_base.iterations
    @test isapprox(x, x_ref; rtol = rtol_ref)
    @test isapprox(x, x_base; rtol)
    solver = init(prob_cfg, Vector(x0), Newton(), Vector(p0))
    x, st = solve!(solver)
    @test st.code === SUCCESS
    @test isapprox(x, x_base; rtol)
end

@testset "R1 dual solve is @inferred, JET.@test_opt clean, and zero-allocating" begin
    for R in REAL_ELTYPES
        x0 = SVector{2, R}(R(1.5), R(2.5))
        d1 = ForwardDiff.Dual{ImplicitTestTagA}(R(4), one(R), zero(R))
        d2 = ForwardDiff.Dual{ImplicitTestTagA}(R(9), zero(R), one(R))
        p_d = SVector(d1, d2)

        res = @inferred solve(PROB_VAL, x0, Newton(), p_d)
        @test res[2].code === SUCCESS

        res_noj = @inferred solve(PROB_VAL_NOJ, x0, Newton(), p_d)
        @test res_noj[2].code === SUCCESS

        res_sc = @inferred solve(PROB_SCALAR, R(1.5), Newton(), d1)
        @test res_sc[2].code === SUCCESS

        if R === Float64
            JET.@test_opt target_modules = (GeometricSolvers,) solve(
                PROB_VAL, x0, Newton(), p_d)
        end

        # Skipped where a run forces `--check-bounds=yes`
        if Base.JLOptions().check_bounds != 1
            @test dual_r1_allocated(PROB_VAL, x0, Newton(), p_d) == 0
            @test dual_r1_allocated(PROB_VAL_NOJ, x0, Newton(), p_d) == 0
            @test dual_r1_allocated(PROB_SCALAR, R(1.5), Newton(), d1) == 0
        end
    end
end
