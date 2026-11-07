# `LevenbergMarquardt`, the damped-step globalization of `Newton`: the parameters and their
# validation, the adapted damping, the rule state and its fail-closed initialisation, the damped
# step against the damped normal equations in every element type and on both array ranks, the one
# factorisation per Jacobian across rejected trials, the one triangularisation of `[J | -F]` per
# outer step of the stacked solve, the truncation of the ranks the factor drops, the damping in the
# units of `Jᴴ J` at both ends of the scale and around the `4 eps` floor, the gain ratio at a
# positive `ρ_accept` and at a trial that leaves the residual where it was, the `STALLED` return at
# `μ_max` and the finite damping it keeps there, the rejection of a non-finite trial residual and
# of a non-finite damped step, the refusal of the stacked solve on a device array and of a sparse
# `jacobian_prototype`, the stacked step against the closed form in `Complex{BigFloat}`, the
# standard reference column and the eleven cells that the standard set excepts, the storage and
# allocations of the state and of every path, and the solve of a `StationarityProblem`.
#
# The cases that need the step in closed form take a linear problem with an analytic `J` and one
# `step!!`, so that the expected `δ` is the solution of `(Jᴴ J + μ D²) δ = -Jᴴ F` computed
# directly, in the working precision or in `Complex{BigFloat}`, with `D` the identity or, under
# `scaling = :diagonal`, the diagonal of the floored column norms. `μ` is absolute, measured in the
# units of `Jᴴ J`, so the reference takes it as it is and a factor's power-of-two scale reaches the
# step only through `Jᴴ J`.

using Adapt: Adapt
using GPUArraysCore: allowscalar
using JET: JET, @test_opt
using JLArrays: JLArray
using LinearAlgebra: I, cond, mul!, norm
using SparseArrays: sparse
using StaticArrays: @SMatrix, @SVector, SMatrix, SVector
using Test

using GeometricSolvers
using GeometricSolvers: LinearMethod, finish!!, init_state, lm_damping_diagonal,
                        lm_svd_coord, norm2, rdot, rnorm, start!!, step!!

include("../helpers/matrix.jl")
include("../helpers/standard_cells.jl")

allowscalar(false)

# An allocation measurement through a function barrier, which calls the function once before it
# measures, and `allocations(collect, x)` is the control that shows the helper sees an allocation.
allocations(f::F, a::A) where {F, A} = (f(a); @allocated f(a))

# A linear method that counts its factorisations, so that a test can see how many times a step
# factorises `J`: the SVD step factorises once per outer step, however many trials it rejects.
# Every other call is the wrapped method's.
struct CountingFactorization{M <: LinearMethod} <: LinearMethod{Nothing}
    method::M
    count::Base.RefValue{Int}
end

GeometricSolvers.lm_svd_path(::CountingFactorization, ::Val{:none}) = true
function GeometricSolvers.factorize!!(fac, m::CountingFactorization, J)
    m.count[] += 1
    return GeometricSolvers.factorize!!(fac, m.method, J)
end
function GeometricSolvers.init_linear(m::CountingFactorization, pattern, x::AbstractVector)
    GeometricSolvers.init_linear(m.method, pattern, x)
end

# A linear method that counts the triangularisation of `[J | -F]`, the `O(n³)` reduction that an
# outer step of the stacked solve does once. That path never calls `factorize!!`, so the count is
# on the hook the reduction goes through.
struct CountingStacked{M <: LinearMethod} <: LinearMethod{Nothing}
    method::M
    count::Base.RefValue{Int}
end

function GeometricSolvers.lm_stacked_factor!(m::CountingStacked, R₀, c₀, J, Fx)
    m.count[] += 1
    return GeometricSolvers.lm_qr_factor!(R₀, c₀, J, Fx)
end
function GeometricSolvers.init_linear(m::CountingStacked, pattern, x::AbstractVector)
    GeometricSolvers.init_linear(m.method, pattern, x)
end

# A dense Jacobian with no zero entry: the imaginary part is separated from the real one per entry.
# `N` is a `Val` so that the matrix is a concrete `SMatrix` at the call site, which an allocation
# measurement needs.
function lm_test_matrix(::Val{N}, ::Type{T}) where {N, T}
    return SMatrix{N, N, T}(ntuple(
        i -> T((i * 7) % 5 + 2) +
             (T <: Complex ? T(im) * T((i * 3) % 4 + 1) : zero(T)),
        Val(N * N)))
end

# The right-hand side `b` of the linear system `F(x) = J x - b`.
function lm_test_rhs(::Val{N}, ::Type{T}) where {N, T}
    return SVector{N, T}(ntuple(i -> T(i) - T(N) / 2, Val(N)))
end

# The Jacobian of the scalar case is negative on a real element type and complex on a complex one:
# the reduction of the 1×1 system keeps the entry's sign in the pivot that `givens` returns, and
# the trial divides by that pivot, so `abs` of the pivot and `|J|` of the entry differ from them.
lm_test_matrix(::Val{1}, ::Type{T}) where {T <: Real} = SMatrix{1, 1, T}((-T(4),))
function lm_test_matrix(::Val{1}, ::Type{T}) where {T <: Complex}
    SMatrix{1, 1, T}((T(-4) + T(4im),))
end

# `δ(μ) = -(Jᴴ J + μ D²)⁻¹ Jᴴ F` for the damping diagonal `d`, computed in `Complex{BigFloat}`: the
# reference that carries none of the step's own rounding, so that the tolerance left is the step's
# and is measured against `eps(real(T)) · κ(J)`.
function lm_stacked_reference(J, F, μ, d)
    B = Complex{BigFloat}
    n = size(J, 1)
    Jb = [B(J[i, j]) for i in 1:n, j in 1:n]
    Fb = [B(F[i]) for i in 1:n]
    A = Jb' * Jb
    for i in 1:n
        A[i, i] += B(μ) * B(d[i])^2
    end
    return -(A \ (Jb' * Fb))
end

# The error of one stacked step against `lm_stacked_reference`, relative to `‖δ‖ · eps(R) · κ(J)`.
function lm_stacked_error(δ, δ_ref, J, ::Type{R}) where {R}
    scale = Float64(norm(δ_ref)) * eps(R) * cond(ComplexF64.(J))
    return norm(ComplexF64.(δ) - ComplexF64.(δ_ref)) / scale
end

# The iterate of the `N`-dimensional case: a `Number` for `N = 1`, which is the scalar path of the
# stacked solve, and an `SVector` above it.
lm_test_iterate(::Val{1}, ::Type{T}) where {T} = zero(T)
lm_test_iterate(::Val{N}, ::Type{T}) where {N, T} = zero(SVector{N, T})

# One accepted trial of the stacked solve on the dense system of `lm_test_matrix`, with its
# `StepInfo` and the error of its step against the closed form.
function lm_stacked_case(::Val{N}, ::Type{T}, ls, sc, μ₀) where {N, T}
    R = real(T)
    J = lm_test_matrix(Val(N), T)
    b = lm_test_rhs(Val(N), T)
    prob = NonlinearProblem((x, _) -> J * x - b, (_, _) -> J)
    m = Newton(;
        linsolve = ls,
        globalization = LevenbergMarquardt(R; μ₀ = μ₀, μ_min = μ₀, scaling = sc))
    x0 = lm_test_iterate(Val(N), T)
    Fx = prob.F(x0, nothing)
    st = start!!(init_state(m, prob, x0, nothing), m, prob, x0, Fx, nothing, rnorm(Fx))
    x1, _, _, info = step!!(st, m, prob, x0, Fx, nothing)
    d_diag = lm_damping_diagonal(J, Val(sc))
    return info, lm_stacked_error(x1 - x0, lm_stacked_reference(J, Fx, μ₀, d_diag), J, R)
end

# The same for the scalar iterate of `Val(1)`, whose Jacobian, residual and step are numbers and
# whose factor is the `1×1` matrix the reduction of `[J | -F]` leaves.
function lm_stacked_case(::Val{1}, ::Type{T}, ls, sc, μ₀) where {T}
    R = real(T)
    J = lm_test_matrix(Val(1), T)[1, 1]
    b = lm_test_rhs(Val(1), T)[1]
    prob = NonlinearProblem((x, _) -> J * x - b, (_, _) -> J)
    m = Newton(;
        linsolve = ls,
        globalization = LevenbergMarquardt(R; μ₀ = μ₀, μ_min = μ₀, scaling = sc))
    x0 = zero(T)
    Fx = prob.F(x0, nothing)
    st = start!!(init_state(m, prob, x0, nothing), m, prob, x0, Fx, nothing, rnorm(Fx))
    x1, _, _, info = step!!(st, m, prob, x0, Fx, nothing)
    Jm = SMatrix{1, 1, T}((J,))
    d_diag = lm_damping_diagonal(Jm, Val(sc))
    # `lm_stacked_error` measures over the iterate's coordinate vector, and the scalar step is its
    # one coordinate.
    err = lm_stacked_error(
        [x1 - x0], lm_stacked_reference(Jm, SVector{1, T}((Fx,)), μ₀, d_diag), Jm, R)
    return info, err
end

# The same trial on an array iterate, whose factor and elimination buffers are the state's.
function lm_stacked_case_r3(::Val{N}, ::Type{T}, sc) where {N, T}
    R = real(T)
    J = lm_test_matrix(Val(N), T)
    b = lm_test_rhs(Val(N), T)
    Jd, bd = Matrix(J), Vector(b)
    prob = NonlinearProblem(
        (F, x, _) -> (mul!(F, Jd, x); F .-= bd; nothing),
        (Jm, _, _) -> (copyto!(Jm, Jd); nothing))
    m = Newton(;
        linsolve = QRFactorization(),
        globalization = LevenbergMarquardt(R; μ₀ = 1.0, μ_min = 1.0, scaling = sc))
    solver = init(prob, zeros(T, N), m)
    prob.F(solver.state.Fx, solver.x, nothing)
    st = start!!(
        solver.state, m, prob, solver.x, solver.state.Fx, nothing, rnorm(solver.state.Fx))
    x1, _, _, info = step!!(st, m, prob, solver.x, solver.state.Fx, nothing)
    d_diag = lm_damping_diagonal(J, Val(sc))
    return info, lm_stacked_error(
        Array(x1), lm_stacked_reference(J, -b, 1.0, d_diag), J, R)
end

# The bytes one `step!!` on a value iterate allocates, through a function barrier: `N` and `T` are
# static parameters, so the barrier's argument has a concrete type.
function lm_r1_allocations(::Val{N}, ::Type{T}, ls, sc) where {N, T}
    J = lm_test_matrix(Val(N), T)
    b = lm_test_rhs(Val(N), T)
    prob = NonlinearProblem((x, _) -> J * x - b, (_, _) -> J)
    m = Newton(; linsolve = ls, globalization = LevenbergMarquardt(real(T); scaling = sc))
    x0 = lm_test_iterate(Val(N), T)
    Fx = prob.F(x0, nothing)
    st = start!!(init_state(m, prob, x0, nothing), m, prob, x0, Fx, nothing, rnorm(Fx))
    args = (st, m, prob, x0, Fx, nothing)
    return allocations(t -> step!!(t[1], t[2], t[3], t[4], t[5], t[6]), args)
end

# The same for a scalar iterate, where the Jacobian is the one entry and not a `1×1` matrix.
function lm_r1_allocations(::Val{1}, ::Type{T}, ls, sc) where {T}
    J = lm_test_matrix(Val(1), T)[1, 1]
    b = lm_test_rhs(Val(1), T)[1]
    prob = NonlinearProblem((x, _) -> J * x - b, (_, _) -> J)
    m = Newton(; linsolve = ls, globalization = LevenbergMarquardt(real(T); scaling = sc))
    x0 = zero(T)
    Fx = prob.F(x0, nothing)
    st = start!!(init_state(m, prob, x0, nothing), m, prob, x0, Fx, nothing, rnorm(Fx))
    args = (st, m, prob, x0, Fx, nothing)
    return allocations(t -> step!!(t[1], t[2], t[3], t[4], t[5], t[6]), args)
end

# The damped step `(Jᴴ J + μ I) δ = -Jᴴ F` solved directly, for a linear problem `F = J x - b`
# from `x₀ = 0` at the start of a solve. `μ` is in the units of `Jᴴ J`.
function damped_reference(A, b, μ)
    return -((transpose(A) * A + μ * SMatrix{2, 2, eltype(A)}(I)) \
             (transpose(A) * (-b)))
end

# The same with Marquardt's column scaling, `D = diag(dᵢ²)`, `dᵢ = max(‖A[:, i]‖, √eps · maxⱼ dⱼ)`.
function damped_reference_diagonal(A, b, μ)
    R = real(eltype(A))
    d = [norm(A[:, i]) for i in 1:2]
    floor = iszero(maximum(d)) ? one(R) : sqrt(eps(R)) * maximum(d)
    D = SMatrix{2, 2, R}(max(d[1], floor)^2, 0, 0, max(d[2], floor)^2)
    return -((transpose(A) * A + μ * D) \ (transpose(A) * (-b)))
end

@testset "LevenbergMarquardt" begin
    @testset "Constructor, validation and Adapt (MU9)" begin
        lm = LevenbergMarquardt()
        @test lm.μ₀ === 1e-6
        @test lm.μ_min === max(1e-15, 4 * eps(Float64))
        @test lm.μ_max === 1e12
        @test lm.μ_decrease === 1.0 / 3.0
        @test lm.μ_increase === 4.0
        @test lm.ρ_accept === 0.0
        @test lm.scaling === Val(:none)

        lm32 = LevenbergMarquardt(Float32; μ₀ = 1e-4, scaling = :diagonal)
        @test lm32 isa LevenbergMarquardt{Float32}
        @test lm32.μ₀ === 1.0f-4
        @test lm32.μ_min === 4 * eps(Float32)
        @test lm32.scaling === Val(:diagonal)

        @test_throws ArgumentError LevenbergMarquardt(; μ_min = 0.0)
        @test_throws ArgumentError LevenbergMarquardt(; μ_min = -1e-12)
        @test_throws ArgumentError LevenbergMarquardt(; μ₀ = 1e-16, μ_min = 1e-12)
        @test_throws ArgumentError LevenbergMarquardt(; μ₀ = 1e13, μ_max = 1e12)
        @test_throws ArgumentError LevenbergMarquardt(; μ_max = Inf)
        @test_throws ArgumentError LevenbergMarquardt(; μ_decrease = 0.0)
        @test_throws ArgumentError LevenbergMarquardt(; μ_decrease = 1.0)
        @test_throws ArgumentError LevenbergMarquardt(; μ_increase = 1.0)
        @test_throws ArgumentError LevenbergMarquardt(; ρ_accept = -0.1)
        @test_throws ArgumentError LevenbergMarquardt(; ρ_accept = 1.0)
        @test_throws ArgumentError LevenbergMarquardt(; scaling = :bad)
        @test_throws ArgumentError Newton(; globalization = LevenbergMarquardt(), reuse = 2)

        # MU9: without `Adapt.@adapt_structure` the fields stay `Float64` under `ToReal{Float32}()`
        adapted = Adapt.adapt(GeometricSolvers.ToReal{Float32}(), lm)
        @test adapted isa LevenbergMarquardt{Float32}
        @test adapted.μ₀ === Float32(1e-6)
        @test adapted.μ_decrease === Float32(1.0 / 3.0)
        @test adapted.scaling === Val(:none)
    end

    @testset "Damped step against the damped normal equations in ELTYPES" begin
        for R in REAL_ELTYPES
            A = @SMatrix [R(3) R(1); R(1) R(2)]
            b = @SVector [R(4), R(3)]
            μ0 = R(1e-2)
            prob = NonlinearProblem((x, _) -> A * x - b, (_, _) -> A)
            opts = Options(R; f_abstol = R(1e-30), f_reltol = R(0), max_iterations = 1)
            δ_none = damped_reference(A, b, μ0)
            δ_diag = damped_reference_diagonal(A, b, μ0)
            for (ls, D) in ((SVDFactorization(), δ_none), (QRFactorization(), δ_none),
                (LUFactorization(), δ_none))
                x, _ = solve(
                    prob, zero(SVector{2, R}),
                    Newton(;
                        linsolve = ls,
                        globalization = LevenbergMarquardt(R; μ₀ = μ0, μ_min = μ0)),
                    nothing; options = opts)
                @test isapprox(x, D; rtol = 100 * eps(R))
            end
            # `scaling = :diagonal` takes the stacked QR path for every linear method.
            for ls in (SVDFactorization(), QRFactorization(), LUFactorization())
                x, _ = solve(
                    prob, zero(SVector{2, R}),
                    Newton(;
                        linsolve = ls,
                        globalization = LevenbergMarquardt(R; μ₀ = μ0, μ_min = μ0,
                            scaling = :diagonal)),
                    nothing; options = opts)
                @test isapprox(x, δ_diag; rtol = 100 * eps(R))
            end
        end
        # A zero Jacobian column still receives positive damping under `:diagonal`.
        A = @SMatrix [3.0 0.0; 1.0 0.0]
        b = @SVector [4.0, 3.0]
        prob = NonlinearProblem((x, _) -> A * x - b, (_, _) -> A)
        x, _ = solve(
            prob, zero(SVector{2, Float64}),
            Newton(;
                linsolve = QRFactorization(),
                globalization = LevenbergMarquardt(; μ₀ = 1e-2, μ_min = 1e-2,
                    scaling = :diagonal)),
            nothing; options = Options(Float64; f_abstol = 1e-30, f_reltol = 0.0,
                max_iterations = 1))
        @test all(isfinite, x)
        @test isapprox(x, damped_reference_diagonal(A, b, 1e-2); rtol = 1e-10)

        # The same zero column on R3, where the stacked QR reads an `Array` Jacobian: without the
        # floor its damping row is zero, the back-substitution meets a zero pivot and the trial is
        # a failure.
        prob_d = NonlinearProblem(
            (F, x, _) -> (F .= A * x .- b; nothing),
            (J, _, _) -> (J .= A; nothing))
        x_d, _ = solve(
            prob_d, zeros(2),
            Newton(;
                linsolve = QRFactorization(),
                globalization = LevenbergMarquardt(; μ₀ = 1e-2, μ_min = 1e-2,
                    scaling = :diagonal)),
            nothing; options = Options(Float64; f_abstol = 1e-30, f_reltol = 0.0,
                max_iterations = 1))
        @test all(isfinite, x_d)
        @test isapprox(Vector(x_d), Vector(damped_reference_diagonal(A, b, 1e-2));
            rtol = 1e-10)

        # D2: a zero Jacobian floors every column norm at `one(R)`.
        @test lm_damping_diagonal(zero(SMatrix{2, 2, Float64}), Val(:diagonal)) ==
              SVector(1.0, 1.0)

        # D2's floor is `√eps(R) · maxⱼ dⱼ`, and not `eps(R) · maxⱼ dⱼ`: a column norm between the
        # two lies on the scale of the matrix and still takes the floor.
        @test lm_damping_diagonal(@SMatrix([1.0 0.0; 0.0 1e-12]), Val(:diagonal)) ==
              SVector(1.0, sqrt(eps(Float64)))
        @test lm_damping_diagonal(@SMatrix([1.0e8 0.0; 0.0 1.0e-4]), Val(:diagonal)) ==
              SVector(1.0e8, sqrt(eps(Float64)) * 1.0e8)
        # A column norm above the floor is kept as it is.
        @test lm_damping_diagonal(@SMatrix([1.0 0.0; 0.0 0.5]), Val(:diagonal)) ==
              SVector(1.0, 0.5)

        # A scalar iterate takes the 1×1 stacked solve: `(J² + μ d²) δ = J b` for `F = J x - b`.
        for ls in (QRFactorization(), LUFactorization()), sc in (:none, :diagonal)

            J, b, μ0 = 3.0, 4.0, 1e-2
            prob = NonlinearProblem((x, _) -> J * x - b, (_, _) -> J)
            x, _ = solve(
                prob, 0.0,
                Newton(;
                    linsolve = ls,
                    globalization = LevenbergMarquardt(; μ₀ = μ0, μ_min = μ0,
                        scaling = sc)),
                nothing; options = Options(Float64; f_abstol = 1e-30, f_reltol = 0.0,
                    max_iterations = 1))
            d = sc === :none ? 1.0 : J
            @test isapprox(x, J * b / (J^2 + μ0 * d^2); rtol = 1e-12)
        end
    end

    @testset "Check 1: quadratic tail at μ_min in REAL_ELTYPES (MU1)" begin
        for R in REAL_ELTYPES
            prob = NonlinearProblem((x, _) -> SVector(x[1]^2 - 2, x[2]^2 - 3))
            # `μ_min` below the floor `4 eps(R)` of the schedule, so the floor is what the damping
            # reaches and not the field.
            μ_bottom = 4 * eps(R)
            m = Newton(;
                linsolve = SVDFactorization(),
                globalization = LevenbergMarquardt(R; μ₀ = 10 * eps(R),
                    μ_min = eps(R) / 16))
            x = SVector{2, R}(2, 2)
            Fx = prob.F(x, nothing)
            st = init_state(m, prob, x, nothing)
            st = start!!(st, m, prob, x, Fx, nothing, rnorm(Fx))
            fnorms = R[rnorm(Fx)]
            mus = R[]
            for _ in 1:30
                x, Fx, st, info = step!!(st, m, prob, x, Fx, nothing)
                push!(fnorms, info.fnorm)
                push!(mus, st.rule.μ)
                info.fnorm <= 100 * eps(R) && break
            end
            @test length(mus) >= 2
            # MU1: a damping that is not lowered after an accepted step stays at `μ₀`
            @test all(==(μ_bottom), mus[(end - 2):end])
            # the residual falls quadratically from the first step at the floor onwards
            for k in max(1, length(mus) - 2):(length(fnorms) - 1)
                fnorms[k] <= 100 * eps(R) && continue
                # the last step of a converged solve sits at the rounding level of `R`
                fnorms[k + 1] <= 100 * eps(R) && continue
                @test fnorms[k + 1] <= 2 * fnorms[k]^2
            end
        end
    end

    @testset "Check 2: rank-one consistent system in R1 and R3 (MU2)" begin
        prob_r1 = NonlinearProblem(
            (x, _) -> SVector(x[1] + x[2] - 2, 2 * (x[1] + x[2]) - 4),
            (_, _) -> @SMatrix [1.0 1.0; 2.0 2.0 + 1e-16])
        x_r1, st_r1 = solve(
            prob_r1, SVector(0.0, 0.0),
            Newton(; linsolve = SVDFactorization(), globalization = LevenbergMarquardt()))
        @test st_r1.code == SUCCESS
        @test norm(x_r1 - SVector(1.0, 1.0)) <= 1e-6

        prob_r3 = NonlinearProblem(
            (F, x, _) -> (F[1] = x[1] + x[2] - 2; F[2] = 2 * (x[1] + x[2]) - 4; nothing),
            (J, _, _) -> (J[1, 1] = 1; J[1, 2] = 1; J[2, 1] = 2; J[2, 2] = 2 + 1e-16;
                nothing))
        x_r3, st_r3 = solve(
            prob_r3, [0.0, 0.0],
            Newton(; linsolve = SVDFactorization(), globalization = LevenbergMarquardt()))
        @test st_r3.code == SUCCESS
        @test norm(x_r3 - [1.0, 1.0]) <= 1e-6
    end

    @testset "Check 2b: a rank the factor drops is not stepped along (MU2)" begin
        # `J = diag(1, 1e-16)` is rank one at the cutoff `√n eps σ₁`, and the residual has a
        # component along the dropped left singular vector: the step must not move along it.
        J1 = @SMatrix [1.0 0.0; 0.0 1e-16]
        b1 = @SVector [1.0, 1.0]
        prob = NonlinearProblem((F, x, _) -> (F .= J1 * x .- b1; nothing),
            (J, _, _) -> (J .= J1; nothing))
        m = Newton(;
            linsolve = SVDFactorization(),
            globalization = LevenbergMarquardt(; μ₀ = 1e-14, μ_min = 1e-14))
        solver = init(prob, [0.0, 0.0], m)
        prob.F(solver.state.Fx, solver.x, nothing)
        state = start!!(
            solver.state, solver.method, solver.prob, solver.x, solver.state.Fx, nothing,
            rnorm(solver.state.Fx))
        x1, _, _, info = step!!(
            state, solver.method, solver.prob, solver.x, solver.state.Fx, nothing)
        @test info.code == SUCCESS
        @test isapprox(x1[1], 1.0; atol = 1e-9)
        @test abs(x1[2]) <= 1e-6

        # The same on R1: the static factor zeroes the column beyond its rank, so this pins the
        # truncated step of the value-iterate path against the closed form.
        prob_r1 = NonlinearProblem((x, _) -> J1 * x - b1, (_, _) -> J1)
        m_r1 = Newton(;
            linsolve = SVDFactorization(),
            globalization = LevenbergMarquardt(; μ₀ = 1e-14, μ_min = 1e-14))
        x0_r1 = zero(SVector{2, Float64})
        Fx0_r1 = prob_r1.F(x0_r1, nothing)
        st_r1 = start!!(
            init_state(m_r1, prob_r1, x0_r1, nothing), m_r1, prob_r1, x0_r1, Fx0_r1, nothing,
            rnorm(Fx0_r1))
        x1_r1, _, _, info_r1 = step!!(st_r1, m_r1, prob_r1, x0_r1, Fx0_r1, nothing)
        @test info_r1.code == SUCCESS
        @test isapprox(x1_r1[1], 1.0; atol = 1e-9)
        @test abs(x1_r1[2]) <= 1e-6

        # A dropped singular value that is exactly zero, on the no-damping path: the static
        # factor zeroes the column beyond its rank, and the coordinate weight of a zero `σ̂` with
        # no damping is `0/0`. The loop must stop at the rank, so the first trial is the truncated
        # step and it is accepted; a sum over `N` forms the `NaN` and rejects the trial.
        J0 = @SMatrix [1.0 0.0; 0.0 0.0]
        b0 = @SVector [1.0, 1.0]
        prob0 = NonlinearProblem((x, _) -> J0 * x - b0, (_, _) -> J0)
        m0 = Newton(;
            linsolve = SVDFactorization(),
            globalization = LevenbergMarquardt(; μ₀ = 1e-16, μ_min = 1e-16))
        x0_s = zero(SVector{2, Float64})
        Fx0_s = prob0.F(x0_s, nothing)
        st0_s = start!!(
            init_state(m0, prob0, x0_s, nothing), m0, prob0, x0_s, Fx0_s, nothing,
            rnorm(Fx0_s))
        x1_s, _, _, info_s = step!!(st0_s, m0, prob0, x0_s, Fx0_s, nothing)
        @test info_s.code == SUCCESS
        @test info_s.failures == 0
        @test all(isfinite, x1_s)
        @test isapprox(x1_s[1], 1.0; atol = 1e-9)
        @test iszero(x1_s[2])
    end

    @testset "Check 3: one Jacobian and one factorisation per step (MU3)" begin
        jac_calls = Ref(0)
        factors = Ref(0)
        prob = NonlinearProblem(
            (x, _) -> norm(x) > 1.5 ? SVector(NaN, NaN) : SVector(x[1] - 1, x[2] - 1),
            (_, _) -> (jac_calls[] += 1; @SMatrix [0.01 0.0; 0.0 0.01]))
        m = Newton(;
            linsolve = CountingFactorization(SVDFactorization(), factors),
            globalization = LevenbergMarquardt())
        x = SVector(0.0, 0.0)
        Fx = prob.F(x, nothing)
        st = init_state(m, prob, x, nothing)
        st = start!!(st, m, prob, x, Fx, nothing, rnorm(Fx))
        _, _, _, info = step!!(st, m, prob, x, Fx, nothing)
        @test info.code == SUCCESS
        @test info.failures >= 2
        # MU3: a factorisation inside the trial loop evaluates the Jacobian once per trial
        @test jac_calls[] == 1
        # MU3: and it factorises once per trial, where the step factorises once per outer step
        @test factors[] == 1
        # the accepted trial's damping had grown past `μ₀`, so the step is a limited one
        @test info.limited
    end

    @testset "Check 3b: the stacked solve reduces [J | -F] once per step (MU3)" begin
        # The stacked path never calls `factorize!!`: its `O(n³)` work per step is the
        # triangularisation of `[J | -F]`, which `CountingStacked` counts. A count above one is a
        # trial that factorises `J` again after a rejected damping.
        F_nan(x) = norm(x) > 1.5 ? SVector(NaN, NaN) : SVector(x[1] - 1, x[2] - 1)
        J_small = @SMatrix [0.01 0.0; 0.0 0.01]
        for (ls, sc) in ((QRFactorization(), :none), (LUFactorization(), :none),
            (SVDFactorization(), :diagonal))
            count = Ref(0)
            prob = NonlinearProblem((x, _) -> F_nan(x), (_, _) -> J_small)
            m = Newton(;
                linsolve = CountingStacked(ls, count),
                globalization = LevenbergMarquardt(; scaling = sc))
            x = SVector(0.0, 0.0)
            Fx = prob.F(x, nothing)
            st = start!!(
                init_state(m, prob, x, nothing), m, prob, x, Fx, nothing, rnorm(Fx))
            _, _, _, info = step!!(st, m, prob, x, Fx, nothing)
            @test info.code == SUCCESS
            @test info.failures >= 2
            @test count[] == 1
        end
        # The same on an array iterate, where the factor lives in the buffers of the state.
        for (ls, sc) in ((QRFactorization(), :none), (LUFactorization(), :none),
            (SVDFactorization(), :diagonal))
            count = Ref(0)
            prob = NonlinearProblem(
                (F, x, _) -> (copyto!(F, F_nan(x)); nothing),
                (J, _, _) -> (copyto!(J, J_small); nothing))
            m = Newton(;
                linsolve = CountingStacked(ls, count),
                globalization = LevenbergMarquardt(; scaling = sc))
            solver = init(prob, [0.0, 0.0], m)
            prob.F(solver.state.Fx, solver.x, nothing)
            st = start!!(
                solver.state, m, prob, solver.x, solver.state.Fx, nothing,
                rnorm(solver.state.Fx))
            _, _, _, info = step!!(st, m, prob, solver.x, solver.state.Fx, nothing)
            @test info.code == SUCCESS
            @test info.failures >= 2
            @test count[] == 1
        end
    end

    @testset "Check 4: STALLED at μ_max and a non-finite trial (MU4, MU5)" begin
        prob = NonlinearProblem((x, _) -> SVector(x[1]^2 + 1, x[2]))
        m = Newton(; linsolve = SVDFactorization(), globalization = LevenbergMarquardt())
        x0 = SVector(0.0, 0.0)
        x, st = solve(prob, x0, m)
        # MU4: another return code in place of `STALLED`
        @test st.code == STALLED
        @test x === x0
        @test st.gradratio == 0
        @test st.step_failures >= 25

        # A non-finite trial residual is a rejection, not an error: `F` is `NaN` above `x₁ = 5`,
        # and the first trials of `x₀ = (0.1, 0)` overshoot to `x₁ ≈ 16` from `μ₀ = 1e-2`.
        # MU5: returning `NONFINITE` at the first non-finite trial instead of rejecting it.
        function F_barrier(x, _)
            return x[1] > 5 ? SVector(NaN, NaN) : SVector(x[1]^2 - 4, x[2])
        end
        function J_barrier(x, _)
            return SMatrix{2, 2, Float64}(2 * x[1], 0.0, 0.0, 1.0)
        end
        prob_nan = NonlinearProblem(F_barrier, J_barrier)
        x0 = SVector(0.1, 0.0)
        m_nan = Newton(;
            linsolve = SVDFactorization(),
            globalization = LevenbergMarquardt(; μ₀ = 1e-2))
        Fx0 = prob_nan.F(x0, nothing)
        st = init_state(m_nan, prob_nan, x0, nothing)
        st = start!!(st, m_nan, prob_nan, x0, Fx0, nothing, rnorm(Fx0))
        x1, _, _, info = step!!(st, m_nan, prob_nan, x0, Fx0, nothing)
        @test info.code == SUCCESS
        @test info.failures >= 1
        @test all(isfinite, x1)
        @test info.fnorm < rnorm(Fx0)
    end

    @testset "Check 4b: STALLED with μ_max above floatmax / μ_increase" begin
        # The constructor allows a `μ_max` above `floatmax(R) / μ_increase`, and the last
        # rejection then overflows `μ`. The state must keep a finite damping: a non-finite one
        # reads as the un-started state on the stall that follows, and the solve would end
        # `NONFINITE` where it stalled.
        prob = NonlinearProblem((x, _) -> SVector(x[1]^2 + 1, x[2]))
        for R in REAL_ELTYPES, μ_max in (floatmax(R) / 2, floatmax(R)),
            ls in (SVDFactorization(), QRFactorization())
            m = Newton(;
                linsolve = ls, globalization = LevenbergMarquardt(R; μ_max = μ_max))
            x0 = zero(SVector{2, R})
            x, st = solve(prob, x0, m)
            @test st.code == STALLED
            @test x === x0
        end
    end

    @testset "Check 5: the standard reference column" begin
        @test length(STANDARD_COLUMNS) == 10
        @test "Newton LevenbergMarquardt SVD" in first.(STANDARD_COLUMNS)
        lm_column = STANDARD_COLUMNS[findfirst(
            c -> first(c) == "Newton LevenbergMarquardt SVD", STANDARD_COLUMNS)]
        @test last(lm_column).linsolve isa SVDFactorization
        @test last(lm_column).globalization isa LevenbergMarquardt
    end

    @testset "Check 1b: the eleven cells the standard set excepts (Done when 1)" begin
        # `Done when 1` reads: on every full-rank problem of the standard set where `Newton()` with
        # `Backtracking()` returns `SUCCESS`, the `"Newton LevenbergMarquardt SVD"` cell returns
        # `SUCCESS` too, except at the eleven cells below. Each is a case the specification states:
        # P1 at `u0` and P8 at `u0`, in both precisions; P11 at `u0`, in both; P15 and P16 at `u0`
        # and `10u0` in `Float64`; and P3 at `u0` in `Float32`. The reference column is `Newton()`
        # with its default `LUFactorization()`, so that no column is chosen to fit the outcome.
        excepted = Set([(1, "u0", Float32), (1, "u0", Float64),
            (8, "u0", Float32), (8, "u0", Float64),
            (11, "u0", Float32), (11, "u0", Float64),
            (15, "u0", Float64), (15, "10u0", Float64),
            (16, "u0", Float64), (16, "10u0", Float64),
            (3, "u0", Float32)])
        @test length(excepted) == 11
        column(name) = last(STANDARD_COLUMNS[findfirst(c -> first(c) == name, STANDARD_COLUMNS)])
        @test column("Newton Backtracking LU").linsolve isa LUFactorization

        # An excepted cell is one that the reference table does not record as `SUCCESS`.
        cells = read_table(STANDARD_REFERENCE)
        reference = Dict(cell_key(c) => c
        for c in cells if c.column == "Newton LevenbergMarquardt SVD")
        for (problem, start, T) in excepted
            cell = reference[(problem, start, "Newton LevenbergMarquardt SVD", string(T))]
            @test cell.code != "SUCCESS"
        end

        # ... and the eleven are the whole of the difference: a cell where the reference column
        # succeeds and the damped SVD column does not is one of them, so that no failure of the
        # damping lies outside the cells the specification states.
        reference_ok = Set((c.problem, c.start, c.eltype)
        for c in cells
        if c.column == "Newton Backtracking LU" && c.code == "SUCCESS")
        svd_ok = Set((c.problem, c.start, c.eltype)
        for c in cells
        if c.column == "Newton LevenbergMarquardt SVD" && c.code == "SUCCESS")
        @test setdiff(reference_ok, svd_ok) ==
              Set((problem, start, string(T)) for (problem, start, T) in excepted)

        # The eleventh cell, run: problem 3 (Powell badly scaled, full rank at its root) from `u0`
        # in `Float32`. `Newton()` with `Backtracking()` returns `SUCCESS`, while the `Float32` SVD
        # truncates `J` to rank one and the damped SVD step stalls at `‖F‖ = 6.7e-3`; the same
        # damping on the stacked QR and LU steps returns `SUCCESS`.
        sp = STANDARD_PROBLEMS[3]
        x₀ = Dict(standard_starts(sp, Float32))["u0"]
        opts = Options(Float32; max_iterations = STANDARD_MAX_ITERATIONS)
        _, st_bt = solve(
            sp.prob, copy(x₀), column("Newton Backtracking LU"), nothing; options = opts)
        _, st_svd = solve(sp.prob, copy(x₀),
            Newton(; linsolve = SVDFactorization(),
                globalization = LevenbergMarquardt(Float32)), nothing; options = opts)
        _, st_qr = solve(sp.prob, copy(x₀),
            Newton(; linsolve = QRFactorization(),
                globalization = LevenbergMarquardt(Float32)), nothing; options = opts)
        _, st_lu = solve(sp.prob, copy(x₀),
            Newton(; linsolve = LUFactorization(),
                globalization = LevenbergMarquardt(Float32)), nothing; options = opts)
        @test st_bt.code == SUCCESS
        @test string(st_svd.code) ==
              reference[(3, "u0", "Newton LevenbergMarquardt SVD", "Float32")].code
        @test st_qr.code == SUCCESS
        @test st_lu.code == SUCCESS
    end

    @testset "Check 6a: stacked QR against the normal equations (MU6)" begin
        J_ill = @SMatrix [1.0 1.0; 1.0 1.0 + 1e-8]
        x_true = @SVector [1.0, -1.0]
        b_ill = J_ill * x_true
        prob = NonlinearProblem((x, _) -> J_ill * x - b_ill, (_, _) -> J_ill)
        # MU6: the normal equations lose this to a relative error of `κ ≈ 1e8`.
        @test norm((transpose(J_ill) * J_ill + 1e-16 * SMatrix{2, 2, Float64}(I)) \
                   (transpose(J_ill) * (-b_ill)) - x_true) > 1e-6
        for ls in (QRFactorization(), LUFactorization())
            x, st = solve(
                prob, SVector(0.0, 0.0),
                Newton(;
                    linsolve = ls,
                    globalization = LevenbergMarquardt(; μ₀ = 1e-16, μ_min = 1e-16)),
                nothing;
                options = Options(Float64; f_abstol = 1e-14, f_reltol = 0.0))
            @test st.code == SUCCESS
            @test norm(x - x_true) < 1e-6
        end
    end

    @testset "Check 6b: the damping is absolute in the units of Jᴴ J (MU7)" begin
        x_star = SVector(1.0, -2.0)
        # One damped step from `x₀ = (3, 3)` solves `(Jᴴ J + μ) δ = -Jᴴ F`. At `2^400 (x - x*)`
        # the damping `μ = 1e-6` is below the rounding of `σ² = 2^800`, so the step is the undamped
        # one; at `2^-400 (x - x*)` it is far above `σ² = 2^-800`, so the iterate does not move and
        # every trial is rejected. A weight or a stacked damping that carries the factor's `2^e`,
        # in either direction, fails one of the two.
        for exponent_shift in (400, -400)
            s = ldexp(1.0, exponent_shift)
            J_ext = @SMatrix [s 0.0; 0.0 s]
            b_ext = J_ext * x_star
            prob = NonlinearProblem((x, _) -> J_ext * x - b_ext, (_, _) -> J_ext)
            x0 = SVector(3.0, 3.0)
            Fx0 = prob.F(x0, nothing)
            for ls in (SVDFactorization(), QRFactorization(), LUFactorization())
                m = Newton(;
                    linsolve = ls,
                    globalization = LevenbergMarquardt(; μ₀ = 1e-6, μ_min = 1e-6))
                state = init_state(m, prob, x0, nothing)
                state = start!!(state, m, prob, x0, Fx0, nothing, rnorm(Fx0))
                x1, _, _, info = step!!(state, m, prob, x0, Fx0, nothing)
                if exponent_shift > 0
                    @test info.code == SUCCESS
                    @test isapprox(x1, x_star; rtol = 1e-12)
                else
                    @test info.code == STALLED
                    @test x1 == x0
                end
            end
        end

        # The dense path damps the same way: the stacked QR of an `Array` Jacobian carries `√μ`
        # and not the factor's `2^e` either.
        for exponent_shift in (400, -400)
            s = ldexp(1.0, exponent_shift)
            J_d = [s 0.0; 0.0 s]
            b_d = J_d * [1.0, -2.0]
            prob_d = NonlinearProblem(
                (F, x, _) -> (F .= J_d * x .- b_d; nothing),
                (Jm, _, _) -> (Jm .= J_d; nothing))
            m_d = Newton(;
                linsolve = QRFactorization(),
                globalization = LevenbergMarquardt(; μ₀ = 1e-6, μ_min = 1e-6))
            solver = init(prob_d, [3.0, 3.0], m_d)
            prob_d.F(solver.state.Fx, solver.x, nothing)
            state = start!!(
                solver.state, solver.method, solver.prob, solver.x, solver.state.Fx, nothing,
                rnorm(solver.state.Fx))
            x1, _, _, info = step!!(
                state, solver.method, solver.prob, solver.x, solver.state.Fx, nothing)
            if exponent_shift > 0
                @test info.code == SUCCESS
                @test isapprox(Array(x1), [1.0, -2.0]; rtol = 1e-12)
            else
                @test info.code == STALLED
                @test Array(x1) == [3.0, 3.0]
            end
        end

        # `2^{-e} √μ` overflows at the bottom of the factor's scale range: `matrix_exponent`
        # clamps `e` at `-1023`, and `2^1023 √(1e12)` is above `floatmax`. The weight is then
        # below the rounding of `x`, so the coordinate is dropped and never formed as a `NaN`.
        @test lm_svd_coord(1.0, 0, 1.0, 1e-6) > 0
        @test iszero(lm_svd_coord(1.0, -1023, 1.0, 1e12))
        # The rescaled form squares neither `σ̂` nor `2^{-e} √μ`: at `e = -561` with `μ = 4 eps`
        # the product `2^{-2e} μ` is above `floatmax`, while the weight
        # `2^{-e} σ̂/(σ̂² + 2^{-2e} μ)` is `2^-511`, which a form that squares either one loses to
        # zero.
        @test lm_svd_coord(1.0, -561, 1.0, 4 * eps(Float64)) == ldexp(1.0, -511)

        # The two assertions of the specification's Check 6b. The default `μ₀` cannot move an
        # iterate of the `2^-400` scale; `μ₀ = 1e-16`, below `4 eps(Float64)`, is no damping at
        # all, which is the truncated Newton step and converges at once.
        s_low = ldexp(1.0, -400)
        J_low = @SMatrix [s_low 0.0; 0.0 s_low]
        b_low = J_low * x_star
        prob_low = NonlinearProblem((x, _) -> J_low * x - b_low, (_, _) -> J_low)
        # The residual of this scale is `√2 · 2^-400`, so the absolute tolerance has to go below
        # it: a fixed `1e-12` would make the stopping test hold before the first step.
        opts_low = Options(Float64; f_abstol = s_low * 1e-12, f_reltol = 1e-12)
        x0_low = zero(SVector{2, Float64})
        Fx0_low = prob_low.F(x0_low, nothing)
        x_default, st_default = solve(
            prob_low, x0_low,
            Newton(; linsolve = SVDFactorization(), globalization = LevenbergMarquardt()),
            nothing; options = opts_low)
        @test st_default.code == STALLED
        @test x_default == x0_low

        m_tiny = Newton(;
            linsolve = SVDFactorization(),
            globalization = LevenbergMarquardt(; μ₀ = 1e-16, μ_min = 1e-16))
        state_tiny = start!!(
            init_state(m_tiny, prob_low, x0_low, nothing), m_tiny, prob_low, x0_low, Fx0_low,
            nothing, rnorm(Fx0_low))
        # `start!!` stores `μ₀` as it is; the trial step, not the schedule, takes it as no damping.
        @test state_tiny.rule.μ === 1e-16
        x_tiny, st_tiny = solve(prob_low, x0_low, m_tiny, nothing; options = opts_low)
        @test st_tiny.code == SUCCESS
        @test norm(x_tiny - x_star) < 1e-6

        # The top of the range converges with `SVDFactorization()`.
        s_high = ldexp(1.0, 400)
        J_high = @SMatrix [s_high 0.0; 0.0 s_high]
        b_high = J_high * x_star
        prob_high = NonlinearProblem((x, _) -> J_high * x - b_high, (_, _) -> J_high)
        x_high, st_high = solve(
            prob_high, x0_low,
            Newton(; linsolve = SVDFactorization(), globalization = LevenbergMarquardt()),
            nothing;
            options = Options(Float64; f_abstol = s_high * 1e-12, f_reltol = 1e-12))
        @test st_high.code == SUCCESS
        @test norm(x_high - x_star) < 1e-6
    end

    @testset "Check 6c: fail-closed rule state (MU8)" begin
        prob = NonlinearProblem(
            (x, _) -> SVector(x[1] - 1, x[2] - 2),
            (_, _) -> SMatrix{2, 2, Float64}(1.0, 0.0, 0.0, 1.0))
        m = Newton(;
            linsolve = SVDFactorization(),
            globalization = LevenbergMarquardt(; μ₀ = 1e-6, μ_min = 5e-7,
                μ_decrease = 0.25))
        x0 = SVector(0.0, 0.0)
        Fx0 = prob.F(x0, nothing)
        st0 = init_state(m, prob, x0, nothing)
        @test isnan(st0.rule.μ)
        # MU8: a rule state initialised to `μ₀` would step here instead of failing closed
        x_bad, Fx_bad, st_bad, info_bad = step!!(st0, m, prob, x0, Fx0, nothing)
        @test info_bad.code == NONFINITE
        @test x_bad === x0
        @test Fx_bad === Fx0
        @test isnan(st_bad.rule.μ)

        st1 = start!!(st0, m, prob, x0, Fx0, nothing, rnorm(Fx0))
        @test st1.rule.μ === 1e-6
        x1, Fx1, st2, info1 = step!!(st1, m, prob, x0, Fx0, nothing)
        @test info1.code == SUCCESS
        @test st2.rule.μ === 5e-7
        # MU8: without the reset in `finish!!` the state keeps the last damping
        _, _, st_fin = finish!!(
            st2, m, prob, x1, Fx1,
            SolverStatus{Float64}(
                SUCCESS, Int32(1), info1.fnorm, info1.stepnorm, Int32(0), false))
        @test isnan(st_fin.rule.μ)

        # A residual that is zero already returns `SUCCESS` with no step, before any trial.
        Fx_zero = zero(x0)
        st_zero = start!!(st0, m, prob, x0, Fx_zero, nothing, rnorm(Fx_zero))
        x_z, _, _, info_z = step!!(st_zero, m, prob, x0, Fx_zero, nothing)
        @test info_z.code == SUCCESS
        @test info_z.stepnorm == 0
        @test x_z === x0

        # And before the Jacobian is read: a non-finite `J` at a root is not a failed step.
        prob_root = NonlinearProblem(
            (x, _) -> zero(x),
            (_, _) -> SMatrix{2, 2, Float64}(NaN, NaN, NaN, NaN))
        st_root = start!!(
            init_state(m, prob_root, x0, nothing), m, prob_root, x0, Fx_zero, nothing,
            rnorm(Fx_zero))
        x_root, _, _, info_root = step!!(st_root, m, prob_root, x0, Fx_zero, nothing)
        @test info_root.code == SUCCESS
        @test info_root.stepnorm == 0
        @test x_root === x0
    end

    @testset "Check 6d: R1 and R3 storage, Float32/Float64, Array and JLArray" begin
        prob = NonlinearProblem((x, _) -> SVector(x[1]^2 - 2, x[2]^2 - 3))
        m = Newton(; linsolve = SVDFactorization(), globalization = LevenbergMarquardt())
        x0 = SVector(2.0, 2.0)
        Fx0 = prob.F(x0, nothing)
        st = @inferred init_state(m, prob, x0, nothing)
        @test isbits(st)
        if Base.JLOptions().check_bounds != 1
            @test allocations(s -> init_state(m, prob, s, nothing), x0) == 0
        end
        st = @inferred start!!(st, m, prob, x0, Fx0, nothing, rnorm(Fx0))
        @inferred step!!(st, m, prob, x0, Fx0, nothing)
        if Base.JLOptions().check_bounds != 1
            @test allocations(s -> step!!(s, m, prob, x0, Fx0, nothing), st) == 0
            @test allocations(collect, x0) > 0
        end
        @test_opt solve(prob, x0, m)
        @test_opt solve(NonlinearProblem((x, _) -> x^2 - 2, (x, _) -> 2x), 1.5, m)

        # A scalar iterate: the one-dimensional case of the same value-iterate path.
        prob_scalar = NonlinearProblem((x, _) -> x^2 - 2, (x, _) -> 2 * x)
        x_sc, st_sc = solve(prob_scalar, 1.5, m)
        @test st_sc.code == SUCCESS
        @test isapprox(x_sc, sqrt(2.0); atol = 1e-8)

        prob_r3 = NonlinearProblem((F, x, _) -> (F[1] = x[1]^2 - 2; F[2] = x[2]^2 - 3; F))
        for ls in (SVDFactorization(), QRFactorization(), LUFactorization())
            solver = init(prob_r3, [2.0, 2.0], Newton(;
                linsolve = ls, globalization = LevenbergMarquardt()))
            _, st_r3 = solve!(solver)
            @test st_r3.code == SUCCESS
            solver.x .= [2.0, 2.0]
            prob_r3.F(solver.state.Fx, solver.x, nothing)
            solver.state = start!!(
                solver.state, solver.method, solver.prob, solver.x, solver.state.Fx,
                nothing, rnorm(solver.state.Fx))
            step!!(solver.state, solver.method, solver.prob, solver.x, solver.state.Fx,
                nothing)
            solver.x .= [2.0, 2.0]
            prob_r3.F(solver.state.Fx, solver.x, nothing)
            solver.state = start!!(
                solver.state, solver.method, solver.prob, solver.x, solver.state.Fx,
                nothing, rnorm(solver.state.Fx))
            args = (solver.state, solver.method, solver.prob, solver.x, solver.state.Fx)
            if Base.JLOptions().check_bounds != 1
                @test allocations(
                    t -> step!!(t[1], t[2], t[3], t[4], t[5], nothing), args) == 0
            end
        end

        # Float32 on R1 and R3, and the device path under `allowscalar(false)`.
        m32 = Newton(;
            linsolve = SVDFactorization(), globalization = LevenbergMarquardt(Float32))
        prob32 = NonlinearProblem((x, _) -> SVector(x[1]^2 - 2.0f0, x[2]^2 - 3.0f0))
        x32, st32 = solve(prob32, SVector(2.0f0, 2.0f0), m32)
        @test st32.code == SUCCESS
        @test isapprox(x32, SVector(sqrt(2.0f0), sqrt(3.0f0)); atol = 1e-5)
        prob32_r3 = NonlinearProblem(
            (F, x, _) -> (F[1] = x[1]^2 - 2.0f0; F[2] = x[2]^2 - 3.0f0; F))
        x32b, st32b = solve(prob32_r3, Float32[2.0, 2.0], m32)
        @test st32b.code == SUCCESS
        @test isapprox(x32b, Float32[sqrt(2.0f0), sqrt(3.0f0)]; atol = 1e-5)

        prob_jl = NonlinearProblem((F, x, _) -> (F .= x .^ 2 .- 4.0; nothing))
        x_jl, st_jl = solve(prob_jl, JLArray([3.0, 3.0]), m)
        @test st_jl.code == SUCCESS
        @test isapprox(Array(x_jl), [2.0, 2.0]; atol = 1e-8)
    end

    @testset "Complex problems in ELTYPES" begin
        for T in filter(T -> T <: Complex, ELTYPES)
            A = SMatrix{2, 2, T}(T(2), T(1) + im, T(1) - im, T(3))
            b = SVector{2, T}(T(1), T(2))
            x_star = A \ b
            prob = NonlinearProblem((x, _) -> A * x - b, (_, _) -> A)
            for ls in (SVDFactorization(), QRFactorization(), LUFactorization())
                x, st = solve(
                    prob, zero(SVector{2, T}),
                    Newton(;
                        linsolve = ls,
                        globalization = LevenbergMarquardt(real(T))))
                @test st.code == SUCCESS
                @test isapprox(x, x_star; atol = sqrt(eps(real(T))))
            end
        end
    end

    @testset "Check 6e: R3 in ELTYPES, both scalings, both backends" begin
        # The damped step of every element type and linear method on an array iterate: the dense
        # SVD, the dense stacked QR and Marquardt's scaling are separate code from the static
        # ones, and a complex `T` separates `conj` from no `conj` in either.
        for AT in ARRAY_BACKENDS, T in ELTYPES,
            ls in (SVDFactorization(), QRFactorization(), LUFactorization()),
            sc in (:none, :diagonal)
            A = T[3 1; 1 2] + (T <: Complex ? T[0 im; -im 0] : zeros(T, 2, 2))
            b = T[4, 3]
            Ad, bd = AT(A), AT(b)
            prob = NonlinearProblem(
                (F, x, _) -> (mul!(F, Ad, x); F .-= bd; nothing),
                (J, _, _) -> (copyto!(J, Ad); nothing))
            m = Newton(;
                linsolve = ls,
                globalization = LevenbergMarquardt(real(T); scaling = sc))
            if AT === JLArray && !(ls isa SVDFactorization && sc === :none)
                # The stacked path indexes scalars: a device iterate is refused at `init` rather
                # than throwing "Scalar indexing is disallowed" at its first trial, and the
                # message names the step a device iterate takes.
                err = try
                    init(prob, AT(zeros(T, 2)), m)
                    nothing
                catch e
                    e
                end
                @test err isa ArgumentError
                msg = err isa ArgumentError ? err.msg : ""
                pattern = "^a device array iterate takes LevenbergMarquardt\\(\\) with " *
                          "SVDFactorization\\(\\) and scaling = :none only"
                @test occursin(Regex(pattern), msg)
            else
                x, st = solve(prob, AT(zeros(T, 2)), m)
                @test st.code == SUCCESS
                @test norm(Array(x) - A \ b) <= sqrt(eps(real(T)))
            end
        end
    end

    @testset "Check 6f: the stacked step against the closed form in ELTYPES (MU10)" begin
        # The Givens rotations of the stacked solve carry `-conj(sn)`; with `-sn` the rotation is
        # not unitary and the step is wrong on a complex element type and only there. One accepted
        # trial at `μ₀` compares the step with `-(Jᴴ J + μ D²)⁻¹ Jᴴ F` computed in
        # `Complex{BigFloat}`. The tolerance is `64 eps(R) κ(J)`, a multiple of the `eps · κ` a
        # factorisation of this size costs: the measured error of the step is at most `1.2` of
        # that unit (`0.19` at `n = 3` and `4`), and the error of the dropped `conj` is above
        # `1e5` units. `Val(1)` is the scalar step, whose pivot is the signed `J` itself, so its
        # fixture is negative on a real element type: a positive entry could not tell that pivot
        # from its modulus.
        for nv in (Val(1), Val(3), Val(4)), T in ELTYPES,
            ls in (QRFactorization(), LUFactorization()), sc in (:none, :diagonal),
            μ₀ in (1.0, 4.0)
            info, err = lm_stacked_case(nv, T, ls, sc, μ₀)
            @test info.code == SUCCESS
            @test info.failures == 0
            @test err <= 64.0
        end
        # The same on an array iterate, whose factor and elimination run on the state's buffers.
        for nv in (Val(3), Val(4)), T in ELTYPES, sc in (:none, :diagonal)
            info, err = lm_stacked_case_r3(nv, T, sc)
            @test info.code == SUCCESS
            @test info.failures == 0
            @test err <= 64.0
        end
    end

    @testset "Check 6g: the R1 step allocates nothing on every path" begin
        # The stacked solve keeps `(R₀, c₀, d_diag)` as values and builds its elimination copies
        # in the call that reads them, so no mutable array crosses a call or a loop iteration.
        # The factor's `SMatrix` field carries its length parameter: without it the field type is
        # the `SMatrix{N, N, T}` alias with `L` free, and the workspace is not `isbits`.
        @test isbits(GeometricSolvers.lm_qr_prepare(
            QRFactorization(), nothing, nothing,
            lm_test_matrix(Val(2), Float64), lm_test_rhs(Val(2), Float64), Val(:none)))
        # `--check-bounds=yes` lets `StaticArrays`' `MMatrix` escape to the heap and skips the
        # assertion, which `run-tests.jl` runs with that flag on a Julia before 1.13.
        if Base.JLOptions().check_bounds != 1
            @test allocations(collect, [1.0, 2.0]) > 0
            for nv in (Val(1), Val(2), Val(4)), T in ELTYPES,
                ls in (SVDFactorization(), QRFactorization(), LUFactorization()),
                sc in (:none, :diagonal)
                @test lm_r1_allocations(nv, T, ls, sc) == 0
            end
        end
    end

    @testset "Check 6h: a non-finite damped step is a rejected trial" begin
        # `J = diag(1, 1e-6)` in `Float32` keeps `σ₂` above the factor's cutoff, and a residual of
        # `1e38` overflows the truncated step at `σ₂`: `1e38 / 1e-6` is `Inf`. That trial is a
        # rejection, and the residual is not evaluated at the non-finite iterate it would build.
        J = SMatrix{2, 2, Float32}([1.0 0.0; 0.0 1e-6])
        b = SVector{2, Float32}(-1.0e38, -1.0e38)
        finite_x = Bool[]
        F(x, _) = (push!(finite_x, all(isfinite, x)); J * x - b)
        prob = NonlinearProblem(F, (_, _) -> J)
        m = Newton(;
            linsolve = SVDFactorization(),
            globalization = LevenbergMarquardt(Float32; μ₀ = 1.0f-14, μ_min = 1.0f-14))
        x0 = zero(SVector{2, Float32})
        Fx = prob.F(x0, nothing)
        st = start!!(init_state(m, prob, x0, nothing), m, prob, x0, Fx, nothing, rnorm(Fx))
        empty!(finite_x)
        x1, _, _, info = step!!(st, m, prob, x0, Fx, nothing)
        @test info.code == STALLED
        @test x1 === x0
        # The rejections of a non-finite step cost no residual evaluation, so the loop evaluates
        # `F` fewer times than it counts a failure.
        @test !isempty(finite_x)
        @test all(finite_x)
        @test length(finite_x) < Int(info.failures)

        # A scalar iterate is the 1×1 case of the same solve, whose step is one rotation and one
        # division rather than a call of `lm_qr_solve!`, and it reports a non-finite step as a
        # rejected trial the same way. `J = 1e-6` and a residual of `3e38` overflow the damped
        # minimiser `-J F / (J² + μ)` in `Float32` until `μ` reaches the size of `J²`, so the
        # smallest trials are non-finite and the loop raises `μ` until it stalls at `μ_max`.
        for ls in (QRFactorization(), LUFactorization()),
            μ₀ in (1.0f-14, 1.0f-20)

            Js = 1.0f-6
            bs = -3.0f38
            scales = Bool[]
            Fs(x, _) = (push!(scales, isfinite(x)); Js * x - bs)
            prob_s = NonlinearProblem(Fs, (_, _) -> Js)
            m_s = Newton(; linsolve = ls,
                globalization = LevenbergMarquardt(Float32; μ₀ = μ₀, μ_min = μ₀))
            x_s = 0.0f0
            Fx_s = prob_s.F(x_s, nothing)
            st_s = start!!(init_state(m_s, prob_s, x_s, nothing), m_s, prob_s, x_s, Fx_s,
                nothing, rnorm(Fx_s))
            empty!(scales)
            x_s1, _, _, info_s = step!!(st_s, m_s, prob_s, x_s, Fx_s, nothing)
            @test info_s.code == STALLED
            @test x_s1 === x_s
            @test !isempty(scales)
            @test all(scales)
            @test length(scales) < Int(info_s.failures)
        end
    end

    @testset "The SVD path allocates no stacked buffer" begin
        # The stacked solve reads its own buffers and the SVD step reads none of them: an `init`
        # of the SVD path allocates none, and one of the stacked path allocates them all.
        prob_r3 = NonlinearProblem(
            (F, x, _) -> (F .= x .^ 2 .- 4.0; nothing),
            (J, x, _) -> (J[1, 1] = 2 * x[1]; J[2, 2] = 2 * x[2]; J[1, 2] = 0.0;
                J[2, 1] = 0.0; nothing))
        for (ls, sc, stacked) in ((SVDFactorization(), :none, false),
            (SVDFactorization(), :diagonal, true), (QRFactorization(), :none, true),
            (LUFactorization(), :none, true))
            m = Newton(;
                linsolve = ls, globalization = LevenbergMarquardt(; scaling = sc))
            g = init(prob_r3, [3.0, 3.0], m).state.g
            @test (g.R₀ !== nothing) === stacked
            @test (g.R !== nothing) === stacked
            @test (g.d_diag !== nothing) === stacked
            @test (g.w !== nothing) === stacked
            @test g.Jδ !== nothing && g.xt !== nothing && g.Ft !== nothing
        end
    end

    @testset "The no-damping floor of 4 eps is bracketed" begin
        # `μ` is absolute, in the units of `Jᴴ J`: on the scale `2^-400` of `J` a `μ` above
        # `4 eps(Float64)` is far above `σ² = 2^-800` and the step stands still, while a `μ`
        # below that floor is no damping at all and the step is the truncated Newton step.
        # The two cases bracket the floor from each side.
        x_star = SVector(1.0, -2.0)
        s = ldexp(1.0, -400)
        J = @SMatrix [s 0.0; 0.0 s]
        b = J * x_star
        prob = NonlinearProblem((x, _) -> J * x - b, (_, _) -> J)
        x0 = zero(SVector{2, Float64})
        opts = Options(Float64; f_abstol = s * 1e-12, f_reltol = 1e-12)
        for (μ₀, code) in ((3 * eps(Float64), SUCCESS), (6 * eps(Float64), STALLED))
            for ls in (SVDFactorization(), QRFactorization())
                m = Newton(;
                    linsolve = ls,
                    globalization = LevenbergMarquardt(; μ₀ = μ₀, μ_min = μ₀))
                _, st = solve(prob, x0, m, nothing; options = opts)
                @test st.code == code
            end
        end
    end

    @testset "A rejected trial raises μ to the floor of 4 eps" begin
        # `μ₀ = μ_min = 1e-300` is below the floor, so the first trial is the undamped step, which
        # overshoots on `atan` from 3. The rejection lifts `μ` to `max(μ_min, 4 eps)` at once:
        # the step needs at most one trial more than a start at that floor, and not the hundreds
        # of identical undamped trials that `μ · μ_increase` takes to climb from `1e-300`.
        evals = Ref(0)
        prob = NonlinearProblem(
            (x, _) -> (evals[] += 1; atan.(x)),
            (x, _) -> @SMatrix [1 / (1 + x[1]^2) 0.0; 0.0 1 / (1 + x[2]^2)])
        x = SVector(3.0, 3.0)
        Fx = prob.F(x, nothing)
        for ls in (SVDFactorization(), QRFactorization())
            counts = map((1e-300, 4 * eps(Float64))) do μ₀
                m = Newton(;
                    linsolve = ls,
                    globalization = LevenbergMarquardt(; μ₀ = μ₀, μ_min = μ₀))
                st = start!!(
                    init_state(m, prob, x, nothing), m, prob, x, Fx, nothing, rnorm(Fx))
                evals[] = 0
                x₊, _, _, info = step!!(st, m, prob, x, Fx, nothing)
                @test info.code == SUCCESS
                @test x₊ != x
                (info.failures, evals[])
            end
            (failures_low, evals_low), (failures_floor, _) = counts
            @test failures_floor >= 1
            @test failures_low <= failures_floor + 1
            @test evals_low == failures_low + 1
        end
    end

    @testset "The gain ratio at a positive ρ_accept" begin
        # D6's predicted decrease carries `-‖Jδ‖²`, and a `ρ_accept` above zero reads it: the
        # ratio of the true `pred` and of the one without that term straddle a `ρ_accept` between
        # them, so the same trial is accepted by the rule and rejected by the other `pred`.
        prob = NonlinearProblem((x, _) -> SVector(x[1]^2 - 2, x[2]^2 - 3))
        x0 = SVector(2.0, 2.0)
        μ₀ = 0.5
        J = @SMatrix [2 * x0[1] 0.0; 0.0 2 * x0[2]]
        Fx = prob.F(x0, nothing)
        fnorm = rnorm(Fx)
        δ = -((transpose(J) * J + μ₀ * SMatrix{2, 2, Float64}(I)) \ (transpose(J) * Fx))
        Jδ = J * δ
        pred = -2 * rdot(Fx, Jδ) - norm2(Jδ)
        pred_mut = -2 * rdot(Fx, Jδ) + norm2(Jδ)
        Ft = prob.F(x0 + δ, nothing)
        ared = (fnorm - rnorm(Ft)) * (fnorm + rnorm(Ft))
        ρ = ared / pred
        ρ_mut = ared / pred_mut
        @test 0 < pred < pred_mut
        @test 0 < ρ_mut < ρ
        @test ρ - ρ_mut > 0.1
        ρ_accept = (ρ + ρ_mut) / 2
        m = Newton(;
            linsolve = SVDFactorization(),
            globalization = LevenbergMarquardt(; μ₀ = μ₀, μ_min = μ₀, ρ_accept = ρ_accept))
        st = start!!(
            init_state(m, prob, x0, nothing), m, prob, x0, Fx, nothing, rnorm(Fx))
        _, _, _, info = step!!(st, m, prob, x0, Fx, nothing)
        @test info.code == SUCCESS
        @test info.failures == 0

        # A trial that leaves the residual where it was has `ared == 0` and `ρ == 0`, which the
        # acceptance edge `ρ > ρ_accept` rejects at `ρ_accept = 0`. `F` is constant here, so
        # every trial has it, and a rule that accepted it would step away from a residual it
        # cannot reduce; the solve can only stall.
        prob_flat = NonlinearProblem(
            (x, _) -> SVector(1.0, 1.0),
            (_, _) -> @SMatrix [0.25 0.0; 0.0 0.25])
        x_flat = SVector(0.0, 0.0)
        for ls in (SVDFactorization(), QRFactorization())
            m_flat = Newton(; linsolve = ls, globalization = LevenbergMarquardt())
            x_s, st_s = solve(prob_flat, x_flat, m_flat)
            @test st_s.code == STALLED
            @test x_s === x_flat
        end
    end

    @testset "Check 7: the Powell singular problem from a fixed start" begin
        function F_powell(x, _)
            return @SVector [x[1], 10 * x[1] / (x[1] + 0.1) + 2 * x[2]^2]
        end
        function J_powell(x, _)
            return @SMatrix [1.0 0.0;
                             1.0 / (x[1] + 0.1)^2 4 * x[2]]
        end
        prob = NonlinearProblem(F_powell, J_powell)
        opts = Options(Float64; f_abstol = 1e-10, f_reltol = 1e-10, max_iterations = 200)
        for ls in (SVDFactorization(), QRFactorization(), LUFactorization())
            x, st = solve(
                prob, SVector(3.0, 1.0),
                Newton(; linsolve = ls, globalization = LevenbergMarquardt()),
                nothing; options = opts)
            @test st.code == SUCCESS
            @test st.fnorm <= 1e-9
            @test norm(x) < 1e-3
        end
    end

    @testset "A sparse jacobian_prototype is refused at init" begin
        # No damped step reads a sparse factor: the stacked step reduces a dense copy of `J(x)`
        # and the SVD step factorises a dense matrix. An array iterate whose problem carries a
        # sparse prototype is refused at `init` with every linear method and both scalings, and
        # the message names the dense alternatives. A static iterate ignores the prototype.
        P = sparse([1, 2], [1, 2], [1.0, 1.0], 2, 2)
        prob = NonlinearProblem(
            (F, x, _) -> (F .= x .^ 2 .- 4.0; nothing); jacobian_prototype = P)
        for ls in (LUFactorization(), QRFactorization(), SVDFactorization(),
                LDLTFactorization()),
            sc in (:none, :diagonal)

            m = Newton(; linsolve = ls, globalization = LevenbergMarquardt(; scaling = sc))
            err = try
                init(prob, [3.0, 3.0], m)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            msg = err isa ArgumentError ? err.msg : ""
            @test occursin("does not use a sparse jacobian_prototype", msg)
            @test occursin("SVDFactorization()", msg)
        end
        prob_r1 = NonlinearProblem((x, _) -> x .^ 2 .- 4.0; jacobian_prototype = P)
        m = Newton(; linsolve = LUFactorization(), globalization = LevenbergMarquardt())
        x, st = solve(prob_r1, SVector(3.0, 3.0), m)
        @test st.code == SUCCESS
        @test norm(x - SVector(2.0, 2.0)) <= sqrt(eps(Float64))
    end

    @testset "StationarityProblem on every linear method, R1 and R3" begin
        # The residual of a `StationarityProblem` is the gradient of `S` and its Jacobian the
        # Hessian, symmetric where `E == U`. The damped step reads no symmetry, so it solves the
        # stationarity equations as any other system, with the merit `‖F‖²` that the problem
        # prescribes. Both actions have their stationary point in closed form: a minimum
        # `(√1.01, √1.01 / 10)` of a double well, and the saddle `(0, -1)` of `S₂`, whose Hessian
        # `diag(1 + x₁², -1)` is indefinite there.
        S₁(x, _) = (x[1]^2 - 1)^2 / 4 + x[2]^2 / 2 - x[1] * x[2] / 10
        S₂(x, _) = x[1]^2 / 2 + x[1]^4 / 12 - x[2]^2 / 2 - x[2]
        cases = ((S₁, (0.8, 0.3), (sqrt(1.01), sqrt(1.01) / 10)),
            (S₂, (0.4, -0.7), (0.0, -1.0)))
        for (S, start, root) in cases,
            ls in (SVDFactorization(), QRFactorization(), LUFactorization(),
                LDLTFactorization()),
            sc in (:none, :diagonal),
            x₀ in (collect(start), SVector(start))
            m = Newton(; linsolve = ls, globalization = LevenbergMarquardt(; scaling = sc))
            x, st = solve(StationarityProblem(S), x₀, m)
            @test st.code == SUCCESS
            @test st.iterations >= 1
            @test norm(collect(x) .- collect(root)) <= sqrt(eps(Float64))
        end
    end
end
