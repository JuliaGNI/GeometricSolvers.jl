using GeometricSolvers
using GeometricSolvers: Options, ReturnCode, SolverStatus,
                        SUCCESS, STALLED, MAXITERS, SINGULAR, NONFINITE,
                        LINESEARCH_FAILED, ToReal, init, solve!!, parameters,
                        isconverged

using Adapt
using GPUArraysCore: allowscalar
using JLArrays
using JET
using KernelAbstractions
using LinearAlgebra
using StaticArrays
using Test

include("../helpers/matrix.jl")

allowscalar(false)

struct SpyOrderVector <: AbstractVector{Int}
    perm::Vector{Int}
    reads::Vector{Int}
end

Base.size(v::SpyOrderVector) = size(v.perm)
Base.IndexStyle(::Type{SpyOrderVector}) = IndexLinear()
function Base.getindex(v::SpyOrderVector, i::Int)
    v.reads[i] += 1
    return v.perm[i]
end
KernelAbstractions.get_backend(::SpyOrderVector) = KernelAbstractions.CPU()

# A host vector with no `get_backend` method.
struct UnreadOrderVector <: AbstractVector{Int}
    perm::Vector{Int}
end

Base.size(v::UnreadOrderVector) = size(v.perm)
Base.IndexStyle(::Type{UnreadOrderVector}) = IndexLinear()
Base.getindex(v::UnreadOrderVector, i::Int) = v.perm[i]

@inline function _r2_hetero_res(x::SVector{2, T}, code::Int) where {T}
    R = real(T)
    if code == 1 # SUCCESS: linear system converges in 1 step
        return SVector{2, T}(x[1] - T(2), x[2] - T(3))
    elseif code == 2 # STALLED: tiny step to stationary nonzero value
        return SVector{2, T}(T(1), T(1))
    elseif code == 3 # MAXITERS: linear contract with slow progress
        return SVector{2, T}(x[1] - T(100), x[2] - T(100))
    elseif code == 4 # SINGULAR: zero Jacobian
        return SVector{2, T}(T(1), T(1))
    elseif code == 5 # NONFINITE: NaN residual
        nan = T(R(NaN))
        return SVector{2, T}(nan, nan)
    else # code == 6: LINESEARCH_FAILED: uphill step in Backtracking
        return SVector{2, T}(x[1]^2 + T(1), x[2]^2 + T(1))
    end
end

@inline function _r2_hetero_jac(x::SVector{2, T}, code::Int) where {T}
    z = zero(T)
    o = one(T)
    if code == 1
        return @SMatrix [o z; z o]
    elseif code == 2
        big = T(real(T)(1e16))
        return @SMatrix [big z; z big]
    elseif code == 3
        two = T(2)
        return @SMatrix [two z; z two]
    elseif code == 4
        return @SMatrix [z z; z z]
    elseif code == 5
        return @SMatrix [o z; z o]
    else
        return @SMatrix [-o z; z -o]
    end
end

@inline function _r2_coupled_res(x::SVector{2, T}, p) where {T}
    return SVector{2, T}(
        x[1] + x[2]^2 / T(8) - T(p[1]),
        x[2] + x[1]^2 / T(8) - T(p[2])
    )
end

@inline function _r2_coupled_jac(x::SVector{2, T}, _) where {T}
    o = one(T)
    return @SMatrix [o x[2] / T(4); x[1] / T(4) o]
end

_r2_coupled_problem() = NonlinearProblem(_r2_coupled_res, _r2_coupled_jac)

@inline _r2_scalar_res(x::T, p) where {T} = x^2 - T(p)
@inline _r2_scalar_jac(x::T, _) where {T} = 2 * x

_r2_scalar_problem() = NonlinearProblem(_r2_scalar_res, _r2_scalar_jac)

@testset "Batched R2 nonlinear" begin
    @testset "Exports and R1 base premises" begin
        @test isdefined(GeometricSolvers, :PerSystem)
        @test isdefined(GeometricSolvers, :solve_batch!)
        @test :PerSystem in names(GeometricSolvers)
        @test :solve_batch! in names(GeometricSolvers)
        @test Int.(instances(ReturnCode)) === Tuple(0:5)
    end

    @testset "PerSystem wrapper and shared vector parameters" begin
        v = [10, 20, 30]
        ps = PerSystem(v)
        @test ps.data === v
        @test parameters(ps, 1) == 10
        @test parameters(ps, 2) == 20
        @test parameters(ps, 3) == 30

        # Shared parameter (plain vector, tuple, scalar, nothing) is returned unchanged
        @test parameters(v, 1) === v
        @test parameters(v, 2) === v
        svec = SVector(1.0, 2.0)
        @test parameters(svec, 3) === svec
        @test parameters(nothing, 5) === nothing
        @test parameters((a = 1, b = 2), 2) === (a = 1, b = 2)

        # Tuple / NTuple and non-Vector PerSystem
        ps_tup = PerSystem((1.0, 2.0, 3.0))
        @test ps_tup.data === (1.0, 2.0, 3.0)
        @test isbits(ps_tup)
        @test parameters(ps_tup, 1) === 1.0
        @test parameters(ps_tup, 2) === 2.0
        @test parameters(ps_tup, 3) === 3.0

        # Adapt.@adapt_structure PerSystem
        jl_ps = Adapt.adapt(JLArray, ps)
        @test jl_ps isa PerSystem{<:JLVector{Int}}
        @test Array(jl_ps.data) == v

        prob = _r2_coupled_problem()
        for B in ARRAY_BACKENDS, T in REAL_ELTYPES

            R = real(T)
            p_shared_svec = SVector{2, T}(T(2.125), T(1.5))
            x0_host = SVector{2, T}[
                SVector{2, T}(T(1.8), T(0.9)),
                SVector{2, T}(T(2.3), T(1.2)),
                SVector{2, T}(T(2.6), T(1.4))
            ]
            # Plain SVector parameter (length 2 != batch length 3) shared across all systems
            X_sh = B(copy(x0_host))
            S_sh = B(Vector{SolverStatus{R}}(undef, 3))
            solve_batch!(X_sh, S_sh, prob, Newton(), p_shared_svec)
            X_sh_h = Array(X_sh)
            S_sh_h = Array(S_sh)
            for i in 1:3
                xr, str = solve(prob, x0_host[i], Newton(), p_shared_svec)
                @test X_sh_h[i] == xr
                @test S_sh_h[i] == str
            end

            # Plain Vector parameter of length 2 != batch length 3 on CPU
            if B === Array
                p_vec_shared = T[T(2.125), T(1.5)]
                X_v = copy(x0_host)
                S_v = Vector{SolverStatus{R}}(undef, 3)
                solve_batch!(X_v, S_v, prob, Newton(), p_vec_shared)
                for i in 1:3
                    xr, str = solve(prob, x0_host[i], Newton(), p_vec_shared)
                    @test X_v[i] == xr
                    @test S_v[i] == str
                end
            end

            # Length mismatch for PerSystem throws DimensionMismatch
            bad_ps = PerSystem(B([p_shared_svec, p_shared_svec]))
            @test_throws DimensionMismatch solve_batch!(
                B(copy(x0_host)), S_sh, prob, Newton(), bad_ps)
            @test_throws DimensionMismatch solve_batch!(
                B(copy(x0_host)), prob, Newton(), bad_ps)
        end
    end

    @testset "Host validations and empty batch" begin
        prob = _r2_coupled_problem()
        p0 = SVector(2.125, 1.5)
        X = [SVector(2.0, 1.0), SVector(1.5, 0.5)]
        S_good = Vector{SolverStatus{Float64}}(undef, 2)
        S_bad = Vector{SolverStatus{Float64}}(undef, 3)

        # length(S) != length(X)
        @test_throws DimensionMismatch solve_batch!(copy(X), S_bad, prob, Newton(), p0)

        # S on another backend than X: a host S with a device X, and a device S with a host X
        @test_throws ArgumentError solve_batch!(
            JLArray(copy(X)), S_good, prob, Newton(), p0)
        @test_throws ArgumentError solve_batch!(
            copy(X), JLArray(S_good), prob, Newton(), p0)

        # options !== nothing && !isempty(kwargs), with the message of `solve` and `init`
        opt = Options(Float64; max_iterations = 20)
        both = ArgumentError("pass either `options` or keyword arguments, not both")
        @test_throws both solve_batch!(
            copy(X), S_good, prob, Newton(), p0; options = opt, max_iterations = 10)
        @test_throws both solve_batch!(
            copy(X), prob, Newton(), p0; options = opt, max_iterations = 10)

        # length(order) != length(X)
        @test_throws DimensionMismatch solve_batch!(
            copy(X), S_good, prob, Newton(), p0; order = [1])
        @test_throws DimensionMismatch solve_batch!(
            copy(X), prob, Newton(), p0; order = [1, 2, 3])

        # order that is not a permutation of 1:length(X): a duplicate, an out-of-range index
        @test_throws ArgumentError solve_batch!(
            copy(X), S_good, prob, Newton(), p0; order = [1, 1])
        @test_throws ArgumentError solve_batch!(
            copy(X), prob, Newton(), p0; order = [1, 3])

        # Empty batch (length(X) == 0)
        X_empty = SVector{2, Float64}[]
        S_empty = SolverStatus{Float64}[]
        @test solve_batch!(X_empty, S_empty, prob, Newton(), p0) === S_empty
        counts_empty = solve_batch!(X_empty, prob, Newton(), p0)
        @test counts_empty === (SUCCESS = 0, STALLED = 0, MAXITERS = 0,
            SINGULAR = 0, NONFINITE = 0, LINESEARCH_FAILED = 0)
        @test isconverged(counts_empty) === true

        # Empty batch still validates S, PerSystem, order, and options+kwargs
        @test_throws DimensionMismatch solve_batch!(X_empty, S_good, prob, Newton(), p0)
        @test_throws DimensionMismatch solve_batch!(
            X_empty, S_empty, prob, Newton(), PerSystem([p0]))
        @test_throws DimensionMismatch solve_batch!(
            X_empty, S_empty, prob, Newton(), p0; order = [1])
        @test_throws ArgumentError solve_batch!(
            X_empty, S_empty, prob, Newton(), p0; options = opt, f_reltol = 1e-4)

        # Mismatched SolverStatus{R} vs real(eltype(eltype(X))) throws ArgumentError on empty and non-empty batch
        X_f32_empty = SVector{2, Float32}[]
        @test_throws ArgumentError solve_batch!(X_f32_empty, S_empty, prob, Newton(), p0)
        X_f32 = [SVector(2.0f0, 1.0f0), SVector(1.5f0, 0.5f0)]
        @test_throws ArgumentError solve_batch!(X_f32, S_good, prob, Newton(), p0)

        # isconverged on ReturnCodeCounts NamedTuple
        all_ok = (SUCCESS = 10, STALLED = 0, MAXITERS = 0, SINGULAR = 0,
            NONFINITE = 0, LINESEARCH_FAILED = 0)
        @test isconverged(all_ok) === true
        for k in (:STALLED, :MAXITERS, :SINGULAR, :NONFINITE, :LINESEARCH_FAILED)
            bad = merge(all_ok, (k => 1,))
            @test isconverged(bad) === false
        end
    end

    @testset "Bitwise agreement, exact root, and reuse across backends, eltypes, methods" begin
        prob = _r2_coupled_problem()
        scalar_prob = _r2_scalar_problem()
        real_methods = (
            Newton(),
            Newton(; globalization = DogLeg()),
            Newton(; reuse = 2),
            Picard(; damping = 0.2)
        )
        complex_methods = (
            Newton(),
            Picard(; damping = 0.2)
        )

        for B in ARRAY_BACKENDS
            @testset "backend $B" begin
                for T in ELTYPES
                    R = real(T)
                    # System 1 starts at the exact dyadic root SVector(2, 1) for p = SVector(2.125, 1.5)
                    x_exact = SVector{2, T}(T(2), T(1))
                    p_exact = SVector{2, T}(T(R(2.125)), T(R(1.5)))
                    p_host = SVector{2, T}[
                        p_exact,
                        SVector{2, T}(T(R(1.25)), T(R(2.0))),
                        SVector{2, T}(T(R(1.5)), T(R(2.5))),
                        SVector{2, T}(T(R(1.75)), T(R(3.0)))
                    ]
                    x0_host = SVector{2, T}[
                        x_exact,
                        SVector{2, T}(T(R(2.5)), T(R(1.5))),
                        SVector{2, T}(T(R(3.5)), T(R(2.5))),
                        SVector{2, T}(T(R(4.0)), T(R(3.0)))
                    ]
                    ms = T <: Real ? real_methods : complex_methods
                    iters_newton = zeros(Int32, 4)
                    iters_reuse = zeros(Int32, 4)

                    for m in ms
                        ref_x = similar(x0_host)
                        ref_st = Vector{SolverStatus{R}}(undef, length(x0_host))
                        for k in eachindex(x0_host)
                            xk, stk = solve(
                                prob, x0_host[k], m, p_host[k]; max_iterations = 150)
                            ref_x[k] = xk
                            ref_st[k] = stk
                        end

                        X_dev = B(copy(x0_host))
                        S_dev = B(Vector{SolverStatus{R}}(undef, length(x0_host)))
                        P_dev = Adapt.adapt(B, PerSystem(copy(p_host)))

                        ret_S = solve_batch!(
                            X_dev, S_dev, prob, m, P_dev; max_iterations = 150)
                        @test ret_S === S_dev

                        X_h = Array(X_dev)
                        S_h = Array(S_dev)
                        @test X_h == ref_x
                        @test S_h == ref_st

                        # System 1 started at exact root: 0 iterations and bit-identical X[1] === x_exact
                        @test X_h[1] === x_exact
                        @test S_h[1].iterations == Int32(0)
                        @test S_h[1].code == SUCCESS
                        @test all(k -> S_h[k].iterations > 0 && S_h[k].code == SUCCESS, 2:4)

                        if m === real_methods[1]
                            iters_newton .= getfield.(S_h, :iterations)
                        elseif m === real_methods[3]
                            iters_reuse .= getfield.(S_h, :iterations)
                        end

                        # Count form
                        X_dev2 = B(copy(x0_host))
                        counts = solve_batch!(X_dev2, prob, m, P_dev; max_iterations = 150)
                        @test Array(X_dev2) == ref_x
                        @test counts === (
                            SUCCESS = 4,
                            STALLED = 0,
                            MAXITERS = 0,
                            SINGULAR = 0,
                            NONFINITE = 0,
                            LINESEARCH_FAILED = 0
                        )
                        @test isconverged(counts) === true
                    end

                    if T <: Real
                        @test iters_reuse[3] > iters_newton[3]
                        @test iters_reuse[4] > iters_newton[4]

                        # One system starts at singular Jacobian SVector(4, 4) while N - 1 converge
                        x0_sing = copy(x0_host)
                        x0_sing[2] = SVector{2, T}(T(4), T(4))
                        X_sing = B(copy(x0_sing))
                        S_sing = B(Vector{SolverStatus{R}}(undef, 4))
                        P_dev = Adapt.adapt(B, PerSystem(copy(p_host)))
                        solve_batch!(
                            X_sing, S_sing, prob, Newton(), P_dev; max_iterations = 60)
                        S_sing_h = Array(S_sing)
                        @test S_sing_h[2].code == SINGULAR
                        @test S_sing_h[1].code == SUCCESS
                        @test S_sing_h[3].code == SUCCESS
                        @test S_sing_h[4].code == SUCCESS

                        X_sing2 = B(copy(x0_sing))
                        counts_sing = solve_batch!(
                            X_sing2, prob, Newton(), P_dev; max_iterations = 60)
                        @test counts_sing.SINGULAR == 1
                        @test counts_sing.SUCCESS == 3
                        @test sum(counts_sing) == 4
                        @test isconverged(counts_sing) === false
                    end

                    # Scalar iterate T (Number)
                    s_x0_host = T[T(R(1.5) + R(0.2) * k) for k in 1:4]
                    s_p_host = T[T(R(2.0) + R(0.5) * k) for k in 1:4]
                    s_ref_x = similar(s_x0_host)
                    s_ref_st = Vector{SolverStatus{R}}(undef, 4)
                    for k in 1:4
                        xk, stk = solve(scalar_prob, s_x0_host[k], Newton(),
                            s_p_host[k]; max_iterations = 40)
                        s_ref_x[k] = xk
                        s_ref_st[k] = stk
                    end
                    s_X = B(copy(s_x0_host))
                    s_S = B(Vector{SolverStatus{R}}(undef, 4))
                    s_P = Adapt.adapt(B, PerSystem(copy(s_p_host)))
                    solve_batch!(s_X, s_S, scalar_prob, Newton(), s_P; max_iterations = 40)
                    @test Array(s_X) == s_ref_x
                    @test Array(s_S) == s_ref_st
                end
            end
        end
    end

    @testset "Host conversion ToReal{R}, and Options of another precision, on Float32 / ComplexF32" begin
        prob_coupled = _r2_coupled_problem()
        for B in ARRAY_BACKENDS, T in filter(T -> real(T) === Float32, ELTYPES)

            R = real(T)
            X = B(SVector{2, T}[
                SVector{2, T}(T(2.0f0), T(1.0f0)),
                SVector{2, T}(T(1.5f0), T(0.8f0))
            ])
            S = B(Vector{SolverStatus{R}}(undef, 2))
            p_coupled = Adapt.adapt(B,
                PerSystem(SVector{2, T}[
                    SVector{2, T}(T(2.125f0), T(1.5f0)),
                    SVector{2, T}(T(1.25f0), T(2.0f0))
                ]))
            # Float64 Picard(; damping = 0.7) converted on host; the options are Options{Float32}
            m_f64 = Picard(; damping = 0.7)
            opt_r = Options(T; f_abstol = 1e-5, f_reltol = 1e-5, max_iterations = 80)
            m_prep, opt_prep, state_prep = GeometricSolvers._prepare_batch(
                X, prob_coupled, m_f64, opt_r, (;))
            @test m_prep === Picard(0.7f0, Static(1.0f0))
            @test typeof(m_prep.damping) === Float32
            @test isbits(m_prep)
            @test opt_prep === opt_r
            @test opt_prep isa Options{Float32}
            @test isbits(opt_prep)
            @test isbits(state_prep)

            # Options of another precision are an error, not a conversion
            opt_f64 = Options(Float64; f_abstol = 1e-5, f_reltol = 1e-5, max_iterations = 80)
            @test_throws ArgumentError solve_batch!(
                X, S, prob_coupled, m_f64, p_coupled; options = opt_f64)

            solve_batch!(X, S, prob_coupled, m_f64, p_coupled; options = opt_r)
            @test all(isconverged, Array(S))
            @test eltype(eltype(X)) === T
            @test eltype(S) === SolverStatus{Float32}

            X2 = B(SVector{2, T}[
                SVector{2, T}(T(2.0f0), T(1.0f0)),
                SVector{2, T}(T(1.5f0), T(0.8f0))
            ])
            counts = solve_batch!(
                X2, prob_coupled, m_f64, p_coupled;
                f_abstol = 1e-5, f_reltol = 1e-5, max_iterations = 80)
            @test isconverged(counts)

            if B === Array && T === Float32
                # Newton(; globalization = Backtracking()) holds Float64 line-search fields
                # that fail dispatch on Float32 unless ToReal{Float32}() runs on the host
                X_bt = B(SVector{2, Float32}[
                    SVector(2.0f0, 1.0f0),
                    SVector(1.5f0, 0.8f0)
                ])
                S_bt = B(Vector{SolverStatus{Float32}}(undef, 2))
                m_bt_f64 = Newton(; globalization = Backtracking())
                solve_batch!(X_bt, S_bt, prob_coupled, m_bt_f64, p_coupled; options = opt_r)
                @test all(isconverged, Array(S_bt))
            end
        end
    end

    @testset "Heterogeneous outcomes and 6-bucket ReturnCode tally" begin
        prob_het = NonlinearProblem(_r2_hetero_res, _r2_hetero_jac)
        for B in ARRAY_BACKENDS
            @testset "backend $B" begin
                codes_p = [1, 2, 3, 4, 5, 6, 1, 3]
                x0 = [SVector(2.5, 3.5) for _ in 1:length(codes_p)]
                X1 = B(copy(x0))
                S1 = B(Vector{SolverStatus{Float64}}(undef, length(codes_p)))
                P1 = Adapt.adapt(B, PerSystem(copy(codes_p)))
                m = Newton(; globalization = Backtracking())

                solve_batch!(X1, S1, prob_het, m, P1; max_iterations = 3)
                S_host = Array(S1)
                @test S_host[1].code == SUCCESS
                @test S_host[2].code == STALLED
                @test S_host[3].code == MAXITERS
                @test S_host[4].code == SINGULAR
                @test S_host[5].code == NONFINITE
                @test S_host[6].code == LINESEARCH_FAILED
                @test S_host[7].code == SUCCESS
                @test S_host[8].code == MAXITERS

                X2 = B(copy(x0))
                counts = solve_batch!(X2, prob_het, m, P1; max_iterations = 3)
                @test counts === (
                    SUCCESS = 2,
                    STALLED = 1,
                    MAXITERS = 2,
                    SINGULAR = 1,
                    NONFINITE = 1,
                    LINESEARCH_FAILED = 1
                )
                @test sum(counts) == length(codes_p)
                @test isconverged(counts) === false
                @test Array(X2)[1] == Array(X1)[1]
                @test Array(X2)[7] == Array(X1)[7]
            end
        end
    end

    @testset "Ordering hook order" begin
        prob = _r2_coupled_problem()
        N = 7
        p_vec = [SVector(1.0 + 0.2 * i, 1.5 + 0.2 * i) for i in 1:N]
        x0_vec = [SVector(1.5 + 0.2 * i, 0.8 + 0.1 * i) for i in 1:N]
        perm = [5, 2, 7, 1, 4, 6, 3]
        inv_perm = invperm(perm)

        for B in ARRAY_BACKENDS
            @testset "backend $B" begin
                X_unord = B(copy(x0_vec))
                S_unord = B(Vector{SolverStatus{Float64}}(undef, N))
                P_dev = Adapt.adapt(B, PerSystem(copy(p_vec)))
                solve_batch!(X_unord, S_unord, prob, Newton(), P_dev)

                # (a) order = perm vs order = nothing on the same input arrays
                X_ord = B(copy(x0_vec))
                S_ord = B(Vector{SolverStatus{Float64}}(undef, N))
                order_dev = B(copy(perm))
                solve_batch!(X_ord, S_ord, prob, Newton(), P_dev; order = order_dev)
                @test Array(X_ord) == Array(X_unord)
                @test Array(S_ord) == Array(S_unord)

                # Host Vector{Int} order passed when X is on B (tests host-to-device adaptation)
                X_ord_host = B(copy(x0_vec))
                S_ord_host = B(Vector{SolverStatus{Float64}}(undef, N))
                solve_batch!(X_ord_host, S_ord_host, prob, Newton(), P_dev; order = perm)
                @test Array(X_ord_host) == Array(X_unord)
                @test Array(S_ord_host) == Array(S_unord)

                # AbstractRange order (1:N and N:-1:1) and non-Array host view under allowscalar(false)
                if B === JLArray
                    for rng in (1:N, N:-1:1)
                        X_rng = B(copy(x0_vec))
                        S_rng = B(Vector{SolverStatus{Float64}}(undef, N))
                        solve_batch!(X_rng, S_rng, prob, Newton(), P_dev; order = rng)
                        @test Array(X_rng) == Array(X_unord)
                        @test Array(S_rng) == Array(S_unord)

                        X_rng_cnt = B(copy(x0_vec))
                        counts_rng = solve_batch!(
                            X_rng_cnt, prob, Newton(), P_dev; order = rng)
                        @test Array(X_rng_cnt) == Array(X_unord)
                        @test counts_rng.SUCCESS == N
                    end

                    perm_view = @view perm[1:N]
                    X_vw = B(copy(x0_vec))
                    S_vw = B(Vector{SolverStatus{Float64}}(undef, N))
                    solve_batch!(X_vw, S_vw, prob, Newton(), P_dev; order = perm_view)
                    @test Array(X_vw) == Array(X_unord)
                    @test Array(S_vw) == Array(S_unord)
                end

                # (b) Permuted inputs X₀[perm] and PerSystem(P[perm]) followed by invperm(perm)
                X_perm_in = B(x0_vec[perm])
                S_perm_in = B(Vector{SolverStatus{Float64}}(undef, N))
                P_perm_in = Adapt.adapt(B, PerSystem(p_vec[perm]))
                solve_batch!(X_perm_in, S_perm_in, prob, Newton(), P_perm_in)
                @test Array(X_perm_in)[inv_perm] == Array(X_unord)
                @test Array(S_perm_in)[inv_perm] == Array(S_unord)

                # Count form with order
                X_cnt = B(copy(x0_vec))
                counts = solve_batch!(X_cnt, prob, Newton(), P_dev; order = perm)
                @test Array(X_cnt) == Array(X_unord)
                @test counts.SUCCESS == N
            end
        end

        # Spy AbstractVector{Int} on CPU() verifies every index 1:length(X) is read once by the
        # permutation check and once by the kernel
        spy = SpyOrderVector(copy(perm), zeros(Int, N))
        X_spy = copy(x0_vec)
        S_spy = Vector{SolverStatus{Float64}}(undef, N)
        solve_batch!(X_spy, S_spy, prob, Newton(), PerSystem(copy(p_vec)); order = spy)
        @test spy.reads == fill(2, N)

        spy2 = SpyOrderVector(copy(perm), zeros(Int, N))
        X_spy2 = copy(x0_vec)
        solve_batch!(X_spy2, prob, Newton(), PerSystem(copy(p_vec)); order = spy2)
        @test spy2.reads == fill(2, N)
    end

    @testset "Order adaptation and kernel synchronization" begin
        ord_cpu = [3, 1, 2]
        @test GeometricSolvers._adapt_order(KernelAbstractions.CPU(), nothing) === nothing
        @test GeometricSolvers._adapt_order(KernelAbstractions.CPU(), ord_cpu) === ord_cpu

        ord_jl = JLArray([3, 1, 2])
        jl_backend = KernelAbstractions.get_backend(ord_jl)
        @test GeometricSolvers._adapt_order(jl_backend, nothing) === nothing
        @test GeometricSolvers._adapt_order(jl_backend, ord_jl) === ord_jl
        @test GeometricSolvers._adapt_order(jl_backend, 3:-1:1) === 3:-1:1
        @test GeometricSolvers._adapt_order(jl_backend, 1:3) === 1:3

        ord_adapted = GeometricSolvers._adapt_order(jl_backend, ord_cpu)
        @test ord_adapted isa JLArray{Int, 1}
        @test Array(ord_adapted) == ord_cpu

        ord_view_adapted = GeometricSolvers._adapt_order(jl_backend, @view(ord_cpu[1:3]))
        @test ord_view_adapted isa JLArray{Int, 1}
        @test Array(ord_view_adapted) == ord_cpu

        # an order on a device, with a batch on the host, is copied to the host
        ord_host = GeometricSolvers._adapt_order(KernelAbstractions.CPU(), ord_jl)
        @test ord_host isa Vector{Int}
        @test ord_host == ord_cpu

        # an order whose backend cannot be read raises the error of `get_backend`
        unread = UnreadOrderVector([3, 1, 2])
        @test_throws ArgumentError KernelAbstractions.get_backend(unread)
        @test_throws ArgumentError GeometricSolvers._adapt_order(
            KernelAbstractions.CPU(), unread)
        @test_throws ArgumentError solve_batch!(
            [SVector(2.0, 1.0) for _ in 1:3], _r2_coupled_problem(), Newton(),
            SVector(2.125, 1.5); order = unread)

        # both forms launch through `_launch!`, which waits for the kernel
        ci = only(Base.code_lowered(GeometricSolvers._launch!))
        @test any(
            stmt -> stmt == GlobalRef(GeometricSolvers, :synchronize) ||
                    stmt == KernelAbstractions.synchronize,
            ci.code
        )
        for sig in (
            Tuple{
            Vector{SVector{2, Float64}},
            Vector{SolverStatus{Float64}},
            NonlinearProblem,
            typeof(Newton()),
            Nothing
        },
            Tuple{
            Vector{SVector{2, Float64}},
            NonlinearProblem,
            typeof(Newton()),
            Nothing
        }
        )
            bf = Base.bodyfunction(which(solve_batch!, sig))
            ci = first(Base.code_lowered(bf))
            @test any(stmt -> stmt == GlobalRef(GeometricSolvers, :_launch!), ci.code)
        end
    end

    @testset "Hostile inputs across batch" begin
        prob = _r2_coupled_problem()
        p_vec = [SVector(2.125, 1.5) for _ in 1:4]
        nan_v = SVector(NaN, 1.0)
        inf_v = SVector(Inf, 1.0)
        sub_v = SVector(floatmin(Float64) / 16, 1.0)
        good_v = SVector(2.0, 1.0)

        X = [nan_v, inf_v, sub_v, good_v]
        S = Vector{SolverStatus{Float64}}(undef, 4)
        solve_batch!(X, S, prob, Newton(), PerSystem(p_vec))
        @test S[1].code == NONFINITE
        @test S[2].code == NONFINITE
        @test S[3].code == SUCCESS
        @test S[4].code == SUCCESS

        X2 = [nan_v, inf_v, sub_v, good_v]
        counts = solve_batch!(X2, prob, Newton(), PerSystem(p_vec))
        @test counts.NONFINITE == 2
        @test counts.SUCCESS == 2
    end

    @testset "Type stability (@inferred), allocations (@allocated == 0), and JET (@test_opt)" begin
        allocations(f::F, a::A) where {F, A} = (f(a); @allocated f(a))
        allocations(f::F, a::A, b::B) where {F, A, B} = (f(a, b); @allocated f(a, b))

        prob = _r2_coupled_problem()
        X = [SVector(2.0, 1.0), SVector(1.5, 0.8)]
        S = Vector{SolverStatus{Float64}}(undef, 2)
        ps = PerSystem([SVector(2.125, 1.5), SVector(1.25, 2.0)])
        ps_tup = PerSystem((SVector(2.125, 1.5), SVector(1.25, 2.0)))
        opt = Options(Float64)
        codes = [SUCCESS, SINGULAR, MAXITERS]

        @test (@inferred parameters(ps, 1)) === SVector(2.125, 1.5)
        @test (@inferred parameters(ps_tup, 1)) === SVector(2.125, 1.5)
        @test (@inferred parameters(1.0, 1)) === 1.0
        @test (@inferred GeometricSolvers._system_index(nothing, 1)) === 1
        @test (@inferred GeometricSolvers._system_index([2, 1], 1)) === 2
        @test (@inferred GeometricSolvers._system_index(2:-1:1, 1)) === 2
        @test (@inferred GeometricSolvers._tally_return_codes(codes)) === (
            SUCCESS = 1, STALLED = 0, MAXITERS = 1,
            SINGULAR = 1, NONFINITE = 0, LINESEARCH_FAILED = 0)
        @test (@inferred solve_batch!(copy(X), S, prob, Newton(), ps; options = opt)) === S
        counts = @inferred solve_batch!(copy(X), prob, Newton(), ps; options = opt)
        @test (@inferred isconverged(counts)) === true

        # Allocation assertions via function barrier after warmup
        @test allocations(collect, 1:4) > 0
        @test allocations(parameters, ps, 1) == 0
        @test allocations(parameters, ps_tup, 1) == 0
        @test allocations(parameters, SVector(2.125, 1.5), 1) == 0
        @test allocations(GeometricSolvers._system_index, nothing, 1) == 0
        @test allocations(GeometricSolvers._system_index, [2, 1], 1) == 0
        @test allocations(GeometricSolvers._system_index, 2:-1:1, 1) == 0
        @test allocations(GeometricSolvers._tally_return_codes, codes) == 0
        @test allocations(isconverged, counts) == 0

        JET.@test_opt target_modules = (GeometricSolvers,) parameters(ps, 1)
        JET.@test_opt target_modules = (GeometricSolvers,) parameters(ps_tup, 1)
        JET.@test_opt target_modules = (GeometricSolvers,) parameters(1.0, 1)
        JET.@test_opt target_modules = (GeometricSolvers,) GeometricSolvers._system_index(
            nothing, 1)
        JET.@test_opt target_modules = (GeometricSolvers,) GeometricSolvers._system_index(
            [2, 1], 1)
        JET.@test_opt target_modules = (GeometricSolvers,) GeometricSolvers._system_index(
            2:-1:1, 1)
        JET.@test_opt target_modules = (GeometricSolvers,) GeometricSolvers._prepare_batch(
            X, prob, Newton(), opt, (;))
        JET.@test_opt target_modules = (GeometricSolvers,) GeometricSolvers._tally_return_codes(
            codes)
        JET.@test_opt target_modules = (GeometricSolvers,) isconverged(counts)
    end
end
