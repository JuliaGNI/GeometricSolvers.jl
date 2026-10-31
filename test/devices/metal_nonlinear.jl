# The R1 nonlinear loop and R2 batched solve! on Metal.

using Adapt: Adapt
using GPUArraysCore: GPUArraysCore
using KernelAbstractions: @index, @kernel, get_backend, synchronize
using Metal: Metal
using StaticArrays: SMatrix, SVector
using Test

using GeometricSolvers
using GeometricSolvers: Backtracking, DogLeg, LUFactorization, Newton, NonlinearProblem,
                        Options, PerSystem, Picard, QRFactorization, SINGULAR, SUCCESS,
                        SolverStatus, Static, ToReal, init_state, isconverged, solve,
                        solve!,
                        solve!!

include("../helpers/matrix.jl")
include("../helpers/r1problems.jl")
include("../helpers/metal_device.jl")

const N_METAL_NL = 4
const B_METAL_NL = 8

@kernel function metal_singular_kernel!(codes, xs, @Const(prob), @Const(x₀))
    i = @index(Global, Linear)
    @inbounds begin
        x, st = solve(prob, x₀, Newton())
        codes[i] = Int32(st.code)
        xs[i] = x
    end
end

@kernel function metal_newton_solve_kernel!(codes, xs, @Const(prob), @Const(x₀), @Const(p))
    i = @index(Global, Linear)
    @inbounds begin
        x, st = solve(prob, x₀, Newton(), p)
        codes[i] = Int32(st.code)
        xs[i] = x
    end
end

@kernel function metal_picard_solve_kernel!(codes, xs, @Const(prob), @Const(x₀), @Const(p))
    i = @index(Global, Linear)
    @inbounds begin
        m = Adapt.adapt(ToReal{Float32}(), Picard(; damping = 0.7))
        x, st = solve(prob, x₀, m, p)
        codes[i] = Int32(st.code)
        xs[i] = x
    end
end

@kernel function solve_nonlinear_kernel!(xs_out, codes_out, iters_out, @Const(xs_in),
        @Const(ps_in), prob, method, state₀, opt)
    i = @index(Global, Linear)
    @inbounds begin
        x, st, _ = solve!!(state₀, prob, xs_in[i], method, ps_in[i], opt)
        xs_out[i] = x
        codes_out[i] = Int32(st.code)
        iters_out[i] = st.iterations
    end
end

@testset "Metal R1 nonlinear kernel" begin
    if Metal.functional() && !on_virtual_gpu()
        for T in filter(T -> real(T) === Float32, ELTYPES)
            R = real(T)

            # 1. Singular start inside a kernel calling solve
            x₀_s = SVector{2, T}(zero(T), zero(T))
            prob_s = NonlinearProblem((y, _) -> y .* y .- one(eltype(y)))
            codes_s = Metal.MtlArray(zeros(Int32, 2))
            xs_s = Metal.MtlArray([x₀_s, x₀_s])
            backend = get_backend(xs_s)
            metal_singular_kernel!(backend)(codes_s, xs_s, prob_s, x₀_s; ndrange = 2)
            synchronize(backend)
            @test all(==(Int32(SINGULAR)), Array(codes_s))
            @test all(x -> x === x₀_s, Array(xs_s))

            # 2. Newton solve inside a kernel on y .* y .- q from (3, 5) with p = (4, 9)
            x₀_c = SVector{2, T}(T(3), T(5))
            p_c = SVector{2, T}(T(4), T(9))
            prob_c = NonlinearProblem((y, q) -> y .* y .- q)
            _, st_host = solve(prob_c, x₀_c, Newton(), p_c)
            codes_c = Metal.MtlArray(zeros(Int32, 2))
            xs_c = Metal.MtlArray([x₀_c, x₀_c])
            metal_newton_solve_kernel!(backend)(
                codes_c, xs_c, prob_c, x₀_c, p_c; ndrange = 2)
            synchronize(backend)
            xs_c_got = Array(xs_c)
            @test all(==(Int32(SUCCESS)), Array(codes_c))
            @test all(==(Int32(st_host.code)), Array(codes_c))
            @test eltype(xs_c_got[1]) === T
            @test all(
                x -> isapprox(x, SVector{2, T}(T(2), T(3)); rtol = 10 * sqrt(eps(R))),
                xs_c_got)

            # 3. solve!! batch kernel on R1Coupled and R1Scaled
            perm = r1_shift(T, N_METAL_NL)
            x₀, p₀ = r1_inputs(T, N_METAL_NL)
            xs_host = [x₀ .+ T(k) / T(20) for k in 1:B_METAL_NL]
            ps_host = [p₀ .+ T(k) / T(30) for k in 1:B_METAL_NL]

            prob_ad = NonlinearProblem(R1Coupled(perm))
            prob_j = NonlinearProblem(
                R1Coupled(perm), (y, q) -> r1_coupled_jacobian(y, q, perm))
            opt = Options(T)

            xs_dev = Metal.MtlArray(xs_host)
            ps_dev = Metal.MtlArray(ps_host)
            xs_out = similar(xs_dev)
            codes_out = Metal.MtlArray(zeros(Int32, B_METAL_NL))
            iters_out = Metal.MtlArray(zeros(Int32, B_METAL_NL))

            for ls in (LUFactorization(), QRFactorization()),
                glob in (Static(), Backtracking()),
                prob in (prob_ad, prob_j)
                m = Adapt.adapt(
                    ToReal{R}(), Newton(; linsolve = ls, globalization = glob))
                state₀ = init_state(m, prob, x₀, p₀)
                @test isbits(prob)
                @test isbits(m)
                @test isbits(state₀)
                @test isbits(opt)

                solve_nonlinear_kernel!(backend)(
                    xs_out, codes_out, iters_out, xs_dev, ps_dev,
                    prob, m, state₀, opt; ndrange = B_METAL_NL)
                synchronize(backend)

                xs_got = Array(xs_out)
                codes_got = Array(codes_out)
                iters_got = Array(iters_out)
                for k in 1:B_METAL_NL
                    x_ref, st_ref, _ = solve!!(
                        state₀, prob, xs_host[k], m, ps_host[k], opt)
                    @test codes_got[k] == Int32(SUCCESS)
                    @test codes_got[k] == Int32(st_ref.code)
                    @test iters_got[k] == st_ref.iterations
                    @test isapprox(
                        xs_got[k], x_ref; rtol = 10 * eps(R), atol = 10 * eps(R))
                end
            end

            # Picard batch kernel on Metal
            prob_pic = NonlinearProblem(R1Scaled())
            ps_pic_host = [SVector{N_METAL_NL, T}([T(1) / T(2) for _ in 1:N_METAL_NL])
                           for _ in 1:B_METAL_NL]
            ps_pic_dev = Metal.MtlArray(ps_pic_host)
            for glob in (Static(), Backtracking())
                m_pic = Adapt.adapt(
                    ToReal{R}(), Picard(; damping = 0.8, globalization = glob))
                state_pic = init_state(m_pic, prob_pic, x₀, ps_pic_host[1])
                @test isbits(m_pic)
                @test isbits(state_pic)

                solve_nonlinear_kernel!(backend)(
                    xs_out, codes_out, iters_out, xs_dev, ps_pic_dev,
                    prob_pic, m_pic, state_pic, opt; ndrange = B_METAL_NL)
                synchronize(backend)

                xs_got = Array(xs_out)
                codes_got = Array(codes_out)
                for k in 1:B_METAL_NL
                    x_ref, _, _ = solve!!(
                        state_pic, prob_pic, xs_host[k], m_pic, ps_pic_host[k], opt)
                    @test codes_got[k] == Int32(SUCCESS)
                    @test isapprox(
                        xs_got[k], x_ref; rtol = 10 * eps(R), atol = 10 * eps(R))
                end
            end
        end

        # 4. Picard solve inside a kernel with Adapt.adapt(ToReal{Float32}(), Picard(; damping = 0.7))
        x₀_p = SVector{2, Float32}(3.0f0, 5.0f0)
        p_p = SVector{2, Float32}(2.0f0, 4.0f0)
        prob_p = NonlinearProblem((y, q) -> y .- q)
        codes_p = Metal.MtlArray(zeros(Int32, 2))
        xs_p = Metal.MtlArray([x₀_p, x₀_p])
        backend_p = get_backend(xs_p)
        metal_picard_solve_kernel!(backend_p)(codes_p, xs_p, prob_p, x₀_p, p_p; ndrange = 2)
        synchronize(backend_p)
        xs_p_got = Array(xs_p)
        @test all(==(Int32(SUCCESS)), Array(codes_p))
        @test eltype(xs_p_got[1]) === Float32
        @test all(
            x -> isapprox(x, p_p; rtol = 10 * sqrt(eps(Float32))), xs_p_got)

        # 5. R2 batched solve! on Metal (per-system status and return-code count forms)
        @testset "Metal R2 batched solve!" begin
            GPUArraysCore.allowscalar(false)
            for T in filter(T -> real(T) === Float32, ELTYPES)
                R = real(T)
                perm = r1_shift(T, N_METAL_NL)
                x₀, p₀ = r1_inputs(T, N_METAL_NL)
                xs_host = [x₀ .+ T(k) / T(20) for k in 1:B_METAL_NL]
                ps_host = [p₀ .+ T(k) / T(30) for k in 1:B_METAL_NL]
                prob_j = NonlinearProblem(
                    R1Coupled(perm), (y, q) -> r1_coupled_jacobian(y, q, perm))
                prob_ad = NonlinearProblem(R1Coupled(perm))

                for m in (
                        Newton(),
                        Newton(; globalization = DogLeg()),
                        Newton(; reuse = 2)
                    ),
                    prob in (prob_j, prob_ad)

                    m_dev = Adapt.adapt(ToReal{R}(), m)
                    opt = Options(R)
                    state₀ = init_state(m_dev, prob, x₀, p₀)

                    X_dev = Metal.MtlArray(xs_host)
                    S_dev = Metal.MtlArray(Vector{SolverStatus{R}}(undef, B_METAL_NL))
                    P_dev = PerSystem(Metal.MtlArray(ps_host))

                    ret_S = solve!(X_dev, S_dev, prob, m, P_dev)
                    @test ret_S === S_dev

                    xs_got = Array(X_dev)
                    st_got = Array(S_dev)
                    for k in 1:B_METAL_NL
                        x_ref, st_ref, _ = solve!!(
                            state₀, prob, xs_host[k], m_dev, ps_host[k], opt)
                        @test st_got[k].code == SUCCESS
                        @test st_got[k].code == st_ref.code
                        @test st_got[k].iterations == st_ref.iterations
                        @test isapprox(
                            xs_got[k], x_ref; rtol = 8 * eps(R), atol = 8 * eps(R))
                    end

                    X_dev2 = Metal.MtlArray(xs_host)
                    counts = solve!(X_dev2, prob, m, P_dev)
                    @test isconverged(counts)
                    @test counts.SUCCESS == B_METAL_NL
                    @test sum(counts) == B_METAL_NL
                    xs_got2 = Array(X_dev2)
                    for k in 1:B_METAL_NL
                        @test isapprox(
                            xs_got2[k], xs_got[k]; rtol = 8 * eps(R), atol = 8 * eps(R))
                    end
                end

                # Picard(; damping = 0.7) on R1Scaled with Float64 Options / keyword tolerances
                prob_pic = NonlinearProblem(R1Scaled())
                ps_pic_host = [SVector{N_METAL_NL, T}([T(1) / T(2) for _ in 1:N_METAL_NL])
                               for _ in 1:B_METAL_NL]
                m_pic = Picard(; damping = 0.7)
                m_pic_dev = Adapt.adapt(ToReal{R}(), m_pic)
                opt_pic_f64 = Options(Float64; f_abstol = 1e-5, f_reltol = 1e-5)
                opt_pic = convert(Options{R}, opt_pic_f64)
                state_pic = init_state(m_pic_dev, prob_pic, x₀, ps_pic_host[1])

                X_pic = Metal.MtlArray(xs_host)
                S_pic = Metal.MtlArray(Vector{SolverStatus{R}}(undef, B_METAL_NL))
                P_pic = PerSystem(Metal.MtlArray(ps_pic_host))
                solve!(X_pic, S_pic, prob_pic, m_pic, P_pic; options = opt_pic_f64)
                xs_pic_got = Array(X_pic)
                st_pic_got = Array(S_pic)
                for k in 1:B_METAL_NL
                    x_ref, st_ref, _ = solve!!(
                        state_pic, prob_pic, xs_host[k], m_pic_dev, ps_pic_host[k], opt_pic)
                    @test st_pic_got[k].code == SUCCESS
                    @test st_pic_got[k].iterations == st_ref.iterations
                    @test isapprox(
                        xs_pic_got[k], x_ref; rtol = 8 * eps(R), atol = 8 * eps(R))
                end

                X_pic2 = Metal.MtlArray(xs_host)
                counts_pic = solve!(
                    X_pic2, prob_pic, m_pic, P_pic; f_abstol = 1e-5, f_reltol = 1e-5)
                @test isconverged(counts_pic)
                @test counts_pic.SUCCESS == B_METAL_NL

                # AbstractRange order (B_METAL_NL:-1:1 and 1:B_METAL_NL) on MtlArray
                for rng in (1:B_METAL_NL, B_METAL_NL:-1:1)
                    X_ord = Metal.MtlArray(xs_host)
                    S_ord = Metal.MtlArray(Vector{SolverStatus{R}}(undef, B_METAL_NL))
                    solve!(X_ord, S_ord, prob_pic, m_pic, P_pic;
                        order = rng, options = opt_pic_f64)
                    xs_ord_got = Array(X_ord)
                    for k in 1:B_METAL_NL
                        @test isapprox(
                            xs_ord_got[k], xs_pic_got[k]; rtol = 8 * eps(R), atol = 8 *
                                                                                    eps(R))
                    end
                end
            end
        end
    elseif Metal.functional()
        @test_skip on_virtual_gpu()  # issue #34
    else
        @test_skip Metal.functional()
    end
end
