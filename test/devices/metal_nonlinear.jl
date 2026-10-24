# The R1 nonlinear loop inside a `@kernel` on Metal.

using Adapt: Adapt
using KernelAbstractions: @index, @kernel, get_backend, synchronize
using Metal: Metal
using StaticArrays: SMatrix, SVector
using Test

using GeometricSolvers
using GeometricSolvers: Backtracking, Bisection, LUFactorization, Newton, NonlinearProblem,
                        Options, Picard, QRFactorization, SUCCESS, Static, StrongWolfe,
                        ToReal, init, solve!!

include("../helpers/matrix.jl")
include("../helpers/r1problems.jl")

const N_METAL_NL = 4
const B_METAL_NL = 8

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
    if Metal.functional()
        for T in filter(T -> real(T) === Float32, ELTYPES)
            R = real(T)
            perm = r1_shift(T, N_METAL_NL)
            x₀, p₀ = r1_inputs(T, N_METAL_NL)
            xs_host = [x₀ .+ T(k) / T(20) for k in 1:B_METAL_NL]
            ps_host = [p₀ .+ T(k) / T(30) for k in 1:B_METAL_NL]

            prob_ad = NonlinearProblem(R1Coupled(perm))
            prob_j = NonlinearProblem(R1Coupled(perm), (y, q) -> r1_coupled_jacobian(y, q, perm))
            opt = Options(T)

            xs_dev = Metal.MtlArray(xs_host)
            ps_dev = Metal.MtlArray(ps_host)
            xs_out = similar(xs_dev)
            codes_out = Metal.MtlArray(zeros(Int32, B_METAL_NL))
            iters_out = Metal.MtlArray(zeros(Int32, B_METAL_NL))
            backend = get_backend(xs_dev)

            for ls in (LUFactorization(), QRFactorization()),
                glob in (Static(), Backtracking(), Bisection(), StrongWolfe()),
                prob in (prob_ad, prob_j)
                m = Adapt.adapt(ToReal{R}(), Newton(; linsolve = ls, globalization = glob))
                state₀ = init(m, prob, x₀, p₀)
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
                    x_ref, st_ref, _ = solve!!(state₀, prob, xs_host[k], m, ps_host[k], opt)
                    @test codes_got[k] == Int32(SUCCESS)
                    @test codes_got[k] == Int32(st_ref.code)
                    @test iters_got[k] == st_ref.iterations
                    @test isapprox(xs_got[k], x_ref; rtol = 10 * eps(R))
                end
            end

            # Picard on Metal
            prob_pic = NonlinearProblem(R1Scaled())
            ps_pic_host = [SVector{N_METAL_NL, T}([T(1) / T(2) for _ in 1:N_METAL_NL])
                           for _ in 1:B_METAL_NL]
            ps_pic_dev = Metal.MtlArray(ps_pic_host)
            for glob in (Static(), Backtracking())
                m_pic = Adapt.adapt(
                    ToReal{R}(), Picard(; damping = 0.8, globalization = glob))
                state_pic = init(m_pic, prob_pic, x₀, ps_pic_host[1])
                @test isbits(m_pic)
                @test isbits(state_pic)

                solve_nonlinear_kernel!(backend)(
                    xs_out, codes_out, iters_out, xs_dev, ps_pic_dev,
                    prob_pic, m_pic, state_pic, opt; ndrange = B_METAL_NL)
                synchronize(backend)

                xs_got = Array(xs_out)
                codes_got = Array(codes_out)
                for k in 1:B_METAL_NL
                    x_ref, st_ref, _ = solve!!(
                        state_pic, prob_pic, xs_host[k], m_pic, ps_pic_host[k], opt)
                    @test codes_got[k] == Int32(SUCCESS)
                    @test isapprox(xs_got[k], x_ref; rtol = 10 * eps(R))
                end
            end
        end
    else
        @test_skip Metal.functional()
    end
end
