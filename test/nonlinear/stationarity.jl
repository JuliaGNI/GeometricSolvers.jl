using ADTypes: AutoFiniteDiff, AutoForwardDiff
using FiniteDiff: FiniteDiff
using ForwardDiff
using GeometricSolvers
using GeometricSolvers: AbstractNonlinearProblem, ChunkedHessianForwardDiff, DIStationarity
using GeometricSolvers: gradient_hessian!!, jacobian!!, jvp!!, prepare_ad, residual!!
using GPUArraysCore: allowscalar
using JLArrays: JLArray
using LinearAlgebra: Diagonal, dot, norm
using StaticArrays: SMatrix, SVector, setindex
using Test

include("../helpers/matrix.jl")

function fd_grad_entry(f, x0::SVector{4, Float64}, p, E_idx, b_s, h1::Float64, a::Int)
    ei = E_idx[a]
    xp = setindex(x0, x0[ei] + h1, ei)
    xm = setindex(x0, x0[ei] - h1, ei)
    return (f(xp, p) - f(xm, p)) / (2 * h1) + b_s[a]
end

function fd_grad_vec(f, x0::SVector{4, Float64}, p, E_idx, b_s, h1::Float64)
    n = length(E_idx)
    vals = ntuple(a -> fd_grad_entry(f, x0, p, E_idx, b_s, h1, a), n)
    return SVector{n, Float64}(vals)
end

function fd_hess_entry(f, x0::SVector{4, Float64}, p, E_idx, U_idx, h2::Float64, idx::Int)
    n = length(E_idx)
    a = mod1(idx, n)
    b_col = cld(idx, n)
    ei = E_idx[a]
    uj = U_idx[b_col]
    xp = setindex(x0, x0[ei] + h2, ei)
    xm = setindex(x0, x0[ei] - h2, ei)
    xpp = setindex(xp, ei == uj ? x0[ei] + 2 * h2 : x0[uj] + h2, uj)
    xpm = setindex(xp, ei == uj ? x0[ei] : x0[uj] - h2, uj)
    xmp = setindex(xm, ei == uj ? x0[ei] : x0[uj] + h2, uj)
    xmm = setindex(xm, ei == uj ? x0[ei] - 2 * h2 : x0[uj] - h2, uj)
    num = f(xpp, p) - f(xpm, p) - f(xmp, p) + f(xmm, p)
    return num / (4 * h2^2)
end

function fd_hess_mat(f, x0::SVector{4, Float64}, p, E_idx, U_idx, h2::Float64)
    n = length(E_idx)
    vals = ntuple(idx -> fd_hess_entry(f, x0, p, E_idx, U_idx, h2, idx), n * n)
    return SMatrix{n, n, Float64, n * n}(vals)
end

@testset "StationarityProblem construction and validation (catches MU8)" begin
    s_quad(x, p) = sum(abs2, x)

    @test StationarityProblem <: AbstractNonlinearProblem
    @test NonlinearProblem <: AbstractNonlinearProblem

    prob_all = StationarityProblem(s_quad)
    @test prob_all.E === Colon()
    @test prob_all.U === Colon()
    @test prob_all.b === nothing
    @test isbits(prob_all)

    prob_tup = StationarityProblem(s_quad; E = (1, 2), U = (2, 3), b = SVector(1.0, -1.0))
    @test prob_tup.E === SVector(1, 2)
    @test prob_tup.U === SVector(2, 3)
    @test prob_tup.b === SVector(1.0, -1.0)
    @test isbits(prob_tup)

    prob_pos = StationarityProblem(s_quad, 1:2, 2:3, SVector(0.5, 1.5))
    @test prob_pos.E == 1:2
    @test prob_pos.U == 2:3
    @test prob_pos.b === SVector(0.5, 1.5)

    # Construction-time validation errors
    @test_throws ArgumentError StationarityProblem(s_quad; E = (1, 2), U = (1, 2, 3))
    @test_throws ArgumentError StationarityProblem(s_quad; E = (), U = ())
    @test_throws ArgumentError StationarityProblem(s_quad; E = :, U = (1, 2))
    @test_throws ArgumentError StationarityProblem(s_quad; E = (1, 2), U = :)
    @test_throws ArgumentError StationarityProblem(s_quad; E = (0, 1), U = (1, 2))
    @test_throws ArgumentError StationarityProblem(s_quad; E = (1, 2), U = (-1, 2))
    @test_throws ArgumentError StationarityProblem(s_quad; E = (1, 1), U = (1, 2))
    @test_throws ArgumentError StationarityProblem(s_quad; E = (1, 2), U = (2, 2))
    @test_throws ArgumentError StationarityProblem(s_quad; E = [1, 2, 1], U = [1, 2, 3])
    @test_throws ArgumentError StationarityProblem(s_quad; E = [1, 2, 3], U = [2, 3, 2])

    # Solve-time validation errors (out-of-bounds E/U and mismatched b length)
    prob_oob_e = StationarityProblem(s_quad; E = (1, 5), U = (1, 2))
    @test_throws ArgumentError solve(prob_oob_e, SVector(1.0, 2.0, 3.0), Newton())
    @test_throws ArgumentError init(prob_oob_e, [1.0, 2.0, 3.0], Newton())

    prob_oob_u = StationarityProblem(s_quad; E = (1, 2), U = (2, 4))
    @test_throws ArgumentError solve(prob_oob_u, SVector(1.0, 2.0, 3.0), Newton())
    @test_throws ArgumentError init(prob_oob_u, [1.0, 2.0, 3.0], Newton())

    b3 = SVector(1.0, 2.0, 3.0)
    prob_bad_b = StationarityProblem(s_quad; E = (1, 2), U = (2, 3), b = b3)
    @test_throws ArgumentError solve(prob_bad_b, SVector(1.0, 2.0, 3.0), Newton())
    @test_throws ArgumentError init(prob_bad_b, [1.0, 2.0, 3.0], Newton())

    prob_ok = StationarityProblem(s_quad; E = (1, 2), U = (2, 3))
    @test_throws ArgumentError solve(prob_ok, SVector(1.0, 2.0, 3.0), Newton(); b = b3)
    solver_ok = init(prob_ok, [1.0, 2.0, 3.0], Newton())
    @test_throws ArgumentError solve!(solver_ok, [1.0, 2.0, 3.0]; b = [1.0, 2.0, 3.0])
end

@testset "Check 1: Gradient, Hessian, and HVP vs central differences (MU1-MU4)" begin
    allowscalar(false)
    calls_s = Ref(0)
    # A smooth non-quadratic action that a host array evaluates without a heap allocation, as the
    # allocation checks below require.
    function action_smooth(x, p)
        calls_s[] += 1
        α = p === nothing ? one(eltype(x)) : eltype(x)(p)
        val = sin(x[1] * x[2]) + 0.5 * x[2] * x[3]^2 - cos(x[1] + 2 * x[4]) + x[3] * x[4]
        return α * val
    end
    # A smooth non-quadratic action written with gathers through an index vector and reductions
    # only, which is how a device array reaches its entries: `x[i]` is scalar indexing there and a
    # gather is not. `perm` cyclically shifts the entries by one; the central-difference oracle of
    # the `JLArray` leg is taken from this same closure on the host.
    function action_smooth_device(perm)
        return function (x, p)
            calls_s[] += 1
            α = p === nothing ? one(eltype(x)) : eltype(x)(p)
            xp = x[perm]
            xq = xp[perm]
            return α * (0.5 * sum(abs2, x) + 0.15 * sum((x .^ 2) .* xp) +
                    0.1 * sum(x .* xp .* xq) + 0.25 * sum(sin.(x .+ xp)))
        end
    end

    x0_s = SVector(1.2, -0.5, 0.8, 0.4)
    h1 = 1e-6
    h2 = 1e-4
    ad_fd = AutoForwardDiff()

    case1 = ((1, 2, 3, 4), (1, 2, 3, 4), SVector(0.1, -0.2, 0.3, -0.4))
    case2 = ((1, 2, 3), (2, 3, 4), SVector(0.3, -0.4, 0.7))
    case3 = ((2, 3, 4), (1, 2, 3), SVector(-0.2, 0.4, 0.1))
    for (E_idx, U_idx, b_s) in (case1, case2, case3)
        n = length(E_idx)
        F_cd = fd_grad_vec(action_smooth, x0_s, 1.5, E_idx, b_s, h1)
        J_cd = fd_hess_mat(action_smooth, x0_s, 1.5, E_idx, U_idx, h2)
        # The oracle of the `JLArray` leg: the gather form of the action, on the host.
        action_dev = action_smooth_device([2, 3, 4, 1])
        F_cd_dev = fd_grad_vec(action_dev, x0_s, 1.5, E_idx, b_s, h1)
        J_cd_dev = fd_hess_mat(action_dev, x0_s, 1.5, E_idx, U_idx, h2)

        if E_idx == (1, 2, 3, 4)
            prob_s = StationarityProblem(action_smooth; b = b_s)
        else
            prob_s = StationarityProblem(action_smooth; E = E_idx, U = U_idx, b = b_s)
        end

        # R1 single-call checks
        calls_s[] = 0
        F_r1 = residual!!(nothing, prob_s, x0_s, 1.5)
        @test calls_s[] == 1
        @test norm(F_r1 - F_cd) / norm(F_cd) <= 1e-6

        calls_s[] = 0
        J_r1 = jacobian!!(nothing, ad_fd, prob_s, x0_s, 1.5)
        @test calls_s[] == 1
        @test norm(J_r1 - J_cd) / norm(J_cd) <= 1e-6

        calls_s[] = 0
        F_gh, J_gh = gradient_hessian!!(nothing, nothing, ad_fd, prob_s, x0_s, 1.5)
        @test calls_s[] == 1
        @test F_gh == F_r1
        @test J_gh == J_r1

        v_r1 = SVector{n, Float64}(ntuple(i -> 0.3 * i - 0.5, n))
        calls_s[] = 0
        jv_r1 = jvp!!(nothing, ad_fd, prob_s, x0_s, v_r1, 1.5)
        @test calls_s[] == 1
        tol_jvp = 10 * n * eps(Float64) * norm(J_r1) * norm(v_r1)
        @test norm(jv_r1 - J_r1 * v_r1) <= tol_jvp

        # R3 checks on Array and JLArray. A device array cannot reach its entries by scalar
        # indexing, so its leg evaluates the gather form of the action and is checked against that
        # form's own central differences, taken on the host.
        for Arr in ARRAY_BACKENDS
            action_a = Arr === Array ? action_smooth :
                       action_smooth_device(Arr([2, 3, 4, 1]))
            F_cd_a = Arr === Array ? F_cd : F_cd_dev
            J_cd_a = Arr === Array ? J_cd : J_cd_dev
            x0_a = Arr(Vector(x0_s))
            b_a = Arr(Vector(b_s))
            if E_idx == (1, 2, 3, 4)
                prob_a = StationarityProblem(action_a; b = b_a)
            else
                e_vec, u_vec = collect(E_idx), collect(U_idx)
                prob_a = StationarityProblem(action_a; E = e_vec, U = u_vec, b = b_a)
            end
            F_a = Arr(zeros(Float64, n))
            J_a = Arr(zeros(Float64, n, n))
            Jv_a = Arr(zeros(Float64, n))
            v_a = Arr(Vector(v_r1))

            prep = prepare_ad(ad_fd, prob_a, F_a, x0_a, 1.5)
            @test prep isa ChunkedHessianForwardDiff

            residual!!(F_a, prep, prob_a, x0_a, 1.5)
            @test norm(Array(F_a) - F_cd_a) / norm(F_cd_a) <= 1e-6

            jacobian!!(J_a, prep, prob_a, x0_a, 1.5)
            @test norm(Array(J_a) - J_cd_a) / norm(J_cd_a) <= 1e-6

            fill!(F_a, 0.0)
            fill!(J_a, 0.0)
            gradient_hessian!!(F_a, J_a, prep, prob_a, x0_a, 1.5)
            @test norm(Array(F_a) - F_cd_a) / norm(F_cd_a) <= 1e-6
            @test norm(Array(J_a) - J_cd_a) / norm(J_cd_a) <= 1e-6

            jvp!!(Jv_a, prep, prob_a, x0_a, v_a, 1.5)
            tol_jvp_a = 10 * n * eps(Float64) * norm(Array(J_a)) * norm(v_r1)
            @test norm(Array(Jv_a) - Array(J_a) * Vector(v_r1)) <= tol_jvp_a

            if Arr === Array && Base.JLOptions().check_bounds != 1
                # skipped where a run forces `--check-bounds=yes`
                @test (@allocated residual!!(F_a, prep, prob_a, x0_a, 1.5)) == 0
                @test (@allocated jacobian!!(J_a, prep, prob_a, x0_a, 1.5)) == 0
                al_gh = @allocated gradient_hessian!!(F_a, J_a, prep, prob_a, x0_a, 1.5)
                @test al_gh == 0
                @test (@allocated jvp!!(Jv_a, prep, prob_a, x0_a, v_a, 1.5)) == 0
            end
        end

        # R3 DifferentiationInterface path (AutoFiniteDiff on CPU Array)
        x0_cpu = Vector(x0_s)
        b_cpu = Vector(b_s)
        if E_idx == (1, 2, 3, 4)
            prob_cpu = StationarityProblem(action_smooth; b = b_cpu)
        else
            ec, uc = collect(E_idx), collect(U_idx)
            prob_cpu = StationarityProblem(action_smooth; E = ec, U = uc, b = b_cpu)
        end
        F_cpu = zeros(Float64, n)
        J_cpu = zeros(Float64, n, n)
        Jv_cpu = zeros(Float64, n)
        v_cpu = Vector(v_r1)
        prep_di = prepare_ad(AutoFiniteDiff(), prob_cpu, F_cpu, x0_cpu, 1.5)
        @test prep_di isa DIStationarity
        residual!!(F_cpu, prep_di, prob_cpu, x0_cpu, 1.5)
        @test norm(F_cpu - F_cd) / norm(F_cd) <= 1e-5
        jacobian!!(J_cpu, prep_di, prob_cpu, x0_cpu, 1.5)
        @test norm(J_cpu - J_cd) / norm(J_cd) <= 1e-4
        jvp!!(Jv_cpu, prep_di, prob_cpu, x0_cpu, v_cpu, 1.5)
        @test norm(Jv_cpu - J_cd * v_cpu) / norm(J_cd * v_cpu) <= 1e-4
    end

    # Scalar (Number) R1 stationarity check and nested ForwardDiff safety
    f_num(x, p) = ForwardDiff.derivative(t -> sin(x * t), 1.0) + 0.25 * x^4
    prob_num = StationarityProblem(f_num; b = -0.5)
    F_num = residual!!(nothing, prob_num, 1.2, nothing)
    J_num = jacobian!!(nothing, ad_fd, prob_num, 1.2, nothing)
    Jv_num = jvp!!(nothing, ad_fd, prob_num, 1.2, 2.0, nothing)
    @test isapprox(F_num, cos(1.2) - 1.2 * sin(1.2) + 1.2^3 - 0.5; rtol = 1e-12)
    @test isapprox(J_num, -2 * sin(1.2) - 1.2 * cos(1.2) + 3 * 1.2^2; rtol = 1e-12)
    @test isapprox(Jv_num, 2.0 * J_num; rtol = 1e-12)
end

@testset "Check 2: Quadratic action converges in 1 Newton step across REAL_ELTYPES" begin
    allowscalar(false)
    m_ldlt, m_lu = LDLTFactorization(), LUFactorization()
    m_qr, m_svd = QRFactorization(), SVDFactorization()
    lmethods = (m_ldlt, m_lu, m_qr, m_svd)
    for T in REAL_ELTYPES
        A_s = SMatrix{3, 3, T, 9}(0, 2, -1, 2, 1, 1 // 2, -1, 1 // 2, -3)
        c_s = SVector{3, T}(1, -2, 1 // 2)
        b_s = SVector{3, T}(1 // 4, -1 // 2, 1)
        x_exact = A_s \ (c_s - b_s)

        f_quad_s(x, p) = T(1 // 2) * dot(x, A_s * x) - dot(c_s, x)
        prob_s = StationarityProblem(f_quad_s; b = b_s)
        for ls in lmethods
            x_sol, st_sol = solve(prob_s, zeros(SVector{3, T}), Newton(; linsolve = ls))
            @test st_sol.code == SUCCESS
            @test st_sol.iterations == Int32(1)
            @test norm(x_sol - x_exact) <= 32 * eps(T)

            solver_s = init(prob_s, zeros(SVector{3, T}), Newton(; linsolve = ls))
            x_sol2, st_sol2 = solve!(solver_s)
            @test st_sol2.code == SUCCESS
            @test st_sol2.iterations == Int32(1)
            @test x_sol2 == x_sol
        end

        # R3 Array and JLArray
        for Arr in ARRAY_BACKENDS
            A_a = Arr(Matrix(A_s))
            c_a = Arr(Vector(c_s))
            b_a = Arr(Vector(b_s))
            f_quad_a(x, p) = T(1 // 2) * sum(x .* (A_a * x)) - sum(c_a .* x)
            prob_a = StationarityProblem(f_quad_a; b = b_a)
            for ls in (m_ldlt, m_lu)
                x0_a = Arr(zeros(T, 3))
                x_sol_a, st_sol_a = solve(prob_a, x0_a, Newton(; linsolve = ls))
                @test st_sol_a.code == SUCCESS
                @test st_sol_a.iterations == Int32(1)
                @test norm(Array(x_sol_a) - Vector(x_exact)) <= 32 * eps(T)

                solver_a = init(prob_a, x0_a, Newton(; linsolve = ls))
                x_in = Arr(zeros(T, 3))
                x_sol_a2, st_sol_a2 = solve!(solver_a, x_in)
                @test x_sol_a2 === x_in
                @test st_sol_a2.code == SUCCESS
                @test st_sol_a2.iterations == Int32(1)
                @test norm(Array(x_sol_a2) - Vector(x_exact)) <= 32 * eps(T)
            end
        end
    end
end

function lobatto_entry(nodes::SVector{4, Float64}, idx::Int)
    m = mod1(idx, 4)
    j = cld(idx, 4)
    tm = nodes[m]
    if m == j
        s = 0.0
        for k in 1:4
            k == j && continue
            s += 1.0 / (nodes[j] - nodes[k])
        end
        return s
    else
        num = 1.0
        den = 1.0
        for k in 1:4
            k == j && continue
            den *= (nodes[j] - nodes[k])
            k == m && continue
            num *= (tm - nodes[k])
        end
        return num / den
    end
end

function ho_nl_res(K_stiff, q0_ho::Float64, p0_ho::Float64, u)
    X = SVector(q0_ho, u[1], u[2], u[3])
    g = K_stiff * X
    return SVector(g[1] + p0_ho, g[2], g[3])
end

@testset "Check 3: Harmonic oscillator cG(3) variational integrator to 8 ulp" begin
    allowscalar(false)
    ω_ho = 1.3
    h_ho = 0.2
    q0_ho = 1.1
    p0_ho = -0.7
    s5 = sqrt(5.0)
    nodes = SVector(0.0, (5.0 - s5) / 10.0, (5.0 + s5) / 10.0, 1.0)
    weights = SVector(1.0 / 12.0, 5.0 / 12.0, 5.0 / 12.0, 1.0 / 12.0)
    D_lob = SMatrix{4, 4, Float64, 16}(ntuple(idx -> lobatto_entry(nodes, idx), 16))

    K_kin = (1.0 / h_ho) * (D_lob' * Diagonal(weights) * D_lob)
    K_pot = (h_ho * ω_ho^2) * Diagonal(weights)
    K_stiff = K_kin - K_pot

    function action_ho(X, p)
        return 0.5 * sum(X .* (K_stiff * X))
    end

    b_ho = SVector(p0_ho, 0.0, 0.0)
    prob_ho_stat = StationarityProblem(action_ho; E = (1, 2, 3), U = (2, 3, 4), b = b_ho)
    prob_ho_nl = NonlinearProblem((u, p) -> ho_nl_res(K_stiff, q0_ho, p0_ho, u))

    X0_ho = SVector(q0_ho, q0_ho, q0_ho, q0_ho)
    X_stat, st_stat = solve(prob_ho_stat, X0_ho, Newton())
    u_nl, st_nl = solve(prob_ho_nl, SVector(q0_ho, q0_ho, q0_ho), Newton())
    X_nl = SVector(q0_ho, u_nl[1], u_nl[2], u_nl[3])
    tol_ho = 8 * eps(Float64) * max(1.0, norm(X_nl, Inf))

    @test st_stat.code == SUCCESS
    @test st_nl.code == SUCCESS
    @test X_stat[1] == q0_ho
    @test norm(X_stat - X_nl, Inf) <= tol_ho

    # Symmetric internal-stage solve E = U = (2, 3) with fixed endpoints
    q1_ho = X_nl[4]
    prob_ho_sym = StationarityProblem(action_ho; E = (2, 3), U = (2, 3))
    X0_sym = SVector(q0_ho, q0_ho, q0_ho, q1_ho)
    m_newt_ldlt = Newton(; linsolve = LDLTFactorization())
    X_sym_ldlt, st_sym_ldlt = solve(prob_ho_sym, X0_sym, m_newt_ldlt)
    @test st_sym_ldlt.code == SUCCESS
    @test X_sym_ldlt[1] == q0_ho && X_sym_ldlt[4] == q1_ho
    @test norm(X_sym_ldlt - X_nl, Inf) <= tol_ho

    # R3 Array and JLArray on the harmonic oscillator variational step
    K_mat = Matrix(K_stiff)
    for Arr in ARRAY_BACKENDS
        K_dev = Arr(K_mat)
        action_ho_r3(X, p) = 0.5 * sum(X .* (K_dev * X))
        b_r3 = Arr([p0_ho, 0.0, 0.0])
        prob_r3 = StationarityProblem(action_ho_r3; E = 1:3, U = 2:4, b = b_r3)
        X0_r3 = Arr([q0_ho, q0_ho, q0_ho, q0_ho])
        X_r3, st_r3 = solve(prob_r3, X0_r3, Newton())
        @test st_r3.code == SUCCESS
        @test Array(X_r3)[1] == q0_ho
        @test norm(Array(X_r3) - Vector(X_nl), Inf) <= tol_ho
    end
end

@testset "Datum b forms, keyword overrides, globalizations, Picard, and solve_batch!" begin
    # Callable b(p) and keyword b override on init / solve / solve!
    A_ind = SMatrix{2, 2, Float64, 4}(0.0, 2.0, 2.0, -1.0)
    f_ind(x, p) = 0.5 * dot(x, A_ind * x)
    prob_callable_b = StationarityProblem(f_ind; b = p -> SVector(p, -2 * p))
    m_newt_ldlt = Newton(; linsolve = LDLTFactorization())
    x_c1, st_c1 = solve(prob_callable_b, SVector(0.0, 0.0), m_newt_ldlt, 1.5)
    @test st_c1.code == SUCCESS
    @test isapprox(x_c1, -(A_ind \ SVector(1.5, -3.0)); rtol = 1e-12)

    # Keyword b override
    b_kw = SVector(-0.5, 1.25)
    x_kw, st_kw = solve(prob_callable_b, SVector(0.0, 0.0), m_newt_ldlt, 1.5; b = b_kw)
    @test st_kw.code == SUCCESS
    @test isapprox(x_kw, -(A_ind \ b_kw); rtol = 1e-12)

    # R3 solve!(solver, x, p; b = b_step) with zero allocations
    A_mat = Matrix(A_ind)
    f_mat(x, p) = 0.5 * dot(x, A_mat, x)
    prob_r3 = StationarityProblem(f_mat; b = [1.0, -1.0])
    x0_vec = [0.0, 0.0]
    solver_r3 = init(prob_r3, x0_vec, m_newt_ldlt)
    b_step = [0.5, -1.5]
    x_step = [0.0, 0.0]
    solve!(solver_r3, x_step, nothing; b = b_step)
    @test isapprox(x_step, -(A_mat \ b_step); rtol = 1e-12)
    if Base.JLOptions().check_bounds != 1
        # skipped where a run forces `--check-bounds=yes`
        @test (@allocated solve!(solver_r3, x_step, nothing; b = b_step)) == 0
    end

    # A non-quadratic action whose stationarity equations are `x₁ + 0.4x₁³ = 1` and
    # `0.4x₂³ - x₂ + 2 = 0`, with the root `(0.7972810583155764, -2.1876029597039026)`, and whose
    # Hessian is indefinite for `|x₂| < 1/√1.2`. `S` is never the merit.
    # S(x, p) = 0.5 * x1^2 - 0.5 * x2^2 + 0.1 * (x1^4 + x2^4) - x1 + 2 * x2
    function saddle_action(x, p)
        return 0.5 * x[1]^2 - 0.5 * x[2]^2 + 0.1 * (x[1]^4 + x[2]^4) - x[1] + 2 * x[2]
    end
    prob_saddle = StationarityProblem(saddle_action)
    x_ref, st_ref = solve(prob_saddle, SVector(0.8, 1.5), m_newt_ldlt)
    @test st_ref.code == SUCCESS
    @test norm(x_ref - SVector(0.7972810583155764, -2.1876029597039026), Inf) <= 1e-9

    # Every globalization runs the same loop on a `StationarityProblem` as on the equivalent
    # hand-written `NonlinearProblem`: the two agree in their return code, and their iterates
    # agree to rounding. The hand-written residual is the analytic gradient of `saddle_action`.
    prob_nl_saddle = NonlinearProblem((x, p) -> SVector(
        x[1] + 0.4 * x[1]^3 - 1, -x[2] + 0.4 * x[2]^3 + 2))
    for g in (Static(), Backtracking(), Bisection(), StrongWolfe(), DogLeg())
        m_g = Newton(; linsolve = LDLTFactorization(), globalization = g)
        x_g, st_g = solve(prob_saddle, SVector(0.8, 1.5), m_g)
        x_nl, st_nl = solve(prob_nl_saddle, SVector(0.8, 1.5), m_g)
        @test st_g.code === st_nl.code
        @test norm(x_g - x_nl, Inf) <= 8 * eps(Float64) * max(1.0, norm(x_nl, Inf))
    end

    # Picard on StationarityProblem
    f_pic(x, p) = 0.5 * ((x[1] - 1.0)^2 + (x[2] + 0.5)^2) + 0.05 * sin(x[1] * x[2])
    prob_pic = StationarityProblem(f_pic)
    x_pic, st_pic = solve(prob_pic, SVector(0.8, -0.4), Picard(; damping = 0.8))
    x_newt, _ = solve(prob_pic, SVector(0.8, -0.4), Newton())
    @test st_pic.code == SUCCESS
    @test isapprox(x_pic, x_newt; atol = 1e-6)

    # Every globalization converges to the same root of a convex action
    for g in (Static(), Backtracking(), Bisection(), StrongWolfe(), DogLeg())
        m_g = Newton(; linsolve = LDLTFactorization(), globalization = g)
        x_g, st_g = solve(prob_pic, SVector(0.8, -0.4), m_g)
        @test st_g.code == SUCCESS
        @test isapprox(x_g, x_newt; atol = 1e-7)
    end

    # R2 solve_batch! on StationarityProblem (both full-state U = : and subset E != U)
    f_b1(x, p) = 0.5 * dot(x, A_ind * x) - p * (x[1] - 2 * x[2])
    prob_b1 = StationarityProblem(f_b1)
    X_batch = [SVector(0.0, 0.0) for _ in 1:4]
    ps = PerSystem([1.0, 2.0, -1.0, 0.5])
    counts = solve_batch!(X_batch, prob_b1, m_newt_ldlt, ps)
    @test counts.SUCCESS == 4
    for i in 1:4
        p_i = ps.data[i]
        @test isapprox(X_batch[i], A_ind \ SVector(p_i, -2 * p_i); rtol = 1e-12)
    end

    # solve_batch! with PerSystem datum b (both keyword b and problem-stored prob.b) and JLArray
    b_per_vec = [SVector(0.25 * i, -0.5 * i) for i in 1:4]
    b_ps = PerSystem(b_per_vec)
    X_b_kw = [SVector(0.0, 0.0) for _ in 1:4]
    counts_b_kw = solve_batch!(X_b_kw, prob_b1, m_newt_ldlt, ps; b = b_ps)
    @test counts_b_kw.SUCCESS == 4
    for i in 1:4
        p_i = ps.data[i]
        rhs_i = SVector(p_i, -2 * p_i) - b_per_vec[i]
        @test isapprox(X_b_kw[i], A_ind \ rhs_i; rtol = 1e-12)
    end

    prob_b_ps = StationarityProblem(f_b1; b = b_ps)
    X_b_prob = [SVector(0.0, 0.0) for _ in 1:4]
    S_b_prob = Vector{SolverStatus{Float64}}(undef, 4)
    ret_s = solve_batch!(X_b_prob, S_b_prob, prob_b_ps, m_newt_ldlt, ps)
    @test ret_s === S_b_prob
    for i in 1:4
        @test S_b_prob[i].code == SUCCESS
        @test isapprox(X_b_prob[i], X_b_kw[i]; rtol = 1e-12)
    end

    allowscalar(false)
    X_jl_b = JLArray([SVector(0.0, 0.0) for _ in 1:4])
    ps_jl = PerSystem(JLArray([1.0, 2.0, -1.0, 0.5]))
    b_ps_jl = PerSystem(JLArray(b_per_vec))
    counts_jl = solve_batch!(X_jl_b, prob_b1, m_newt_ldlt, ps_jl; b = b_ps_jl)
    @test counts_jl.SUCCESS == 4
    @test isapprox(Array(X_jl_b), X_b_kw; rtol = 1e-12)

    # solve_batch! with status vector S and subset U = (2, 3) preserving X[idx][1]
    f_b2(x, p) = 0.5 * ((x[2] - x[1])^2 + (x[3] - p)^2)
    prob_b2 = StationarityProblem(f_b2; E = (2, 3), U = (2, 3))
    X_sub = [SVector(Float64(i), 0.0, 0.0) for i in 1:4]
    S_sub = Vector{SolverStatus{Float64}}(undef, 4)
    ret_sub = solve_batch!(X_sub, S_sub, prob_b2, m_newt_ldlt, ps)
    @test ret_sub === S_sub
    for i in 1:4
        @test S_sub[i].code == SUCCESS
        @test X_sub[i][1] == Float64(i)
        @test isapprox(X_sub[i][2], Float64(i); atol = 1e-12)
        @test isapprox(X_sub[i][3], ps.data[i]; atol = 1e-12)
    end
end

@testset "D1 runtime index rejection, 7-arg gradient_hessian!!, and D3 chunk symmetry" begin
    # D1: runtime Vector{Int} indices must throw ArgumentError on R1/R2 StaticVector or Number
    f_d1(x, p) = 0.5 * sum(abs2, x)
    prob_rt_both = StationarityProblem(f_d1; E = [1, 2], U = [1, 2])
    prob_rt_e = StationarityProblem(f_d1; E = [1, 2], U = (1, 2))
    prob_rt_u = StationarityProblem(f_d1; E = (1, 2), U = [1, 2])
    x_sv3 = SVector(1.0, 2.0, 3.0)
    @test_throws ArgumentError solve(prob_rt_both, x_sv3, Newton())
    @test_throws ArgumentError solve(prob_rt_e, x_sv3, Newton())
    @test_throws ArgumentError solve(prob_rt_u, x_sv3, Newton())
    @test_throws ArgumentError init(prob_rt_both, x_sv3, Newton())
    @test_throws ArgumentError solve_batch!([x_sv3, x_sv3], prob_rt_both, Newton())
    prob_rt_num = StationarityProblem((x, p) -> 0.5 * x^2; E = [1], U = [1])
    @test_throws ArgumentError solve(prob_rt_num, 1.0, Newton())

    # 7-argument gradient_hessian!!(F, J, prep, backend, prob, x, p) in R1 and R3
    ad_fd = AutoForwardDiff()
    prob_gh = StationarityProblem(f_d1; b = SVector(0.5, -1.0))
    x_sv2 = SVector(1.5, -2.0)
    F_r1_7, J_r1_7 = gradient_hessian!!(
        nothing, nothing, nothing, ad_fd, prob_gh, x_sv2, nothing)
    F_r1_6, J_r1_6 = gradient_hessian!!(nothing, nothing, ad_fd, prob_gh, x_sv2, nothing)
    @test F_r1_7 == F_r1_6
    @test J_r1_7 == J_r1_6

    prob_gh_vec = StationarityProblem(f_d1; b = [0.5, -1.0])
    x_v2 = [1.5, -2.0]
    F_v2 = zeros(2)
    J_v2 = zeros(2, 2)
    prep_v2 = prepare_ad(ad_fd, prob_gh_vec, F_v2, x_v2, nothing)
    gradient_hessian!!(F_v2, J_v2, prep_v2, ad_fd, prob_gh_vec, x_v2, nothing)
    @test isapprox(F_v2, Vector(F_r1_6); rtol = 1e-12)
    @test isapprox(J_v2, Matrix(J_r1_6); rtol = 1e-12)

    # D3: When E == U and C = cld(n, N) > 1, ChunkedHessianForwardDiff evaluates S C*(C+1)/2 times
    calls_d3 = Ref(0)
    function f_d3(x, p)
        calls_d3[] += 1
        s = zero(eltype(x))
        for i in eachindex(x)
            s += i * x[i]^2
        end
        for i in 1:(length(x) - 1)
            s += 0.3 * x[i] * x[i + 1]
        end
        return 0.5 * s
    end
    prob_d3 = StationarityProblem(f_d3)
    x_d3 = collect(range(0.5, 1.5; length = 6))
    F_d3 = zeros(6)
    J_d3 = zeros(6, 6)
    ad_c2 = AutoForwardDiff(; chunksize = 2)
    prep_d3 = prepare_ad(ad_c2, prob_d3, F_d3, x_d3, nothing)
    @test prep_d3.sym === true

    calls_d3[] = 0
    jacobian!!(J_d3, prep_d3, prob_d3, x_d3, nothing)
    @test calls_d3[] == 6

    F_full = zeros(6)
    J_full = zeros(6, 6)
    prep_full = prepare_ad(AutoForwardDiff(; chunksize = 6), prob_d3, F_full, x_d3, nothing)
    gradient_hessian!!(F_full, J_full, prep_full, prob_d3, x_d3, nothing)
    @test isapprox(J_d3, J_full; rtol = 1e-12)

    fill!(F_d3, 0.0)
    fill!(J_d3, 0.0)
    calls_d3[] = 0
    gradient_hessian!!(F_d3, J_d3, prep_d3, prob_d3, x_d3, nothing)
    @test calls_d3[] == 6
    @test isapprox(F_d3, F_full; rtol = 1e-12)
    @test isapprox(J_d3, J_full; rtol = 1e-12)
end
