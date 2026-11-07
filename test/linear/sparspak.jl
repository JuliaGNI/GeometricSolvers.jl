# Sparse LU factorisation via Sparspak and colored sparse AD via DifferentiationInterface.

using ADTypes: AutoForwardDiff, AutoSparse, NoSparsityDetector
using ForwardDiff: ForwardDiff
using GeometricSolvers
using GeometricSolvers: Backtracking, Bisection, DogLeg, LUFactorization, Newton,
                        NonlinearProblem, Picard, QRFactorization, SINGULAR, SUCCESS,
                        SVDFactorization, init, returncode, solve, solve!, solve_batch!
using LinearAlgebra: norm, transpose
using SparseArrays: SparseArrays, SparseMatrixCSC, getcolptr, nnz, nonzeros, nzrange,
                    rowvals, sparse, spzeros
using SparseMatrixColorings: GreedyColoringAlgorithm, column_groups
using Sparspak: Sparspak
using StaticArrays: SVector
using Test

include("../helpers/matrix.jl")
include("../helpers/sparse_problems.jl")

@testset "Sparspak extension and dispatch" begin
    ext = Base.get_extension(GeometricSolvers, :GeometricSolversSparspakExt)
    @test ext !== nothing

    prob_analytic, _, z0, p_val, z_star = sparse_bracket_problem(125, Float64)
    r_star = similar(z0)
    prob_analytic.F(r_star, z_star, p_val)
    @test norm(r_star, Inf) <= 10 * eps(eltype(z0))

    P = GeometricSolvers.jacobian_prototype(prob_analytic)
    @test P isa SparseMatrixCSC{Float64, Int}
    @test size(P) == (250, 250)
    @test nnz(P) == 28 * 125
    @test P != transpose(P)
    @test GeometricSolvers.jacobian_pattern(prob_analytic, nothing) === P
    @test GeometricSolvers.jacobian_pattern(nothing, nothing) === nothing

    m = Newton()
    solver = init(prob_analytic, copy(z0), m, p_val)
    st = solver.state
    @test st.J === st.fac.factors
    @test st.J !== P
    which_fac = which(
        GeometricSolvers.factorize!!, (typeof(st.fac), typeof(m.linsolve), typeof(st.J))
    )
    @test parentmodule(which_fac) === ext
    @test GeometricSolvers.dense_factor(LUFactorization(), prob_analytic, z0, p_val) isa
          ext.SparspakLU
    @test_throws MethodError GeometricSolvers.init_linear(
        LUFactorization(), [1.0 0.0; 0.0 1.0], [1.0, 2.0]
    )

    prob_analytic.J(st.J, z0, p_val)
    fac = GeometricSolvers.factorize!!(st.fac, LUFactorization(), st.J)
    @test fac.info == 0
    @test returncode(fac) === SUCCESS

    # Out-of-place and in-place (d === r) linear solves
    rhs = collect(range(0.1, 1.0; length = 250))
    d_oop = zeros(eltype(z0), 250)
    GeometricSolvers.ldiv!!(d_oop, fac, rhs)
    @test isapprox(st.J * d_oop, rhs; rtol = 1e-10)
    d_ip = copy(rhs)
    GeometricSolvers.ldiv!!(d_ip, fac, d_ip)
    @test d_ip == d_oop

    # factorize!! when J !== fac.factors copies nonzeros into fac.factors
    J_other = copy(st.J)
    nonzeros(J_other) .*= 2.0
    fac2 = GeometricSolvers.factorize!!(st.fac, LUFactorization(), J_other)
    @test fac2.info == 0
    @test nonzeros(fac2.factors) == nonzeros(J_other)
end

@testset "Explicit LUFactorization(T) with a sparse prototype: T = $T" for T in REAL_ELTYPES
    tol = T === Float64 ? 1e-10 : 1.0f-4
    prob_analytic, _, z0, p_val, z_star = sparse_bracket_problem(125, T)
    solver = init(prob_analytic, copy(z0), Newton(; linsolve = LUFactorization(T)),
        p_val; f_abstol = tol, f_reltol = zero(T))
    x, st = solve!(solver)
    @test st.code === SUCCESS
    @test x != z0
    r = similar(x)
    prob_analytic.F(r, x, p_val)
    @test norm(r, 2) <= tol
    @test isapprox(x, z_star; rtol = T === Float64 ? 1e-9 : 1.0f-3)
end

@testset "DogLeg Jacobian buffer: T = $T" for T in REAL_ELTYPES
    tol = T === Float64 ? 1e-10 : 1.0f-4
    prob_analytic, _, z0, p_val, _ = sparse_bracket_problem(125, T)
    P = GeometricSolvers.jacobian_prototype(prob_analytic)

    # DogLeg reads the Jacobian from a buffer of its own, so that buffer carries the prototype's
    # pattern and is the matrix that `jacobian!!` fills
    solver = init(prob_analytic, copy(z0), Newton(), p_val)
    J_dl = GeometricSolvers.init_newton_J(
        Newton(; globalization = DogLeg()), solver.state.fac, nothing, z0)
    @test J_dl isa SparseMatrixCSC{T, Int}
    @test J_dl !== solver.state.fac.factors
    @test getcolptr(J_dl) == getcolptr(P)
    @test rowvals(J_dl) == rowvals(P)

    solver_dl = init(
        prob_analytic, copy(z0), Newton(; globalization = DogLeg()), p_val;
        f_abstol = tol, f_reltol = zero(T)
    )
    _, st_dl = solve!(solver_dl)
    @test st_dl.code === SUCCESS
    @test st_dl.fnorm <= tol
    @test solver_dl.state.J !== solver_dl.state.fac.factors
    @test getcolptr(solver_dl.state.J) == getcolptr(P)
    @test rowvals(solver_dl.state.J) == rowvals(P)
end

@testset "Sparse block problem: N = $N, T = $T" for N in (125, 768), T in REAL_ELTYPES

    tol = T === Float64 ? 1e-10 : 1.0f-4
    prob_analytic, prob_ad, z0, p_val, z_star = sparse_bracket_problem(N, T)
    P = GeometricSolvers.jacobian_prototype(prob_analytic)
    nz0 = copy(nonzeros(P))
    colptr0 = copy(getcolptr(P))
    rowval0 = copy(rowvals(P))

    # Analytic Jacobian solve
    solver_a = init(
        prob_analytic, copy(z0), Newton(), p_val; f_abstol = tol, f_reltol = zero(T))
    x_a, st_a = solve!(solver_a)
    @test st_a.code === SUCCESS
    @test x_a != z0
    r_a = similar(x_a)
    prob_analytic.F(r_a, x_a, p_val)
    @test norm(r_a, 2) <= tol
    @test isapprox(x_a, z_star; rtol = T === Float64 ? 1e-9 : 1.0f-3)
    @test solver_a.state.J !== P
    @test getcolptr(solver_a.state.J) == colptr0
    @test rowvals(solver_a.state.J) == rowval0
    @test any(iszero, nonzeros(solver_a.state.J))
    @test nonzeros(P) == nz0

    # AutoSparse AD solve
    ad_backend = AutoSparse(
        AutoForwardDiff(); coloring_algorithm = GreedyColoringAlgorithm())
    m_ad = Newton(; jacobian = ad_backend)
    solver_d = init(prob_ad, copy(z0), m_ad, p_val; f_abstol = tol, f_reltol = zero(T))
    x_d, st_d = solve!(solver_d)
    @test st_d.code === SUCCESS
    @test x_d != z0
    r_d = similar(x_d)
    prob_ad.F(r_d, x_d, p_val)
    @test norm(r_d, 2) <= tol
    @test isapprox(x_d, x_a; rtol = T === Float64 ? 1e-10 : 1.0f-4)
    @test solver_d.state.J !== P
    @test getcolptr(solver_d.state.J) == colptr0
    @test rowvals(solver_d.state.J) == rowval0
    @test any(iszero, nonzeros(solver_d.state.J))
    @test nonzeros(P) == nz0

    ncols = length(column_groups(solver_d.state.jacprep.jacprep))
    @test ncols <= 24
    if N == 768
        @test ncols < (2 * N) ÷ 12
    end

    calls = Ref(0)
    prob_cnt = NonlinearProblem(
        (r, z, p) -> (calls[] += 1; prob_ad.F(r, z, p));
        jacobian_prototype = P
    )
    prep_cnt = GeometricSolvers.prepare_ad(ad_backend, prob_cnt, similar(z0), z0, p_val)
    f_fd = (r, x) -> prob_ad.F(r, x, p_val)
    J_ad = copy(P)
    J_ana = copy(P)
    for pt in (z0, x_d)
        calls[] = 0
        GeometricSolvers.jacobian!!(J_ad, prep_cnt, prob_cnt, pt, p_val)
        @test 1 <= calls[] <= ncols
        if N == 768
            @test calls[] < (2 * N) ÷ 12
        end

        prob_analytic.J(J_ana, pt, p_val)
        @test any(iszero, nonzeros(J_ad))
        @test any(iszero, nonzeros(J_ana))
        J_fd = ForwardDiff.jacobian(f_fd, similar(pt), pt)
        M_ad = Matrix(J_ad)
        @test isapprox(M_ad, J_fd; rtol = 8 * eps(T), atol = 8 * eps(T))
        @test isapprox(M_ad, Matrix(J_ana); rtol = 8 * eps(T), atol = 8 * eps(T))
    end
    GeometricSolvers.jacobian!!(J_ad, prep_cnt, prob_cnt, z_star, p_val)
    prob_analytic.J(J_ana, z_star, p_val)
    @test any(iszero, nonzeros(J_ad))
    @test any(iszero, nonzeros(J_ana))
end

@testset "Detector override: N = $N, T = $T" for N in (125, 768), T in REAL_ELTYPES

    prob_analytic, prob_ad, z0, p_val, _ = sparse_bracket_problem(N, T)
    P = GeometricSolvers.jacobian_prototype(prob_analytic)

    # KnownJacobianSparsityDetector(P) overrides a user-supplied detector (DenseSparsityDetector or NoSparsityDetector)
    for det in (
        GeometricSolvers.DI.DenseSparsityDetector(AutoForwardDiff(); atol = 0.0),
        NoSparsityDetector()
    )
        calls = Ref(0)
        F_cnt = (r, z, p) -> (calls[] += 1; prob_ad.F(r, z, p))
        prob_cnt = NonlinearProblem(F_cnt; jacobian_prototype = P)
        m_det = Newton(;
            jacobian = AutoSparse(
            AutoForwardDiff();
            sparsity_detector = det,
            coloring_algorithm = GreedyColoringAlgorithm()
        )
        )
        solver_det = init(prob_cnt, copy(z0), m_det, p_val)
        ncols = length(column_groups(solver_det.state.jacprep.jacprep))
        @test ncols <= 24
        @test calls[] <= ncols
        if N == 768
            @test ncols < (2 * N) ÷ 12
        end
        calls[] = 0
        J_det = copy(P)
        GeometricSolvers.jacobian!!(
            J_det, solver_det.state.jacprep, prob_cnt, z0, p_val
        )
        @test 1 <= calls[] <= ncols
        if N == 768
            @test calls[] < (2 * N) ÷ 12
        end
    end
end

@testset "Prototype element types and structural zeros" begin
    prob_analytic, prob_ad, z0, p_val, _ = sparse_bracket_problem(125, Float64)
    P = GeometricSolvers.jacobian_prototype(prob_analytic)

    # Prototype with different element type (Bool and Float32 and Float64-for-Float32) is never mutated
    P_bool = SparseMatrixCSC(
        size(P, 1),
        size(P, 2),
        copy(getcolptr(P)),
        copy(rowvals(P)),
        fill(true, nnz(P))
    )
    nz_bool = copy(nonzeros(P_bool))
    prob_bool = NonlinearProblem(
        prob_analytic.F, prob_analytic.J; jacobian_prototype = P_bool)
    _, st_bool = solve(
        prob_bool, copy(z0), Newton(), p_val; f_abstol = 1e-10, f_reltol = 0.0)
    @test st_bool.code === SUCCESS
    @test nonzeros(P_bool) == nz_bool

    P_f32 = similar(P, Float32)
    fill!(nonzeros(P_f32), 1.0f0)
    nz_f32 = copy(nonzeros(P_f32))
    prob_f32 = NonlinearProblem(prob_ad.F; jacobian_prototype = P_f32)
    m_ad = Newton(;
        jacobian = AutoSparse(
        AutoForwardDiff(); coloring_algorithm = GreedyColoringAlgorithm())
    )
    _, st_f32 = solve(prob_f32, copy(z0), m_ad, p_val; f_abstol = 1e-10, f_reltol = 0.0)
    @test st_f32.code === SUCCESS
    @test nonzeros(P_f32) == nz_f32

    _, prob_ad32, z0_32, p32, _ = sparse_bracket_problem(125, Float32)
    nz_f64 = copy(nonzeros(P))
    prob_64_for_32 = NonlinearProblem(prob_ad32.F; jacobian_prototype = P)
    solver_32 = init(
        prob_64_for_32, copy(z0_32), m_ad, p32; f_abstol = 1.0f-4, f_reltol = 0.0f0)
    @test eltype(solver_32.state.J) === Float32
    _, st_32 = solve!(solver_32)
    @test st_32.code === SUCCESS
    @test nonzeros(P) == nz_f64

    # Structural zero at a point where an entry evaluates to 0 stays stored
    z_zero = copy(z0)
    z_zero[126] = 0.0
    solver_z = init(prob_ad, z_zero, m_ad, p_val)
    J_z = copy(P)
    GeometricSolvers.jacobian!!(J_z, solver_z.state.jacprep, prob_ad, z_zero, p_val)
    @test nnz(J_z) == 28 * 125
    @test getcolptr(J_z) == getcolptr(P)
    @test rowvals(J_z) == rowvals(P)
    @test J_z[1, 126] == 0.0
end

@testset "Globalization, chord reuse, and R1/R2 ignore prototype" begin
    ad_b = AutoSparse(
        AutoForwardDiff(); coloring_algorithm = GreedyColoringAlgorithm())
    for T in REAL_ELTYPES, g in (Backtracking(), Bisection())

        tol = T === Float64 ? 1e-10 : 1.0f-4
        prob_analytic, prob_ad, z0, p_val, _ = sparse_bracket_problem(125, T)
        sol_ga = init(
            prob_analytic, copy(z0), Newton(; globalization = g), p_val;
            f_abstol = tol, f_reltol = zero(T)
        )
        @test sol_ga.state.jvpprep.J isa SparseMatrixCSC{T, Int}
        Jv_a = similar(z0)
        v_dir = ones(T, length(z0))
        GeometricSolvers.jvp!!(Jv_a, sol_ga.state.jvpprep, prob_analytic, z0, v_dir, p_val)
        _, st_ga = solve!(sol_ga)
        @test st_ga.code === SUCCESS
        @test st_ga.fnorm <= tol

        sol_gd = init(
            prob_ad, copy(z0), Newton(; jacobian = ad_b, globalization = g), p_val;
            f_abstol = tol, f_reltol = zero(T)
        )
        Jv_d = similar(z0)
        GeometricSolvers.jvp!!(Jv_d, sol_gd.state.jvpprep, prob_ad, z0, v_dir, p_val)
        @test isapprox(Jv_d, Jv_a; rtol = 8 * eps(T), atol = 8 * eps(T))
        _, st_gd = solve!(sol_gd)
        @test st_gd.code === SUCCESS
        @test st_gd.fnorm <= tol
    end

    # Chord-Newton reuse = 2 across REAL_ELTYPES
    for T in REAL_ELTYPES
        tol = T === Float64 ? 1e-10 : 1.0f-4
        prob_an_t, _, z0_t, p_t, _ = sparse_bracket_problem(125, T)
        jcalls = Ref(0)
        prob_cnt_j = NonlinearProblem(
            prob_an_t.F,
            (J, z, p) -> (jcalls[] += 1; prob_an_t.J(J, z, p));
            jacobian_prototype = prob_an_t.jacobian_prototype
        )
        _, st_ch = solve(
            prob_cnt_j, copy(z0_t), Newton(; reuse = 2), p_t;
            f_abstol = tol, f_reltol = zero(T)
        )
        @test st_ch.code === SUCCESS
        @test st_ch.fnorm <= tol
        @test 1 <= jcalls[] < st_ch.iterations
    end

    # R1, R2, and Picard ignore jacobian_prototype
    P_small = sparse([1, 2], [1, 2], [1.0, 1.0], 2, 2)
    prob_r1 = NonlinearProblem((x, p) -> x .^ 2 .- 2.0; jacobian_prototype = P_small)
    _, st_r1 = solve(prob_r1, SVector(1.5, 1.5), Newton())
    @test st_r1.code === SUCCESS
    X_r2 = [SVector(1.5, 1.5), SVector(1.5, 1.5)]
    counts_r2 = solve_batch!(X_r2, prob_r1, Newton())
    @test counts_r2.SUCCESS == 2

    prob_pic = NonlinearProblem(
        (r, x, p) -> (r .= x .- 0.5 .* sin.(x) .- 1.0; r);
        jacobian_prototype = P_small
    )
    _, st_pic = solve(prob_pic, [0.5, 0.5], Picard(); f_abstol = 1e-10)
    @test st_pic.code === SUCCESS
end

@testset "Singular and non-finite Jacobians: N = $N, T = $T" for N in (125, 768),
    T in REAL_ELTYPES

    prob_analytic, _, z0, p_val, _ = sparse_bracket_problem(N, T)
    P = GeometricSolvers.jacobian_prototype(prob_analytic)
    n = 2 * N
    J_base = copy(P)
    prob_analytic.J(J_base, z0, p_val)

    # Row 1 stored zeros: Sparspak throws ErrorException directly; extension catches it
    J_row1 = copy(J_base)
    for j in 1:n, k in nzrange(J_row1, j)

        rowvals(J_row1)[k] == 1 && (nonzeros(J_row1)[k] = zero(T))
    end
    threw = try
        Sparspak.SparseCSCInterface.sparspaklu(copy(J_row1))
        false
    catch e
        e isa ErrorException
    end
    @test threw

    # Row N stored zeros: Sparspak returns errflag == 0 with non-finite factors; extension catches it
    J_rowN = copy(J_base)
    for j in 1:n, k in nzrange(J_rowN, j)

        rowvals(J_rowN)[k] == N && (nonzeros(J_rowN)[k] = zero(T))
    end
    lu_rowN = Sparspak.SparseCSCInterface.sparspaklu(copy(J_rowN))
    @test iszero(lu_rowN.slvr.errflag)
    @test !all(isfinite, lu_rowN.slvr.lnz) || !all(isfinite, lu_rowN.slvr.unz)

    J_zero = copy(J_base)
    fill!(nonzeros(J_zero), zero(T))

    J_nan = copy(J_base)
    nonzeros(J_nan)[5] = T(NaN)

    J_inf = copy(J_base)
    nonzeros(J_inf)[7] = T(Inf)

    J_ninf = copy(J_base)
    nonzeros(J_ninf)[9] = -T(Inf)

    solver = init(prob_analytic, copy(z0), Newton(), p_val)
    st = solver.state
    J_bad_ref = Ref(J_row1)
    prob_bad = NonlinearProblem(
        prob_analytic.F,
        (J, x, p) -> (copyto!(nonzeros(J), nonzeros(J_bad_ref[])); J);
        jacobian_prototype = P
    )
    solver_bad = init(prob_bad, copy(z0), Newton(), p_val)
    for J_bad in (J_row1, J_rowN, J_zero, J_nan, J_inf, J_ninf)
        copyto!(nonzeros(st.fac.factors), nonzeros(J_bad))
        fac_out = GeometricSolvers.factorize!!(st.fac, LUFactorization(), st.fac.factors)
        @test fac_out.info == 1
        @test returncode(fac_out) === SINGULAR

        J_bad_ref[] = J_bad
        copyto!(solver_bad.x, z0)
        _, st_bad = solve!(solver_bad)
        @test st_bad.code === SINGULAR
        @test st_bad.rank == -1
        @test isnan(st_bad.nullnorm)
    end

    # A non-finite `J` returns before `sparspaklu!`, so `lnz` and `unz` keep the values of the
    # last valid factorisation
    copyto!(nonzeros(st.fac.factors), nonzeros(J_base))
    fac_ok = GeometricSolvers.factorize!!(st.fac, LUFactorization(), st.fac.factors)
    @test fac_ok.info == 0
    for J_nf in (J_nan, J_inf, J_ninf)
        fac_nf = GeometricSolvers.factorize!!(fac_ok, LUFactorization(), J_nf)
        @test fac_nf.info == 1
        @test returncode(fac_nf) === SINGULAR
        @test all(isfinite, fac_nf.lu.slvr.lnz)
        @test all(isfinite, fac_nf.lu.slvr.unz)
    end
end

@testset "Refusals and unsupported configurations" begin
    prob_analytic, prob_ad, z0, p_val, _ = sparse_bracket_problem(125, Float64)
    P = GeometricSolvers.jacobian_prototype(prob_analytic)
    n = length(z0)

    # Subprocess without Sparspak loaded
    pkg_root = normpath(joinpath(@__DIR__, "..", ".."))
    julia_bin = joinpath(Sys.BINDIR, "julia")
    sub_code = """
    using GeometricSolvers, SparseArrays
    @assert Base.get_extension(GeometricSolvers, :GeometricSolversSparspakExt) === nothing
    P = sparse([1, 2], [1, 2], [1.0, 1.0], 2, 2)
    prob = NonlinearProblem((r, x, p) -> (r .= x; r), (J, x, p) -> J; jacobian_prototype = P)
    threw = try
        init(prob, [1.0, 2.0], Newton())
        false
    catch e
        e isa MethodError
    end
    @assert threw
    """
    env_sub = Dict(
        k => v
    for (k, v) in ENV
    if k != "JULIA_LOAD_PATH" && k != "JULIA_PROJECT" && !startswith(k, "GIT_")
    )
    @test success(
        setenv(
        `$julia_bin --startup-file=no --compiled-modules=existing --project=$pkg_root -e $sub_code`,
        env_sub
    )
    )

    # Sparse QRFactorization and SVDFactorization are not defined
    @test_throws MethodError init(
        prob_analytic, copy(z0), Newton(; linsolve = QRFactorization()), p_val)
    @test_throws MethodError init(
        prob_analytic, copy(z0), Newton(; linsolve = SVDFactorization()), p_val)

    # Mixed precision TF != T is refused with ArgumentError naming jacobian_prototype
    err_tf = try
        init(prob_analytic, copy(z0), Newton(; linsolve = LUFactorization(Float32)), p_val)
        nothing
    catch e
        e
    end
    @test err_tf isa ArgumentError
    @test occursin("jacobian_prototype", sprint(showerror, err_tf))

    # Wrong-size prototype is refused in both init_linear and prepare_ad
    for P_bad in (
        spzeros(Float64, n + 1, n + 1),
        spzeros(Float64, n, n + 1),
        spzeros(Float64, n + 1, n)
    )
        prob_bad_ana = NonlinearProblem(
            prob_analytic.F, prob_analytic.J; jacobian_prototype = P_bad)
        err_sz1 = try
            init(prob_bad_ana, copy(z0), Newton(), p_val)
            nothing
        catch e
            e
        end
        @test err_sz1 isa ArgumentError
        @test occursin("jacobian_prototype", sprint(showerror, err_sz1))

        prob_bad_ad = NonlinearProblem(prob_ad.F; jacobian_prototype = P_bad)
        m_ad = Newton(;
            jacobian = AutoSparse(
            AutoForwardDiff(); coloring_algorithm = GreedyColoringAlgorithm())
        )
        err_sz2 = try
            GeometricSolvers.prepare_ad(
                m_ad.jacobian, prob_bad_ad, similar(z0), copy(z0), p_val
            )
            nothing
        catch e
            e
        end
        @test err_sz2 isa ArgumentError
        @test occursin("jacobian_prototype", sprint(showerror, err_sz2))
    end

    # Device iterate (JLArray) with sparse prototype raises MethodError
    @test_throws MethodError init(prob_analytic, JLArray(z0), Newton(), p_val)

    # Default Newton() (ChunkedForwardDiff) with sparse prototype and no analytic J raises MethodError
    @test_throws MethodError solve(prob_ad, copy(z0), Newton(), p_val)

    # Analytic J! that mutates rowvals raises ArgumentError
    prob_mut_row = NonlinearProblem(
        prob_analytic.F,
        (J, x, p) -> (prob_analytic.J(J, x, p); rowvals(J)[1] = 250; J);
        jacobian_prototype = P
    )
    err_mut = try
        solve(prob_mut_row, copy(z0), Newton(), p_val)
        nothing
    catch e
        e
    end
    @test err_mut isa ArgumentError
    @test occursin("jacobian_prototype", sprint(showerror, err_mut))

    P_small = sparse([1, 2], [1, 2], [1.0, 1.0], 2, 2)
    prob_out = NonlinearProblem(
        (r, x, p) -> (r .= x .- 1.0; r),
        (J, x, p) -> (J[1, 1] = 1.0; J[2, 2] = 1.0; J[1, 2] = 3.0; J);
        jacobian_prototype = P_small
    )
    @test_throws ArgumentError solve(prob_out, [2.0, 2.0], Newton())

    # Moving one entry to another column leaves `rowvals` the same and changes only `colptr`: the
    # stored entries (1, 1) and (2, 2) become (1, 1) and (2, 1)
    prob_mut_colptr = NonlinearProblem(
        (r, x, p) -> (r .= x .- 1.0; r),
        (J, x, p) -> (copyto!(nonzeros(J), (1.0, 1.0)); getcolptr(J)[2] += 1; J);
        jacobian_prototype = P_small
    )
    err_colptr = try
        solve(prob_mut_colptr, [2.0, 2.0], Newton())
        nothing
    catch e
        e
    end
    @test err_colptr isa ArgumentError
    @test occursin("jacobian_prototype", sprint(showerror, err_colptr))

    # Regularisation > 0 on a prototype missing a diagonal entry mutates the pattern and raises ArgumentError
    P_nodiag = sparse([1, 2], [2, 1], [1.0, 1.0], 2, 2)
    prob_nodiag = NonlinearProblem(
        (r, x, p) -> (r .= x; r),
        (J, x, p) -> J;
        jacobian_prototype = P_nodiag
    )
    @test_throws ArgumentError solve(
        prob_nodiag, [1.0, 2.0], Newton(; regularization = 1e-3))
end
