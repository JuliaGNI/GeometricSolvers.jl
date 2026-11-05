# Automatic Jacobian sparsity detection and colouring for the R3 path: the
# `GeometricSolversSparseADExt` extension of `jacobian_pattern`, which returns the pattern that a
# sparsity detector found inside an `AutoSparse` preparation, and the ways that pattern can be
# wrong.
#
# The coloured Jacobian is compared to `ForwardDiff.jacobian` of the same residual with `==`, not a
# tolerance: colouring changes which columns share a dual pass, never the arithmetic of an entry.

using Test
using LinearAlgebra
using SparseArrays
using ForwardDiff
using FiniteDiff: FiniteDiff
using ADTypes: AutoFiniteDiff, AutoForwardDiff, AutoSparse, KnownJacobianSparsityDetector
using SparseConnectivityTracer: TracerSparsityDetector
using SparseMatrixColorings: GreedyColoringAlgorithm, column_groups, sparsity_pattern
using Sparspak
using JLArrays: JLArray
using GeometricSolvers

include(joinpath(@__DIR__, "..", "helpers", "matrix.jl"))
include(joinpath(@__DIR__, "..", "helpers", "sparse_problems.jl"))

const GS = GeometricSolvers

@testset "GeometricSolversSparseADExt automatic sparsity detection" begin
    ext_sparspak = Base.get_extension(GeometricSolvers, :GeometricSolversSparspakExt)
    @test ext_sparspak !== nothing

    @testset "Extension loading and jacobian_pattern dispatch" begin
        ext = Base.get_extension(GeometricSolvers, :GeometricSolversSparseADExt)
        @test ext !== nothing

        F_diag!(r, x, p) = (r .= x .^ 2 .- one(eltype(x)); r)
        prob = NonlinearProblem(F_diag!)
        @test prob.jacobian_prototype === nothing
        backend = AutoSparse(
            AutoForwardDiff();
            sparsity_detector = TracerSparsityDetector(),
            coloring_algorithm = GreedyColoringAlgorithm()
        )
        x0 = [1.5, 2.0]
        prep = GS.prepare_ad(backend, prob, similar(x0), x0, nothing)

        m = which(GS.jacobian_pattern, Tuple{typeof(prob), typeof(prep)})
        @test parentmodule(m) === ext

        pat = GS.jacobian_pattern(prob, prep)
        @test pat isa SparseMatrixCSC{Bool, Int}
        @test size(pat) == (2, 2)
        @test nnz(pat) == 2
        @test pat == SparseMatrixCSC(sparsity_pattern(prep.jacprep))

        # A preparation of another back end is not the extension's to read: the core method
        # returns the problem's prototype, which is `nothing` here.
        prep_fd = GS.prepare_ad(AutoFiniteDiff(), prob, similar(x0), x0, nothing)
        @test GS.jacobian_pattern(prob, prep_fd) === nothing
    end

    @testset "Sparse bracket problem without jacobian_prototype (T = $T, N = $N)" for T in REAL_ELTYPES,
        N in (125, 768)

        _, prob_with_proto, z0, p_val, z_star = sparse_bracket_problem(N, T)
        prob = NonlinearProblem(prob_with_proto.F)
        @test prob.jacobian_prototype === nothing

        n = length(z0)
        @test n == 2 * N
        tol = T === Float32 ? 1.0f-4 : 1.0e-10

        backend = AutoSparse(
            AutoForwardDiff();
            sparsity_detector = TracerSparsityDetector(),
            coloring_algorithm = GreedyColoringAlgorithm()
        )

        # Counting wrapper on F!
        eval_count = Ref(0)
        F_counted!(r, x, p) = (eval_count[] += 1; prob.F(r, x, p))
        prob_counted = NonlinearProblem(F_counted!)

        prep = GS.prepare_ad(backend, prob_counted, similar(z0), z0, p_val)
        pat = GS.jacobian_pattern(prob_counted, prep)
        @test pat isa SparseMatrixCSC{<:Any, Int}
        @test size(pat) == (n, n)
        @test nnz(pat) > 0

        ncolors = length(column_groups(prep.jacprep))
        @test 0 < ncolors < n / 10

        # Jacobian at z0 matches ForwardDiff.jacobian and uses <= ncolors chunked evaluations
        J_z0 = similar(pat, T)
        eval_count[] = 0
        GS.jacobian!!(J_z0, prep, prob_counted, z0, p_val)
        @test 1 <= eval_count[] <= ncolors
        @test eval_count[] < n / 10

        J_fd_z0 = ForwardDiff.jacobian(
            (r, x) -> prob.F(r, x, p_val), similar(z0), z0)
        @test Matrix(J_z0) == J_fd_z0

        # Newton solve with detected pattern and SparspakLU default factorisation
        solver = init(
            prob, copy(z0), Newton(; jacobian = backend), p_val;
            f_abstol = tol, f_reltol = zero(T)
        )
        @test solver.state.fac isa ext_sparspak.SparspakLU{SparseMatrixCSC{T, Int}}
        @test solver.state.J isa SparseMatrixCSC{T, Int}

        x_sol, st = solve!(solver)
        @test st.code === SUCCESS
        r_sol = similar(x_sol)
        prob.F(r_sol, x_sol, p_val)
        @test norm(r_sol) <= tol
        @test isapprox(x_sol, z_star; rtol = T === Float32 ? 1.0f-3 : 1.0e-7)

        # Jacobian at x* matches ForwardDiff.jacobian and uses <= ncolors chunked evaluations
        J_star = similar(pat, T)
        eval_count[] = 0
        GS.jacobian!!(J_star, prep, prob_counted, x_sol, p_val)
        @test 1 <= eval_count[] <= ncolors
        @test eval_count[] < n / 10

        J_fd_star = ForwardDiff.jacobian(
            (r, x) -> prob.F(r, x, p_val), similar(x_sol), x_sol)
        @test Matrix(J_star) == J_fd_star

        # `AutoForwardDiff(; chunksize = 1)` seeds one column per dual pass, so one `jacobian!!`
        # evaluates the residual exactly once per colour. The default chunk size packs several
        # colour groups into one pass and needs fewer evaluations than there are colours.
        backend_c1 = AutoSparse(
            AutoForwardDiff(; chunksize = 1);
            sparsity_detector = TracerSparsityDetector(),
            coloring_algorithm = GreedyColoringAlgorithm()
        )
        prep_c1 = GS.prepare_ad(backend_c1, prob_counted, similar(z0), z0, p_val)
        @test length(column_groups(prep_c1.jacprep)) == ncolors

        J_z0_c1 = similar(pat, T)
        eval_count[] = 0
        GS.jacobian!!(J_z0_c1, prep_c1, prob_counted, z0, p_val)
        @test eval_count[] == ncolors
        @test eval_count[] < n / 10
        @test Matrix(J_z0_c1) == J_fd_z0

        J_star_c1 = similar(pat, T)
        eval_count[] = 0
        GS.jacobian!!(J_star_c1, prep_c1, prob_counted, x_sol, p_val)
        @test eval_count[] == ncolors
        @test eval_count[] < n / 10
        @test Matrix(J_star_c1) == J_fd_star
    end

    @testset "All-coupled dense residual detected as n x n SparseMatrixCSC with n colours" begin
        n = 6
        function F_dense!(r, x, p)
            s = sum(x)
            @inbounds for i in eachindex(r, x)
                r[i] = x[i]^2 + T_val(0.05, eltype(x)) * s - one(eltype(x))
            end
            return r
        end
        T_val(v, ::Type{T}) where {T} = T(v)

        prob_dense = NonlinearProblem(F_dense!)
        backend = AutoSparse(
            AutoForwardDiff();
            sparsity_detector = TracerSparsityDetector(),
            coloring_algorithm = GreedyColoringAlgorithm()
        )
        x0 = fill(1.2, n)
        prep = GS.prepare_ad(backend, prob_dense, similar(x0), x0, nothing)
        pat = GS.jacobian_pattern(prob_dense, prep)
        @test pat isa SparseMatrixCSC{<:Any, Int}
        @test size(pat) == (n, n)
        @test nnz(pat) == n * n
        @test length(column_groups(prep.jacprep)) == n

        solver = init(
            prob_dense, copy(x0), Newton(; jacobian = backend);
            f_abstol = 1.0e-10, f_reltol = 0.0
        )
        @test solver.state.fac isa ext_sparspak.SparspakLU{SparseMatrixCSC{Float64, Int}}
        x_sol, st = solve!(solver)
        @test st.code === SUCCESS
        r_sol = similar(x_sol)
        F_dense!(r_sol, x_sol, nothing)
        @test norm(r_sol) <= 1.0e-10
    end

    @testset "Explicit jacobian_prototype wins over TracerSparsityDetector" begin
        F_diag!(r, x, p) = (r .= x .^ 2 .- 1.0; r)
        # Prototype with an extra stored structural entry at (1, 2)
        P_explicit = sparse([1, 1, 2], [1, 2, 2], [1.0, 0.0, 1.0], 2, 2)
        prob_proto = NonlinearProblem(F_diag!; jacobian_prototype = P_explicit)
        backend = AutoSparse(
            AutoForwardDiff();
            sparsity_detector = TracerSparsityDetector(),
            coloring_algorithm = GreedyColoringAlgorithm()
        )
        x0 = [1.5, 2.0]
        prep = GS.prepare_ad(backend, prob_proto, similar(x0), x0, nothing)
        @test prep.backend.sparsity_detector isa KnownJacobianSparsityDetector
        @test GS.jacobian_pattern(prob_proto, prep) === P_explicit

        # Explicit prototype also works when AutoSparse has no sparsity_detector
        backend_nodet = AutoSparse(
            AutoForwardDiff();
            coloring_algorithm = GreedyColoringAlgorithm()
        )
        prep_nodet = GS.prepare_ad(
            backend_nodet, prob_proto, similar(x0), x0, nothing)
        @test GS.jacobian_pattern(prob_proto, prep_nodet) === P_explicit

        solver = init(
            prob_proto, copy(x0), Newton(; jacobian = backend);
            f_abstol = 1.0e-12, f_reltol = 0.0
        )
        @test findnz(solver.state.J)[1:2] == findnz(P_explicit)[1:2]
        x_sol, st = solve!(solver)
        @test st.code === SUCCESS
        @test isapprox(x_sol, [1.0, 1.0]; atol = 1.0e-11)
    end

    @testset "Globalization (Backtracking JVP) with TracerSparsityDetector" begin
        _, prob_with_proto, z0, p_val, z_star = sparse_bracket_problem(125, Float64)
        prob = NonlinearProblem(prob_with_proto.F)
        backend = AutoSparse(
            AutoForwardDiff();
            sparsity_detector = TracerSparsityDetector(),
            coloring_algorithm = GreedyColoringAlgorithm()
        )
        solver = init(
            prob, copy(z0),
            Newton(; jacobian = backend, globalization = Backtracking()),
            p_val;
            f_abstol = 1.0e-10, f_reltol = 0.0
        )
        x_sol, st = solve!(solver)
        @test st.code === SUCCESS
        @test isapprox(x_sol, z_star; rtol = 1.0e-7)
    end

    @testset "Guard rails: device array and invalid shapes" begin
        F_simple!(r, x, p) = (r .= x .^ 2 .- one(eltype(x)); r)
        prob = NonlinearProblem(F_simple!)
        backend = AutoSparse(
            AutoForwardDiff();
            sparsity_detector = TracerSparsityDetector(),
            coloring_algorithm = GreedyColoringAlgorithm()
        )
        x_dev = JLArray(Float32[1.5, 2.0])
        # The refusal is `prepare_ad`'s own, not whichever error the back end raises on a device
        # array afterwards: the message names the AutoForwardDiff-only rule.
        @test_throws "a device array iterate" GS.prepare_ad(
            backend, prob, similar(x_dev), x_dev, nothing)
        @test_throws "a device array iterate" init(
            prob, x_dev, Newton(; jacobian = backend))

        # Matrix-shaped CPU iterate raises ArgumentError in prepare_ad
        x_mat = [1.5 2.0; 3.0 4.0]
        @test_throws ArgumentError GS.prepare_ad(
            backend, prob, similar(x_mat), x_mat, nothing)

        # Wrong-size jacobian_prototype raises ArgumentError in prepare_ad
        prob_badsz = NonlinearProblem(F_simple!; jacobian_prototype = spzeros(3, 3))
        x_cpu = [1.5, 2.0]
        @test_throws ArgumentError GS.prepare_ad(
            backend, prob_badsz, similar(x_cpu), x_cpu, nothing)
    end

    @testset "Clean subprocess without SparseConnectivityTracer has extension === nothing" begin
        pkg_root = normpath(joinpath(@__DIR__, "..", ".."))
        sub_code = """
        using GeometricSolvers
        ext = Base.get_extension(GeometricSolvers, :GeometricSolversSparseADExt)
        ext === nothing || error("expected nothing, got \$ext")
        """
        julia_exe = joinpath(Sys.BINDIR, "julia")
        env_clean = filter(
            p -> !startswith(p.first, "JULIA_") && !startswith(p.first, "GIT_"),
            copy(ENV)
        )
        proc = run(ignorestatus(setenv(
            `$julia_exe --startup-file=no --compiled-modules=existing --project=$pkg_root -e $sub_code`,
            env_clean
        )))
        @test proc.exitcode == 0
    end
end
