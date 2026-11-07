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
using ADTypes: AutoFiniteDiff, AutoForwardDiff, AutoSparse, KnownJacobianSparsityDetector,
               NoSparsityDetector
using SparseConnectivityTracer: TracerSparsityDetector
using SparseMatrixColorings: GreedyColoringAlgorithm, column_groups, sparsity_pattern
using Sparspak
using JLArrays: JLArray
using GeometricSolvers

include(joinpath(@__DIR__, "..", "helpers", "matrix.jl"))
include(joinpath(@__DIR__, "..", "helpers", "sparse_problems.jl"))

const GS = GeometricSolvers

# The `AutoSparse` back end that every testset here uses: `TracerSparsityDetector` finds the
# pattern and `GreedyColoringAlgorithm` colours it. `chunksize = nothing` leaves the chunk size to
# `AutoForwardDiff`; `chunksize = 1` seeds one column per dual pass.
function sparse_backend(; chunksize = nothing)
    AutoSparse(
        AutoForwardDiff(; chunksize);
        sparsity_detector = TracerSparsityDetector(),
        coloring_algorithm = GreedyColoringAlgorithm()
    )
end

@testset "GeometricSolversSparseADExt automatic sparsity detection" begin
    ext_sparspak = Base.get_extension(GeometricSolvers, :GeometricSolversSparspakExt)
    @test ext_sparspak !== nothing

    @testset "Extension loading and jacobian_pattern dispatch" begin
        ext = Base.get_extension(GeometricSolvers, :GeometricSolversSparseADExt)
        @test ext !== nothing

        F_diag!(r, x, p) = (r .= x .^ 2 .- one(eltype(x)); r)
        prob = NonlinearProblem(F_diag!)
        @test prob.jacobian_prototype === nothing
        backend = sparse_backend()
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

        # An `AutoSparse` back end with no sparsity detector prepares a pattern that is true
        # everywhere, which has no sparsity to exploit: the extension returns `nothing` there, so
        # the solve keeps a dense factor and a dense Jacobian.
        backend_nodet = AutoSparse(
            AutoForwardDiff(); coloring_algorithm = GreedyColoringAlgorithm())
        prep_nodet = GS.prepare_ad(backend_nodet, prob, similar(x0), x0, nothing)
        @test prep_nodet.backend.sparsity_detector isa NoSparsityDetector
        @test all(sparsity_pattern(prep_nodet.jacprep))
        @test GS.jacobian_pattern(prob, prep_nodet) === nothing
        solver_nodet = init(prob, copy(x0), Newton(; jacobian = backend_nodet))
        @test solver_nodet.state.fac isa GS.DenseLU
        @test solver_nodet.state.J isa Matrix
        x_nodet, st_nodet = solve!(solver_nodet)
        @test st_nodet.code === SUCCESS
        @test isapprox(x_nodet, ones(2); atol = 1.0e-11)
    end

    @testset "Sparse bracket problem without jacobian_prototype (T = $T, N = $N)" for T in REAL_ELTYPES,
        N in (125, 768)

        _, prob_with_proto, z0, p_val, z_star = sparse_bracket_problem(N, T)
        prob = NonlinearProblem(prob_with_proto.F)
        @test prob.jacobian_prototype === nothing

        n = length(z0)
        @test n == 2 * N
        tol = T === Float32 ? 1.0f-4 : 1.0e-10

        backend = sparse_backend()

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
        backend_c1 = sparse_backend(; chunksize = 1)
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

    @testset "All-coupled dense residual detected as n x n SparseMatrixCSC with n colours (T = $T)" for T in REAL_ELTYPES
        n = 6
        function F_dense!(r, x, p)
            s = sum(x)
            @inbounds for i in eachindex(r, x)
                r[i] = x[i]^2 + T(0.05) * s - one(T)
            end
            return r
        end

        prob_dense = NonlinearProblem(F_dense!)
        backend = sparse_backend()
        x0 = fill(T(1.2), n)
        prep = GS.prepare_ad(backend, prob_dense, similar(x0), x0, nothing)
        pat = GS.jacobian_pattern(prob_dense, prep)
        @test pat isa SparseMatrixCSC{<:Any, Int}
        @test size(pat) == (n, n)
        @test nnz(pat) == n * n
        @test length(column_groups(prep.jacprep)) == n

        # The final residual norm is 1.75f-6 in `Float32` and 3.2e-13 in `Float64`, below `tol`.
        tol = T === Float32 ? 1.0f-4 : 1.0e-10
        solver = init(
            prob_dense, copy(x0), Newton(; jacobian = backend);
            f_abstol = tol, f_reltol = zero(T)
        )
        @test solver.state.fac isa ext_sparspak.SparspakLU{SparseMatrixCSC{T, Int}}
        x_sol, st = solve!(solver)
        @test st.code === SUCCESS
        r_sol = similar(x_sol)
        F_dense!(r_sol, x_sol, nothing)
        @test norm(r_sol) <= tol
    end

    @testset "Explicit jacobian_prototype wins over TracerSparsityDetector (T = $T)" for T in REAL_ELTYPES
        F_diag!(r, x, p) = (r .= x .^ 2 .- one(T); r)
        # Prototype with an extra stored structural entry at (1, 2)
        P_explicit = sparse([1, 1, 2], [1, 2, 2], T[1, 0, 1], 2, 2)
        prob_proto = NonlinearProblem(F_diag!; jacobian_prototype = P_explicit)
        backend = sparse_backend()
        x0 = T[1.5, 2.0]
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
            f_abstol = T === Float32 ? 1.0f-6 : 1.0e-12, f_reltol = zero(T)
        )
        @test findnz(solver.state.J)[1:2] == findnz(P_explicit)[1:2]
        x_sol, st = solve!(solver)
        @test st.code === SUCCESS
        @test isapprox(x_sol, ones(T, 2); atol = T === Float32 ? 1.0f-5 : 1.0e-11)
    end

    @testset "Backtracking globalization with TracerSparsityDetector (T = $T)" for T in REAL_ELTYPES
        _, prob_with_proto, z0, p_val, z_star = sparse_bracket_problem(125, T)
        prob = NonlinearProblem(prob_with_proto.F)
        backend = sparse_backend()
        solver = init(
            prob, copy(z0),
            Newton(; jacobian = backend, globalization = Backtracking()),
            p_val;
            f_abstol = T === Float32 ? 1.0f-4 : 1.0e-10, f_reltol = zero(T)
        )
        x_sol, st = solve!(solver)
        @test st.code === SUCCESS
        @test isapprox(x_sol, z_star; rtol = T === Float32 ? 1.0f-3 : 1.0e-7)
    end

    # Only `GeometricSolversSparspakExt` builds a sparse factor from a detected pattern, for an
    # `LUFactorization` of the iterate's own real element type on a `Vector`. Every other linear
    # method takes the dense factor, as it does with no pattern, and solves to the same iterate as
    # the dense `AutoForwardDiff` Jacobian: colouring changes no entry of the Jacobian.
    @testset "Detected pattern without a sparse factor solves dense" begin
        F_sq!(r, x, p) = (r .= x .^ 2 .- p; r)
        prob = NonlinearProblem(F_sq!)

        @test GS.has_sparse_factor(LUFactorization(), Float64[])
        @test GS.has_sparse_factor(LUFactorization(Float32), Float32[])
        @test !GS.has_sparse_factor(LUFactorization(Float64), Float32[])
        @test !GS.has_sparse_factor(QRFactorization(), Float64[])
        @test !GS.has_sparse_factor(SVDFactorization(), Float64[])
        @test !GS.has_sparse_factor(LUFactorization(), ComplexF64[])

        function check_dense(prob, x0, p, linsolve, backend, Fac)
            prep = GS.prepare_ad(backend, prob, similar(x0), x0, p)
            @test GS.jacobian_pattern(prob, prep) isa SparseMatrixCSC
            solver = init(prob, copy(x0), Newton(; linsolve, jacobian = backend), p)
            @test solver.state.fac isa Fac
            @test solver.state.J isa Matrix
            x, st = solve!(solver)
            @test st.code === SUCCESS
            dense = Newton(; linsolve, jacobian = AutoForwardDiff())
            x_ref, st_ref = solve!(init(prob, copy(x0), dense, p))
            @test st_ref.code === SUCCESS
            @test isapprox(x, x_ref; rtol = 8 * eps(real(eltype(x0))))
            @test x != x0
        end

        @testset "$(nameof(typeof(linsolve))) (T = $T)" for T in REAL_ELTYPES,
            (linsolve, Fac) in ((QRFactorization(), GS.DenseQR), (
                SVDFactorization(), GS.DenseSVD))

            check_dense(
                prob, T[1.5, 2.0, 0.7], T[2, 3, 0.5], linsolve, sparse_backend(), Fac)
        end

        @testset "Mixed precision LUFactorization(Float64) on Float32" begin
            check_dense(prob, Float32[1.5, 2.0, 0.7], Float32[2, 3, 0.5],
                LUFactorization(Float64), sparse_backend(), GS.DenseLU)
        end

        # `ForwardDiff` and the tracer take no complex number, so the complex iterate goes through
        # `AutoFiniteDiff` with a known pattern on a problem that has no prototype: the pattern is
        # still the detector's, read from the preparation.
        @testset "ComplexF64 iterate" begin
            backend_c = AutoSparse(AutoFiniteDiff();
                sparsity_detector = KnownJacobianSparsityDetector(sparse(1.0I, 3, 3)),
                coloring_algorithm = GreedyColoringAlgorithm())
            x0 = ComplexF64[1.5, 2.0, 0.7]
            p = ComplexF64[2, 3, 0.5]
            solver = init(prob, copy(x0), Newton(; jacobian = backend_c), p)
            @test solver.state.fac isa GS.DenseLU
            x, st = solve!(solver)
            @test st.code === SUCCESS
            @test isapprox(x, sqrt.(p); rtol = 1.0e-8)
        end

        @testset "LUFactorization of the iterate's element type keeps SparspakLU (T = $T)" for T in REAL_ELTYPES
            solver = init(prob, T[1.5, 2.0, 0.7],
                Newton(; linsolve = LUFactorization(T), jacobian = sparse_backend()), T[2, 3, 0.5])
            @test solver.state.fac isa ext_sparspak.SparspakLU{SparseMatrixCSC{T, Int}}
            @test solver.state.J isa SparseMatrixCSC{T, Int}
            @test solve!(solver)[2].code === SUCCESS
        end
    end

    # `regularize!!` writes `μ` onto the diagonal of `J`, which a structural entry of the factor's
    # pattern must hold. A detected pattern without the whole diagonal therefore takes the dense
    # factor under `μ ≠ 0`, and keeps `SparspakLU` under `μ = 0` or with the diagonal present.
    @testset "Regularization and a detected pattern without the diagonal" begin
        F_perm!(r, x, p) = (r[1] = x[2]^3 + x[2] - 2; r[2] = x[1]^3 + x[1] - 2; r)
        prob = NonlinearProblem(F_perm!)
        x0 = [0.5, 0.5]
        prep = GS.prepare_ad(sparse_backend(), prob, similar(x0), x0, nothing)
        @test !GS.has_structural_diagonal(GS.jacobian_pattern(prob, prep))

        solver = init(prob, copy(x0),
            Newton(; jacobian = sparse_backend(), regularization = 1.0e-10))
        @test solver.state.fac isa GS.DenseLU
        x, st = solve!(solver)
        @test st.code === SUCCESS
        @test isapprox(x, ones(2); rtol = 1.0e-8)

        solver_μ0 = init(prob, copy(x0), Newton(; jacobian = sparse_backend()))
        @test solver_μ0.state.fac isa ext_sparspak.SparspakLU
        @test solve!(solver_μ0)[2].code === SUCCESS

        F_sq!(r, x, p) = (r .= x .^ 2 .- p; r)
        solver_diag = init(NonlinearProblem(F_sq!), [1.5, 2.0],
            Newton(; jacobian = sparse_backend(), regularization = 1.0e-10), [2.0, 3.0])
        @test solver_diag.state.fac isa ext_sparspak.SparspakLU
        @test solve!(solver_diag)[2].code === SUCCESS
    end

    @testset "Guard rails: device array and invalid shapes" begin
        F_simple!(r, x, p) = (r .= x .^ 2 .- one(eltype(x)); r)
        prob = NonlinearProblem(F_simple!)
        backend = sparse_backend()
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

    # A child process loads only the packages its code names, so it sees the extensions that
    # those packages trigger and no others. `JULIA_DEPOT_PATH` stays, so that a child of a run in
    # a custom depot finds its packages.
    function run_child(code)
        julia_exe = joinpath(Sys.BINDIR, "julia")
        env_clean = filter(
            p -> (p.first == "JULIA_DEPOT_PATH" || !startswith(p.first, "JULIA_")) &&
                 !startswith(p.first, "GIT_"),
            copy(ENV)
        )
        return run(ignorestatus(setenv(
            `$julia_exe --startup-file=no --compiled-modules=existing --project=$(Base.active_project()) -e $code`,
            env_clean
        )))
    end

    @testset "Subprocess with SparseMatrixColorings and no SparseConnectivityTracer" begin
        # `SparseMatrixColorings` alone triggers the extension, which reads nothing else: a
        # detector from any package, here a known pattern on a problem with no prototype, then
        # gives the detected pattern.
        sub_code = """
        using GeometricSolvers, SparseMatrixColorings, SparseArrays
        using LinearAlgebra: I
        using ADTypes: AutoForwardDiff, AutoSparse, KnownJacobianSparsityDetector
        ext = Base.get_extension(GeometricSolvers, :GeometricSolversSparseADExt)
        ext === nothing && error("GeometricSolversSparseADExt is not loaded")
        any(k -> k.name == "SparseConnectivityTracer", keys(Base.loaded_modules)) &&
            error("SparseConnectivityTracer is loaded")
        prob = NonlinearProblem((r, x, p) -> (r .= x .^ 2 .- 1; r))
        backend = AutoSparse(AutoForwardDiff();
            sparsity_detector = KnownJacobianSparsityDetector(sparse(1.0I, 2, 2)),
            coloring_algorithm = GreedyColoringAlgorithm())
        x0 = [1.5, 2.0]
        prep = GeometricSolvers.prepare_ad(backend, prob, similar(x0), x0, nothing)
        GeometricSolvers.jacobian_pattern(prob, prep) isa SparseMatrixCSC ||
            error("no detected pattern")
        """
        @test run_child(sub_code).exitcode == 0
    end

    @testset "Subprocess with a detected pattern and no Sparspak" begin
        # With no extension that builds a sparse factor, a detected pattern gives the dense
        # factor and solves to the iterate of the dense `AutoForwardDiff` Jacobian.
        sub_code = """
        using GeometricSolvers, SparseConnectivityTracer, SparseMatrixColorings, SparseArrays
        using ADTypes: AutoForwardDiff, AutoSparse
        Base.get_extension(GeometricSolvers, :GeometricSolversSparspakExt) === nothing ||
            error("GeometricSolversSparspakExt is loaded")
        prob = NonlinearProblem((r, x, p) -> (r .= x .^ 2 .- p; r))
        backend = AutoSparse(AutoForwardDiff(); sparsity_detector = TracerSparsityDetector(),
            coloring_algorithm = GreedyColoringAlgorithm())
        x0 = [1.5, 2.0, 0.7]
        p = [2.0, 3.0, 0.5]
        prep = GeometricSolvers.prepare_ad(backend, prob, similar(x0), x0, p)
        GeometricSolvers.jacobian_pattern(prob, prep) isa SparseMatrixCSC ||
            error("no detected pattern")
        solver = init(prob, copy(x0), Newton(; jacobian = backend), p)
        solver.state.fac isa GeometricSolvers.DenseLU || error("factor \$(typeof(solver.state.fac))")
        x, st = solve!(solver)
        st.code === SUCCESS || error("return code \$(st.code)")
        x_ref, _ = solve!(init(prob, copy(x0), Newton(; jacobian = AutoForwardDiff()), p))
        isapprox(x, x_ref; rtol = 8 * eps()) || error("\$x differs from \$x_ref")
        """
        @test run_child(sub_code).exitcode == 0
    end
end
