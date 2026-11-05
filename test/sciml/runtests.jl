# The `SciMLBase` package extension (`GeometricSolversSciMLBaseExt`), in an environment of its own.
#
# `SciMLBase` is a weak dependency of `GeometricSolvers` and is not in `test/Project.toml`, so the
# main test suite tests the package with no extension loaded. This suite loads `SciMLBase` and
# tests `CommonSolve.solve`, `CommonSolve.init`, `CommonSolve.solve!` and
# `GeometricSolvers.sciml_retcode` across `ELTYPES` and `ARRAY_BACKENDS`.
#
# Usage, from the repository root:
#   julia --startup-file=no --project=test/sciml test/sciml/runtests.jl

using ADTypes: AutoForwardDiff
using GeometricSolvers
using JLArrays: JLArrays, JLArray
using KernelAbstractions: KernelAbstractions
using LinearAlgebra: norm
using SciMLBase: SciMLBase
using StaticArrays: MVector, SMatrix, SVector
using Test

const GS = GeometricSolvers
const GNP = GeometricSolvers.NonlinearProblem
const SNP = SciMLBase.NonlinearProblem
const SNF = SciMLBase.NonlinearFunction
const RC = SciMLBase.ReturnCode

JLArrays.allowscalar(false)

include("../helpers/matrix.jl")

err_msg(f) =
    try
        f()
        "no throw"
    catch e
        e isa ArgumentError ? e.msg : string(typeof(e))
    end

close8(a, b) = isapprox(Array(a), Array(b); rtol = 8 * eps(real(eltype(a))))
same_result(a, b) = a[1] == b[1] && a[2] == b[2]

sq(u, p) = u .* u .- p
sq!(du, u, p) = (du .= u .* u .- p; nothing)

@testset "GeometricSolversSciMLBaseExt" begin
    @testset "Extension loading, method signatures, and sciml_retcode" begin
        ext = Base.get_extension(GS, :GeometricSolversSciMLBaseExt)
        @test ext isa Module
        @test Base.ispublic(GS, :sciml_retcode)
        @test !Base.isexported(GS, :sciml_retcode)
        @test Base.Docs.hasdoc(GS, :sciml_retcode)

        solve_ms = [m for m in methods(GS.solve) if parentmodule(m) === ext]
        @test !isempty(solve_ms)
        for m in solve_ms
            sig = Base.unwrap_unionall(m.sig)
            @test sig.parameters[2] <: SciMLBase.NonlinearProblem
            @test sig.parameters[3] !== Any
            @test sig.parameters[3] == Union{Newton, Picard}
        end

        init_ms = [m for m in methods(GS.init) if parentmodule(m) === ext]
        @test !isempty(init_ms)
        for m in init_ms
            sig = Base.unwrap_unionall(m.sig)
            @test sig.parameters[2] <: SciMLBase.NonlinearProblem
            @test sig.parameters[3] !== Any
            @test sig.parameters[3] == Union{Newton, Picard}
        end
        pr_sig = SNP(sq, SVector(3.0, 5.0), SVector(4.0, 9.0))
        @test parentmodule(which(solve, (typeof(pr_sig), Picard))) === ext
        @test parentmodule(which(init, (typeof(pr_sig), Picard))) === ext

        solvebang_ms = [m for m in methods(GS.solve!) if parentmodule(m) === ext]
        @test isempty(solvebang_ms)

        expected = (
            SUCCESS => RC.Success,
            STALLED => RC.Stalled,
            MAXITERS => RC.MaxIters,
            SINGULAR => RC.InternalLinearSolveFailed,
            NONFINITE => RC.Unstable,
            LINESEARCH_FAILED => RC.InternalLineSearchFailed
        )
        for (code, sret) in expected
            @test GS.sciml_retcode(code) === sret
        end
        images = [GS.sciml_retcode(c) for c in instances(GS.ReturnCode)]
        @test length(unique(images)) == 6
        @test count(SciMLBase.successful_retcode, images) == 1
        @test SciMLBase.successful_retcode(GS.sciml_retcode(SUCCESS))

        _, st_sing = solve(SNP((u, p) -> u * u + 1, 0.0), Newton())
        @test st_sing.code == SINGULAR
        @test GS.sciml_retcode(st_sing.code) === RC.InternalLinearSolveFailed

        _, st_nan = solve(SNP((u, p) -> u * u - 2 + NaN, 1.0), Newton())
        @test st_nan.code == NONFINITE
        @test st_nan.iterations == 0
        @test GS.sciml_retcode(st_nan.code) === RC.Unstable

        _, st_ls = solve(
            SNP((u, p) -> u * u + 1, 0.1),
            Newton(; globalization = Backtracking(; maxiter = 1))
        )
        @test st_ls.code == LINESEARCH_FAILED
        @test GS.sciml_retcode(st_ls.code) === RC.InternalLineSearchFailed
    end

    @testset "Refusals and check order" begin
        e_inplace(u0) = "an in-place SciML `f` needs an array `u0` that is not a `StaticArray`; got $(typeof(u0))"
        for u0 in (SVector(3.0, 5.0), MVector(3.0, 5.0), 3.0)
            @test err_msg(() -> solve(SNP((du, u, p) -> (du .= u .* u .- 2), u0), Newton())) ==
                  e_inplace(u0)
        end

        e_field(name) = "GeometricSolvers does not support a SciML `NonlinearProblem` with `$name` set"
        field_cases = (
            "jac_prototype" =>
                SNP(SNF(sq!; jac_prototype = zeros(2, 2)), [3.0, 5.0], [4.0, 9.0]),
            "sparsity" => SNP(SNF(sq!; sparsity = zeros(2, 2)), [3.0, 5.0], [4.0, 9.0]),
            "lb" => SNP(sq!, [3.0, 5.0], [4.0, 9.0]; lb = [0.0, 0.0]),
            "ub" => SNP(sq!, [3.0, 5.0], [4.0, 9.0]; ub = [9.0, 9.0])
        )
        for (name, pr) in field_cases
            @test err_msg(() -> solve(pr, Newton())) == e_field(name)
        end

        # Order of refusals: unknown keyword before refused fields, refused fields in order
        # (jac_prototype, sparsity, lb, ub) before call form.
        e_kw(list) = "a SciML solve with GeometricSolvers takes the keywords `abstol`, `reltol`, `maxiters` and `options`; got $list"
        base = SNP(sq!, [3.0, 5.0], [4.0, 9.0])
        @test err_msg(() -> solve(base, Newton(); foo = 1)) == e_kw("`foo`")
        @test err_msg(() -> solve(SNP(sq!, [3.0, 5.0], [4.0, 9.0]; foo = 1), Newton())) ==
              e_kw("`foo`")
        @test err_msg(
            () -> solve(SNP(sq!, [3.0, 5.0], [4.0, 9.0]; foo = 1), Newton(); bar = 2)
        ) == e_kw("`foo`, `bar`")

        pr_all_bad = SNP(
            SNF(sq!; jac_prototype = zeros(2, 2), sparsity = zeros(2, 2)),
            SVector(3.0, 5.0),
            SVector(4.0, 9.0);
            lb = SVector(0.0, 0.0),
            ub = SVector(9.0, 9.0)
        )
        @test err_msg(() -> solve(pr_all_bad, Newton(); badkw = 1)) == e_kw("`badkw`")
        @test err_msg(() -> solve(pr_all_bad, Newton())) == e_field("jac_prototype")

        pr_sp_lb_ub = SNP(
            SNF(sq!; sparsity = zeros(2, 2)),
            SVector(3.0, 5.0),
            SVector(4.0, 9.0);
            lb = SVector(0.0, 0.0),
            ub = SVector(9.0, 9.0)
        )
        @test err_msg(() -> solve(pr_sp_lb_ub, Newton())) == e_field("sparsity")

        pr_lb_ub = SNP(
            sq!,
            SVector(3.0, 5.0),
            SVector(4.0, 9.0);
            lb = SVector(0.0, 0.0),
            ub = SVector(9.0, 9.0)
        )
        @test err_msg(() -> solve(pr_lb_ub, Newton())) == e_field("lb")

        pr_ub_iip = SNP(sq!, SVector(3.0, 5.0), SVector(4.0, 9.0); ub = SVector(9.0, 9.0))
        @test err_msg(() -> solve(pr_ub_iip, Newton())) == e_field("ub")

        @test err_msg(() -> solve(base, Newton(); options = Options(Float64), abstol = 1e-8)) ==
              "pass either `options` or keyword arguments, not both"
        @test err_msg(() -> solve(base, Newton(); options = Options(Float64), reltol = 1e-8)) ==
              "pass either `options` or keyword arguments, not both"
        @test err_msg(() -> solve(base, Newton(); options = Options(Float64), maxiters = 10)) ==
              "pass either `options` or keyword arguments, not both"
        @test err_msg(() -> solve(base, Newton(); abstol = -1.0)) ==
              "f_abstol must be non-negative, got -1.0"
        @test err_msg(() -> solve(base, Newton(); options = Options(Float32))) ==
              "options are Options{Float32}, but the iterate has element type Float64: pass Options(Float64; …)"
    end

    @testset "Keywords, Options, and prob.kwargs precedence" begin
        base = SNP(sq!, [3.0, 5.0], [4.0, 9.0])

        _, st_opt = solve(base, Newton(); options = Options(Float64; max_iterations = 1))
        @test st_opt.code == MAXITERS
        @test st_opt.iterations == 1

        _, st_m1 = solve(base, Newton(); maxiters = 1)
        @test st_m1.code == MAXITERS
        @test st_m1.iterations == 1

        _, st_m0 = solve(base, Newton(); maxiters = 0)
        @test st_m0.code == MAXITERS
        @test st_m0.iterations == 0

        prob_m1 = SNP(sq!, [3.0, 5.0], [4.0, 9.0]; maxiters = 1)
        _, st_pm1 = solve(prob_m1, Newton())
        @test st_pm1.code == MAXITERS
        @test st_pm1.iterations == 1

        _, st_override = solve(prob_m1, Newton(); maxiters = 100)
        @test st_override.code == SUCCESS
        @test st_override.iterations > 1

        prob_abs = SNP(sq!, [3.0, 5.0], [4.0, 9.0]; abstol = 100.0)
        _, st_pabs = solve(prob_abs, Newton())
        @test st_pabs.code == SUCCESS
        @test st_pabs.iterations == 0
    end

    @testset "NullParameters, Picard, and analytic Jacobians" begin
        seen = Ref{Any}(missing)
        _, st_null = solve(SNP((du, u, p) -> (seen[] = p; du .= u .* u .- 2), [1.0, 2.0]), Newton())
        @test st_null.code == SUCCESS
        @test seen[] === SciMLBase.NullParameters()

        seen_r1 = Ref{Any}(missing)
        _, st_null_r1 = solve(SNP((u, p) -> (seen_r1[] = p; u .* u .- 2), SVector(1.0, 2.0)), Newton())
        @test st_null_r1.code == SUCCESS
        @test seen_r1[] === SciMLBase.NullParameters()

        xp, stp = solve(SNP((u, p) -> u .- p, [3.0, 5.0], [2.0, 3.0]), Picard(;
            damping = 0.5))
        @test stp.code == SUCCESS
        @test isapprox(xp, [2.0, 3.0]; rtol = 1e-6)

        xp_r1, stp_r1 = solve(
            SNP((u, p) -> u .- p, SVector(3.0, 5.0), SVector(2.0, 3.0)),
            Picard(; damping = 0.5)
        )
        @test stp_r1.code == SUCCESS
        @test isapprox(xp_r1, SVector(2.0, 3.0); rtol = 1e-6)
        @test same_result(
            solve!(
                init(
                SNP((u, p) -> u .- p, SVector(3.0, 5.0), SVector(2.0, 3.0)),
                Picard(; damping = 0.5)
            )
            ),
            (xp_r1, stp_r1)
        )

        calls = Ref(0)
        jac!(J, u, p) = (calls[] += 1; J .= 0; J[1, 1] = 2u[1]; J[2, 2] = 2u[2]; nothing)
        jac_oop(u, p) = (calls[] += 1; [2u[1] 0; 0 2u[2]])
        jac_r1(u, p) = (calls[] += 1; SMatrix{2, 2, Float64}(2u[1], 0, 0, 2u[2]))

        for (f, j) in ((sq!, jac!), (sq, jac_oop))
            calls[] = 0
            x_j, st_j = solve(SNP(SNF(f; jac = j), [3.0, 5.0], [4.0, 9.0]), Newton())
            @test st_j.code == SUCCESS
            @test calls[] > 0
            @test isapprox(x_j, [2.0, 3.0]; rtol = 1e-7)

            calls[] = 0
            x_ad, st_ad = solve(
                SNP(SNF(f; jac = j), [3.0, 5.0], [4.0, 9.0]),
                Newton(; jacobian = AutoForwardDiff())
            )
            @test st_ad.code == SUCCESS
            @test calls[] == 0
            @test close8(x_ad, x_j)
        end

        calls[] = 0
        x_jr1, st_jr1 = solve(
            SNP(SNF(sq; jac = jac_r1), SVector(3.0, 5.0), SVector(4.0, 9.0)),
            Newton()
        )
        @test st_jr1.code == SUCCESS
        @test calls[] > 0
        @test isapprox(x_jr1, SVector(2.0, 3.0); rtol = 1e-7)

        calls[] = 0
        _, st_adr1 = solve(
            SNP(SNF(sq; jac = jac_r1), SVector(3.0, 5.0), SVector(4.0, 9.0)),
            Newton(; jacobian = AutoForwardDiff())
        )
        @test st_adr1.code == SUCCESS
        @test calls[] == 0
    end

    @testset "R1 and R3 solve across ELTYPES and ARRAY_BACKENDS" begin
        for T in ELTYPES
            R = real(T)
            p_svec = SVector{2, T}(4, 9)
            for u0 in (SVector{2, T}(3, 5), MVector{2, T}(3, 5))
                u0_snapshot = SVector{2, T}(u0)
                x, st = solve(SNP(sq, u0, p_svec), Newton())
                xn, stn = solve(GNP(sq), u0, Newton(), p_svec)
                @test x isa typeof(u0)
                @test SVector{2, T}(u0) == u0_snapshot
                @test st isa SolverStatus{R}
                @test st.code == SUCCESS
                @test st.code == stn.code
                @test st.iterations == stn.iterations
                @test close8(x, xn)
                @test isapprox(st.fnorm, norm(sq(x, p_svec)); rtol = 8 * eps(R))
            end

            x_num, st_num = solve(SNP((u, p) -> u * u - p, T(3), T(4)), Newton())
            xn_num, stn_num = solve(GNP((u, p) -> u * u - p), T(3), Newton(), T(4))
            @test x_num isa T
            @test st_num isa SolverStatus{R}
            @test st_num.code == SUCCESS
            @test st_num.code == stn_num.code
            @test st_num.iterations == stn_num.iterations
            @test isapprox(x_num, xn_num; rtol = 8 * eps(R))
            @test isapprox(x_num, T(2); rtol = 10 * sqrt(eps(R)))

            for AT in ARRAY_BACKENDS
                u0 = AT(T[3, 5])
                p = AT(T[4, 9])
                x, st = solve(SNP(sq!, u0, p), Newton())
                xn, stn = solve(GNP(sq!), AT(T[3, 5]), Newton(), p)
                @test x isa typeof(u0)
                @test x !== u0
                @test Array(u0) == T[3, 5]
                @test st isa SolverStatus{R}
                @test st.code == SUCCESS
                @test st.code == stn.code
                @test st.iterations == stn.iterations
                @test close8(x, xn)
                @test isapprox(st.fnorm, norm(Array(sq(Array(x), Array(p)))); rtol = 8 *
                                                                                     eps(R))

                xo, sto = solve(SNP(sq, u0, p), Newton())
                @test xo isa typeof(u0)
                @test xo !== u0
                @test Array(u0) == T[3, 5]
                @test sto isa SolverStatus{R}
                @test sto.code == SUCCESS
                @test sto.iterations == st.iterations
                @test close8(xo, x)
            end

            s = R(1e6)
            g!(du, u, q) = (du .= s .* (u .* u .- 2); nothing)
            at = 64 * eps(R) * s
            _, st_abs = solve(SNP(g!, T[3, 4]), Newton(); abstol = at)
            @test st_abs.code == SUCCESS
            @test st_abs.fnorm <= at

            _, st_def = solve(SNP(g!, T[3, 4]), Newton())
            _, st_rel = solve(SNP(g!, T[3, 4]), Newton(); reltol = R(1e-2))
            @test st_rel.code == SUCCESS
            @test st_rel.iterations < st_def.iterations

            _, st_both = solve(SNP(g!, T[3, 4]), Newton(); abstol = R(0), reltol = R(1e-2))
            @test st_both.code == SUCCESS
            @test st_both.iterations == st_rel.iterations
        end
    end

    @testset "CommonSolve.init and solve!" begin
        # Analytic Jacobians through init + solve!, both in-place and out-of-place.
        icalls = Ref(0)
        ijac!(J, u, p) = (icalls[] += 1; J .= 0; J[1, 1] = 2u[1]; J[2, 2] = 2u[2]; nothing)
        ijac_oop(u, p) = (icalls[] += 1; [2u[1] 0; 0 2u[2]])
        for (f, j) in ((sq!, ijac!), (sq, ijac_oop))
            icalls[] = 0
            pr_j = SNP(SNF(f; jac = j), [3.0, 5.0], [4.0, 9.0])
            r_init = solve!(init(pr_j, Newton()))
            @test r_init[2].code == SUCCESS
            @test icalls[] > 0
            @test same_result(r_init, solve(pr_j, Newton()))
        end

        # Rebinding with solve!(solver, x, p).
        solver = init(SNP(sq!, [3.0, 5.0], [4.0, 9.0]), Newton())
        x2, p2 = [5.0, 6.0], [16.0, 25.0]
        xr, str = solve!(solver, x2, p2)
        @test str.code == SUCCESS
        @test xr === x2
        @test solver.x === x2
        @test solver.p === p2
        @test isapprox(x2, [4.0, 5.0]; rtol = 1e-7)
        @test err_msg(() -> solve!(solver, [1.0, 1.0], Float32[16, 25])) ==
              "`solve!` needs parameters of type Vector{Float64}, like those given to `init`; got type Vector{Float32}"

        # Refusals raise at init with the same messages as solve.
        e_inplace(u0) = "an in-place SciML `f` needs an array `u0` that is not a `StaticArray`; got $(typeof(u0))"
        for u0 in (SVector(3.0, 5.0), MVector(3.0, 5.0), 3.0)
            @test err_msg(() -> init(SNP((du, u, p) -> (du .= u .* u .- 2), u0), Newton())) ==
                  e_inplace(u0)
        end
        e_field(name) = "GeometricSolvers does not support a SciML `NonlinearProblem` with `$name` set"
        field_cases = (
            "jac_prototype" =>
                SNP(SNF(sq!; jac_prototype = zeros(2, 2)), [3.0, 5.0], [4.0, 9.0]),
            "sparsity" => SNP(SNF(sq!; sparsity = zeros(2, 2)), [3.0, 5.0], [4.0, 9.0]),
            "lb" => SNP(sq!, [3.0, 5.0], [4.0, 9.0]; lb = [0.0, 0.0]),
            "ub" => SNP(sq!, [3.0, 5.0], [4.0, 9.0]; ub = [9.0, 9.0])
        )
        for (name, pr) in field_cases
            @test err_msg(() -> init(pr, Newton())) == e_field(name)
        end
        e_kw(list) = "a SciML solve with GeometricSolvers takes the keywords `abstol`, `reltol`, `maxiters` and `options`; got $list"
        base = SNP(sq!, [3.0, 5.0], [4.0, 9.0])
        @test err_msg(() -> init(base, Newton(); foo = 1)) == e_kw("`foo`")
        @test err_msg(
            () -> init(SNP(sq!, [3.0, 5.0], [4.0, 9.0]; foo = 1), Newton(); bar = 2)
        ) == e_kw("`foo`, `bar`")
        @test err_msg(() -> init(base, Newton(); options = Options(Float64), abstol = 1e-8)) ==
              "pass either `options` or keyword arguments, not both"
        @test err_msg(() -> init(base, Newton(); abstol = -1.0)) ==
              "f_abstol must be non-negative, got -1.0"

        for T in ELTYPES
            R = real(T)
            for u0 in (SVector{2, T}(3, 5), MVector{2, T}(3, 5))
                pr = SNP(sq, u0, SVector{2, T}(4, 9))
                solver = init(pr, Newton())
                @test solver isa NonlinearSolver
                @test same_result(solve!(solver), solve(pr, Newton()))
            end
            prn = SNP((u, p) -> u * u - p, T(3), T(4))
            @test same_result(solve!(init(prn, Newton())), solve(prn, Newton()))

            for AT in ARRAY_BACKENDS, f in (sq!, sq)

                u0 = AT(T[3, 5])
                pr = SNP(f, u0, AT(T[4, 9]))
                solver = init(pr, Newton())
                @test solver isa NonlinearSolver
                @test solver.x !== u0
                @test Array(u0) == T[3, 5]
                r = solve!(solver)
                @test Array(u0) == T[3, 5]
                @test r[1] isa typeof(u0)
                @test r[1] === solver.x
                @test same_result(r, solve(pr, Newton()))
            end

            s = R(1e6)
            g!(du, u, q) = (du .= s .* (u .* u .- 2); nothing)
            at = 64 * eps(R) * s
            _, st_abs = solve!(init(SNP(g!, T[3, 4]), Newton(); abstol = at))
            @test st_abs.code == SUCCESS
            @test st_abs.fnorm <= at
        end
    end
end
