# S1 fix 2: the CUDA base run and the CUDA mutants, for the DGX Spark (not for merge).
# Run from the root of a GeometricSolvers.jl checkout on the branch under test:
#   julia --startup-file=no device-run/s1_cuda_mutants.jl ../results/s1-spark
# Each mutant must make test/gpu/cuda/runtests.jl fail (CAUGHT). MU14e and MU14g only move rows
# between two routes that give the same ranks, so both are expected to survive.
# The script restores the extension after every mutant, also on an error.

const EXT = "ext/GeometricSolversCUDAExt.jl"
const TEST = `julia --startup-file=no --project=test/gpu/cuda test/gpu/cuda/runtests.jl`

const MUTANTS = [
    ("MU14d", "the own-kernel branch deleted, every cutoff runs on gesvdj!",
        "    GeometricSolvers.rank_tolerance(method.rtol, m_rows, RT) <\n" *
        "    GeometricSolvers.jacobi_tolerance(m_rows, RT) &&\n" *
        "        return GeometricSolvers.ka_factorize_svd!(fac, J, method)\n",
        ""),
    ("MU14e", "every cutoff runs on the own kernel (< for <=)",
        "GeometricSolvers.rank_tolerance(method.rtol, m_rows, RT) <\n",
        "GeometricSolvers.rank_tolerance(method.rtol, m_rows, RT) <=\n"),
    ("MU14g", "every cutoff runs on the own kernel (the condition always true)",
        "    GeometricSolvers.rank_tolerance(method.rtol, m_rows, RT) <\n" *
        "    GeometricSolvers.jacobi_tolerance(m_rows, RT) &&\n",
        "    true &&\n"),
    ("MU14f", "the own kernel of the new branch drops rtol",
        "        return GeometricSolvers.ka_factorize_svd!(fac, J, method)\n    e = GeometricSolvers.matrix_exponent(J)",
        "        return GeometricSolvers.ka_factorize_svd!(fac, J)\n    e = GeometricSolvers.matrix_exponent(J)"),
    ("MU14a", "the gesvdj! rank ignores rtol",
        "m_rows, method.rtol)\n    r < n",
        "m_rows, nothing)\n    r < n"),
    ("MU14b", "the Jacobi fallback of gesvdj! drops the method",
        "return GeometricSolvers.ka_factorize_svd!(fac, J, method)\n    end",
        "return GeometricSolvers.ka_factorize_svd!(fac, J)\n    end"),
    ("MU14c", "the zeroing of U after gesvdj! deleted",
        "    r < n && fill!(@view(fac.U[:, (r + 1):n]), zero(T))\n",
        "")
]

function runtest(log)
    p = run(pipeline(ignorestatus(TEST); stdout = log, stderr = log))
    return success(p)
end

function main(outdir)
    mkpath(outdir)
    head = readchomp(`git rev-parse --short HEAD`)
    println("head: ", head)
    isempty(readchomp(`git status --porcelain -- $EXT`)) || error("$EXT has local changes")
    orig = read(EXT, String)

    base_log = joinpath(outdir, "base-$head.log")
    base_ok = runtest(base_log)
    println("BASE: ", base_ok ? "PASS" : "FAIL", "  (", base_log, ")")
    base_ok || println("  the base fails: the mutant verdicts below say nothing")

    for (tag, what, from, to) in MUTANTS
        n = length(findall(from, orig))
        if n != 1
            println(tag, ": INVALID (pattern found ", n, " times) — ", what)
            continue
        end
        log = joinpath(outdir, "$tag-$head.log")
        try
            write(EXT, replace(orig, from => to))
            ok = runtest(log)
            println(tag, ": ", ok ? "SURVIVED" : "CAUGHT", " — ", what, "  (", log, ")")
        finally
            write(EXT, orig)
        end
    end
    isempty(readchomp(`git status --porcelain -- $EXT`)) || error("$EXT was not restored")
    println("done; $EXT restored")
end

main(length(ARGS) >= 1 ? ARGS[1] : "../results/s1-spark")
