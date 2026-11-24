using ParallelTestRunner
using GeometricSolvers

const TESTS = [
    "core" => [
        "Backends" => "integration/backends.jl",
        "Return codes and status" => "base/status.jl",
        "Options" => "base/options.jl",
        "Reductions" => "base/reductions.jl",
        "ToReal adaptor" => "base/adapt.jl",
        "Storage seam" => "base/bang.jl",
        "Linear-solver methods" => "linear/methods.jl",
        "Static factors" => "linear/factors.jl",
        "Static rectangular QR" => "linear/static/qr_rect.jl",
        "Static rectangular SVD" => "linear/static/svd_rect.jl",
        "Jacobi eigensolver" => "linear/static/eigen.jl",
        "LDLT factorisation" => "linear/ldlt.jl",
        "Cholesky factorisation" => "linear/cholesky.jl",
        "Dense linear solvers" => "linear/dense.jl",
        "Recursive host LU" => "linear/recursive_lu.jl",
        "KA dense linear solvers" => "linear/ka.jl",
        "Own-kernel reference solve" => "devices/own_kernel.jl",
        "Mixed-precision linear solvers" => "linear/mixed.jl",
        "Factor status and rank" => "linear/status.jl",
        "Linear and spectral stability" => "linear/stability.jl",
        "R1 Jacobian" => "ad/forwarddiff_static.jl",
        "R3 Jacobian" => "ad/jacobian.jl",
        "Jacobian-vector product" => "ad/jvp.jl",
        "Sparse LU and AutoSparse" => "linear/sparspak.jl",
        "Sparse LU of UMFPACK" => "linear/sparse.jl",
        "Sparse AD sparsity detection" => "ad/sparse_ad.jl"
    ],
    "quality" => [
        "Aqua Quality Assurance" => "quality/aqua.jl",
        "JET Static Analysis" => "quality/jet.jl",
        "Element-type matrix" => "quality/matrix.jl"
    ],
    "nonlinear" => [
        "Line searches" => "globalization/linesearch.jl",
        "Line function and globalization" => "globalization/linefunction.jl",
        "Merit passes and search trips" => "globalization/passes.jl",
        "DogLeg" => "globalization/dogleg.jl",
        "LevenbergMarquardt" => "globalization/levenbergmarquardt.jl",
        "Nonlinear problem" => "nonlinear/problem.jl",
        "Stationarity problem" => "nonlinear/stationarity.jl",
        "Reparametrized problem" => "nonlinear/reparametrized.jl",
        "Deflated problem" => "nonlinear/deflation.jl",
        "Block elimination and variable projection" => "nonlinear/eliminate.jl",
        "Nonlinear solver loop" => "nonlinear/solver.jl",
        "Default method" => "nonlinear/default.jl",
        "Newton" => "nonlinear/newton.jl",
        "Iterative refinement" => "nonlinear/refinement.jl",
        "Picard" => "nonlinear/picard.jl",
        "Solver loop hooks" => "nonlinear/hooks.jl",
        "Batched R2 nonlinear" => "nonlinear/batched.jl",
        "Verified root certificates" => "nonlinear/certify.jl"
    ],
    "leastsquares" => [
        "R3 nonlinear" => "nonlinear/r3.jl",
        "R5 matrix-free Newton-GMRES" => "nonlinear/krylov.jl",
        "GMRES-IR" => "nonlinear/gmres_ir.jl",
        "Implicit differentiation" => "nonlinear/implicit.jl",
        "Rank-deficient Newton" => "nonlinear/rank_deficient.jl",
        "Range and spectrum diagnostics" => "nonlinear/diagnostics.jl",
        "Nonlinear least squares" => "nonlinear/leastsquares.jl",
        "CommonSolve interface" => "nonlinear/commonsolve.jl"
    ],
    "composition" => [
        "Composition weights" => "composition/weights.jl",
        "Additive combinator" => "composition/additive.jl",
        "Multiplicative combinator" => "composition/multiplicative.jl",
        "Repeat combinator" => "composition/repeat.jl",
        "Switch combinator" => "composition/switch.jl",
        "Fallback combinator" => "composition/fallback.jl",
        "Anderson acceleration" => "composition/anderson.jl",
        "Combinators refuse ForwardDiff.Dual inputs" => "composition/implicit.jl",
        "Left preconditioning and RAS" => "composition/preconditioning.jl"
    ],
    "verification" => [
        "Standard test set" => "verification/standard.jl"
    ],
    "linear" => [
        "Dense linear solvers (large)" => "linear/dense_large.jl",
        "Static LU" => "linear/static/lu.jl",
        "Static QR" => "linear/static/qr.jl"
    ],
    "slow" => [
        "KA dense linear solvers (large)" => "linear/ka_large.jl",
        "Static SVD" => "linear/static/svd.jl",
        "Broyden multisecant" => "composition/broyden.jl",
        "Measurement scripts" => "integration/hardware_scripts.jl"
    ],
    "doctests" => [
        "Doctests" => "quality/doctests.jl"
    ],
    "metal" => [
        "Metal" => "devices/metal.jl",
        "Metal static LU and QR" => "devices/metal_linear.jl",
        "Metal KA LU and QR" => "devices/metal_ka.jl",
        "Metal vendor fast paths" => "devices/metal_vendor.jl",
        "Metal static SVD and Jacobi eigensolver" => "devices/metal_svd.jl",
        "Metal R1 nonlinear" => "devices/metal_nonlinear.jl",
        "Metal R5 Newton-GMRES" => "devices/metal_krylov.jl"
    ]
]

const SELECTED = filter(!startswith("--"), ARGS)
const GROUPS = isempty(SELECTED) ?
               (Sys.isapple() && Sys.ARCH === :aarch64 ?
                [
    "core", "quality", "nonlinear", "leastsquares", "composition",
    "verification", "linear", "slow", "metal"] :
                [
    "core", "quality", "nonlinear", "leastsquares", "composition",
    "verification", "linear", "slow"]) : SELECTED
const UNKNOWN = setdiff(SELECTED, first.(TESTS))
isempty(UNKNOWN) ||
    error("no test group ", join(UNKNOWN, ", "), " in runtests.jl; its groups are ",
        join(first.(TESTS), ", "))

runtests(GeometricSolvers, filter(startswith("--"), ARGS);
    testsuite = Dict(label => :(include($(joinpath(@__DIR__, path))))
    for (group, files) in TESTS if group in GROUPS for (label, path) in files),
    serial = [label
              for (group, files) in TESTS if group in ("metal", "cuda")
              for (label, _) in files],
    serial_position = :after,
    retries = 0,
    exeflags = ["--threads=$(Threads.nthreads())"],
    max_default_jobs = max(1, Sys.CPU_THREADS ÷ Threads.nthreads()))
