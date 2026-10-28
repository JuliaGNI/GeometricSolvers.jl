using SafeTestsets

const GROUPS = isempty(ARGS) ?
               (Sys.isapple() && Sys.ARCH === :aarch64 ? ["core", "slow", "metal"] :
                ["core", "slow"]) : ARGS

if "core" in GROUPS
    @safetestset "Aqua Quality Assurance" include("quality/aqua.jl")
    @safetestset "JET Static Analysis" include("quality/jet.jl")
    @safetestset "Element-type matrix" include("quality/matrix.jl")
    @safetestset "Backends" include("integration/backends.jl")
    @safetestset "Return codes and status" include("base/status.jl")
    @safetestset "Options" include("base/options.jl")
    @safetestset "Reductions" include("base/reductions.jl")
    @safetestset "ToReal adaptor" include("base/adapt.jl")
    @safetestset "Storage seam" include("base/bang.jl")
    @safetestset "Linear-solver methods" include("linear/methods.jl")
    @safetestset "Static factors" include("linear/factors.jl")
    @safetestset "Static LU" include("linear/static/lu.jl")
    @safetestset "Static QR" include("linear/static/qr.jl")
    @safetestset "Jacobi eigensolver" include("linear/static/eigen.jl")
    @safetestset "Dense linear solvers" include("linear/dense.jl")
    @safetestset "KA dense linear solvers" include("linear/ka.jl")
    @safetestset "Mixed-precision linear solvers" include("linear/mixed.jl")
    @safetestset "Line searches" include("globalization/linesearch.jl")
    @safetestset "Line function and globalization" include("globalization/linefunction.jl")
    @safetestset "DogLeg" include("globalization/dogleg.jl")
    @safetestset "Nonlinear problem" include("nonlinear/problem.jl")
    @safetestset "Nonlinear solver loop" include("nonlinear/solver.jl")
    @safetestset "Newton" include("nonlinear/newton.jl")
    @safetestset "Picard" include("nonlinear/picard.jl")
    @safetestset "Batched R2 nonlinear" include("nonlinear/batched.jl")
    @safetestset "R1 Jacobian" include("ad/forwarddiff_static.jl")
    @safetestset "R3 Jacobian" include("ad/jacobian.jl")
    @safetestset "Jacobian-vector product" include("ad/jvp.jl")
end
if "slow" in GROUPS
    @safetestset "Dense linear solvers (large)" include("linear/dense_large.jl")
    @safetestset "Static SVD" include("linear/static/svd.jl")
end
if "doctests" in GROUPS
    @safetestset "Doctests" include("quality/doctests.jl")
end
if "metal" in GROUPS
    @safetestset "Metal" include("devices/metal.jl")
    @safetestset "Metal static LU and QR" include("devices/metal_linear.jl")
    @safetestset "Metal KA LU and QR" include("devices/metal_ka.jl")
    @safetestset "Metal static SVD and Jacobi eigensolver" include("devices/metal_svd.jl")
    @safetestset "Metal R1 nonlinear" include("devices/metal_nonlinear.jl")
end
