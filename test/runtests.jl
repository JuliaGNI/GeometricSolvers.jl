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
    @safetestset "Linear-solver methods" include("linear/methods.jl")
    @safetestset "Static factors" include("linear/factors.jl")
    @safetestset "Static LU" include("linear/static/lu.jl")
    @safetestset "Static QR" include("linear/static/qr.jl")
    @safetestset "Jacobi eigensolver" include("linear/static/eigen.jl")
    @safetestset "Line searches" include("globalization/linesearch.jl")
    @safetestset "R3 Jacobian" include("ad/jacobian.jl")
    @safetestset "R3 Jacobian-vector product" include("ad/jvp.jl")
end
# The Jacobi SVD cold, in the order of this file, is 77 s, past the 60 s of a `core` file.
if "slow" in GROUPS
    @safetestset "Static SVD" include("linear/static/svd.jl")
end
if "doctests" in GROUPS
    @safetestset "Doctests" include("quality/doctests.jl")
end
if "metal" in GROUPS
    @safetestset "Metal" include("devices/metal.jl")
    @safetestset "Metal static LU and QR" include("devices/metal_linear.jl")
    @safetestset "Metal static SVD and Jacobi eigensolver" include("devices/metal_svd.jl")
end
