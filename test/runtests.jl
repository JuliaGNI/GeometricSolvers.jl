using SafeTestsets

const GROUPS = isempty(ARGS) ?
               (Sys.isapple() && Sys.ARCH === :aarch64 ? ["core", "slow", "metal"] :
                ["core", "slow"]) : ARGS

if "core" in GROUPS
    @safetestset "Aqua Quality Assurance" include("quality/aqua.jl")
    @safetestset "JET Static Analysis" include("quality/jet.jl")
    @safetestset "Element-type matrix" include("quality/matrix.jl")
    @safetestset "Backends" include("backends.jl")
    @safetestset "Return codes and status" include("base/status.jl")
    @safetestset "Options" include("base/options.jl")
    @safetestset "Reductions" include("base/reductions.jl")
    @safetestset "ToReal adaptor" include("base/adapt.jl")
    @safetestset "Linear-solver methods" include("linear/methods.jl")
    @safetestset "Line searches" include("globalization/linesearch.jl")
    @safetestset "R3 Jacobian" include("ad/jacobian.jl")
    @safetestset "R3 Jacobian-vector product" include("ad/jvp.jl")
end
if "doctests" in GROUPS
    @safetestset "Doctests" include("quality/doctests.jl")
end
if "metal" in GROUPS
    @safetestset "Metal" include("devices/metal.jl")
end
