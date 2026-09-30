using SafeTestsets

const GROUPS = isempty(ARGS) ? ["core", "slow"] : ARGS

if "core" in GROUPS
    @safetestset "Aqua Quality Assurance" include("quality/aqua.jl")
    @safetestset "JET Static Analysis" include("quality/jet.jl")
    @safetestset "Backends" include("backends.jl")
    @safetestset "Return codes and status" include("base/status.jl")
    @safetestset "Options" include("base/options.jl")
    @safetestset "Reductions" include("base/reductions.jl")
    @safetestset "ToReal adaptor" include("base/adapt.jl")
    @safetestset "Linear-solver methods" include("linear/methods.jl")
    @safetestset "Line searches" include("globalization/linesearch.jl")
end
if "slow" in GROUPS
    @safetestset "Doctests" include("quality/doctests.jl")
end
