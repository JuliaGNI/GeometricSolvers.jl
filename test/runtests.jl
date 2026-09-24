using SafeTestsets

@safetestset "Aqua Quality Assurance" begin
    include("aqua_tests.jl")
end
@safetestset "JET Static Analysis" begin
    include("jet_tests.jl")
end
@safetestset "Backends" begin
    include("backends.jl")
end
@safetestset "Return codes and status" begin
    include("base/status.jl")
end
@safetestset "Options" begin
    include("base/options.jl")
end
@safetestset "Reductions" begin
    include("base/reductions.jl")
end
@safetestset "ToReal adaptor" begin
    include("base/adapt.jl")
end
@safetestset "Linear-solver methods" begin
    include("linear/methods.jl")
end
