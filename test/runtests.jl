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
