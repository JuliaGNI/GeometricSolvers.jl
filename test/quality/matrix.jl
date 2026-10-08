# The element types and array backends a test loops over are named once, in
# `test/helpers/matrix.jl`, and nowhere else: a test that writes its own `(Float32, Float64)` can
# drift from the matrix the package claims to support, and a reader can check the matrix only by
# reading every loop. This file makes that a command: it parses every test file and fails with
# `file:line` for each literal tuple of two or more element-type names.
#
# A test that needs one element type writes that type -- only a tuple of two or more is a matrix.
# A subset of a named set is `REAL_ELTYPES` or a `filter` of `ELTYPES`. A type expression
# `Tuple{Float32, Float64}` and a call `promote_type(Float32, Float64)` are not tuple expressions
# and pass. `test/helpers/` holds the sets themselves, and `test/gpu/` is the separate suite with
# its own environment (it is not run from `runtests.jl`), so neither is scanned.

using Test

# The element-type names a tuple is checked for. `Float16`, `BFloat16` and `ComplexF16` stay in
# the set although 16-bit element types are not supported, so that a return of one is seen.
const ELTYPE_NAMES = (:Float16, :BFloat16, :Float32, :Float64, :ComplexF16, :ComplexF32,
    :ComplexF64)

# The top-level directories of `test/` that are not scanned.
const UNSCANNED = ("helpers", "gpu")

"Whether `ex` is a literal tuple expression of two or more element-type names."
function is_eltype_tuple(ex)
    ex isa Expr || return false
    ex.head === :tuple || return false
    return count(a -> a isa Symbol && a in ELTYPE_NAMES, ex.args) >= 2
end

"Walk `ex`, pushing `file:line` for each literal element-type tuple; `line` is the line in force."
function walk_tuples!(bad, ex, line::Int, file)
    ex isa Expr || return nothing
    is_eltype_tuple(ex) && push!(bad, "$file:$line")
    for a in ex.args
        if a isa LineNumberNode
            line = a.line
        else
            walk_tuples!(bad, a, line, file)
        end
    end
    return nothing
end

"Walk the complete AST, pushing a parse-error finding for each error or incomplete expression."
function walk_parse_errors!(bad, ex, file)
    ex isa QuoteNode && return walk_parse_errors!(bad, ex.value, file)
    ex isa Expr || return nothing
    ex.head in (:error, :incomplete) && push!(bad, "$file: does not parse: $ex")
    for a in ex.args
        walk_parse_errors!(bad, a, file)
    end
    return nothing
end

"""
    eltype_tuples(code_or_ast, file = "snippet") -> Vector{String}

Every `file:line` of the code or AST at which a literal tuple of two or more element-type names stands.
Code that does not parse yields its parse error instead, so that a broken file fails rather than
being skipped.
"""
function eltype_tuples(code::AbstractString, file::AbstractString = "snippet")
    return eltype_tuples(Meta.parseall(code; filename = file), file)
end

function eltype_tuples(ast::Expr, file::AbstractString = "snippet")
    bad = String[]
    walk_parse_errors!(bad, ast, file)
    isempty(bad) || return bad
    walk_tuples!(bad, ast, 0, file)
    return bad
end

"The `.jl` files under `dir` that are scanned, relative to `dir`, sorted."
function scanned_files(dir)
    files = String[]
    for (root, _, names) in walkdir(dir)
        rel = relpath(root, dir)
        first(splitpath(rel)) in UNSCANNED && continue
        for n in names
            endswith(n, ".jl") && push!(files, relpath(joinpath(root, n), dir))
        end
    end
    return sort!(files)
end

@testset "the checker sees a literal element-type tuple, and only that" begin
    # the two inline snippets: one with such a tuple, one without
    @test eltype_tuples("for T in (Float32, Float64)\n    @test T === T\nend\n") ==
          ["snippet:1"]
    @test isempty(eltype_tuples("for T in ELTYPES\n    @test T === T\nend\n"))

    # the line reported is the line of the tuple, not of the file
    @test eltype_tuples("x = 1\n\nconst E = (ComplexF32, ComplexF64)\n") == ["snippet:3"]
    @test eltype_tuples("f() = (Float32, Float64)\n", "a/b.jl") == ["a/b.jl:1"]

    # one element type is not a matrix; a type expression and a call are not tuple expressions
    @test isempty(eltype_tuples("g(x) = (Float32,)\nh = Float64\n"))
    @test isempty(eltype_tuples("t = Tuple{Float32, Float64}\n"))
    @test isempty(eltype_tuples("p = promote_type(Float32, Float64)\n"))
    # a tuple in a string or a comment is not code
    @test isempty(eltype_tuples("s = \"(Float32, Float64)\"\n# (Float32, Float64)\n"))

    # every name of the set counts, including the three that are not supported
    @test eltype_tuples("u = (Float16, BFloat16)\n") == ["snippet:1"]
    @test eltype_tuples("v = (ComplexF16, Float32, Int)\n") == ["snippet:1"]

    # a file that does not parse fails with its parse error; it is never skipped
    broken = eltype_tuples("function f(\n", "broken.jl")
    @test length(broken) == 1
    @test occursin("does not parse", only(broken))

    # Parser recovery can put an error inside a block or call, with no element-type tuple.
    for head in (:error, :incomplete), wrap in (identity, QuoteNode)

        ast = Expr(:toplevel,
            Expr(:block, Expr(:call, :f, wrap(Expr(head, "nested parse failure")))))
        broken = eltype_tuples(ast, "nested.jl")
        @test length(broken) == 1
        @test startswith(only(broken), "nested.jl: does not parse:")
        @test occursin("nested parse failure", only(broken))
    end
end

@testset "no test file writes a literal element-type tuple" begin
    dir = normpath(joinpath(@__DIR__, ".."))
    files = scanned_files(dir)
    # The scan set is checked against a walk of `test/` made here, which excludes `helpers` and
    # `gpu` by name and nothing else: a narrowed `UNSCANNED`, a walk that stops early, or a filter
    # that drops a suffix would otherwise leave whole directories unchecked while this testset
    # stays green. The comparison is of the whole set, so it needs no list of file names and
    # covers a directory added later.
    expected = [relpath(joinpath(root, n), dir)
                for (root, _, names) in walkdir(dir) for n in names]
    filter!(f -> endswith(f, ".jl") && !(first(splitpath(f)) in ("helpers", "gpu")), expected)
    sort!(expected)
    @test !isempty(expected)
    @test files == expected
    # and the two excluded directories are indeed excluded, so that the walk above is not simply
    # agreeing with an empty exclusion on both sides
    @test !any(f -> first(splitpath(f)) in ("helpers", "gpu"), files)
    @test isfile(joinpath(dir, "helpers", "matrix.jl"))
    bad = String[]
    for f in files
        append!(bad, eltype_tuples(read(joinpath(dir, f), String), f))
    end
    # each entry is a `file:line` that must loop over a set of `test/helpers/matrix.jl` instead
    @test bad == String[]
end
