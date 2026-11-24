include(joinpath(get(ENV, "Y3_CLONE", pwd()), "benchmark", "devices.jl"))
backend, machine = ARGS
vendor = load_vendor(backend)
dev = Base.invokelatest(device_of, backend, machine, vendor)
println("threads: ", Threads.nthreads(), "; mode: warm (standard/1/u0 first, as in phase B)")
vendor === nothing || println("vendor: ", nameof(vendor), " ", pkgversion(vendor))
println("KernelAbstractions ", pkgversion(KernelAbstractions), ", SimpleNonlinearSolve ",
    pkgversion(SimpleNonlinearSolve))
rows = device_rows(DT)

function launch(row)
    X = dev.r2array(fill(row.x0, R2_SYSTEMS))
    backend_ka = get_backend(X)
    codes = KernelAbstractions.allocate(backend_ka, Int32, length(X))
    sns_kernel!(backend_ka)(X, codes, row.F, row.p, sqrt(eps(DT)), MAX_ITERATIONS;
        ndrange = length(X))
    KernelAbstractions.synchronize(backend_ka)
    return sns_code(Array(codes)[1]), Array(X)[1]
end

for name in ("1", "2")
    k = findfirst(r -> r.set == "standard" && r.problem == name && r.start == "u0", rows)
    println("now: the phase C kernel, standard/$name/u0, n = ", length(rows[k].x0))
    try
        code, x = launch(rows[k])
        println("RESULT standard/$name/u0: code = ", code, ", X[1] = ", x)
    catch e
        println("RESULT standard/$name/u0: ", typeof(e), ": ", sprint(showerror, e)[1:min(end, 200)])
        name == "2" && rethrow()
    end
end
