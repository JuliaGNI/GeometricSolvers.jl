include(joinpath(get(ENV, "Y3_CLONE", pwd()), "benchmark", "devices.jl"))
backend, machine, mode = ARGS
vendor = load_vendor(backend)
dev = Base.invokelatest(device_of, backend, machine, vendor)
println("threads: ", Threads.nthreads(), "; mode: ", mode)
vendor === nothing || println("vendor: ", nameof(vendor), " ", pkgversion(vendor))
println("KernelAbstractions ", pkgversion(KernelAbstractions), ", SimpleNonlinearSolve ",
    pkgversion(SimpleNonlinearSolve))
rows = device_rows(DT)
k2 = findfirst(r -> r.set == "standard" && r.problem == "2" && r.start == "u0", rows)
row = rows[k2]
@kernel function sns_kernel_b!(X, codes, f, p)
    i = @index(Global)
    prob = SciMLBase.NonlinearProblem{false}(f, X[i], p)
    sol = SciMLBase.solve(prob, SimpleNewtonRaphson(); abstol = 0.00034526698f0, maxiters = 200)
    X[i] = sol.u
    codes[i] = Int32(sol.retcode)
end
X = dev.r2array(fill(row.x0, R2_SYSTEMS))
backend_ka = get_backend(X)
codes = KernelAbstractions.allocate(backend_ka, Int32, length(X))
if mode == "phasec"
    println("now: the phase C kernel, standard/2/u0")
    sns_kernel!(backend_ka)(X, codes, row.F, row.p, sqrt(eps(DT)), MAX_ITERATIONS;
        ndrange = length(X))
elseif mode == "phaseb"
    println("now: the phase B kernel, standard/2/u0")
    sns_kernel_b!(backend_ka)(X, codes, row.F, row.p; ndrange = length(X))
else
    error("mode must be phasec or phaseb")
end
KernelAbstractions.synchronize(backend_ka)
println("RESULT: code = ", sns_code(Array(codes)[1]), ", X[1] = ", Array(X)[1])
