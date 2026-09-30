using Aqua
using GeometricSolvers
using Test

# Package-level quality assurance: type piracy, method ambiguities, stale and duplicated
# dependencies, undefined exports, unbound type parameters, `Project.toml` validity. These are
# the faults the rest of the suite is structurally unable to see -- it exercises behaviour, and
# every one of these is a property of the package as a whole.
Aqua.test_all(GeometricSolvers)
