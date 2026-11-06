module GeometricSolversSparseADExt

using ADTypes: AutoSparse, NoSparsityDetector
using GeometricSolvers: GeometricSolvers, DIJacobian, NonlinearProblem
using SparseArrays: SparseMatrixCSC
using SparseMatrixColorings: SparseMatrixColorings

# The pattern a sparsity detector found inside the preparation, as the `SparseMatrixCSC{Bool, Int}`
# that `init_linear` dispatches on. The problem is restricted to one with no prototype, so this
# method never matches a problem that carries one: the core method returns the prototype, which
# then wins over detection. A back end with no sparsity detector finds only a pattern that is true
# everywhere, which no dense `init_linear` accepts, so that preparation keeps the dense path.
function GeometricSolvers.jacobian_pattern(
        ::NonlinearProblem{<:Any, <:Any, Nothing},
        prep::DIJacobian{<:AutoSparse}
)
    prep.backend.sparsity_detector isa NoSparsityDetector && return nothing
    return SparseMatrixCSC(SparseMatrixColorings.sparsity_pattern(prep.jacprep))
end

end
