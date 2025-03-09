module DAPAO

using LinearAlgebra, Printf

# Export types and functions
export OptimizationFunction, OptimizationProblem, solve

# Export Hessian-free algorithms
export SteepestDescent, BFGS, LBFGS

# Export Newton-based algorithms
export Newton, ModifiedNewton, NewtonCG

include("solve.jl")


end