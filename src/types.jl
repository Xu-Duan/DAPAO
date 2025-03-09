struct OptimizationFunction{F,G,H}
    f::F
    grad::G
    Hess::H
end
OptimizationFunction(f, grad; Hess=nothing) = OptimizationFunction(f, grad, Hess)

struct OptimizationProblem{T,F,G,H, P}
    optfunc::OptimizationFunction{F,G,H}
    x0::T
    p::P
end
OptimizationProblem(optfunc, x0, p=nothing) = OptimizationProblem(optfunc, x0, p)

# Solution type to store results
struct OptimizationSolution{T}
    x::T           # Final solution
    f_val::Float64 # Final objective value
    converged::Bool# Whether the solver converged
    iterations::Int# Number of iterations
    status::String # result status
end

abstract type OptimizationAlgorithm end
abstract type GradientBasedAlgorithm <: OptimizationAlgorithm end
abstract type HessianBasedAlgorithm <: OptimizationAlgorithm end

struct SteepestDescent <: GradientBasedAlgorithm end
struct BFGS <: GradientBasedAlgorithm end

# L-BFGS algorithm
struct LBFGS <: GradientBasedAlgorithm
    m::Int  # Number of corrections to store
    
    # Constructor with default value
    LBFGS(m::Int=10) = new(m)
end

struct Newton <: HessianBasedAlgorithm end
# Modified Newton algorithm (handles non-positive definite Hessians)
struct ModifiedNewton <: HessianBasedAlgorithm
    min_eig::Float64  # Minimum eigenvalue threshold
    
    # Constructor with default value
    ModifiedNewton(min_eig::Float64=1e-6) = new(min_eig)
end
# Newton-CG algorithm (uses conjugate gradient for solving the Newton system)
struct NewtonCG <: HessianBasedAlgorithm
    tol::Float64      # Tolerance for CG method
    max_cg_iter::Int  # Maximum CG iterations
    
    # Constructor with default values
    NewtonCG(tol::Float64=1e-6, max_cg_iter::Int=100) = new(tol, max_cg_iter)
end