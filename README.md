# DAPAO (Distributed Advanced PArallel Optimization)

A high-performance optimization library that leverages parallel computing for large-scale optimization problems. DAPAO provides efficient implementations of classical and modern optimization algorithms including Steepest Descent, Newton, Modified Newton, BFGS, L-BFGS, and Newton-CG.

## Key Features

- **Parallel Execution**: Utilizes Julia's native parallel computing capabilities for faster convergence on multi-core systems
- **Distributed Computing Support**: Scales to multiple nodes for handling very large optimization problems
- **High Performance**: Optimized implementations of gradient-based algorithms
- **Flexible API**: Simple interface that works with both simple and complex optimization problems
- **Extensible Design**: Easy to add custom optimization methods

## Installation

You can install DAPAO from the Julia REPL using the package manager:

```julia
using Pkg
Pkg.add("DAPAO")
```

Or, in pkg mode (press `]` in the REPL):

```julia
add DAPAO
```

## Direction Search Methods
- [x] Steepest Descent
- [x] Newton
- [x] Modified Newton
- [ ] BFGS
- [ ] L-BFGS
- [x] Newton-CG

## TO-DO
- [ ] Implement BFGS method
- [ ] Implement L-BFGS method
- [ ] Pass all tests for different functions in test/runtest.jl
- [ ] Test scalability


## How to use

Basic usage example:

```julia
using DAPAO
using LinearAlgebra

# Define your objective function and its gradient
function f(x, p)
    return x[1]^2 + 2x[2]^2
end

function grad(x, p)
    g = zeros(2)
    g[1] = 2x[1]
    g[2] = 4x[2]
    return g
end

# Optimize using Steepest Descent
optfunc = OptimizationFunction(f, grad)
x0 = [-1.2, 1.0]
prob = OptimizationProblem(optfunc, x0)
sol = solve(prob, SteepestDescent())
println("Minimum found at: ", sol.x)
println("Minimum value: ", sol.f_val)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
