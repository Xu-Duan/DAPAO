# DAPAO (Distributed Advanced PArallel Optimization)

A optimization library that supports Steepest Descent, Newton, Modified Newton, BFGS, L-BFGS, Newton-CG

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
- [ ] Modified Newton
- [ ] BFGS
- [ ] L-BFGS
- [ ] Newton-CG

## TO-DO
- [ ] Implement Modified Newton method
- [ ] Implement BFGS method
- [ ] Implement L-BFGS method
- [ ] Implement Newton-CG method
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
