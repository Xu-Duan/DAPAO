include("types.jl")
"""
    algorithms.jl

This module contains the algorithms for the DAPAO project.
"""

"""
    compute_general_direction(alg::OptimizationAlgorithm, x, g, H, f, grad!, p)

Compute the search direction `pk` for a given algorithm.
- `alg`: The optimization algorithm (e.g., BFGS, SteepestDescent).
- `x`: Current point.
- `g`: Gradient at `x`.
- `H`: Hessian or approximation (used by some algorithms).
- `f`: Objective function.
- `grad!`: Gradient function.
- `p`: Parameters (optional).
Returns the search direction `pk`.
"""
function compute_general_direction(alg::OptimizationAlgorithm, x, g, f=nothing, grad=nothing, H=nothing, p=nothing)
    if alg isa GradientBasedAlgorithm
        if H !== nothing
            @warn "Hessian is provided, but Hessian-free algorithm is used."
        end
        return compute_direction(alg::GradientBasedAlgorithm, x, g, p)
    elseif alg isa HessianBasedAlgorithm
        if H === nothing
            error("Hessian is not provided, but Newton-based algorithm is used.")
        end
        return compute_direction(alg::HessianBasedAlgorithm, x, g, f, grad, H, p)
    else
        throw(ArgumentError("Unsupported algorithm type"))
    end
end

function compute_direction(alg::SteepestDescent, x, g, p)
    return -g  # Steepest descent: direction is negative gradient
end

function compute_direction(alg::BFGS, x, g, p)
    return -g  # BFGS: direction uses inverse Hessian approximation
end

function compute_direction(alg::LBFGS, x, g, p)
    # For L-BFGS, we would normally use a history of positions and gradients
    # For simplicity, we'll just return the negative gradient for now
    return -g
end

function compute_direction(alg::Newton, x, g, f, grad, H, p)
    # Standard Newton's method: pk = -H⁻¹ * g
    # For simplicity, assume H is the Hessian and invertible
    return -H(x, p) \ g  # Backslash operator solves H * pk = -g
end

function compute_direction(alg::ModifiedNewton, x, g, f, grad, H, p)
    # Modified Newton's method: Ensure H is positive definite
    # by modifying eigenvalues if necessary
    
    # Compute Hessian
    hessk = H(x, p)
    beta = alg.min_eig

    if minimum(diag(hessk)) > 0
        delta = 0
    else
        delta = -minimum(diag(hessk)) + beta
    end
    
    # Try Cholesky factorization, if it fails, modify the Hessian
    local C
    while true
        try
            # Try Cholesky factorization
            C = cholesky(hessk + delta * I, check=false)
            break  # Exit the loop if successful
        catch
            # Subsequent failures: double delta
            delta = max(2 * delta, beta)
        end
    end

    # Solve the system using Cholesky factorization
    pk = C \ (-g)     # Solve using the Cholesky factorization
    return pk
end

function compute_direction(alg::NewtonCG, x, g, f, grad, H, p)
    # Newton-CG method: Use conjugate gradient to solve H * pk = -g
    
    # Get Hessian at current point
    hessk = H(x, p)
    
    # Extract parameters from alg (assuming NewtonCG is a struct)
    cg_tol = alg.tol  # CG tolerance
    max_iter = alg.max_cg_iter
    
    # Initialize CG variables
    z = zeros(eltype(x), length(x))  # Match type of x (e.g., Float64)
    r = copy(g)                      # Initial residual = gradient
    d = -r                           # Initial CG direction
    
    # Conjugate Gradient iterations
    for j in 0:max_iter
        # Compute Hessian-vector product
        Hd = hessk * d
        
        # Check for negative curvature
        dHd = dot(d, Hd)  # d' * Hd in Julia
        if dHd <= 0
            # If negative curvature on first iteration, use steepest descent
            return (j == 0) ? -g : z
        end
        
        # Compute step size
        rr = dot(r, r)    # r' * r
        alpha = rr / dHd
        
        # Update solution and residual
        z .+= alpha .* d     # In-place update
        r_new = r + alpha * Hd
        
        # Check for convergence based on norm of residual
        if norm(r_new) < cg_tol * norm(g)
            return z
        end
        
        # Compute beta using Polak-Ribiere formula
        r_new_r_new = dot(r_new, r_new)
        beta = r_new_r_new / rr
        
        # Update direction
        d = -r_new + beta * d
        
        # Update residual
        r = r_new
    end
    
    # If max iterations reached without convergence, return current solution
    return z
end

"""
    compute_lbfgs_direction(g, s_history, y_history, m)

Compute the L-BFGS search direction using the two-loop recursion algorithm.
- `g`: Current gradient.
- `s_history`: History of position differences.
- `y_history`: History of gradient differences.
- `m`: Maximum number of correction pairs to store.
Returns the search direction.
"""
function compute_lbfgs_direction(g, s_history, y_history, m)
    if isempty(s_history)
        return -g  # If no history yet, use steepest descent
    end
    
    # Two-loop recursion algorithm for L-BFGS
    q = copy(g)
    k = length(s_history)
    α = zeros(k)
    ρ = [1.0 / dot(y_history[i], s_history[i]) for i in 1:k]
    
    # First loop
    for i in k:-1:1
        α[i] = ρ[i] * dot(s_history[i], q)
        q = q - α[i] * y_history[i]
    end
    
    # Initial Hessian approximation H_0 = (y'*s)/(y'*y) * I
    γ = dot(s_history[end], y_history[end]) / dot(y_history[end], y_history[end])
    r = γ * q
    
    # Second loop
    for i in 1:k
        β = ρ[i] * dot(y_history[i], r)
        r = r + (α[i] - β) * s_history[i]
    end
    
    return -r  # Return the negative direction for minimization
end

function wolfe_line_search(xk, pk, f, grad, p, c1 = 1e-4, c2 = 0.9, alpha_init = 1.0)
    # Get current values
    fk = f(xk, p)
    gradk = grad(xk, p)
    # Store initial gradient dot product for second Wolfe condition
    initial_grad_dot_pk = gradk' * pk
    
    alpha = alpha_init  # Initial step size
    alpha_l = 0
    alpha_u = Inf

    # Wolfe line search
    while true
        # Check function value condition (first Wolfe condition)
        if f(xk + alpha*pk, p) > fk + c1*alpha*(initial_grad_dot_pk)
            alpha_u = alpha
        else
            # Check curvature condition (second Wolfe condition)
            new_gradk = grad(xk + alpha*pk, p)
            if new_gradk' * pk < c2 * initial_grad_dot_pk
                alpha_l = alpha
            else
                # Both conditions satisfied
                break
            end
        end
        
        # Update alpha
        if alpha_u < Inf
            alpha = (alpha_l + alpha_u) / 2  # Bisection
        else
            alpha = 2 * alpha  # Double step size
        end
        
        # Safety check for minimum step size
        if alpha < 1e-10
            break
        end
    end
    return alpha
end