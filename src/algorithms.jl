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

function compute_direction(alg::LBFGS, x, g, H, f, grad, p)
    # For L-BFGS, H is actually a tuple of (s_history, y_history)
    #s_history, y_history = H
    #return compute_lbfgs_direction(g, s_history, y_history, alg.m)
    return -g
end

function compute_direction(alg::Newton, x, g, f, grad, H, p)
    # Standard Newton's method: pk = -H⁻¹ * g
    # For simplicity, assume H is the Hessian and invertible
    return -H(x, p) \ g  # Backslash operator solves H * pk = -g
end

function compute_direction(alg::ModifiedNewton, x, g, H, f, grad!, p)
    # Modified Newton's method: Ensure H is positive definite
    # by modifying eigenvalues if necessary
    
    # Compute eigendecomposition
    F = eigen(Symmetric(H))
    vals, vecs = F.values, F.vectors
    
    # Modify eigenvalues to ensure positive definiteness
    modified_vals = max.(vals, alg.min_eig)
    
    # Reconstruct modified Hessian
    H_modified = vecs * Diagonal(modified_vals) * vecs'
    
    # Solve the system with modified Hessian
    return -H_modified \ g
end

function compute_direction(alg::NewtonCG, x, g, H, f, grad!, p)
    # Newton-CG method: Use conjugate gradient to solve H * pk = -g
    
    # Implementation of conjugate gradient method
    function cg_solve(A, b, tol, max_iter)
        x = zeros(length(b))
        r = b - A * x
        p = copy(r)
        rsold = dot(r, r)
        
        for i in 1:max_iter
            Ap = A * p
            alpha = rsold / dot(p, Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = dot(r, r)
            
            if sqrt(rsnew) < tol
                break
            end
            
            p = r + (rsnew / rsold) * p
            rsold = rsnew
        end
        
        return x
    end
    
    # Solve the Newton system using CG
    return cg_solve(H, -g, alg.tol, alg.max_cg_iter)
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