include("algorithms.jl")
"""
    solve.jl

Implementation of the solve function for DAPAO.
"""

# Main solve function
function solve(prob::OptimizationProblem, alg::OptimizationAlgorithm; 
               g_tol=1e-6, f_tol=1e-8, x_tol=1e-8, maxiters=1000, kwargs...)
    # Extract components
    f = prob.optfunc.f
    grad = prob.optfunc.grad
    Hess = prob.optfunc.Hess
    x0 = prob.x0
    p = prob.p
    iter_counter = 0
    xk = x0
    x_prev = x0
    objk = f(xk, p)
    gradk = grad(xk, p)
    grad_prev = gradk

    while iter_counter < maxiters
        # Get search direction and step size
        pk = compute_general_direction(alg, xk, gradk, f, grad, Hess, p);
        alpha = wolfe_line_search(xk, pk, f, grad, p);

        # Update iterate
        x_prev = xk;
        obj_prev = objk;
        grad_prev = gradk;
        xk = xk + alpha * pk;
        objk = f(xk, p);
        gradk = grad(xk, p);
        
        # Increment counter
        iter_counter = iter_counter + 1;
        
        # Check termination conditions
        grad_cond = norm(gradk) < g_tol;                    # Gradient norm small enough
        fval_cond = abs(objk - obj_prev) < f_tol*(1 + abs(obj_prev)); # Relative change in f small
        step_cond = norm(xk - x_prev) < x_tol*(1 + norm(x_prev));     # Relative step size small
        
        # Set status based on which condition was met
        if grad_cond
            status = @sprintf("Gradient norm tolerance reached (%.2e < %.2e)", norm(gradk), g_tol);
            break;
        elseif fval_cond
            status = @sprintf("Function value tolerance reached (%.2e)", abs(objk - obj_prev)/(1 + abs(obj_prev)));
            break;
        elseif step_cond
            status = @sprintf("Step size tolerance reached (%.2e)", norm(xk - x_prev)/(1 + norm(x_prev)));
            break;
        elseif alpha < 1e-6
            status = @sprintf("Step size too small (%.2e)", alpha);
            break;
        end
    end

    # Check if max iterations was reached
    if iter_counter >= maxiters
        status = @sprintf("Maximum iterations (%d) reached", maxiters);
    end

    # Max iterations reached
    return OptimizationSolution(xk, objk, iter_counter < maxiters, iter_counter, status)
end