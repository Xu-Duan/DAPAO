using DAPAO
using Test
using LinearAlgebra, Printf

@testset "DAPAO Tests" begin
    # Rosenbrock test function
    rosenbrockfunc(x, p=nothing) = (1 - x[1])^2 + 100 * (x[2] - x[1]^2)^2
    function rosenbrockgrad(x, p=nothing)
        g = zeros(2)
        g[1] = -2 * (1 - x[1]) - 400 * x[1] * (x[2] - x[1]^2)
        g[2] = 200 * (x[2] - x[1]^2)
        return g
    end
    function rosenbrockhess(x, p=nothing)
        n = length(x)
        H = zeros(n,n)
        
        # First row/column
        H[1,1] = 1200*x[1]^2 - 400*x[2] + 2
        H[1,2] = -400*x[1]
        if n > 2
            H[1,3:end] .= 0
        end
        
        # Last row/column
        H[n,n] = 200
        H[n,n-1] = -400*x[n-1]
        if n > 2
            H[n,1:n-2] .= 0
        end
        return H
    end
    # Create optimization problem
    optprob = OptimizationFunction(rosenbrockfunc, rosenbrockgrad)
    x0 = [-1.2, 1.0]
    prob = OptimizationProblem(optprob, x0)
    
    @testset "Steepest Descent" begin
        sol = solve(prob, SteepestDescent(), maxiters=10000)
        @test sol.converged
        @test isapprox(sol.x, [1.0, 1.0], atol=0.001)
        @test isapprox(sol.f_val, 0.0, atol=0.001)
    end
    """
    # Test Hessian-free algorithms
    @testset "Hessian-free Algorithms" begin
        # Test Steepest Descent
        @testset "Steepest Descent" begin
            sol = solve(prob, SteepestDescent(), g_tol=1e-6, maxiters=10000)
            @test sol.converged
            @test isapprox(sol.x, [1.0, 1.0], atol=0.4)
            @test isapprox(sol.f_val, 0.0, atol=0.01)
        end
        # Test BFGS
        @testset "BFGS" begin
            sol = solve(prob, BFGS(), g_tol=1e-6, maxiters=1000)
            @test sol.converged
            @test isapprox(sol.x, [1.0, 1.0], atol=1e-3)
            @test isapprox(sol.f_val, 0.0, atol=1e-3)
        end
        
        # Test L-BFGS
        @testset "L-BFGS" begin
            sol = solve(prob, LBFGS(5), g_tol=1e-6, maxiters=1000)
            @test sol.converged
            @test isapprox(sol.x, [1.0, 1.0], atol=1e-3)
            @test isapprox(sol.f_val, 0.0, atol=1e-3)
        end
        
    end
    """

    # Create optimization problem
    optprob = OptimizationFunction(rosenbrockfunc, rosenbrockgrad, rosenbrockhess)
    x0 = [-1.2, 1.0]
    prob = OptimizationProblem(optprob, x0)
    @testset "Standard Newton" begin
        sol = solve(prob, Newton(), g_tol=1e-6, maxiters=100)
        @test sol.converged
        @test isapprox(sol.x, [1.0, 1.0], atol=1e-3)
        @test isapprox(sol.f_val, 0.0, atol=1e-3)
    end

    """
    # Test Newton variants
    @testset "Newton-based Algorithms" begin
        # Standard Newton
        @testset "Standard Newton" begin
            sol = solve(prob, Newton(), g_tol=1e-6, maxiters=100)
            @test sol.converged
            @test isapprox(sol.x, [1.0, 1.0], atol=1e-3)
            @test isapprox(sol.f_val, 0.0, atol=1e-3)
        end
        # Modified Newton
        @testset "Modified Newton" begin
            sol = solve(prob, ModifiedNewton(1e-6), g_tol=1e-6, maxiters=100)
            @test sol.converged
            @test isapprox(sol.x, [1.0, 1.0], atol=1e-3)
            @test isapprox(sol.f_val, 0.0, atol=1e-3)
        end
        
        # Newton CG
        @testset "Newton CG" begin
            sol = solve(prob, NewtonCG(1e-6, 50), g_tol=1e-6, maxiters=100)
            @test sol.converged
            @test isapprox(sol.x, [1.0, 1.0], atol=1e-3)
            @test isapprox(sol.f_val, 0.0, atol=1e-3)
        end
        
    end
    """

    println("All tests completed")
end