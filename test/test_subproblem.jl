using MultiObjectiveDescent
using Test
using LinearAlgebra
using Clarabel

@testset "Subproblem" begin

    # ---------------------------------------------------------------
    # The steepest descent subproblem:
    #   min   alpha + (1/2)||d||^2
    #   s.t.  [JF * d]_i <= alpha,   i = 1,...,m
    #
    # Returns v (optimal d) and theta = max_i(JF*v)_i + ||v||^2/2
    # At Pareto critical: v = 0, theta = 0
    # At non-critical: v ≠ 0, theta < 0
    # ---------------------------------------------------------------

    @testset "Pareto critical point → v = 0, theta = 0" begin
        # f1(x) = sum(x.^2), f2(x) = sum((x-1).^2)
        # At x = [0.5, 0.5, 0.5]:
        #   grad_f1 = [1, 1, 1], grad_f2 = [-1, -1, -1]
        # Gradients oppose each other → Pareto critical
        JF = [1.0 1.0 1.0;
              -1.0 -1.0 -1.0]

        v, theta, info = solve_steepest_descent(JF; optimizer=Clarabel.Optimizer)

        @test norm(v) < 1e-6
        @test abs(theta) < 1e-6
        @test info == 0
    end

    @testset "Non-critical point → v ≠ 0, theta < 0" begin
        # At x = [2, 2, 2]:
        #   grad_f1 = [4, 4, 4], grad_f2 = [2, 2, 2]
        # Both gradients point in same direction → not Pareto critical
        JF = [4.0 4.0 4.0;
              2.0 2.0 2.0]

        v, theta, info = solve_steepest_descent(JF; optimizer=Clarabel.Optimizer)

        @test norm(v) > 1e-6
        @test theta < -1e-6
        @test info == 0

        # v must be a descent direction for ALL objectives
        directional_derivs = JF * v
        @test all(directional_derivs .< -1e-8)
    end

    @testset "Descent direction property" begin
        # For any non-critical JF, max_i(JF * v)_i < 0
        JF = [1.0 0.0;
              0.0 1.0;
              -0.5 0.5]

        v, theta, info = solve_steepest_descent(JF; optimizer=Clarabel.Optimizer)

        max_deriv = maximum(JF * v)
        @test max_deriv < 0
        @test theta ≈ max_deriv + norm(v)^2 / 2
    end

    @testset "Single objective reduces to negative gradient" begin
        # With m=1: v should be -grad (normalized by the QP)
        # QP: min alpha + (1/2)||d||^2  s.t. g'*d <= alpha
        # Solution: d = -g, alpha = -||g||^2
        g = [3.0, 4.0]
        JF = reshape(g, 1, 2)

        v, theta, info = solve_steepest_descent(JF; optimizer=Clarabel.Optimizer)

        @test v ≈ -g atol=1e-6
        @test theta ≈ -norm(g)^2 / 2 atol=1e-6
    end

    @testset "Theta formula: theta = max(JF*v) + ||v||^2/2" begin
        JF = [2.0 1.0 0.0;
              0.0 1.0 3.0]

        v, theta, info = solve_steepest_descent(JF; optimizer=Clarabel.Optimizer)

        expected_theta = maximum(JF * v) + norm(v)^2 / 2
        @test theta ≈ expected_theta atol=1e-6
    end

    @testset "Pluggable optimizer" begin
        # Should work with any JuMP-compatible optimizer
        JF = [1.0 2.0;
              3.0 -1.0]

        v1, theta1, _ = solve_steepest_descent(JF; optimizer=Clarabel.Optimizer)
        # Same result with explicit Clarabel (just confirming the interface works)
        v2, theta2, _ = solve_steepest_descent(JF; optimizer=Clarabel.Optimizer)

        @test v1 ≈ v2 atol=1e-6
        @test theta1 ≈ theta2 atol=1e-6
    end

    @testset "solve_subproblem! updates state" begin
        f(x, i) = i == 1 ? sum(x .^ 2) : sum((x .- 1) .^ 2)
        function grad!(G, x, i)
            if i == 1
                G .= 2 .* x
            else
                G .= 2 .* (x .- 1)
            end
        end

        prob = MOProblem(; n=3, m=2, f=f, grad=grad!)

        # Non-critical point
        x0 = [2.0, 2.0, 2.0]
        state = MOPState(prob, x0)
        evaluate_jacobian!(state, prob)

        solve_subproblem!(state; optimizer=Clarabel.Optimizer)

        @test norm(state.v) > 1e-6
        @test state.theta < -1e-6
        @test state.v_evals == 1

        # Call again → counter increments
        solve_subproblem!(state; optimizer=Clarabel.Optimizer)
        @test state.v_evals == 2
    end

    @testset "solve_subproblem! at Pareto critical" begin
        f(x, i) = i == 1 ? sum(x .^ 2) : sum((x .- 1) .^ 2)
        function grad!(G, x, i)
            if i == 1
                G .= 2 .* x
            else
                G .= 2 .* (x .- 1)
            end
        end

        prob = MOProblem(; n=3, m=2, f=f, grad=grad!)

        # Pareto critical point
        x0 = [0.5, 0.5, 0.5]
        state = MOPState(prob, x0)
        evaluate_jacobian!(state, prob)

        solve_subproblem!(state; optimizer=Clarabel.Optimizer)

        @test norm(state.v) < 1e-6
        @test abs(state.theta) < 1e-6
    end

end
