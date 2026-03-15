using MultiObjectiveDescent
using Test

@testset "More-Thuente scalar line search" begin

    # ---------------------------------------------------------------
    # The More-Thuente algorithm finds a step α satisfying:
    #   Sufficient decrease: f(α) ≤ f(0) + ftol·α·f'(0)
    #   Curvature condition:  |f'(α)| ≤ gtol·|f'(0)|
    # ---------------------------------------------------------------

    @testset "Simple quadratic: f(α) = (α - 2)²" begin
        # f(α) = (α-2)², f'(α) = 2(α-2)
        # f(0) = 4, f'(0) = -4
        # Minimum at α = 2
        phi(α) = (α - 2.0)^2
        dphi(α) = 2.0 * (α - 2.0)

        stp, info = mt_linesearch(phi, dphi, 1.0;
                                   ftol=1e-4, gtol=0.9,
                                   stpmin=1e-15, stpmax=1e10)

        @test info == 0
        @test stp > 0
        # Should satisfy Wolfe conditions
        @test phi(stp) <= phi(0.0) + 1e-4 * stp * dphi(0.0)
        @test abs(dphi(stp)) <= 0.9 * abs(dphi(0.0))
    end

    @testset "Steep quadratic: f(α) = 100·(α - 0.01)²" begin
        phi(α) = 100.0 * (α - 0.01)^2
        dphi(α) = 200.0 * (α - 0.01)

        stp, info = mt_linesearch(phi, dphi, 1.0;
                                   ftol=1e-4, gtol=0.9,
                                   stpmin=1e-15, stpmax=1e10)

        @test info == 0
        @test phi(stp) <= phi(0.0) + 1e-4 * stp * dphi(0.0)
        @test abs(dphi(stp)) <= 0.9 * abs(dphi(0.0))
    end

    @testset "Cubic function" begin
        # f(α) = α³ - 6α² + 11α - 6, f'(α) = 3α² - 12α + 11
        # f(0) = -6, f'(0) = 11 > 0 ... not a descent direction
        # Instead use: f(α) = -α + α²/2
        # f(0) = 0, f'(0) = -1, minimum at α = 1
        phi(α) = -α + 0.5 * α^2
        dphi(α) = -1.0 + α

        stp, info = mt_linesearch(phi, dphi, 0.5;
                                   ftol=1e-4, gtol=0.1,
                                   stpmin=1e-15, stpmax=1e10)

        @test info == 0
        @test phi(stp) <= phi(0.0) + 1e-4 * stp * dphi(0.0)
        @test abs(dphi(stp)) <= 0.1 * abs(dphi(0.0))
    end

    @testset "Respects stpmin and stpmax bounds" begin
        phi(α) = (α - 100.0)^2
        dphi(α) = 2.0 * (α - 100.0)

        # stpmax = 5 should prevent reaching the minimum at α=100
        stp, info = mt_linesearch(phi, dphi, 1.0;
                                   ftol=1e-4, gtol=0.9,
                                   stpmin=1e-15, stpmax=5.0)

        @test stp <= 5.0
    end

    @testset "Rosenbrock-like along search direction" begin
        # f(α) = 100(α - 1)⁴ + (1 - α)²
        # f(0) = 101, f'(0) = -400 - 2 + 2 = -402 (descent)
        phi(α) = 100.0 * (α - 1.0)^4 + (1.0 - α)^2
        dphi(α) = 400.0 * (α - 1.0)^3 - 2.0 * (1.0 - α)

        stp, info = mt_linesearch(phi, dphi, 0.1;
                                   ftol=1e-4, gtol=0.9,
                                   stpmin=1e-15, stpmax=1e10)

        @test info == 0
        @test phi(stp) <= phi(0.0) + 1e-4 * stp * dphi(0.0)
    end

end
