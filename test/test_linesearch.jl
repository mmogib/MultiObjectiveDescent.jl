using MultiObjectiveDescent
using Test
using LinearAlgebra

@testset "Vector Wolfe line search" begin

    # ---------------------------------------------------------------
    # Helper: create line functions for a 2-objective problem
    #   f1(x) = sum(x.^2),  f2(x) = sum((x .- a).^2)
    # along direction d from point x0
    # ---------------------------------------------------------------
    function make_line_fns(x0, d, a)
        evalf = (stp, i) -> begin
            xp = x0 .+ stp .* d
            i == 1 ? sum(xp .^ 2) : sum((xp .- a) .^ 2)
        end
        evalgphi = (stp, i) -> begin
            xp = x0 .+ stp .* d
            g = i == 1 ? 2.0 .* xp : 2.0 .* (xp .- a)
            dot(g, d)
        end
        return evalf, evalgphi
    end

    @testset "Standard Wolfe (lstype=:standard)" begin
        x0 = [3.0, 3.0]
        d = [-1.0, -1.0]  # descent direction
        a = [1.0, 1.0]

        evalf, evalgphi = make_line_fns(x0, d, a)
        m = 2
        phi0 = [evalf(0.0, i) for i in 1:m]
        gphi0 = [evalgphi(0.0, i) for i in 1:m]

        @test maximum(gphi0) < 0  # must be descent

        stp, nfev, ngev, info = vector_wolfe_linesearch(
            evalf, evalgphi, 1.0, phi0, gphi0;
            m=m, ftol=1e-4, gtol=0.1, stpmin=1e-15, stpmax=1e10,
            lstype=:standard,
        )

        @test info == 0
        @test stp > 0

        # Sufficient decrease for ALL objectives
        maxg0 = maximum(gphi0)
        for i in 1:m
            @test evalf(stp, i) <= phi0[i] + 1e-4 * stp * maxg0 + 1e-10
        end

        # Standard Wolfe curvature: max_i{gphi_i(stp)} >= gtol * maxg0
        maxg = maximum(evalgphi(stp, i) for i in 1:m)
        @test maxg >= 0.1 * maxg0 - 1e-10
    end

    @testset "Strong Wolfe (lstype=:strong)" begin
        x0 = [3.0, 3.0]
        d = [-1.0, -1.0]
        a = [1.0, 1.0]

        evalf, evalgphi = make_line_fns(x0, d, a)
        m = 2
        phi0 = [evalf(0.0, i) for i in 1:m]
        gphi0 = [evalgphi(0.0, i) for i in 1:m]

        stp, nfev, ngev, info = vector_wolfe_linesearch(
            evalf, evalgphi, 1.0, phi0, gphi0;
            m=m, ftol=1e-4, gtol=0.1, stpmin=1e-15, stpmax=1e10,
            lstype=:strong,
        )

        @test info == 0
        @test stp > 0

        # Sufficient decrease
        maxg0 = maximum(gphi0)
        for i in 1:m
            @test evalf(stp, i) <= phi0[i] + 1e-4 * stp * maxg0 + 1e-10
        end

        # Strong Wolfe curvature: |max_i{gphi_i(stp)}| <= -gtol * maxg0
        maxg = maximum(evalgphi(stp, i) for i in 1:m)
        @test abs(maxg) <= -0.1 * maxg0 + 1e-10
    end

    @testset "Restricted Wolfe (lstype=:restricted)" begin
        x0 = [3.0, 3.0]
        d = [-1.0, -1.0]
        a = [1.0, 1.0]

        evalf, evalgphi = make_line_fns(x0, d, a)
        m = 2
        phi0 = [evalf(0.0, i) for i in 1:m]
        gphi0 = [evalgphi(0.0, i) for i in 1:m]
        maxg0 = maximum(gphi0)

        tol = 1e6  # generous upper bound

        stp, nfev, ngev, info = vector_wolfe_linesearch(
            evalf, evalgphi, 1.0, phi0, gphi0;
            m=m, ftol=1e-4, gtol=0.1, stpmin=1e-15, stpmax=1e10,
            lstype=:restricted, tol=tol,
        )

        @test info == 0
        @test stp > 0

        # Sufficient decrease
        for i in 1:m
            @test evalf(stp, i) <= phi0[i] + 1e-4 * stp * maxg0 + 1e-10
        end

        # Restricted Wolfe: gtol*maxg0 <= max_i{gphi_i(stp)} <= tol
        maxg = maximum(evalgphi(stp, i) for i in 1:m)
        @test maxg >= 0.1 * maxg0 - 1e-10
        @test maxg <= tol + 1e-10
    end

    @testset "Three objectives" begin
        x0 = [2.0, 2.0, 2.0]
        d = [-1.0, -1.0, -1.0]

        evalf = (stp, i) -> begin
            xp = x0 .+ stp .* d
            if i == 1
                sum(xp .^ 2)
            elseif i == 2
                sum((xp .- 1.0) .^ 2)
            else
                sum((xp .- 0.5) .^ 2)
            end
        end
        evalgphi = (stp, i) -> begin
            xp = x0 .+ stp .* d
            if i == 1
                dot(2.0 .* xp, d)
            elseif i == 2
                dot(2.0 .* (xp .- 1.0), d)
            else
                dot(2.0 .* (xp .- 0.5), d)
            end
        end

        m = 3
        phi0 = [evalf(0.0, i) for i in 1:m]
        gphi0 = [evalgphi(0.0, i) for i in 1:m]
        maxg0 = maximum(gphi0)

        stp, nfev, ngev, info = vector_wolfe_linesearch(
            evalf, evalgphi, 1.0, phi0, gphi0;
            m=m, ftol=1e-4, gtol=0.1, stpmin=1e-15, stpmax=1e10,
            lstype=:standard,
        )

        @test info == 0
        # Sufficient decrease for all 3
        for i in 1:m
            @test evalf(stp, i) <= phi0[i] + 1e-4 * stp * maxg0 + 1e-10
        end
    end

    @testset "Not a descent direction → error" begin
        # All gphi0 non-negative → not descent
        evalf = (stp, i) -> stp^2
        evalgphi = (stp, i) -> 2.0 * stp
        phi0 = [0.0]
        gphi0 = [1.0]  # positive → not descent

        stp, nfev, ngev, info = vector_wolfe_linesearch(
            evalf, evalgphi, 1.0, phi0, gphi0;
            m=1, ftol=1e-4, gtol=0.1, stpmin=1e-15, stpmax=1e10,
            lstype=:standard,
        )

        @test info == -1
    end

    @testset "Function evaluation counts" begin
        x0 = [3.0, 3.0]
        d = [-1.0, -1.0]
        a = [1.0, 1.0]
        evalf, evalgphi = make_line_fns(x0, d, a)
        m = 2
        phi0 = [evalf(0.0, i) for i in 1:m]
        gphi0 = [evalgphi(0.0, i) for i in 1:m]

        stp, nfev, ngev, info = vector_wolfe_linesearch(
            evalf, evalgphi, 1.0, phi0, gphi0;
            m=m, ftol=1e-4, gtol=0.1, stpmin=1e-15, stpmax=1e10,
            lstype=:standard,
        )

        @test nfev >= 0
        @test ngev >= 0
    end

end
