using MultiObjectiveDescent
using Test
using LinearAlgebra

@testset "Types" begin

    # ---------------------------------------------------------------
    # Simple 2-objective quadratic problem for testing:
    #   f1(x) = sum(x.^2)
    #   f2(x) = sum((x .- 1).^2)
    # ---------------------------------------------------------------

    f(x, i) = i == 1 ? sum(x .^ 2) : sum((x .- 1) .^ 2)
    function grad!(G, x, i)
        if i == 1
            G .= 2 .* x
        else
            G .= 2 .* (x .- 1)
        end
    end

    @testset "MOProblem construction" begin
        prob = MOProblem(; n=3, m=2, f=f, grad=grad!)
        @test prob.n == 3
        @test prob.m == 2

        # Evaluate objectives through the problem
        x = [1.0, 2.0, 3.0]
        @test prob.f(x, 1) ≈ 14.0   # 1 + 4 + 9
        @test prob.f(x, 2) ≈ 5.0    # 0 + 1 + 4

        # Evaluate gradients through the problem
        G = zeros(3)
        prob.grad(G, x, 1)
        @test G ≈ [2.0, 4.0, 6.0]

        prob.grad(G, x, 2)
        @test G ≈ [0.0, 2.0, 4.0]
    end

    @testset "MOProblem with Jacobian callback" begin
        function jac!(JF, x)
            JF[1, :] .= 2 .* x
            JF[2, :] .= 2 .* (x .- 1)
        end

        prob = MOProblem(; n=3, m=2, f=f, jac=jac!)
        @test prob.n == 3
        @test prob.m == 2

        # Should be able to fill Jacobian
        JF = zeros(2, 3)
        x = [1.0, 2.0, 3.0]
        prob.jac(JF, x)
        @test JF[1, :] ≈ [2.0, 4.0, 6.0]
        @test JF[2, :] ≈ [0.0, 2.0, 4.0]
    end

    @testset "MOProblem requires f and at least one gradient method" begin
        @test_throws ArgumentError MOProblem(; n=3, m=2, grad=grad!)
        @test_throws ArgumentError MOProblem(; n=3, m=2, f=f)
        @test_throws ArgumentError MOProblem(; n=0, m=2, f=f, grad=grad!)
        @test_throws ArgumentError MOProblem(; n=3, m=0, f=f, grad=grad!)
    end

    @testset "MOProblem quadratic flags" begin
        prob = MOProblem(; n=3, m=2, f=f, grad=grad!)
        @test all(.!prob.quadratic)

        prob2 = MOProblem(; n=3, m=2, f=f, grad=grad!, quadratic=[true, true])
        @test all(prob2.quadratic)

        @test_throws ArgumentError MOProblem(; n=3, m=2, f=f, grad=grad!, quadratic=[true])
    end

    @testset "SolverOptions defaults" begin
        opts = SolverOptions()

        @test opts.epsopt ≈ 5.0 * sqrt(eps())
        @test opts.ftol ≈ 1e-4
        @test opts.gtol ≈ 0.1
        @test opts.ctol ≈ 0.4
        @test opts.max_iter == 5000
        @test opts.stpmin ≈ 1e-15
        @test opts.stpmax ≈ 1e10
        @test opts.verbose == false
        @test opts.scale == false
    end

    @testset "SolverOptions custom values" begin
        opts = SolverOptions(; epsopt=1e-6, max_iter=10000, verbose=true)
        @test opts.epsopt ≈ 1e-6
        @test opts.max_iter == 10000
        @test opts.verbose == true
        @test opts.ftol ≈ 1e-4
    end

    @testset "SolverOptions validation" begin
        @test_throws ArgumentError SolverOptions(; ftol=0.5, gtol=0.1)
        @test_throws ArgumentError SolverOptions(; epsopt=-1.0)
        @test_throws ArgumentError SolverOptions(; ftol=-0.1)
        @test_throws ArgumentError SolverOptions(; max_iter=0)
    end

    @testset "MOPResult" begin
        x = [0.5, 0.5, 0.5]
        result = MOPResult(;
            x=x,
            fx=[0.75, 0.75],
            theta=-1e-9,
            iterations=42,
            f_evals=100,
            g_evals=80,
            v_evals=43,
            time=1.23,
            status=:optimal,
        )

        @test result.x ≈ x
        @test result.fx ≈ [0.75, 0.75]
        @test result.theta ≈ -1e-9
        @test result.iterations == 42
        @test result.f_evals == 100
        @test result.g_evals == 80
        @test result.v_evals == 43
        @test result.time ≈ 1.23
        @test result.status == :optimal
    end

    @testset "MOPState initialization" begin
        prob = MOProblem(; n=3, m=2, f=f, grad=grad!)
        x0 = [1.0, 2.0, 3.0]
        state = MOPState(prob, x0)

        @test state.x ≈ x0
        state.x[1] = 999.0
        @test x0[1] ≈ 1.0   # original not mutated

        @test size(state.JF) == (2, 3)
        @test length(state.v) == 3
        @test length(state.d) == 3
        @test state.iter == 0
        @test state.f_evals == 0
        @test state.g_evals == 0
        @test state.v_evals == 0
        @test length(state.sF) == 2
    end

    @testset "MOPState evaluate_jacobian!" begin
        prob = MOProblem(; n=3, m=2, f=f, grad=grad!)
        x0 = [1.0, 2.0, 3.0]
        state = MOPState(prob, x0)

        evaluate_jacobian!(state, prob)

        @test state.JF[1, :] ≈ [2.0, 4.0, 6.0]
        @test state.JF[2, :] ≈ [0.0, 2.0, 4.0]
        @test state.g_evals == 2
    end

    @testset "MOPState evaluate_jacobian! with jac callback" begin
        function jac!(JF, x)
            JF[1, :] .= 2 .* x
            JF[2, :] .= 2 .* (x .- 1)
        end
        prob = MOProblem(; n=3, m=2, f=f, jac=jac!)
        x0 = [1.0, 2.0, 3.0]
        state = MOPState(prob, x0)

        evaluate_jacobian!(state, prob)

        @test state.JF[1, :] ≈ [2.0, 4.0, 6.0]
        @test state.JF[2, :] ≈ [0.0, 2.0, 4.0]
        @test state.g_evals == 2
    end

end
