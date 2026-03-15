"""
    MOProblem(; n, m, f, grad=nothing, jac=nothing, quadratic=nothing, quadstep=nothing)

Define a multiobjective optimization problem: minimize F(x) = [f₁(x), ..., fₘ(x)].

# Arguments
- `n::Int`: number of variables
- `m::Int`: number of objectives
- `f`: objective function `f(x, i) → scalar` returning the value of the i-th objective
- `grad`: gradient function `grad(G, x, i)` filling the gradient of the i-th objective (mutating)
- `jac`: Jacobian function `jac(JF, x)` filling the full m×n Jacobian (alternative to `grad`)
- `quadratic::Vector{Bool}`: flags indicating which objectives are quadratic (default: all false)
- `quadstep`: function `quadstep(x, d, i) → step` returning the exact minimizer along d for quadratic objective i

At least one of `grad` or `jac` must be provided.
"""
struct MOProblem{F,G,J,Q}
    n::Int
    m::Int
    f::F
    grad::G
    jac::J
    quadratic::AbstractVector{Bool}
    quadstep::Q
end

function MOProblem(; n::Int, m::Int, f=nothing, grad=nothing, jac=nothing,
                     quadratic::Union{Nothing,Vector{Bool}}=nothing,
                     quadstep=nothing)
    f === nothing && throw(ArgumentError("objective function `f` is required"))
    grad === nothing && jac === nothing && throw(ArgumentError("either `grad` or `jac` must be provided"))
    n <= 0 && throw(ArgumentError("`n` must be positive, got $n"))
    m <= 0 && throw(ArgumentError("`m` must be positive, got $m"))

    quad = if quadratic === nothing
        falses(m)
    else
        length(quadratic) != m && throw(ArgumentError("`quadratic` length $(length(quadratic)) must equal m=$m"))
        quadratic
    end

    # If only jac is provided, derive grad from it
    g_func = if grad !== nothing
        grad
    else
        let J_fn = jac, n_vars = n, m_objs = m
            (G, x, i) -> begin
                JF_tmp = zeros(eltype(x), m_objs, n_vars)
                J_fn(JF_tmp, x)
                G .= @view JF_tmp[i, :]
            end
        end
    end

    return MOProblem(n, m, f, g_func, jac, quad, quadstep)
end

"""
    SolverOptions(; kwargs...)

Options for the multiobjective descent solver.

# Keyword Arguments
- `epsopt::Float64`: optimality tolerance (default: `5√ε` ≈ 3.33e-8)
- `ftol::Float64`: sufficient decrease tolerance (default: 1e-4)
- `gtol::Float64`: curvature condition tolerance (default: 0.1)
- `ctol::Float64`: sufficient descent condition tolerance (default: 0.4)
- `max_iter::Int`: maximum outer iterations (default: 5000)
- `stpmin::Float64`: minimum step size (default: 1e-15)
- `stpmax::Float64`: maximum step size (default: 1e10)
- `verbose::Bool`: print iteration info (default: false)
- `scale::Bool`: scale objectives by gradient norms at x₀ (default: false)
"""
struct SolverOptions
    epsopt::Float64
    ftol::Float64
    gtol::Float64
    ctol::Float64
    max_iter::Int
    stpmin::Float64
    stpmax::Float64
    verbose::Bool
    scale::Bool
end

function SolverOptions(;
    epsopt::Float64 = 5.0 * sqrt(eps()),
    ftol::Float64 = 1e-4,
    gtol::Float64 = 0.1,
    ctol::Float64 = 0.4,
    max_iter::Int = 5000,
    stpmin::Float64 = 1e-15,
    stpmax::Float64 = 1e10,
    verbose::Bool = false,
    scale::Bool = false,
)
    epsopt <= 0 && throw(ArgumentError("`epsopt` must be positive, got $epsopt"))
    ftol <= 0 && throw(ArgumentError("`ftol` must be positive, got $ftol"))
    gtol <= 0 && throw(ArgumentError("`gtol` must be positive, got $gtol"))
    ftol >= gtol && throw(ArgumentError("`ftol` must be less than `gtol` (got ftol=$ftol, gtol=$gtol)"))
    max_iter <= 0 && throw(ArgumentError("`max_iter` must be positive, got $max_iter"))

    return SolverOptions(epsopt, ftol, gtol, ctol, max_iter, stpmin, stpmax, verbose, scale)
end

"""
    MOPResult

Result returned by the solver.

# Fields
- `x::Vector{Float64}`: solution point
- `fx::Vector{Float64}`: objective values at solution
- `theta::Float64`: optimality measure at solution (θ=0 means Pareto critical)
- `iterations::Int`: number of outer iterations
- `f_evals::Int`: number of objective function evaluations
- `g_evals::Int`: number of gradient evaluations
- `v_evals::Int`: number of steepest descent direction evaluations
- `time::Float64`: CPU time in seconds
- `status::Symbol`: `:optimal`, `:max_iter`, or `:error`
"""
struct MOPResult
    x::Vector{Float64}
    fx::Vector{Float64}
    theta::Float64
    iterations::Int
    f_evals::Int
    g_evals::Int
    v_evals::Int
    time::Float64
    status::Symbol
end

function MOPResult(; x, fx, theta, iterations, f_evals, g_evals, v_evals, time, status)
    return MOPResult(x, fx, theta, iterations, f_evals, g_evals, v_evals, time, status)
end

"""
    MOPState

Mutable iteration state for the solver.
"""
mutable struct MOPState
    # Current point
    x::Vector{Float64}

    # Jacobian (m × n)
    JF::Matrix{Float64}

    # Steepest descent direction
    v::Vector{Float64}

    # Search direction
    d::Vector{Float64}

    # Optimality measure
    theta::Float64

    # Scaling factors
    sF::Vector{Float64}

    # Previous iteration data
    x_prev::Vector{Float64}
    v_prev::Vector{Float64}
    d_prev::Vector{Float64}
    JF_prev::Matrix{Float64}

    # Counters
    iter::Int
    f_evals::Int
    g_evals::Int
    v_evals::Int
end

function MOPState(prob::MOProblem, x0::Vector{Float64})
    n, m = prob.n, prob.m

    return MOPState(
        copy(x0),                  # x
        zeros(m, n),               # JF
        zeros(n),                  # v
        zeros(n),                  # d
        0.0,                       # theta
        ones(m),                   # sF (default: no scaling)
        zeros(n),                  # x_prev
        zeros(n),                  # v_prev
        zeros(n),                  # d_prev
        zeros(m, n),               # JF_prev
        0, 0, 0, 0,               # counters
    )
end

"""
    evaluate_jacobian!(state, prob)

Fill `state.JF` with the Jacobian at `state.x` and increment `state.g_evals`.
Uses `prob.jac` if available, otherwise calls `prob.grad` for each objective.
"""
function evaluate_jacobian!(state::MOPState, prob::MOProblem)
    if prob.jac !== nothing
        prob.jac(state.JF, state.x)
    else
        G = zeros(prob.n)
        for i in 1:prob.m
            prob.grad(G, state.x, i)
            state.JF[i, :] .= G
        end
    end
    # Apply scaling
    for i in 1:prob.m
        state.JF[i, :] .*= state.sF[i]
    end
    state.g_evals += prob.m
    return nothing
end
