"""
    solve_steepest_descent(JF; optimizer=Clarabel.Optimizer) → (v, theta, info)

Compute the steepest descent direction by solving the QP subproblem:

    min   α + (1/2)||d||²
    s.t.  [JF · d]ᵢ ≤ α,   i = 1, ..., m

Returns:
- `v::Vector{Float64}`: steepest descent direction (optimal d)
- `theta::Float64`: optimality measure θ = max_i(JF·v)_i + ||v||²/2
- `info::Int`: 0 = optimal, nonzero = error

At a Pareto critical point: v = 0, θ = 0.
At a non-critical point: v ≠ 0, θ < 0.
"""
function solve_steepest_descent(JF::AbstractMatrix{Float64};
                                 optimizer=Clarabel.Optimizer)
    m, n = size(JF)

    model = Model(optimizer)
    set_silent(model)

    @variable(model, d[1:n])
    @variable(model, α)

    # Objective: min α + (1/2)||d||²
    @objective(model, Min, α + 0.5 * sum(d[j]^2 for j in 1:n))

    # Constraints: [JF · d]ᵢ ≤ α for each objective
    for i in 1:m
        @constraint(model, sum(JF[i, j] * d[j] for j in 1:n) <= α)
    end

    optimize!(model)

    info = if termination_status(model) == OPTIMAL
        0
    else
        -1
    end

    v = value.(d)
    alpha_val = value(α)

    # theta = max_i(JF*v)_i + ||v||^2/2
    # At optimality, alpha_val = max_i(JF*v)_i, so theta = alpha_val + ||v||^2/2
    theta = alpha_val + 0.5 * dot(v, v)

    return v, theta, info
end

"""
    solve_subproblem!(state; optimizer=Clarabel.Optimizer)

Compute the steepest descent direction from `state.JF` and store the result
in `state.v` and `state.theta`. Increments `state.v_evals`.
"""
function solve_subproblem!(state::MOPState; optimizer=Clarabel.Optimizer)
    v, theta, info = solve_steepest_descent(state.JF; optimizer=optimizer)
    state.v .= v
    state.theta = theta
    state.v_evals += 1
    return info
end
