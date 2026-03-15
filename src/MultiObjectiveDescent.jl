module MultiObjectiveDescent

using LinearAlgebra
using Printf
using JuMP
using Clarabel

# Types
include("types.jl")

# Steepest descent subproblem
include("subproblem.jl")

# Public API
export MOProblem, SolverOptions, MOPResult, MOPState
export evaluate_jacobian!
export solve_steepest_descent, solve_subproblem!

end
