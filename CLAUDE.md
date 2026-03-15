# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with the MultiObjectiveDescent.jl package.

## Package Overview

Julia package for descent methods for unconstrained multiobjective (vector) optimization problems. Supports pluggable search directions (steepest descent, conjugate gradient, BFGS, L-BFGS, Newton), line searches, and QP subproblem solvers.

## Development

```bash
# Run tests
julia --project=. -e 'using Pkg; Pkg.test()'

# Run a specific test file
julia --project=. -e 'using Pkg; Pkg.activate("test"); include("test/test_subproblem.jl")'

# Build docs locally
julia --project=docs docs/make.jl
```

## Architecture

### Unified Solver Loop (all methods share this)

```
for k = 0, 1, 2, ...
    1. Compute Jacobian JF(x_k)
    2. Compute steepest descent v(x_k) via QP subproblem  → subproblem.jl
    3. Check optimality |theta(x_k)| <= eps
    4. Compute search direction d_k                        → directions/*.jl
    5. Line search for step alpha_k                        → linesearch.jl
    6. Update x_{k+1} = x_k + alpha_k * d_k
    7. Update method-specific state                        → directions/*.jl
end
```

### Source Layout

```
src/
├── MultiObjectiveDescent.jl    # Module: exports, includes
├── types.jl                    # MOProblem, MOPState, MOPResult, SolverOptions
├── subproblem.jl               # Steepest descent QP via JuMP (pluggable solver)
├── linesearch.jl               # AbstractLineSearch + VectorWolfeLineSearch
├── morethuente.jl              # More-Thuente scalar line search (translated from Fortran)
├── scaling.jl                  # Scaling utilities
├── checkderivatives.jl         # Finite-difference gradient checker
├── problems.jl                 # Built-in test problems
├── solve.jl                    # Main solver loop
└── directions/
    ├── interface.jl            # AbstractDirectionMethod API
    ├── steepest.jl             # SteepestDescent
    ├── conjugate_gradient.jl   # CG + restart/SDC logic
    ├── beta_formulas.jl        # AbstractBetaFormula + all variants
    ├── bfgs.jl                 # BFGS
    ├── lbfgs.jl                # L-BFGS
    └── newton.jl               # Newton with safeguards
```

### Pluggable Components

1. **Direction method** — `AbstractDirectionMethod`: steepest descent, CG, BFGS, L-BFGS, Newton
2. **Beta formula** (CG family) — `AbstractBetaFormula`: FR, CD, DY, mDY, PRP+, HS+, HZ, HZ+, plus extended variants
3. **Line search** — `AbstractLineSearch`: Standard/Strong/Restricted Wolfe
4. **QP solver** — any JuMP-compatible optimizer (default: Clarabel.Optimizer)
5. **Stopping criterion** — theta-based, iteration limit, user callbacks

### Key Types (planned)

```julia
# User defines their problem
MOProblem(; n, m, f, g!, J!)  # f(x,i)→scalar, g!(G,x,i)→gradient, or J!(JF,x)→Jacobian

# User calls solver
result = solve(problem, ConjugateGradient(HagerZhang()); x0=..., optimizer=Clarabel.Optimizer)
result = solve(problem, BFGS(); x0=...)
result = solve(problem, SteepestDescent(); x0=...)
```

### Dependencies

- **JuMP.jl** — QP subproblem modeling
- **Clarabel.jl** — default QP solver (pure Julia, high precision)
- **LinearAlgebra** (stdlib)
- **Printf** (stdlib)

## Testing (TDD)

Tests are written first as specifications. Each source file has a corresponding test file.

```
test/
├── runtests.jl            # Runs all test files
├── test_types.jl          # Problem/state construction
├── test_subproblem.jl     # QP solver correctness
├── test_morethuente.jl    # Scalar line search
├── test_linesearch.jl     # Vector Wolfe conditions
├── test_beta.jl           # Beta formula values
├── test_directions.jl     # Direction computation
├── test_solver.jl         # Integration tests
└── test_problems.jl       # Test problem gradients
```

## Reference Implementation

The Fortran source in `../CGMOP/` serves as the reference for CG methods. Key default parameters for validation:
- `epsopt = 5 * sqrt(2^(-52))` ≈ 3.33e-8
- `ftol = 1e-4`, `gtol = 0.1`, `ctol = 0.4`
- `maxoutiter = 5000`
- `stpmin = 1e-15`, `stpmax = 1e10`

See `../CGMOP/CLAUDE.md` for detailed Fortran architecture.
