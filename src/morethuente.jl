#=
    More-Thuente line search algorithm.

    Translation of dcsrch/dcstep from MINPACK-2 (Fortran).

    J. J. More and D. J. Thuente, "Line Search Algorithms with Guaranteed
    Sufficient Decrease", ACM Trans. Math. Softw., 20 (1994), pp. 286-307.

    Note: The convergence test in this implementation uses the form from CGMOP:
      f(stp) <= f(0) + ftol*stp*f'(0)  AND  gtol*f'(0) <= f'(stp) <= tol
    Setting tol = -gtol*f'(0) recovers the standard strong Wolfe conditions.
=#

"""
Internal mutable state for the More-Thuente line search.
"""
mutable struct MTState
    brackt::Bool
    stage::Int
    finit::Float64
    ginit::Float64
    gtest::Float64
    width::Float64
    width1::Float64
    stx::Float64
    fx::Float64
    gx::Float64
    sty::Float64
    fy::Float64
    gy::Float64
    stmin::Float64
    stmax::Float64
end

"""
    _dcstep!(state, stp, fp, dp, stpmin, stpmax) → stp

Compute a safeguarded step for a line search and update the interval
[stx, sty] that contains a minimizer. Direct translation of MINPACK-2 dcstep.
"""
function _dcstep!(st::MTState, stp::Float64, fp::Float64, dp::Float64,
                   stpmin::Float64, stpmax::Float64)
    p66 = 0.66

    sgnd = dp * (st.gx / abs(st.gx))

    if fp > st.fx
        # Case 1: Higher function value. Minimum is bracketed.
        theta = 3.0 * (st.fx - fp) / (stp - st.stx) + st.gx + dp
        s = max(abs(theta), abs(st.gx), abs(dp))
        gamma = s * sqrt((theta / s)^2 - (st.gx / s) * (dp / s))
        if stp < st.stx
            gamma = -gamma
        end
        p = (gamma - st.gx) + theta
        q = ((gamma - st.gx) + gamma) + dp
        r = p / q
        stpc = st.stx + r * (stp - st.stx)
        stpq = st.stx + ((st.gx / ((st.fx - fp) / (stp - st.stx) + st.gx)) / 2.0) * (stp - st.stx)
        if abs(stpc - st.stx) < abs(stpq - st.stx)
            stpf = stpc
        else
            stpf = stpc + (stpq - stpc) / 2.0
        end
        st.brackt = true

    elseif sgnd < 0.0
        # Case 2: Lower function value, derivatives of opposite sign. Bracketed.
        theta = 3.0 * (st.fx - fp) / (stp - st.stx) + st.gx + dp
        s = max(abs(theta), abs(st.gx), abs(dp))
        gamma = s * sqrt((theta / s)^2 - (st.gx / s) * (dp / s))
        if stp > st.stx
            gamma = -gamma
        end
        p = (gamma - dp) + theta
        q = ((gamma - dp) + gamma) + st.gx
        r = p / q
        stpc = stp + r * (st.stx - stp)
        stpq = stp + (dp / (dp - st.gx)) * (st.stx - stp)
        if abs(stpc - stp) > abs(stpq - stp)
            stpf = stpc
        else
            stpf = stpq
        end
        st.brackt = true

    elseif abs(dp) < abs(st.gx)
        # Case 3: Lower function value, same sign derivatives, magnitude decreases.
        theta = 3.0 * (st.fx - fp) / (stp - st.stx) + st.gx + dp
        s = max(abs(theta), abs(st.gx), abs(dp))
        gamma = s * sqrt(max(0.0, (theta / s)^2 - (st.gx / s) * (dp / s)))
        if stp > st.stx
            gamma = -gamma
        end
        p = (gamma - dp) + theta
        q = (gamma + (st.gx - dp)) + gamma
        r = p / q
        if r < 0.0 && gamma != 0.0
            stpc = stp + r * (st.stx - stp)
        elseif stp > st.stx
            stpc = stpmax
        else
            stpc = stpmin
        end
        stpq = stp + (dp / (dp - st.gx)) * (st.stx - stp)

        if st.brackt
            if abs(stpc - stp) < abs(stpq - stp)
                stpf = stpc
            else
                stpf = stpq
            end
            if stp > st.stx
                stpf = min(stp + p66 * (st.sty - stp), stpf)
            else
                stpf = max(stp + p66 * (st.sty - stp), stpf)
            end
        else
            if abs(stpc - stp) > abs(stpq - stp)
                stpf = stpc
            else
                stpf = stpq
            end
            stpf = min(stpmax, stpf)
            stpf = max(stpmin, stpf)
        end

    else
        # Case 4: Lower function value, same sign derivatives, magnitude does not decrease.
        if st.brackt
            theta = 3.0 * (fp - st.fy) / (st.sty - stp) + st.gy + dp
            s = max(abs(theta), abs(st.gy), abs(dp))
            gamma = s * sqrt((theta / s)^2 - (st.gy / s) * (dp / s))
            if stp > st.sty
                gamma = -gamma
            end
            p = (gamma - dp) + theta
            q = ((gamma - dp) + gamma) + st.gy
            r = p / q
            stpc = stp + r * (st.sty - stp)
            stpf = stpc
        elseif stp > st.stx
            stpf = stpmax
        else
            stpf = stpmin
        end
    end

    # Update the interval
    if fp > st.fx
        st.sty = stp
        st.fy = fp
        st.gy = dp
    else
        if sgnd < 0.0
            st.sty = st.stx
            st.fy = st.fx
            st.gy = st.gx
        end
        st.stx = stp
        st.fx = fp
        st.gx = dp
    end

    return stpf
end

"""
    mt_linesearch(phi, dphi, stp0; ftol, gtol, stpmin, stpmax, maxiter=20) → (stp, info)

More-Thuente line search for a scalar function.

Finds a step `stp` satisfying the strong Wolfe conditions:
- Sufficient decrease: `phi(stp) ≤ phi(0) + ftol·stp·dphi(0)`
- Curvature condition: `|dphi(stp)| ≤ gtol·|dphi(0)|`

# Arguments
- `phi`: function `phi(α) → f(α)` (scalar line function)
- `dphi`: derivative `dphi(α) → f'(α)`
- `stp0`: initial step size (must be positive)
- `ftol`: sufficient decrease parameter (default: 1e-4)
- `gtol`: curvature condition parameter (default: 0.9)
- `stpmin`: minimum step (default: 1e-15)
- `stpmax`: maximum step (default: 1e10)
- `maxiter`: maximum iterations (default: 20)

# Returns
- `stp::Float64`: the computed step
- `info::Int`: 0 = converged, 1 = stp=stpmin, 2 = stp=stpmax,
               3 = rounding errors, 4 = xtol test, 5 = max iterations
"""
function mt_linesearch(phi, dphi, stp0::Float64;
                        ftol::Float64=1e-4, gtol::Float64=0.9,
                        stpmin::Float64=1e-15, stpmax::Float64=1e10,
                        maxiter::Int=20)
    xtrapl = 1.1
    xtrapu = 4.0
    xtol = 1e-20

    stp = stp0
    f0 = phi(0.0)
    g0 = dphi(0.0)

    # Use tol = -gtol*g0 = gtol*|g0| for strong Wolfe
    tol = -gtol * g0

    st = MTState(
        false,              # brackt
        1,                  # stage
        f0,                 # finit
        g0,                 # ginit
        ftol * g0,          # gtest
        stpmax - stpmin,    # width
        2.0 * (stpmax - stpmin),  # width1
        0.0, f0, g0,       # stx, fx, gx
        0.0, f0, g0,       # sty, fy, gy
        0.0,                # stmin
        stp + xtrapu * stp, # stmax
    )

    for iter in 1:maxiter
        f = phi(stp)
        g = dphi(stp)

        ftest = st.finit + stp * st.gtest

        # Stage transition
        if st.stage == 1 && f <= ftest && g >= 0.0
            st.stage = 2
        end

        # Test for warnings
        if st.brackt && (stp <= st.stmin || stp >= st.stmax)
            return stp, 3  # rounding errors
        end
        if st.brackt && st.stmax - st.stmin <= xtol * st.stmax
            return stp, 4  # xtol test
        end
        if stp == stpmax && f <= ftest && g <= st.gtest
            return stp, 2  # stp = stpmax
        end
        if stp == stpmin && (f > ftest || g >= st.gtest)
            return stp, 1  # stp = stpmin
        end

        # Test for convergence (restricted Wolfe with tol)
        if f <= ftest && g >= gtol * st.ginit && g <= tol
            return stp, 0  # converged
        end

        # Compute new step using modified function in stage 1
        if st.stage == 1 && f <= st.fx && f > ftest
            fm = f - stp * st.gtest
            fxm = st.fx - st.stx * st.gtest
            fym = st.fy - st.sty * st.gtest
            gm = g - st.gtest
            gxm = st.gx - st.gtest
            gym = st.gy - st.gtest

            # Save current values, swap in modified
            fx_save, gx_save = st.fx, st.gx
            fy_save, gy_save = st.fy, st.gy
            st.fx, st.gx = fxm, gxm
            st.fy, st.gy = fym, gym

            stp = _dcstep!(st, stp, fm, gm, st.stmin, st.stmax)

            # Restore from modified
            st.fx = st.fx + st.stx * st.gtest
            st.fy = st.fy + st.sty * st.gtest
            st.gx = st.gx + st.gtest
            st.gy = st.gy + st.gtest
        else
            stp = _dcstep!(st, stp, f, g, st.stmin, st.stmax)
        end

        # Bisection step if needed
        if st.brackt
            if abs(st.sty - st.stx) >= 0.66 * st.width1
                stp = st.stx + 0.5 * (st.sty - st.stx)
            end
            st.width1 = st.width
            st.width = abs(st.sty - st.stx)
        end

        # Set step bounds
        if st.brackt
            st.stmin = min(st.stx, st.sty)
            st.stmax = max(st.stx, st.sty)
        else
            st.stmin = stp + xtrapl * (stp - st.stx)
            st.stmax = stp + xtrapu * (stp - st.stx)
        end

        # Force step within global bounds
        stp = max(stp, stpmin)
        stp = min(stp, stpmax)

        # If no progress possible, use best point
        if (st.brackt && (stp <= st.stmin || stp >= st.stmax)) ||
           (st.brackt && st.stmax - st.stmin <= xtol * st.stmax)
            stp = st.stx
        end
    end

    return stp, 5  # max iterations
end
