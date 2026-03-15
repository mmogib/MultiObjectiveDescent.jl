#=
    Vector Wolfe line search for multiobjective optimization.

    Translation of lsvecopt.f90 from CGMOP.

    Reference:
    L. R. Lucambio Pérez and L. F. Prudente, "A Wolfe line search algorithm
    for vector optimization", ACM Trans. Math. Softw., 2019.

    Finds a step α > 0 satisfying:
      Sufficient decrease: f_i(α) ≤ f_i(0) + ftol·α·maxg0  for all i
      Curvature condition (one of):
        - Standard Wolfe:    max_i{f'_i(α)} ≥ gtol·maxg0
        - Strong Wolfe:      |max_i{f'_i(α)}| ≤ -gtol·maxg0
        - Restricted Wolfe:  gtol·maxg0 ≤ max_i{f'_i(α)} ≤ tol
    where maxg0 = max_i{f'_i(0)} < 0 (descent condition).
=#

"""
    vector_wolfe_linesearch(evalf, evalgphi, stp0, phi0, gphi0; kwargs...) → (stp, nfev, ngev, info)

Wolfe line search for vector optimization.

Finds a step satisfying the vector Wolfe conditions across all `m` objectives.
Uses the More-Thuente algorithm as the inner scalar line search engine.

# Arguments
- `evalf(stp, i)`: returns the value of objective `i` at step `stp`
- `evalgphi(stp, i)`: returns the directional derivative of objective `i` at step `stp`
- `stp0`: initial trial step (positive)
- `phi0::Vector{Float64}`: objective values at step 0 (`[f_1(0), ..., f_m(0)]`)
- `gphi0::Vector{Float64}`: directional derivatives at step 0 (`[f'_1(0), ..., f'_m(0)]`)

# Keyword Arguments
- `m::Int`: number of objectives
- `ftol::Float64`: sufficient decrease parameter (default: 1e-4)
- `gtol::Float64`: curvature condition parameter (default: 0.1)
- `stpmin::Float64`: minimum step (default: 1e-15)
- `stpmax::Float64`: maximum step (default: 1e10)
- `lstype::Symbol`: `:standard`, `:strong`, or `:restricted` (default: `:standard`)
- `tol::Float64`: upper tolerance for restricted Wolfe (required when `lstype=:restricted`)
- `maxoutiter::Int`: maximum outer iterations (default: 100)

# Returns
- `stp::Float64`: the computed step
- `nfev::Int`: number of function evaluations
- `ngev::Int`: number of gradient evaluations
- `info::Int`: 0 = success, 1 = stp=stpmin, 2 = stp=stpmax,
               3 = rounding errors, 4 = interval too small, 5 = max iterations, -1 = error
"""
function vector_wolfe_linesearch(evalf, evalgphi, stp0::Float64,
                                  phi0::Vector{Float64}, gphi0::Vector{Float64};
                                  m::Int,
                                  ftol::Float64=1e-4, gtol::Float64=0.1,
                                  stpmin::Float64=1e-15, stpmax::Float64=1e10,
                                  lstype::Symbol=:standard,
                                  tol::Float64=Inf,
                                  maxoutiter::Int=100)
    xtol = 1e-20
    smallnum = -1e20
    xtrapl = 1.1
    xtrapu = 4.0

    # Validate inputs
    if ftol >= gtol
        return stp0, 0, 0, -1
    end
    if stp0 < stpmin || stp0 > stpmax
        return stp0, 0, 0, -1
    end
    if lstype == :restricted && tol < 0.0
        return stp0, 0, 0, -1
    end

    # Counters
    nfev = 0
    ngev = 0

    # Compute maxg0 and validate descent
    maxg0 = maximum(gphi0)
    if maxg0 >= 0.0
        return stp0, 0, 0, -1  # not a descent direction
    end

    ftest = ftol * maxg0
    gtest = -gtol * maxg0  # positive

    # Determine tolerance for the inner More-Thuente line search
    tolLS = if lstype == :standard
        Inf
    elseif lstype == :strong
        gtest
    elseif lstype == :restricted
        tol
    else
        return stp0, 0, 0, -1
    end

    # Inner ftol/gtol for More-Thuente (slightly relaxed to avoid cycling)
    ftolinner = min(1.1 * ftol, 0.75 * ftol + 0.25 * gtol)
    gtolinner = max(0.9 * gtol, 0.25 * ftol + 0.75 * gtol)

    stp = stp0
    brackt = false

    # All objectives are non-quadratic in this implementation
    # (quadratic exploitation can be added later)
    obj_indices = collect(1:m)

    # Select initial working index: objective with most negative derivative
    ind = argmin(gphi0)

    # ----- Initial check of SDC and CC at stp -----

    # Sufficient decrease condition (SDC)
    sdc = true
    for i in obj_indices
        f = evalf(stp, i)
        nfev += 1
        if f > phi0[i] + ftest * stp
            sdc = false
            brackt = true
            ind = i
            stpmax = stp
            break
        end
    end

    # Curvature condition (CC)
    cc = false
    if sdc
        maxg = smallnum
        for i in obj_indices
            g = evalgphi(stp, i)
            ngev += 1
            maxg = max(maxg, g)
            if g > tolLS
                brackt = true
                ind = i
                stpmax = stp
                break
            end
        end
        if maxg >= -gtest && maxg <= tolLS
            cc = true
        end
    end

    # ----- Main loop -----
    outiter = 0
    MTinfo = -1

    stpmin_local = stpmin
    stpmax_local = stpmax

    while true
        # Test convergence
        if sdc && cc
            return stp, nfev, ngev, 0
        end

        # Test for stopping at stpmin
        if outiter > 0 && MTinfo == 1
            return stp, nfev, ngev, 1
        end

        # Test stpmax (not bracketed)
        if !brackt && stp == stpmax
            return stp, nfev, ngev, 2
        end

        # Rounding errors
        if outiter > 0 && MTinfo == 3
            return stp, nfev, ngev, 3
        end

        # Interval too small
        if outiter > 0 && MTinfo == 4
            return stp, nfev, ngev, 4
        end

        # Max iterations
        if outiter >= maxoutiter
            return stp, nfev, ngev, 5
        end

        outiter += 1

        # ----- Inner More-Thuente on objective `ind` -----
        # Set up scalar line search for objective ind
        f_mt = phi0[ind]
        g_mt = maxg0

        st = MTState(
            false, 1,
            f_mt, g_mt, ftolinner * g_mt,
            stpmax_local - stpmin_local,
            2.0 * (stpmax_local - stpmin_local),
            0.0, f_mt, g_mt,
            0.0, f_mt, g_mt,
            0.0,
            stp + xtrapu * stp,
        )

        initer = 0
        inner_maxiter = 100

        for _ in 1:inner_maxiter
            f_val = evalf(stp, ind)
            g_val = evalgphi(stp, ind)
            nfev += 1
            ngev += 1

            ftest_inner = st.finit + stp * st.gtest

            # Stage transition
            if st.stage == 1 && f_val <= ftest_inner && g_val >= 0.0
                st.stage = 2
            end

            # Check convergence/warnings
            local converged = false
            if st.brackt && (stp <= st.stmin || stp >= st.stmax)
                MTinfo = 3; break
            end
            if st.brackt && st.stmax - st.stmin <= xtol * st.stmax
                MTinfo = 4; break
            end
            if stp == stpmax_local && f_val <= ftest_inner && g_val <= st.gtest
                MTinfo = 2; break
            end
            if stp == stpmin_local && (f_val > ftest_inner || g_val >= st.gtest)
                MTinfo = 1; break
            end

            # Convergence of inner: sufficient decrease + curvature for this objective
            if f_val <= ftest_inner && g_val >= gtolinner * st.ginit && g_val <= tolLS
                MTinfo = 0; break
            end

            # Compute new step
            if st.stage == 1 && f_val <= st.fx && f_val > ftest_inner
                fm = f_val - stp * st.gtest
                st.fx -= st.stx * st.gtest
                st.fy -= st.sty * st.gtest
                gm = g_val - st.gtest
                st.gx -= st.gtest
                st.gy -= st.gtest

                stp = _dcstep!(st, stp, fm, gm, st.stmin, st.stmax)

                st.fx += st.stx * st.gtest
                st.fy += st.sty * st.gtest
                st.gx += st.gtest
                st.gy += st.gtest
            else
                stp = _dcstep!(st, stp, f_val, g_val, st.stmin, st.stmax)
            end

            # Bisection
            if st.brackt
                if abs(st.sty - st.stx) >= 0.66 * st.width1
                    stp = st.stx + 0.5 * (st.sty - st.stx)
                end
                st.width1 = st.width
                st.width = abs(st.sty - st.stx)
            end

            # Update bounds
            if st.brackt
                st.stmin = min(st.stx, st.sty)
                st.stmax = max(st.stx, st.sty)
            else
                st.stmin = stp + xtrapl * (stp - st.stx)
                st.stmax = stp + xtrapu * (stp - st.stx)
            end

            stp = max(stp, stpmin_local)
            stp = min(stp, stpmax_local)

            if (st.brackt && (stp <= st.stmin || stp >= st.stmax)) ||
               (st.brackt && st.stmax - st.stmin <= xtol * st.stmax)
                stp = st.stx
            end

            if st.brackt
                brackt = true
            end

            initer += 1
            # If not bracketed and initer > 0, exit inner loop
            if initer > 0 && !st.brackt
                break
            end
        end

        # ----- Prepare next outer iteration -----
        if MTinfo == 1 || MTinfo == 3 || MTinfo == 4
            continue
        end

        # Re-check SDC and CC across all objectives
        sdc = true
        for i in obj_indices
            if MTinfo == 0 && i == ind
                continue  # already checked in inner loop
            end
            f = evalf(stp, i)
            nfev += 1
            if f > phi0[i] + ftest * stp
                sdc = false
                brackt = true
                ind = i
                stpmax_local = stp
                break
            end
        end

        cc = false
        if sdc
            if MTinfo == 0
                maxg = evalgphi(stp, ind)
                ngev += 1
            else
                maxg = smallnum
            end

            for i in obj_indices
                if MTinfo == 0 && i == ind
                    continue
                end
                g = evalgphi(stp, i)
                ngev += 1
                maxg = max(maxg, g)
                if g > tolLS
                    brackt = true
                    ind = i
                    stpmax_local = stp
                    break
                end
            end

            if maxg >= -gtest && maxg <= tolLS
                cc = true
            end
        end
    end
end
