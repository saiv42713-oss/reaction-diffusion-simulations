from __future__ import annotations
from typing import Mapping, Tuple
import numpy as np


def hill_with_grads(
    a: float,
    i: float,
    ka: float,
    ki: float,
    n: float,
    m: float,
    basal: float = 0.0,
    activator_type: str = "juxtacrine",
) -> Tuple[float, float, float]:
    """
    Hill response with one activator and one inhibitor, plus analytic gradients.

    Parameters
    ----------
    a, i
        Activator and inhibitor inputs.
    ka, ki
        Half-saturation constants. Must be strictly positive.
    n, m
        Hill coefficients for activator and inhibitor.
    basal
        Basal production offset.
    activator_type
        Reserved for API compatibility. Currently not used in the computation.

    Returns
    -------
    H
        Hill output.
    dH_da
        Partial derivative of H with respect to a.
    dH_di
        Partial derivative of H with respect to i.
    """
    # Guard against negative inputs and ensure numeric type.
    ka = float(ka)
    ki = float(ki)
    a = float(max(a, 0.0))
    i = float(max(i, 0.0))

    if ka <= 0 or ki <= 0:
        raise ValueError("ka and ki must be > 0")

    # Nonlinear Hill terms:
    # aa = (a / ka)^n, ii = (i / ki)^m
    aa = (a / ka) ** n if a > 0 else 0.0
    ii = (i / ki) ** m if i > 0 else 0.0

    # Output is normalized so that it stays in a bounded range.
    denom = basal + 1.0 + aa + ii
    H = basal + aa / denom

    # Analytic derivatives of the Hill terms.
    daa_da = (n * aa / a) if a > 0 else 0.0
    dii_di = (m * ii / i) if i > 0 else 0.0

    # H = basal + aa / denom
    # dH/d(aa) = (denom - aa) / denom^2
    # dH/d(ii) = -aa / denom^2
    inv_denom2 = 1.0 / (denom * denom)
    dH_daa = (denom - aa) * inv_denom2
    dH_dii = (-aa) * inv_denom2

    dH_da = dH_daa * daa_da
    dH_di = dH_dii * dii_di

    return H, dH_da, dH_di


def fast_stable_steady_state(
    p: Mapping[str, float],
    activator_type: str = "juxtacrine",
    tol: float = 5e-4,
    max_newton: int = 12,
) -> Tuple[float, float, float]:
    """
    Find the non-null, reaction-stable steady state.

    The fixed point is found in terms of the scalar Hill output H, then mapped back to
    activator/inhibitor concentrations:
        a* = A * H*
        i* = I * H*

    Parameters
    ----------
    p
        Parameter dictionary with keys:
        - act_prod_rate
        - act_decay_rate
        - inh_prod_rate
        - inh_decay_rate
        - act_half_sat
        - inh_half_sat
        - act_hill_coeff
        - inh_hill_coeff
        - basal_prod (optional)
    activator_type
        Reserved for API compatibility with hill_with_grads.
    tol
        Root-finding tolerance.
    max_newton
        Maximum safeguarded Newton iterations per initial guess.

    Returns
    -------
    a_star, i_star, H_star
        Rounded steady-state values. Returns (0.0, 0.0, 0.0) if no stable
        non-null steady state is found.
    """
    # Convert production / decay ratios into linear scaling factors.
    A = float(p["act_prod_rate"]) / float(p["act_decay_rate"])
    I = float(p["inh_prod_rate"]) / float(p["inh_decay_rate"])

    # Hill parameters.
    ka = p["act_half_sat"]
    ki = p["inh_half_sat"]
    n = p["act_hill_coeff"]
    m = p["inh_hill_coeff"]
    basal = float(p.get("basal_prod", 0.0))

    # In this model, H is typically confined to [basal, basal + 1].
    # Keep the lower bound slightly above zero when basal is zero.
    H_lo = basal + (1e-9 if basal == 0.0 else 0.0)
    H_hi = basal + 1.0 - 1e-9

    def F(H: float) -> float:
        Hval, _, _ = hill_with_grads(
            A * H, I * H, ka, ki, n, m, basal, activator_type
        )
        return Hval

    def g(H: float) -> float:
        # Solve H = F(H), i.e. g(H) = F(H) - H = 0
        return F(H) - H

    def gprime(H: float) -> float:
        # Chain rule:
        # g'(H) = A * dH/da + I * dH/di - 1
        _, dH_da, dH_di = hill_with_grads(
            A * H, I * H, ka, ki, n, m, basal, activator_type
        )
        return A * dH_da + I * dH_di - 1.0

    # Try to converge toward the upper non-null branch first.
    for H0 in (0.9 * H_hi + 0.1 * H_lo, 0.2 * H_hi + 0.8 * H_lo):
        H = H0
        lo, hi = H_lo, H_hi

        for _ in range(max_newton):
            f = g(H)
            if abs(f) < tol:
                break

            gp = gprime(H)

            # If the derivative is unusable, fall back to bisection.
            if not np.isfinite(gp) or abs(gp) < 1e-8:
                H = 0.5 * (lo + hi)
            else:
                Hn = H - f / gp  # Newton step

                # Keep the iterate inside the bracket.
                if Hn <= lo or Hn >= hi:
                    Hn = 0.5 * (H + (lo if f > 0 else hi))
                H = Hn

            # Maintain a simple sign bracket.
            try:
                if g(lo) * g(H) <= 0:
                    hi = H
                else:
                    lo = H
            except Exception:
                pass

    converged = abs(g(H)) < tol

    # If Newton did not converge, refine with a small bracketed solver.
    if not converged:
        glo, ghi = g(lo), g(hi)
        if not (np.isfinite(glo) and np.isfinite(ghi) and glo * ghi <= 0):
            span = max(1e-3, 0.05 * (H_hi - H_lo))
            lo, hi = max(H_lo, H - span), min(H_hi, H + span)

        H = _mini_brent(g, lo, hi, tol)

    # Final safety check: if this is still weak, scan for sign changes.
    if abs(g(H)) >= 5 * tol:
        Hs = np.linspace(H_lo, H_hi, 64)
        gs = np.array([g(h) for h in Hs])

        candidates = []
        for k in range(len(Hs) - 1):
            if np.isfinite(gs[k]) and np.isfinite(gs[k + 1]) and gs[k] * gs[k + 1] <= 0:
                Hr = _mini_brent(g, Hs[k], Hs[k + 1], tol)
                candidates.append(Hr)

        # Prefer the largest positive non-trivial root.
        candidates = [h for h in candidates if h > (basal + 10 * tol)]
        if candidates:
            H = max(candidates)

    a = max(A * H, 0.0)
    i = max(I * H, 0.0)

    if _is_reaction_stable(a, i, p) and (a > 0 or i > 0) and abs(g(H)) < 5 * tol:
        return _round_if_needed(a, tol), _round_if_needed(i, tol), H

    # Fallback: scan the interval for sign-change brackets and test each root.
    Nscan = 64
    Hs = np.linspace(H_lo, H_hi, Nscan)
    gs = np.array([g(h) for h in Hs])

    intervals = []
    for k in range(Nscan - 1):
        if not np.isfinite(gs[k]) or not np.isfinite(gs[k + 1]):
            continue
        if gs[k] == 0.0:
            intervals.append((max(H_lo, Hs[k] - 1e-6), min(H_hi, Hs[k] + 1e-6)))
        elif gs[k] * gs[k + 1] < 0:
            intervals.append((Hs[k], Hs[k + 1]))

    for lo, hi in intervals:
        H = _mini_brent(g, lo, hi, tol)
        a = max(A * H, 0.0)
        i = max(I * H, 0.0)
        if _is_reaction_stable(a, i, p) and (a > 0 or i > 0):
            return _round_if_needed(a, tol), _round_if_needed(i, tol), H

    # No stable non-null root found.
    return 0.0, 0.0, 0.0


def _mini_brent(g, a: float, b: float, tol: float) -> float:
    """
    Lightweight bracketed root finder.

    This is intentionally small and dependency-free, and is sufficient for a few
    decimals of accuracy on well-behaved functions.
    """
    fa, fb = g(a), g(b)

    if fa == 0.0:
        return a
    if fb == 0.0:
        return b

    # If no sign change exists, return the midpoint as a safe fallback.
    if fa * fb > 0:
        return 0.5 * (a + b)

    c, fc = a, fa
    d = e = b - a

    for _ in range(50):
        if fb == 0.0:
            return b

        # Re-bracket if the sign is wrong.
        if fa * fb > 0:
            a, fa = c, fc
            d = e = b - a

        if abs(fa) < abs(fb):
            c, fc = b, fb
            b, fb = a, fa
            a, fa = c, fc

        # Current bracket midpoint.
        m = 0.5 * (a - b)
        if abs(m) < tol:
            return b

        # Secant step when possible, otherwise fallback to bisection.
        if (fb - fa) != 0:
            s = b - fb * (b - a) / (fb - fa)
        else:
            s = b + m

        # Keep the candidate inside the bracket; otherwise bisect.
        if not (min(a, b) < s < max(a, b)):
            s = b + m

        fs = g(s)

        c, fc = a, fa
        a, fa = b, fb
        b, fb = s, fs

    return b


def _is_reaction_stable(a: float, i: float, p: Mapping[str, float]) -> bool:
    """
    Check reaction-only local stability from the Jacobian eigenvalues.

    A fixed point is considered stable if all eigenvalues have negative real part.
    """
    _, dH_da, dH_di = hill_with_grads(
        a,
        i,
        p["act_half_sat"],
        p["inh_half_sat"],
        p["act_hill_coeff"],
        p["inh_hill_coeff"],
        p.get("basal_prod", 0.0),
    )

    ba, bi = p["act_prod_rate"], p["inh_prod_rate"]
    la, li = p["act_decay_rate"], p["inh_decay_rate"]

    J = np.array(
        [
            [ba * dH_da - la, ba * dH_di],
            [bi * dH_da, bi * dH_di - li],
        ],
        dtype=float,
    )

    ev = np.linalg.eigvals(J)
    return np.all(np.real(ev) < 0)


def _round_if_needed(x: float, tol: float) -> float:
    """
    Round output to match the requested tolerance.
    """
    dec = 3 if tol <= 1e-3 else 2
    return float(np.round(x, dec))
