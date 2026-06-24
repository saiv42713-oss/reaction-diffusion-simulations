import math
import numpy as np

# ---------- Hill with gradients (scalar-fast) ----------
def hill_with_grads(a, i, ka, ki, n, m, basal=0.0, activator_type = "juxtacrine"):
    # Guard against zero/negatives
    ka = float(ka); ki = float(ki)
    a = float(max(a, 0.0)); i = float(max(i, 0.0))


    # aa=(a/ka)^n, ii=(i/ki)^m
    if ka <= 0 or ki <= 0:
        raise ValueError("ka and ki must be > 0")
    aa = (a/ka)**n if a > 0 else 0.0
    ii = (i/ki)**m if i > 0 else 0.0

    denom = basal + 1.0 + aa + ii
    H = basal + aa/denom

    # d(aa)/da = n*aa/a (if a>0), d(ii)/di = m*ii/i (if i>0)
    daa_da = (n * aa / a) if a > 0 else 0.0
    dii_di = (m * ii / i) if i > 0 else 0.0

    # H = basal + aa/denom  => dH/d(aa) = (denom - aa)/denom^2 ; dH/d(ii) = -aa/denom^2
    inv_denom2 = 1.0 / (denom * denom)
    dH_daa = (denom - aa) * inv_denom2
    dH_dii = (-aa) * inv_denom2

    dH_da = dH_daa * daa_da
    dH_di = dH_dii * dii_di
    return H, dH_da, dH_di

# ---------- Fast non-null, stable steady-state finder ----------
def fast_stable_steady_state(p, activator_type = "juxtacrine", tol=5e-4, max_newton=12):
    """
    Returns (a*, i*, H*) for the non-null reaction-stable fixed point.
    Target precision ~1e-3 on a*, i* (2-3 decimals).
    """
    # Ratios A, I
    A = float(p["act_prod_rate"]) / float(p["act_decay_rate"])
    I = float(p["inh_prod_rate"]) / float(p["inh_decay_rate"])

    # Params for Hill
    ka = p["act_half_sat"]; ki = p["inh_half_sat"]
    n  = p["act_hill_coeff"]; m  = p["inh_hill_coeff"]
    basal = float(p.get("basal_prod", 0.0))

    # Bracket for H (typical Hill in [basal, basal+1]); keep away from 0 if basal=0
    H_lo = basal + (1e-9 if basal == 0.0 else 0.0)
    H_hi = basal + 1.0 - 1e-9

    def F(H):
        Hval, _, _ = hill_with_grads(A*H, I*H, ka, ki, n, m, basal, activator_type)
        return Hval

    def g(H):
        return F(H) - H

    def gprime(H):
        # g'(H) = A*dH/da + I*dH/di - 1 evaluated at (a=AH, i=IH)
        Hval, dH_da, dH_di = hill_with_grads(A*H, I*H, ka, ki, n, m, basal)
        return A*dH_da + I*dH_di - 1.0

    # Try to land on the upper non-null stable branch first (often the one you want)
    # Start near the top of the feasible H range
    for H0 in (0.9*H_hi + 0.1*H_lo, 0.2*H_hi + 0.8*H_lo):
        H = H0
        # Safeguarded Newton iterations
        lo, hi = H_lo, H_hi
        for _ in range(max_newton):
            f = g(H)
            if abs(f) < tol:
                break
            gp = gprime(H)
            # If derivative is tiny or NaN, take a bisection-ish step
            if not np.isfinite(gp) or abs(gp) < 1e-8:
                H = 0.5*(lo + hi)
            else:
                Hn = H - f/gp  # Newton step
                # keep it inside [lo, hi]; if not, take a secant/bisect blend
                if Hn <= lo or Hn >= hi:
                    Hn = 0.5*(H + (lo if f > 0 else hi))
                H = Hn
            # maintain a simple bracket using sign of g
            try:
                if g(lo) * g(H) <= 0:
                    hi = H
                else:
                    lo = H
            except Exception:
                pass

    # after Newton loop:
    converged = abs(g(H)) < tol

    # If Newton didn't converge, fall back to a tiny bracket refine
    if not converged:
        # We already kept a running [lo,hi] bracket; if it's valid use it,
        # otherwise build a quick bracket around H
        glo, ghi = g(lo), g(hi)
        if not (np.isfinite(glo) and np.isfinite(ghi) and glo*ghi <= 0):
            # build a small local bracket around H
            span = max(1e-3, 0.05*(H_hi - H_lo))
            lo, hi = max(H_lo, H - span), min(H_hi, H + span)
        H = _mini_brent(g, lo, hi, tol)

    # Now enforce that we actually have a root
    if abs(g(H)) >= 5*tol:
        # as a last resort, coarse scan for a sign-change and refine the *largest* positive root
        Hs = np.linspace(H_lo, H_hi, 64)
        gs = np.array([g(h) for h in Hs])
        cand = []
        for k in range(len(Hs)-1):
            if np.isfinite(gs[k]) and np.isfinite(gs[k+1]) and gs[k]*gs[k+1] <= 0:
                Hr = _mini_brent(g, Hs[k], Hs[k+1], tol)
                cand.append(Hr)
        # prefer nonzero roots; take the largest positive one
        cand = [h for h in cand if h > (basal + 10*tol)]
        if cand:
            H = max(cand)

    a = max(A*H, 0.0)
    i = max(I*H, 0.0)
    if _is_reaction_stable(a, i, p) and (a > 0 or i > 0) and abs(g(H)) < 5*tol:
        return _round_if_needed(a, tol), _round_if_needed(i, tol), H


    # Fallback: tiny bracketed search to nearest non-null root, then pick stable
    # Sample a coarse grid and refine the best sign-change interval
    Nscan = 64
    Hs = np.linspace(H_lo, H_hi, Nscan)
    gs = np.array([g(h) for h in Hs])
    intervals = []
    for k in range(Nscan-1):
        if not np.isfinite(gs[k]) or not np.isfinite(gs[k+1]):
            continue
        if gs[k] == 0.0:
            intervals.append((max(H_lo, Hs[k]-1e-6), min(H_hi, Hs[k]+1e-6)))
        elif gs[k]*gs[k+1] < 0:
            intervals.append((Hs[k], Hs[k+1]))

    for (lo, hi) in intervals:
        H = _mini_brent(g, lo, hi, tol)
        a = max(A*H, 0.0); i = max(I*H, 0.0)
        if _is_reaction_stable(a, i, p) and (a > 0 or i > 0):
            return _round_if_needed(a, tol), _round_if_needed(i, tol), H

    # If nothing stable/non-null found, return zeros (caller can handle)
    return 0.0, 0.0, 0.0

# --- helpers ---
def _mini_brent(g, a, b, tol):
    # Very small Brent-ish solver: enough for 3 decimals, ~20 evals worst-case
    fa, fb = g(a), g(b)
    if fa == 0.0: return a
    if fb == 0.0: return b
    if fa*fb > 0:
        return 0.5*(a+b)
    c, fc = a, fa
    d = e = b - a
    for _ in range(50):
        if fb == 0.0: return b
        if fa*fb > 0:
            a, fa = c, fc
            d = e = b - a
        if abs(fa) < abs(fb):
            c, fc = b, fb
            b, fb = a, fa
            a, fa = c, fc
        # Convergence?
        m = 0.5*(a - b)
        if abs(m) < tol:  # enough for 2–3 decimals in H
            return b
        # Bisection step by default
        s = b - fb*(b - a)/(fb - fa) if (fb - fa) != 0 else b + m
        # Safeguards
        if (s < (3*a + b)/4 and s > b) or (s > (3*a + b)/4 and s < b):
            s = b + m
        fs = g(s)
        c, fc = a, fa
        a, fa = b, fb
        b, fb = s, fs
    return b

def _is_reaction_stable(a, i, p):
    # Jacobian eigenvalues for reaction-only system
    H, dH_da, dH_di = hill_with_grads(a, i,
                                      p["act_half_sat"], p["inh_half_sat"],
                                      p["act_hill_coeff"], p["inh_hill_coeff"],
                                      p.get("basal_prod", 0.0))
    ba, bi = p["act_prod_rate"], p["inh_prod_rate"]
    la, li = p["act_decay_rate"], p["inh_decay_rate"]
    J = np.array([[ba*dH_da - la,  ba*dH_di],
                  [bi*dH_da,       bi*dH_di - li]], dtype=float)
    ev = np.linalg.eigvals(J)
    return np.all(np.real(ev) < 0)

def _round_if_needed(x, tol):
    # If you target ~1e-3 in H, a and i inherit that scale; round to 1e-3
    dec = 3 if tol <= 1e-3 else 2
    return float(np.round(x, dec))
