"""'Intuition': pre-analyse the data to guess basis functions, coordinates, method and settings
BEFORE any model fitting, the way an experienced modeller eyeballs data. Public data only, no LLM.

    intuit(meta, data) -> {"facts": {...}, "hypotheses": [ {hypothesis, evidence, confidence, try: {tool, args}} ],
                           "recommended_config": {...}}

ODE: noise & sampling, positivity/boundedness/decades, oscillation spectra, fixed points + local linearisation
     (eigenvalues -> spiral/node/saddle), ADDITIVE partial-dependence shapes of every dx_i/dt on every x_j
     (linear, quadratic, cubic, sin, saturating x/(K+x), exp, ...), amplitude-dependent frequency
     (anharmonicity), and a quick sum-conservation check.
PDE: noise & resolved bandwidth, measured DISPERSION RELATION / growth rates of Fourier modes
     (lambda(k) fitted by sum_n a_n (ik)^n identifies the linear derivative terms and their coefficients),
     travelling-wave speed (cross-correlation) and its dependence on amplitude (nonlinear advection),
     conservation of the mean (flux form), and u -> -u parity (odd vs even nonlinearity).
"""
import itertools

import numpy as np
from scipy.signal import savgol_filter

from . import toolbox as tb


def _r(x, s=3):
    return float(f"{float(x):.{s}g}") if np.isfinite(x) else None


# ============================================================================ ODE
SHAPES = {
    "linear": lambda x, K: x,
    "quadratic": lambda x, K: x ** 2,
    "cubic": lambda x, K: x ** 3,
    "sin": lambda x, K: np.sin(x),
    "cos": lambda x, K: np.cos(x),
    "saturating x/(K+x)": lambda x, K: x / (K + x),
    "exp": lambda x, K: np.exp(np.clip(x / K, -30, 30)),
    "tanh": lambda x, K: np.tanh(x / K),
}


def _shape_fit(x, g):
    """Best 1-D template for g(x) (with constant + linear backbone), chosen by how much it shrinks the residual
    relative to the linear fit: log(res_shape / res_linear) + 0.15 per extra nonlinear constant. A shape is only
    accepted if it removes >= 40% of the linear residual. Returns (name, K, log-ratio score, R2)."""
    def res(cols):
        A = np.column_stack([np.ones_like(x)] + cols)
        c, *_ = np.linalg.lstsq(A, g, rcond=None)
        return float(np.var(g - A @ c)) + 1e-30
    r_lin = res([x])
    span = np.ptp(x) + 1e-12
    best = ("linear", None, np.log(0.6), 1 - r_lin / (np.var(g) + 1e-30))   # acceptance bar
    for name, f in SHAPES.items():
        if name == "linear":
            continue
        k_par = 1 if ("K" in name or name in ("exp", "tanh")) else 0
        if name in ("sin", "cos") and span < np.pi:
            continue                     # less than half a period observed: periodicity is not identifiable
        Ks = list(np.geomspace(0.05, 5, 12) * span) if k_par else [None]
        for K in Ks:
            if name.startswith("saturating") and (x.min() + K) * (x.max() + K) <= 0:
                continue
            with np.errstate(all="ignore"):
                col = f(x, K)
            if not np.all(np.isfinite(col)):
                continue
            r = res([x, col])
            score = np.log(r / r_lin) + 0.3 * k_par + (0.1 if name in ('sin', 'cos') else 0.0)
            if score < best[2]:
                best = (name, K, score, 1 - r / (np.var(g) + 1e-30))
    return best


def _additive_components(X, y, nbins=24, iters=3):
    """Backfitting additive model y ~ sum_j g_j(x_j) with binned means; returns list of (centres, g_j values)."""
    n, q = X.shape
    edges = [np.quantile(X[:, j], np.linspace(0, 1, nbins + 1)) for j in range(q)]
    idx = [np.clip(np.searchsorted(edges[j][1:-1], X[:, j]), 0, nbins - 1) for j in range(q)]
    G = np.zeros((n, q))
    mu = y.mean()
    for _ in range(iters):
        for j in range(q):
            partial = y - mu - G.sum(1) + G[:, j]
            means = np.array([partial[idx[j] == b].mean() if np.any(idx[j] == b) else 0.0 for b in range(nbins)])
            G[:, j] = means[idx[j]]
            G[:, j] -= G[:, j].mean()
    return G, idx


def _ode(meta, data, out):
    U, t, dt = data["U"], data["t"], meta["dt"]
    names = meta["variables"]
    q = len(names)
    diag = tb.diagnose(meta, data)
    noise = max(diag["noise_rel_estimate"].values())
    step = max(diag["mean_change_per_step_rel"].values())
    Us, dU = tb.smooth_and_differentiate(meta, U, window=11 if noise > 0.02 else 7)
    X, Y = Us[:, 3:-3].reshape(-1, q), dU[:, 3:-3].reshape(-1, q)
    facts = {"noise_rel": _r(noise), "change_per_step_rel": _r(step), "n_traj": int(U.shape[0])}
    H = out["hypotheses"]

    # --- positivity, boundedness, decades
    for i, v in enumerate(names):
        lo, hi = X[:, i].min(), X[:, i].max()
        if lo > 0:
            dec = np.log10(hi / max(lo, 1e-12))
            facts[f"{v}_positive_decades"] = _r(dec)
            if dec > 0.8:
                H.append({"hypothesis": f"{v} is positive and spans {dec:.1f} decades: multiplicative dynamics are likely; "
                          f"try log({v}) coordinates", "confidence": "medium",
                          "try": {"tool": "transform", "args": {"name": f"log_{v}", "forward": {f"l{v}": f"log({v})"}}}})
        if lo >= -0.02 and hi <= 1.02 and hi > 0.3:
            H.append({"hypothesis": f"{v} stays in [0, 1]: a fraction or probability; expect logistic-type terms {v}*(1-{v})",
                      "confidence": "low", "try": {"tool": "run_sindy", "args": {"custom_terms": [f"{v}*(1-{v})"]}}})
    # --- sum conservation
    if q > 1:
        s = U.sum(-1)
        rv = float(s.std(axis=1).mean() / (np.abs(s).mean() + 1e-12))
        facts["sum_rel_variation"] = _r(rv)
        if rv < max(0.02, 2 * noise):
            H.append({"hypothesis": f"sum of all variables is conserved (variation {rv:.3f}): closed system; drop one "
                      "variable via the constraint", "confidence": "high",
                      "try": {"tool": "find_invariants", "args": {"poly_degree": 1}}})
    # --- oscillation
    osc = {}
    for i, v in enumerate(names):
        spec = np.abs(np.fft.rfft(Us[..., i] - Us[..., i].mean(1, keepdims=True), axis=1)) ** 2
        P = spec.mean(0)
        if len(P) > 4:
            k = int(np.argmax(P[1:]) + 1)
            frac = float(P[k - 1:k + 2].sum() / (P[1:].sum() + 1e-30))
            if frac > 0.4:
                osc[v] = {"period": _r(len(t) * dt / k), "peak_power_fraction": _r(frac)}
    facts["oscillation"] = osc
    # --- fixed points & linearisation
    speed = np.linalg.norm(Y / (Y.std(0) + 1e-12), axis=1)
    slow = speed < np.quantile(speed, 0.03)
    if slow.sum() > 5:
        xs = np.median(X[slow], axis=0)
        near = np.linalg.norm((X - xs) / (X.std(0) + 1e-12), axis=1) < np.quantile(
            np.linalg.norm((X - xs) / (X.std(0) + 1e-12), axis=1), 0.15)
        A = np.column_stack([np.ones(near.sum()), X[near] - xs])
        J = np.linalg.lstsq(A, Y[near], rcond=None)[0][1:].T
        ev = np.linalg.eigvals(J)
        kind = ("spiral" if np.any(np.abs(ev.imag) > 0.2 * np.abs(ev.real) + 1e-9) else "node/saddle")
        stab = "stable" if np.all(ev.real < 0) else "unstable" if np.all(ev.real > 0) else "saddle-type"
        facts["fixed_point_estimate"] = [_r(x) for x in xs]
        facts["linearisation_eigenvalues"] = [f"{e.real:.3g}{e.imag:+.3g}j" for e in ev]
        facts["fixed_point_type"] = f"{stab} {kind}"
        if kind == "spiral" and q == 2:
            fx = [f"({names[j]} - ({xs[j]:.4g}))" for j in range(2)]
            H.append({"hypothesis": f"trajectories rotate about ({xs[0]:.3g}, {xs[1]:.3g}) ({stab} spiral): polar coordinates "
                      "centred there may separate amplitude and phase dynamics", "confidence": "medium",
                      "try": {"tool": "transform", "args": {"name": "polar", "forward": {
                          "r": f"sqrt({fx[0]}**2 + {fx[1]}**2)", "theta": f"atan2({fx[1]}, {fx[0]})"},
                          "inverse": {names[0]: f"{xs[0]:.6g} + r*cos(theta)", names[1]: f"{xs[1]:.6g} + r*sin(theta)"}}}})
    # --- interactions: full quadratic (with products) vs additive quadratic (no products)
    sub = np.random.default_rng(0).choice(len(X), min(20000, len(X)), replace=False)
    Xs = (X[sub] - X[sub].mean(0)) / (X[sub].std(0) + 1e-12)
    add_cols = [np.ones(len(sub))] + [Xs[:, j] for j in range(q)] + [Xs[:, j] ** 2 for j in range(q)]
    prod_cols = [Xs[:, a_] * Xs[:, b_] for a_, b_ in itertools.combinations(range(q), 2)]
    interacts = {}
    for i, v in enumerate(names):
        y = Y[sub, i]
        def rv(cols):
            A = np.column_stack(cols)
            c, *_ = np.linalg.lstsq(A, y, rcond=None)
            return float(np.var(y - A @ c)) + 1e-30
        r_add = rv(add_cols)
        r_full = rv(add_cols + prod_cols) if prod_cols else r_add
        interacts[v] = r_full / r_add < 0.7
        if interacts[v]:
            H.append({"hypothesis": f"d{v}/dt depends on products of variables (adding cross terms cuts the residual by "
                      f"{1 - r_full / r_add:.0%}): include interaction terms such as x*y", "confidence": "high"
                      if r_full / r_add < 0.3 else "medium", "try": {"tool": "run_sindy", "args": {"poly_degree": 2}}})
            poly_deg = 2
    # --- single-variable shapes (only meaningful where the dependence is additive)
    shapes, customs = [], []
    poly_deg = max(1, 2 if any(interacts.values()) else 1)
    for i, v in enumerate(names):
        if interacts[v]:
            continue
        G, _ = _additive_components(X[sub], Y[sub, i])
        tot = Y[sub, i].var() + 1e-30
        for j, w in enumerate(names):
            share = float(G[:, j].var() / tot)
            if share < 0.02:
                continue
            name, K, score, r2 = _shape_fit(X[sub, j], G[:, j])
            rec = {"d/dt": v, "of": w, "variance_share": _r(share), "shape": name, "fit_R2": _r(r2)}
            if K is not None:
                rec["K"] = _r(K)
            shapes.append(rec)
            if name == "quadratic":
                poly_deg = max(poly_deg, 2)
            elif name == "cubic":
                poly_deg = max(poly_deg, 3)
            elif name in ("sin", "cos"):
                customs.append(f"{name}({w})")
            elif name.startswith("saturating") or name == "tanh":
                # saturating family: the exact form is not identifiable from shape alone -> offer the usual suspects
                lo_ = X[sub, j].min()
                fam = [f"tanh({w}/{_r(K)})", f"{w}**3"] + ([f"{w}/({_r(K)} + {w})"] if lo_ > 0 else [])
                customs.extend(fam)
                rec["shape"] = "saturating (tanh, x/(K+x) or cubic softening; fit decides)"
            elif name == "exp":
                customs.append(f"exp({w}/{_r(K)})")
    facts["partial_dependence_shapes"] = shapes
    customs = sorted(set(customs))
    for c in customs:
        H.append({"hypothesis": f"non-polynomial dependence detected: {c}", "confidence": "medium",
                  "try": {"tool": "run_sindy", "args": {"custom_terms": [c]}}})
    # --- anharmonicity: frequency vs amplitude across trajectories
    if osc and U.shape[0] >= 3:
        v = next(iter(osc))
        i = names.index(v)
        per, amp = [], []
        for j in range(U.shape[0]):
            sig = Us[j, :, i] - Us[j, :, i].mean()
            P = np.abs(np.fft.rfft(sig)) ** 2
            k = int(np.argmax(P[1:]) + 1)
            per.append(len(t) * dt / k)
            amp.append(np.ptp(sig))
        cc = np.corrcoef(amp, per)[0, 1] if np.std(per) > 0 else 0
        facts["amplitude_period_correlation"] = _r(cc)
        if abs(cc) > 0.7:
            H.append({"hypothesis": f"oscillation period depends on amplitude (corr {cc:.2f}): the restoring force is "
                      "nonlinear (e.g. periodic or cubic in the displacement)", "confidence": "medium",
                      "try": {"tool": "run_pysr", "args": {"template": "f(...) + g(...)", "model_selection": "rollout"}}})
    out["facts"].update(facts)
    out["recommended_config"] = {
        "method": "weak_sindy" if (noise > 0.01 or step > 0.15) else "run_sindy",
        "poly_degree": poly_deg, "custom_terms": customs, "include_trig": any("sin" in c or "cos" in c for c in customs),
        "window": int(2 * round((7 + 30 * noise) / 2) + 1)}


# ============================================================================ PDE
def _pde_positivity(meta, data, out):
    """A positive field spanning decades is often observed through exp (or has multiplicative noise): suggest a
    pointwise log transform, which keeps the other fields as they are."""
    U, names = data["U"], meta["variables"]
    for i, v in enumerate(names):
        f = U[..., i]
        lo, hi, med = float(f.min()), float(f.max()), float(np.median(f))
        if lo <= 0:
            continue
        dec = np.log10(hi / max(lo, 1e-12))
        out["facts"][f"{v}_positive_decades"] = _r(dec)
        if dec > 1.5 or hi / max(med, 1e-12) > 5:
            fwd = {(f"l{v}" if w == v else w): (f"log({v})" if w == v else w) for w in names}
            out["hypotheses"].append({
                "hypothesis": f"field {v} is positive and spans {dec:.1f} decades (max/median {hi / med:.1f}): it may be "
                              f"observed through exp or have multiplicative noise; try log({v}) coordinates, where the law "
                              "can be polynomial even if it is not in the observed field",
                "confidence": "medium", "try": {"tool": "transform", "args": {"name": f"log_{v}", "forward": fwd}}})


def _pde(meta, data, out):
    U, t, dt = data["U"], data["t"], meta["dt"]
    _pde_positivity(meta, data, out)
    if U.ndim != 4 or meta.get("boundary", "periodic") != "periodic":
        out["facts"]["note"] = "dispersion analysis implemented for 1-D periodic data"
        return
    L, nx = meta["L"], U.shape[2]
    names = meta["variables"]
    diag = tb.diagnose(meta, data)
    noise = max(diag["noise_rel_estimate"].values())
    H = out["hypotheses"]
    k = 2 * np.pi * np.fft.rfftfreq(nx, d=L / nx)
    for i, f in enumerate(names):
        uh = np.fft.rfft(U[..., i], axis=2)               # (traj, nt, nk)
        P = (np.abs(uh) ** 2).mean((0, 1))
        floor = np.median(P[int(0.75 * len(P)):])
        good = np.where(P > 30 * floor)[0]
        good = good[good > 0]
        # growth / frequency per mode: lambda = log(uh(t+dt)/uh(t))/dt, robust median over time, weighted to
        # small-amplitude epochs where linear dynamics dominate
        lam = []
        for kk in good:
            z = uh[:, :, kk]
            ratio = z[:, 1:] / (z[:, :-1] + 1e-30)
            amp = np.abs(z[:, :-1])
            w = amp > np.quantile(amp, 0.2)
            lr = np.log(np.abs(ratio[w]) + 1e-30) / dt
            ph = np.angle(ratio[w]) / dt
            lam.append((k[kk], np.median(lr), np.median(ph)))
        if len(lam) >= 4:
            kk, gr, om = map(np.array, zip(*lam))
            # fit Re lambda = c2*(-k^2) + c4*(k^4) ... from L = sum a_n (ik)^n:
            #   n=2: a2*(-k^2) real; n=4: a4*k^4 real; n=1: a1*(ik) -> imaginary a1 k; n=3: a3*(-i k^3)
            Ar = np.column_stack([np.ones_like(kk), -kk ** 2, kk ** 4])
            cr, *_ = np.linalg.lstsq(Ar, gr, rcond=None)
            Ai = np.column_stack([kk, -kk ** 3])
            ci, *_ = np.linalg.lstsq(Ai, om, rcond=None)
            def r2(A, c, y):
                return 1 - np.var(y - A @ c) / (np.var(y) + 1e-30)
            disp = {"growth_rate_fit": {"const": _r(cr[0]), f"{f}_xx coef": _r(cr[1]), f"{f}_xxxx coef": _r(cr[2]),
                                        "R2": _r(r2(Ar, cr, gr))},
                    "frequency_fit": {f"{f}_x coef": _r(ci[0]), f"{f}_xxx coef": _r(ci[1]), "R2": _r(r2(Ai, ci, om))},
                    "k_range": [_r(kk.min()), _r(kk.max())], "n_modes": int(len(kk))}
            out["facts"][f"dispersion_{f}"] = disp
            lin = []
            scale_r = np.abs(gr).max() + 1e-12
            scale_i = np.abs(om).max() + 1e-12
            ok_r = (disp["growth_rate_fit"]["R2"] or 0) > 0.5
            ok_i = (disp["frequency_fit"]["R2"] or 0) > 0.5
            if ok_r and abs(cr[1]) * kk.max() ** 2 > 0.2 * scale_r:
                lin.append(f"{f}_xx")
            if ok_r and abs(cr[2]) * kk.max() ** 4 > 0.2 * scale_r:
                lin.append(f"{f}_xxxx")
            if ok_i and abs(ci[0]) * kk.max() > 0.2 * scale_i:
                lin.append(f"{f}_x (or advection by the mean)")
            if ok_i and abs(ci[1]) * kk.max() ** 3 > 0.2 * scale_i:
                lin.append(f"{f}_xxx (dispersion)")
            if lin:
                H.append({"hypothesis": f"mode growth/frequency vs wavenumber suggests linear terms {lin} "
                          f"(growth fit R2={disp['growth_rate_fit']['R2']}, frequency fit R2={disp['frequency_fit']['R2']}); "
                          "coefficients are biased by nonlinear transfer but signs are informative",
                          "confidence": "medium" if max(disp['growth_rate_fit']['R2'] or 0, disp['frequency_fit']['R2'] or 0) > 0.6 else "low",
                          "try": {"tool": "weak_sindy", "args": {"max_deriv": 4 if any("xxxx" in x for x in lin) else 3}}})
            if cr[1] < 0 and cr[2] < 0 and abs(cr[1]) > 0:
                H.append({"hypothesis": "long waves grow and short waves decay (negative diffusion plus higher-order damping): "
                          "a long-wave instability saturated by short-wave damping", "confidence": "medium",
                          "try": {"tool": "weak_sindy", "args": {"max_deriv": 4}}})
        # travelling waves: shift maximising correlation between consecutive snapshots
        x = np.arange(nx) * L / nx
        sp_, am_ = [], []
        for j in range(U.shape[0]):
            for n in range(0, U.shape[1] - 1, max(1, U.shape[1] // 10)):
                a, b = U[j, n, :, i] - U[j, n, :, i].mean(), U[j, n + 1, :, i] - U[j, n + 1, :, i].mean()
                cc = np.fft.irfft(np.conj(np.fft.rfft(a)) * np.fft.rfft(b), n=nx)
                s = int(np.argmax(cc))
                s = s - nx if s > nx // 2 else s
                sp_.append(s * (L / nx) / dt)
                am_.append(np.ptp(U[j, n, :, i]))
        sp_, am_ = np.array(sp_), np.array(am_)
        if np.median(np.abs(sp_)) > 0:
            out["facts"][f"travelling_speed_{f}"] = {"median": _r(np.median(sp_)), "iqr": _r(np.subtract(*np.percentile(sp_, [75, 25])))}
            if len(sp_) > 5 and np.std(am_) > 0 and np.std(sp_) > 0:
                cc = float(np.corrcoef(am_, sp_)[0, 1])
                out["facts"][f"speed_amplitude_corr_{f}"] = _r(cc)
                if abs(cc) > 0.5:
                    H.append({"hypothesis": f"wave speed grows with amplitude (corr {cc:.2f}): nonlinear advection "
                              f"{f}*{f}_x", "confidence": "medium",
                              "try": {"tool": "weak_sindy", "args": {"poly_degree": 1}}})
        # conservation of mean, parity
        m = U[..., i].mean(2)
        drift = m.std(1).mean() / (U[..., i].std() + 1e-12)
        out["facts"][f"mean_drift_rel_{f}"] = _r(drift)
        if drift < 0.02 + 2 * noise / np.sqrt(nx):
            H.append({"hypothesis": f"the spatial mean of {f} is conserved: the right-hand side is a total derivative "
                      f"(flux form), e.g. ({f}^2)_x, {f}_xx, {f}_xxx; no source terms like {f} or {f}^2",
                      "confidence": "high" if drift < 0.005 else "medium", "try": {"tool": "weak_sindy", "args": {"exclude_terms": ["1", f, f"{f}*{f}"]}}})
        else:
            H.append({"hypothesis": f"the spatial mean of {f} changes: there are source/reaction terms (e.g. {f}, {f}^2, {f}^3)",
                      "confidence": "medium", "try": {"tool": "weak_sindy", "args": {"poly_degree": 3}}})
        skew = float(((U[..., i] - U[..., i].mean()) ** 3).mean() / (U[..., i].std() ** 3 + 1e-30))
        out["facts"][f"skewness_{f}"] = _r(skew)
        if abs(skew) < 0.15 and abs(U[..., i].mean()) < 0.1 * U[..., i].std():
            H.append({"hypothesis": f"{f} is statistically symmetric under {f} -> -{f}: nonlinearities are likely odd "
                      f"({f}^3) or of advective type; quadratic sources are unlikely", "confidence": "low",
                      "try": {"tool": "detect_symmetries", "args": {}}})
    out["facts"]["noise_rel"] = _r(noise)
    out["recommended_config"] = {"method": "weak_sindy", "max_deriv": 4,
                                 "note": "weak form avoids differentiating noisy fields"}


def intuit(meta, data):
    out = {"facts": {}, "hypotheses": []}
    if meta["kind"] == "ode":
        _ode(meta, data, out)
    else:
        _pde(meta, data, out)
    order = {"high": 0, "medium": 1, "low": 2}
    out["hypotheses"].sort(key=lambda h: order.get(h["confidence"], 3))
    return out
