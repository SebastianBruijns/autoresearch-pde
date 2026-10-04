"""Structural priors from the data: measured patterns -> implied term FAMILIES (never named equations).
Public data only, no LLM. Part of the evidence layer (data stage; see eqdisc/audit/__init__.py).

    audit(meta, data)  -> list[Finding]   (severity info, except identifiability -> warn)
    card(meta, data)   -> {"applicable", "detectors": [...], "programme": {...}, "summary": [...]}
    programme(meta, detectors) -> {"tool": "weak_sindy", "args": {...}}   first typed research programme

Each detector returns
    {"name", "verdict", "confidence": high|medium|low, "evidence": {numbers},
     "implies": {"exclude": [terms], "require_any": [terms], "prefer": {tool args}}}
and its implication is derived from first principles (linear theory, conservation, energy balance), so
it holds for any PDE of the V1 class  u_t = F(u, u_x, ..., u_xxxx), not just for benchmark systems.
The `programme` is the first typed research programme these implications suggest. Thresholds were set on
development systems only (prior_* keys in thresholds.json, defaults in code).

Detectors (1-D periodic scalar fields; other layouts return applicable=False):
  resolution      noise level, resolved band k_c, per-derivative-order SNR inside the band. Clean data
                  -> wide test functions (quadrature error, not noise, limits the weak form).
  mean            spatial mean conserved -> rhs is a total x-derivative: no source monomials (1, u, u^2, u^3).
  energy          d/dt mean(u'^2): conserved -> no even-order linear damping terms; decaying -> some damping
                  term; growing or sustained with a conserved mean -> an energy source at some scales
                  (with flux-form nonlinearity this must be linear: anti-diffusion balanced by u_xxxx).
  linear_response growth rate and frequency of Fourier modes vs k (Re: -a2 k^2 + a4 k^4, Im: a1 k - a3 k^3)
                  -> which linear derivative orders are active, and their signs.
  identifiability column conditioning of the candidate library on these data: groups of terms the
                  data cannot tell apart (He, Zhao & Zhong 2024: a solution may not determine its PDE).
"""
import numpy as np

from .. import toolbox as tb
from . import finding, threshold


def _r(x, s=3):
    try:
        return float(f"{float(x):.{s}g}")
    except (TypeError, ValueError):
        return None


def applicable(meta, data):
    return (meta.get("kind") == "pde" and np.asarray(data["U"]).ndim == 4
            and (meta.get("boundary") or "periodic") == "periodic"
            and len(meta.get("spatial_dims") or ["x"]) == 1)


def spectral_noise(meta, data):
    """Relative white-noise level per field from the flat high-k plateau of the spatial spectrum
    (robust to coarse time sampling, unlike a temporal-smoothing residual)."""
    out = {}
    for i, f in enumerate(meta["variables"]):
        P, _, floor, _ = _spectrum(data["U"][..., i], meta["L"])
        out[f] = float(np.sqrt(min(floor * len(P) / (P.sum() + 1e-300), 1.0)))
    return out


def _spectrum(Ui, L):
    """Mean power per rfft mode over (traj, time), angular wavenumbers, noise floor, corner index."""
    P = (np.abs(np.fft.rfft(Ui, axis=2)) ** 2).mean((0, 1))
    k = 2 * np.pi * np.fft.rfftfreq(Ui.shape[2], d=L / Ui.shape[2])
    floor = float(np.median(P[int(0.75 * len(P)):])) + 1e-300
    above = np.where(P > 10 * floor)[0]
    kc = int(above.max()) if above.size else len(P) - 1
    return P, k, floor, kc


# ----------------------------------------------------------------------------- detectors
def resolution(meta, data, noise_rel):
    out = []
    for i, f in enumerate(meta["variables"]):
        P, k, floor, kc = _spectrum(data["U"][..., i], meta["L"])
        band = slice(1, kc + 1)
        snr = {}
        for n in range(1, 5):
            sig = np.sum(k[band] ** (2 * n) * np.maximum(P[band] - floor, 0))
            noi = np.sum(k[band] ** (2 * n) * floor)
            snr[f"{f}_{'x' * n}"] = _r(min(np.sqrt(sig / (noi + 1e-300)), 1e6))
        clean = noise_rel[f] < threshold("prior_clean_noise", 1e-3)
        weak_orders = [d for d, s in snr.items() if s is not None and s < 3]
        prefer = {"width_factor": threshold("prior_clean_width_factor", 12)} if clean else {}
        verdict = ("no measurable noise: quadrature error limits the weak form, use wide test functions"
                   if clean else f"noise {noise_rel[f]:.1%}; strong-form derivatives unreliable for {weak_orders}"
                   if weak_orders else f"noise {noise_rel[f]:.1%}; all derivative orders resolved")
        out.append({"name": f"resolution[{f}]", "verdict": verdict, "confidence": "high",
                    "evidence": {"noise_rel": _r(noise_rel[f]), "k_corner": _r(k[kc]),
                                 "modes_resolved": int(kc), "n_modes": int(len(P)), "derivative_snr": snr},
                    "implies": {"prefer": {"tool": "weak_sindy", **prefer}}})
    return out


def mean(meta, data, noise_rel):
    out = []
    U, nx = data["U"], data["U"].shape[2]
    for i, f in enumerate(meta["variables"]):
        m = U[..., i].mean(2)
        drift = float(m.std(1).mean() / (U[..., i].std() + 1e-12))
        tol = 0.02 + 2 * noise_rel[f] / np.sqrt(nx)
        if drift < tol:
            conf = "high" if drift < 0.005 else "medium"
            out.append({"name": f"mean[{f}]", "verdict": "spatial mean conserved: rhs is a total x-derivative",
                        "confidence": conf, "evidence": {"mean_drift_rel": _r(drift), "tolerance": _r(tol)},
                        "implies": {"exclude": ["1", f, f"{f}**2", f"{f}**3"]}})
        else:
            out.append({"name": f"mean[{f}]", "verdict": "spatial mean changes: source/reaction terms present",
                        "confidence": "medium", "evidence": {"mean_drift_rel": _r(drift), "tolerance": _r(tol)},
                        "implies": {"require_any": [f, f"{f}**2", f"{f}**3", "1"]}})
    return out


def energy(meta, data, noise_rel, mean_conserved):
    """Fluctuation energy E(t) = mean_x (u - mean_x u)^2 minus the white-noise variance."""
    out = []
    U = data["U"]
    t = np.asarray(data["t"], float)
    for i, f in enumerate(meta["variables"]):
        X = U[..., i] - U[..., i].mean(2, keepdims=True)
        sig2 = (noise_rel[f] * U[..., i].std()) ** 2
        E = np.maximum((X ** 2).mean(2) - sig2, 1e-300)                  # (traj, nt)
        rel_var = float(np.median(E.std(1) / E.mean(1)))
        ratio = float(np.median(E[:, -1] / E[:, 0]))
        w = max(1, E.shape[1] // 20)
        Es = np.stack([np.convolve(e, np.ones(w) / w, mode="valid") for e in E])
        frac_down = float(np.mean(np.diff(Es, axis=1) < 0))
        # noise in E from finite nx: ~ 2 sigma |u'| / sqrt(nx) relative
        tol = 0.01 + 4 * noise_rel[f] / np.sqrt(U.shape[2])
        fs = [f"{f}_xx", f"{f}_xxxx"]
        if rel_var < tol:
            out.append({"name": f"energy[{f}]", "verdict": "fluctuation energy conserved: no net linear damping",
                        "confidence": "high" if rel_var < tol / 3 else "medium",
                        "evidence": {"rel_variation": _r(rel_var), "end_over_start": _r(ratio), "tolerance": _r(tol)},
                        "implies": {"exclude": fs}})
        elif ratio < 0.8 and frac_down > 0.75:
            out.append({"name": f"energy[{f}]", "verdict": "fluctuation energy decays: a damping term is present",
                        "confidence": "medium",
                        "evidence": {"rel_variation": _r(rel_var), "end_over_start": _r(ratio),
                                     "frac_steps_decreasing": _r(frac_down)},
                        "implies": {"require_any": fs}})
        else:
            imp = {"require_any": [f"{f}_xxxx"], "prefer": {"max_deriv": 4}} if mean_conserved.get(f) else {}
            out.append({"name": f"energy[{f}]",
                        "verdict": "fluctuation energy grows or is sustained: an energy source at some scales"
                                   + (" (mean conserved: a linear long-wave instability saturated by short-wave "
                                      "damping is the only flux-form option)" if imp else ""),
                        "confidence": "medium",
                        "evidence": {"rel_variation": _r(rel_var), "end_over_start": _r(ratio),
                                     "frac_steps_decreasing": _r(frac_down)},
                        "implies": imp})
    return out


def linear_response(meta, data, i):
    """Growth rate and frequency of resolved Fourier modes, weighted to small-amplitude epochs.
    Returns {"k", "growth", "freq", "growth_fit", "freq_fit"} or None. Coefficients are biased by
    nonlinear energy transfer; signs are informative."""
    U, dt, L = data["U"], data["t"][1] - data["t"][0], meta["L"]
    uh = np.fft.rfft(U[..., i], axis=2)
    P, k, floor, _ = _spectrum(U[..., i], L)
    good = np.where(P > 30 * floor)[0]
    good = good[good > 0]
    lam = []
    for kk in good:
        z = uh[:, :, kk]
        ratio = z[:, 1:] / (z[:, :-1] + 1e-30)
        amp = np.abs(z[:, :-1])
        w = amp > np.quantile(amp, 0.2)
        lam.append((k[kk], np.median(np.log(np.abs(ratio[w]) + 1e-30) / dt), np.median(np.angle(ratio[w]) / dt)))
    if len(lam) < 4:
        return None
    kk, gr, om = map(np.array, zip(*lam))
    Ar = np.column_stack([np.ones_like(kk), -kk ** 2, kk ** 4])
    cr = np.linalg.lstsq(Ar, gr, rcond=None)[0]
    Ai = np.column_stack([kk, -kk ** 3])
    ci = np.linalg.lstsq(Ai, om, rcond=None)[0]

    def r2(A, c, y):
        return float(1 - np.var(y - A @ c) / (np.var(y) + 1e-30))
    return {"k": kk, "growth": gr, "freq": om, "cr": cr, "ci": ci, "r2_growth": r2(Ar, cr, gr),
            "r2_freq": r2(Ai, ci, om)}


def dispersion(meta, data):
    out = []
    for i, f in enumerate(meta["variables"]):
        lr = linear_response(meta, data, i)
        if lr is None:
            continue
        kk, cr, ci = lr["k"], lr["cr"], lr["ci"]
        sr = np.abs(lr["growth"]).max() + 1e-12
        si = np.abs(lr["freq"]).max() + 1e-12
        okr, oki = lr["r2_growth"] > 0.5, lr["r2_freq"] > 0.5
        active = {}
        if okr and abs(cr[1]) * kk.max() ** 2 > 0.2 * sr:
            active[f"{f}_xx"] = "damping" if cr[1] > 0 else "anti-diffusion"
        if okr and abs(cr[2]) * kk.max() ** 4 > 0.2 * sr:
            active[f"{f}_xxxx"] = "damping" if cr[2] < 0 else "amplifying"
        if oki and abs(ci[0]) * kk.max() > 0.2 * si:
            active[f"{f}_x"] = "advection (or advection by a nonzero mean)"
        if oki and abs(ci[1]) * kk.max() ** 3 > 0.2 * si:
            active[f"{f}_xxx"] = "dispersion"
        conf = "medium" if min(lr["r2_growth"] if (f"{f}_xx" in active or f"{f}_xxxx" in active) else 1,
                               lr["r2_freq"] if (f"{f}_x" in active or f"{f}_xxx" in active) else 1) > 0.95 else "low"
        out.append({"name": f"linear_response[{f}]",
                    "verdict": (f"active linear orders (small-amplitude modes; biased by nonlinear transfer): {active}"
                                if active else "no clear linear signature"),
                    "confidence": conf,
                    "evidence": {"growth_fit": {"a2 (u_xx)": _r(cr[1]), "a4 (u_xxxx)": _r(cr[2]),
                                                "R2": _r(lr["r2_growth"])},
                                 "freq_fit": {"a1 (u_x)": _r(ci[0]), "a3 (u_xxx)": _r(ci[1]), "R2": _r(lr["r2_freq"])},
                                 "n_modes": int(len(kk))},
                    "implies": {"require_any": [t for t in active if t.endswith("xxx")]} if f"{f}_xxx" in active
                    else {}})
    return out


def identifiability(meta, data, max_rows=20000, cos_tol=0.98, seed=0):
    """Column geometry of the default candidate library (degree 3, derivatives to 4th order)."""
    from .baselines import lowpass
    U = np.asarray(data["U"], float)
    rng = np.random.default_rng(seed)
    nt = U.shape[1]
    frames = rng.choice(U.shape[0] * nt, min(U.shape[0] * nt, max(1, max_rows // U.shape[2])), replace=False)
    S = np.stack([U[j // nt, j % nt] for j in frames])[None]          # (1, n, nx, nf)
    Ss = lowpass(S, 0.3)
    feats = tb.feature_arrays(meta, Ss, np.zeros(Ss.shape[1]))
    terms, cols = tb.build_library(meta, feats, 3, 4)
    A = np.stack([np.asarray(c, float).ravel() for c in cols], 1)
    keep = np.linalg.norm(A, axis=0) > 0
    terms, A = [t for t, k in zip(terms, keep) if k], A[:, keep]
    A = A / np.linalg.norm(A, axis=0)
    s = np.linalg.svd(A, compute_uv=False)
    C = np.abs(A.T @ A)
    pairs = [(terms[a], terms[b], _r(C[a, b])) for a in range(len(terms)) for b in range(a + 1, len(terms))
             if C[a, b] > cos_tol]
    pairs.sort(key=lambda p: -p[2])
    cond = float(s[0] / max(s[-1], 1e-300))
    return [{"name": "identifiability",
             "verdict": (f"{len(pairs)} nearly collinear term pairs: these data cannot separate them; prefer "
                         "ensemble/stability checks and data with different amplitudes or scales"
                         if pairs else "library columns well separated on these data"),
             "confidence": "high", "evidence": {"condition_number": _r(cond), "collinear_pairs": pairs[:8]},
             "implies": {}}]


# ----------------------------------------------------------------------------- card
def card(meta, data):
    if not applicable(meta, data):
        return {"applicable": False, "detectors": [], "summary": ["detectors cover 1-D periodic PDE data"]}
    noise_rel = spectral_noise(meta, data)
    dets = resolution(meta, data, noise_rel)
    mdet = mean(meta, data, noise_rel)
    dets += mdet
    conserved = {d["name"][5:-1]: d["confidence"] == "high" and "exclude" in d["implies"] for d in mdet}
    dets += energy(meta, data, noise_rel, conserved)
    dets += dispersion(meta, data)
    try:
        dets += identifiability(meta, data)
    except Exception as e:  # noqa: BLE001
        dets.append({"name": "identifiability", "verdict": f"not computed: {e}", "confidence": "low",
                     "evidence": {}, "implies": {}})
    return {"applicable": True, "detectors": dets, "programme": programme(meta, dets),
            "summary": [f"[{d['confidence']}] {d['verdict']}" for d in dets]}


def programme(meta, dets, include_medium=False):
    """First typed research programme implied by the detectors."""
    excl, args = [], {"poly_degree": 3, "max_deriv": 4}
    for d in dets:
        imp = d.get("implies", {})
        if d["confidence"] == "high" or (include_medium and d["confidence"] == "medium"):
            excl += [t for t in imp.get("exclude", []) if t not in excl]
        args.update({k: v for k, v in imp.get("prefer", {}).items() if k != "tool"})
    if excl:
        args["exclude_terms"] = excl
    return {"tool": "weak_sindy", "args": args,
            "why": "weak form by default (robust to noise; MDBench 2025); exclusions only from high-confidence detectors"}


def audit(meta, data):
    """Findings contract adapter: one info Finding per detector, implications in details."""
    c = card(meta, data)
    out = []
    for d in c["detectors"]:
        base = d["name"].split("[")[0]
        ev = d["evidence"]
        stat, thr = {"resolution": (ev.get("noise_rel"), threshold("prior_clean_noise", 1e-3)),
                     "mean": (ev.get("mean_drift_rel"), ev.get("tolerance")),
                     "energy": (ev.get("rel_variation"), ev.get("tolerance")),
                     "identifiability": (len(ev.get("collinear_pairs", [])), 0)}.get(base, (None, None))
        collinear = base == "identifiability" and stat
        out.append(finding(
            f"prior_{base}", "data", stat, thr, bool(d["implies"]) or bool(collinear),
            "warn" if collinear else "info", "widen" if collinear else None,
            message=f"{d['verdict']} [{d['confidence']}]",
            details={"field": d["name"], "confidence": d["confidence"], "implies": d["implies"], "evidence": ev}))
    if c.get("programme"):
        out.append(finding("prior_programme", "data", None, None, True, "info", None,
                           fix=c["programme"], message="first research programme implied by the data priors",
                           details={"programme": c["programme"]}))
    return out
