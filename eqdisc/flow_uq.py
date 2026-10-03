"""Confidence checks for ODE laws fitted by flow-map shooting (coarse sampling, e.g. hourly satellite positions),
in the same slim format the demo's confidence panel uses for eqdisc assessments:

  terms        each fitted constant: value, 90% interval (Gauss-Newton), significance, dBIC if its term is removed
  missing      candidate extra terms (generic, not domain-specific): dBIC if added and one-step error reduction
  validation   internal hold-out: fit on the first 80% of the TRAINING window, forecast the last 20%
  experiments  where to measure next: candidate new trajectories ranked by information about each constant,
               relative to recording the same amount of data along the existing trajectory
"""
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

from .flow import fit_flow, make_flow_fn


def _bic(rss, n, k):
    return n * np.log(rss / n) + k * np.log(n)


def _rss(f):
    return (f["one_step_rel_err_heldout"] ** 2) * f["n_pairs"]


def _drop_param(withp, p):
    return {k: str(sp.sympify(v).subs(sp.Symbol(p), 0)) for k, v in withp.items()}


def flow_uq(meta, data, withp, init, max_pairs=1200, candidates=None, holdout_frac=0.2, valid_tol=0.01,
            experiments=None, names=None):
    """withp: law with constants p0.. (e.g. from oos.parametrize_floats); init: their values."""
    base = fit_flow(meta, data, withp, max_pairs=max_pairs, init=init)
    n = base["n_pairs"] * len(meta["variables"])
    rss0, k0 = _rss(base), len(init)
    terms = []
    for i, (p, v) in enumerate(base["params"].items()):
        s = base["param_sigma"].get(p, np.nan)
        ci = [v - 1.645 * s, v + 1.645 * s]
        try:
            fd = fit_flow(meta, data, _drop_param(withp, p), max_pairs=max_pairs,
                          init=[x for j, x in enumerate(init) if j != i])
            dbic = float(_bic(_rss(fd), n, k0 - 1) - _bic(rss0, n, k0))
        except Exception:  # noqa: BLE001
            dbic = None
        terms.append({"var": (names or {}).get(p, {}).get("var", "·"), "term": (names or {}).get(p, {}).get("term", p),
                      "coef": v, "ci90": ci, "rel_uncertainty": float(1.645 * s / abs(v)) if v else None,
                      "significant": bool(ci[0] > 0 or ci[1] < 0), "dBIC_if_removed": dbic})
    missing = []
    for cand_name, cand in (candidates or {}).items():
        q = f"p{len(init)}"
        aug = {k: (f"({v}) + {q}*({cand[k]})" if k in cand else v) for k, v in withp.items()}
        try:
            fa = fit_flow(meta, data, aug, max_pairs=max_pairs, init=list(init) + [0.0])
            missing.append({"var": "·", "term": cand_name, "dBIC_if_added": float(_bic(_rss(fa), n, k0 + 1) - _bic(rss0, n, k0)),
                            "error_reduction": float(1 - fa["one_step_rel_err_heldout"] / base["one_step_rel_err_heldout"])})
        except Exception:  # noqa: BLE001
            pass
    # internal hold-out inside the training window
    U, t = data["U"][0], data["t"]
    k = int(len(t) * (1 - holdout_frac))
    fh = fit_flow(meta, {"U": U[None, :k], "t": t[:k]}, withp, max_pairs=max_pairs, init=init)
    rhs = make_flow_fn(meta["variables"], fh["rhs"], [])
    tt = t[k - 1:] - t[k - 1]
    sol = solve_ivp(lambda s_, y: rhs(y[None], [])[0], (0, tt[-1]), U[k - 1], t_eval=tt, rtol=1e-10, atol=1e-12,
                    method="DOP853")
    npos = len(meta["variables"]) // 2
    rel = np.linalg.norm(sol.y.T[:, :npos] - U[k - 1:, :npos], axis=1) / np.linalg.norm(U[k - 1:, :npos], axis=1)
    bad = np.nonzero(rel > valid_tol)[0]
    vt = float(tt[bad[0]]) if len(bad) else float(tt[-1])
    validation = {"rollout_valid_time": vt, "rollout_horizon": float(tt[-1]), "rollout_blew_up": not np.all(np.isfinite(rel)),
                  "criterion": f"position error < {valid_tol:.0%} of radius", "max_rel_error": float(np.nanmax(rel))}
    return {"terms": terms, "missing": missing, "validation": validation, "fit": base,
            "experiments": experiments or []}


def orbit_oed(withp, params, x0, dt, span, candidates, pidx_names, n_steps=400):
    """Information (sum of squared trajectory sensitivities) about each constant from a new trajectory, relative to
    the same-length record along the existing one (x0). candidates: {description: initial state}."""
    variables = [f"u{i}" for i in range(1, 7)]
    pn = list(params)
    f = make_flow_fn(variables, withp, pn)
    p0 = np.array([params[k] for k in pn], float)

    def traj(p, y0):
        tt = np.linspace(0, span, n_steps)
        return solve_ivp(lambda s_, y: f(y[None], p)[0], (0, span), y0, t_eval=tt, rtol=1e-10, atol=1e-12).y.T[:, :3]

    def info(y0):
        out = {}
        base = traj(p0, y0)
        for i, k in enumerate(pn):
            h = 1e-4 * abs(p0[i]) or 1e-8
            pp = p0.copy()
            pp[i] += h
            S = (traj(pp, y0) - base) / h * p0[i]          # sensitivity to a relative change of the constant
            out[k] = float(np.mean(S ** 2))
        return out
    ref = info(x0)
    rows = []
    for desc, y0 in candidates.items():
        inf = info(np.asarray(y0, float))
        gains = [{"coefficient": pidx_names.get(k, k), "info_gain_vs_existing": inf[k] / ref[k]} for k in pn]
        gains.sort(key=lambda g: -g["info_gain_vs_existing"])
        rows.append({"description": desc, "informs_coefficients": gains, "score": max(g["info_gain_vs_existing"] for g in gains)})
    rows.sort(key=lambda r: -r["score"])
    return rows
