"""Score a candidate model against a dataset's hidden test set.

A candidate is {"rhs": {var: "expression", ...}} using the dataset's
meta["allowed_symbols"] (PDE derivatives as u_x, u_xx, ...; 2-D: u_x, u_y, u_xy, ...).

PDE scoring: 1-D periodic datasets use the original spectral/ETDRK4 path (unchanged). 2-D and
non-periodic datasets use solvers.make_pde_rhs_general (spectral if periodic, finite differences
otherwise; the vector-field error excludes the boundary layer of non-periodic grids) and
solvers.integrate_pde_general; on non-periodic grids the rollout's boundary layer is driven by the
clean test trajectory (Dirichlet from data), so candidates are scored on interior dynamics only.

Metrics (all computed on *clean, held-out* trajectories from unseen ICs):
    vf_nrmse       vector-field error: ||f_cand(U) - f_true(U)|| / ||f_true(U)||
    rollout_nrmse  simulate candidate from test ICs over the eval horizon
    valid_frac     fraction of the horizon before rollout error exceeds 0.3
    n_terms        additive terms in the expanded model (parsimony)
    score          scalar fitness for search (higher is better)
With reveal=True also: term precision / recall / F1 and coefficient error vs truth.

    python -m eqdisc.evaluate datasets/lorenz_n0.01_dt1_s0 cand.json --reveal
"""
import argparse
import json
from pathlib import Path

import numpy as np
import sympy as sp

from .solvers import (MAX_DERIV, derivative_symbols, integrate_ode, integrate_pde, integrate_pde_general,
                      is_legacy_pde, make_ode_rhs, make_pde_rhs, make_pde_rhs_general, parse, pde_layout)

CAP = 10.0
PARSIMONY = 0.02


def load(dataset):
    d = Path(dataset)
    meta = json.loads((d / "meta.json").read_text())
    data = dict(np.load(d / "data.npz"))
    return meta, data


def _load_hidden(d):
    truth = json.loads((d / "hidden" / "truth.json").read_text())
    test = dict(np.load(d / "hidden" / "test.npz"))
    return truth, test


def _names(truth):
    if truth["kind"] == "pde":
        return derivative_symbols(truth["variables"], truth.get("spatial_dims") or ["x"], MAX_DERIV)
    return truth["variables"] + ["t"]


def _pde_meta(truth, test):
    """Grid / boundary layout of a PDE dataset (legacy truth.json files have only L; nx from test x)."""
    m = dict(truth)
    m.setdefault("nx", len(test["x"]))
    return pde_layout(m)


def _nrmse(pred, true, axis_reduce):
    """Per-variable normalised RMSE averaged over variables; NaN/inf -> CAP."""
    with np.errstate(all="ignore"):
        err = np.sqrt(np.mean((pred - true) ** 2, axis=axis_reduce))
        ref = np.sqrt(np.mean((true - true.mean(axis=axis_reduce, keepdims=True)) ** 2, axis=axis_reduce))
        ref = np.where(ref > 1e-12, ref, np.sqrt(np.mean(true ** 2, axis=axis_reduce)) + 1e-12)
        r = err / ref
    r = np.where(np.isfinite(r), r, CAP)
    return float(np.minimum(r, CAP).mean())


def terms(expr):
    """{monomial: coefficient} of an expanded expression."""
    out = {}
    for t in sp.Add.make_args(sp.expand(expr)):
        c, m = t.as_coeff_Mul()
        if c.is_number and abs(float(c)) > 1e-10:
            out[m] = out.get(m, 0.0) + float(c)
    return out


def structure_metrics(cand_rhs, true_rhs, names):
    tp = fp = fn = 0
    coef_err = []
    for v, e in true_rhs.items():
        T = terms(parse(e, names))
        C = terms(parse(cand_rhs.get(v, "0"), names))
        tp += len(T.keys() & C.keys())
        fp += len(C.keys() - T.keys())
        fn += len(T.keys() - C.keys())
        coef_err += [abs(C[m] - T[m]) / abs(T[m]) for m in T.keys() & C.keys()]
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-12),
            "exact_structure": fp == 0 and fn == 0,
            "coef_rel_err": float(np.mean(coef_err)) if coef_err else None}


def evaluate(dataset, candidate, reveal=False):
    d = Path(dataset)
    truth, test = _load_hidden(d)
    names = _names(truth)
    variables = truth["variables"]
    rhs = candidate["rhs"] if "rhs" in candidate else candidate
    res = {"dataset": d.name}

    # ---- validate
    try:
        exprs = {v: parse(rhs.get(v, "0"), names) for v in variables}
        bad = set().union(*[e.free_symbols for e in exprs.values()]) - {sp.Symbol(n) for n in names}
        if bad:
            raise ValueError(f"unknown symbols {sorted(map(str, bad))}; allowed: {names}")
    except Exception as e:  # noqa: BLE001
        return {**res, "error": f"parse: {e}", "score": -10.0}
    rhs = {v: str(exprs[v]) for v in variables}
    res["rhs"] = rhs
    res["n_terms"] = int(sum(len(sp.Add.make_args(sp.expand(e))) for e in exprs.values()))

    U, t = test["U"], test["t"]
    nh = int(np.searchsorted(t, truth["eval_horizon"] + 1e-9))
    th = t[:nh]
    with np.errstate(all="ignore"):
        if truth["kind"] == "ode":
            ft = make_ode_rhs(variables, truth["rhs"])(U, t[None, :])
            fc = make_ode_rhs(variables, rhs)(U, t[None, :])
            roll = np.stack([integrate_ode(variables, rhs, u[0], th) for u in U])
            vf_axes, roll_axes = (0, 1), (1,)
        elif is_legacy_pde(_pde_meta(truth, test)):
            x = test["x"]
            ft = make_pde_rhs(variables, truth["rhs"], truth["L"])(U, x, t[None, :, None])
            fc = make_pde_rhs(variables, rhs, truth["L"])(U, x, t[None, :, None])
            roll = np.stack([integrate_pde(variables, rhs, truth["L"], u[0], th, truth["dt_sim"])
                             for u in U])
            vf_axes, roll_axes = (0, 1, 2), (1, 2)
        else:
            lay = _pde_meta(truth, test)
            periodic = lay["boundary"] == "periodic"
            ft = make_pde_rhs_general(variables, truth["rhs"], lay)(U)
            fc = make_pde_rhs_general(variables, rhs, lay)(U)
            if not periodic:      # boundary nodes are imposed, not governed by the PDE
                inner = (slice(None), slice(None)) + (slice(1, -1),) * len(lay["spatial_dims"])
                ft, fc = ft[inner], fc[inner]
            roll = np.stack([integrate_pde_general(variables, rhs, lay, u[0], th, truth.get("dt_sim"),
                                                   boundary_data=None if periodic else u[:nh], max_seconds=60.0)
                             for u in U])
            vf_axes = tuple(range(U.ndim - 1))
            roll_axes = None      # per-variable NRMSE over (traj, time, space)

    res["vf_nrmse"] = _nrmse(fc, ft, vf_axes)
    ref = U[:, :nh]
    if roll_axes is None:
        res["rollout_nrmse"] = _nrmse(roll, ref, tuple(range(roll.ndim - 1)))
    else:   # legacy reduction (kept so that existing scores are unchanged)
        res["rollout_nrmse"] = _nrmse(roll, ref, (0, 1) + tuple(a + 1 for a in roll_axes[1:]))
    # per-time error -> valid fraction of horizon
    red = tuple(range(2, roll.ndim))
    with np.errstate(all="ignore"):
        e_t = np.sqrt(np.mean((roll - ref) ** 2, axis=red)) / (np.sqrt(np.mean(ref ** 2, axis=red)) + 1e-12)
    e_t = np.where(np.isfinite(e_t), e_t, np.inf)
    bad_t = np.argmax(e_t > 0.3, axis=1)
    bad_t = np.where((e_t > 0.3).any(axis=1), bad_t, nh)
    res["valid_frac"] = float(np.mean(bad_t / nh))

    s_vf = min(-np.log10(max(res["vf_nrmse"], 1e-6)), 6)
    s_ro = min(-np.log10(max(res["rollout_nrmse"], 1e-6)), 6)
    res["score"] = float(0.5 * s_vf + 0.5 * s_ro + 0.5 * res["valid_frac"] - PARSIMONY * res["n_terms"])
    if reveal:
        res["truth"] = truth["rhs"]
        res.update(structure_metrics(rhs, truth["rhs"], names))
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset")
    p.add_argument("candidate", help="JSON file with {'rhs': {...}}, or 'truth' to score the true model")
    p.add_argument("--reveal", action="store_true", help="include structure metrics vs ground truth")
    a = p.parse_args()
    if a.candidate == "truth":
        cand = json.loads((Path(a.dataset) / "hidden" / "truth.json").read_text())
    else:
        cand = json.loads(Path(a.candidate).read_text())
    print(json.dumps(evaluate(a.dataset, cand, a.reveal), indent=2, default=str))


if __name__ == "__main__":
    main()
