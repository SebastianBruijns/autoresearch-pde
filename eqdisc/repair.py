"""Local structural repair of a near-correct model (STRIDE-style REMOVE / ADD / REPLACE edits).

Given a model rhs = sum_k c_k g_k per variable, try every single-term removal and every single-term
addition from a candidate pool, refitting the linear coefficients by least squares each time.
Two-stage screening: (1) a cheap held-out derivative-error screen on a row subset, then (2) full
`toolbox.validate` (rollout) for the best few edits. Returns ranked edits and the best repaired model.
"""
import itertools

import numpy as np
import sympy as sp

from . import toolbox as tb
from .solvers import parse


def _terms(expr):
    out = []
    for t in sp.Add.make_args(sp.expand(expr)):
        c, m = t.as_coeff_Mul()
        if m == 1:
            out.append(sp.Integer(1))
        else:
            out.append(m)
    return out


def default_pool(meta, degree=3, include_trig=True):
    v = meta["variables"]
    if meta["kind"] == "ode":
        pool = ["1"] + ["*".join(c) for d in range(1, degree + 1) for c in itertools.combinations_with_replacement(v, d)]
        if include_trig:
            pool += [f"{f}({x})" for x in v for f in ("sin", "cos")]
        return pool
    base = [f"{f}_{'x' * k}" if k else f for f in v for k in range(0, 5)]
    pool = list(base) + [f"{a}*{b}" for a in v for b in base] + [f"{a}*{a}*{b}" for a in v for b in base[:3]]
    return pool


def _bic(err2, n, k):
    return n * np.log(err2 + 1e-300) + k * np.log(n)


def local_search(meta, data, rhs, pool=None, max_rows=20000, top=3, window=9, lowpass_frac=0.3, seed=0):
    names = tb.symbols(meta)
    U, t = data["U"], data["t"]
    Us, dU = tb.smooth_and_differentiate(meta, U, window, 3, lowpass_frac if meta["kind"] == "pde" else None)
    sl = slice(3, -3)
    Us, dU = Us[:, sl], dU[:, sl]
    feats = tb.feature_arrays(meta, Us, t[sl])
    tr, va = tb.split_rows(meta, Us.shape[:-1], max_train=max_rows, max_val=max_rows // 4, seed=seed)
    pool = pool or default_pool(meta)
    cache = {}

    def col(term):
        key = str(term)
        if key not in cache:
            v = tb.eval_exprs([key], feats, names)[0].ravel()
            cache[key] = v if np.all(np.isfinite(v)) else None
        return cache[key]

    def fit(terms, y):
        cols = [col(tm) for tm in terms]
        if any(c is None for c in cols):
            return None, np.inf
        if not cols:
            return np.zeros(0), float(np.mean(y[va] ** 2) / np.mean(y[va] ** 2))
        A = np.stack(cols, 1)
        c, *_ = np.linalg.lstsq(A[tr], y[tr], rcond=None)
        return c, float(np.mean((A[va] @ c - y[va]) ** 2) / (np.mean(y[va] ** 2) + 1e-30))

    def expr_of(terms, c):
        return " + ".join(f"({ci:.6g})*({tm})" for ci, tm in zip(c, terms)) or "0"

    edits = []
    base_terms = {}
    for i, v in enumerate(meta["variables"]):
        y = dU[..., i].ravel()
        terms = [str(m) for m in _terms(parse(rhs.get(v, "0"), names))]
        base_terms[v] = terms
        c0, e0 = fit(terms, y)
        n = len(va)
        b0 = _bic(e0, n, len(terms))
        cands = [("remove", tm, [x for x in terms if x != tm]) for tm in terms]
        have = {str(sp.expand(parse(x, names))) for x in terms}
        cands += [("add", tm, terms + [tm]) for tm in pool if str(sp.expand(parse(tm, names))) not in have]
        for kind, tm, new in cands:
            c, e = fit(new, y)
            if c is None:
                continue
            edits.append({"var": v, "edit": kind, "term": tm, "val_err": e, "base_val_err": e0,
                          "delta_bic": float(_bic(e, n, len(new)) - b0), "n_terms": len(new),
                          "expr": expr_of(new, c)})
    edits.sort(key=lambda r: r["delta_bic"])
    improving = [e for e in edits if e["delta_bic"] < -10]
    # stage 2: full validation of the best few, applied one at a time and greedily combined per variable
    finalists = []
    for e in improving[:top]:
        model = dict(rhs)
        model[e["var"]] = e["expr"]
        finalists.append({**{k: e[k] for k in ("var", "edit", "term", "delta_bic")}, "rhs": model,
                          "validation": tb.validate(meta, data, model)})
    combined = dict(rhs)
    used = set()
    for e in improving:
        if e["var"] not in used:
            combined[e["var"]] = e["expr"]
            used.add(e["var"])
    out = {"base_terms": base_terms,
           "removal_evidence": [{"var": e["var"], "term": e["term"], "delta_bic": round(e["delta_bic"], 2)}
                                for e in edits if e["edit"] == "remove"],
           "top_edits": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items() if k != "expr"}
                         for e in edits[:12]],
           "finalists": finalists,
           "note": "delta_bic < -10: strong evidence for the edit; > +10: strong evidence against."}
    if used:
        out["combined_best_edits"] = {"rhs": combined, "validation": tb.validate(meta, data, combined)}
    return out
