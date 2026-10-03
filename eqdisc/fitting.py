"""Variable-projection ("mixed") fitting of parametrised skeletons (STRIDE, arXiv 2605.17790).

A skeleton like  p0 - p1*s/(p2 + s)  has LINEAR parameters (p0, p1: the model is linear in them
once the others are fixed) and NONLINEAR ones (p2). VarPro solves the linear ones exactly by least
squares inside an outer search over the nonlinear ones. That is far more reliable than optimising
everything jointly from random starts, and it judges a skeleton by its best fit, not by a lucky start.
"""
import numpy as np
import sympy as sp
from scipy.optimize import least_squares


def split_params(exprs, pnames):
    """Return (linear, nonlinear) parameter names across a dict of sympy expressions."""
    psyms = [sp.Symbol(p) for p in pnames]
    linear = [p for p in psyms if all(sp.simplify(sp.diff(e, p, 2)) == 0 for e in exprs.values())]
    # a 'linear' parameter whose coefficient involves another linear parameter is really bilinear
    changed = True
    while changed:
        changed = False
        for p in list(linear):
            others = set(linear) - {p}
            if any(sp.diff(e, p).free_symbols & others for e in exprs.values()):
                linear.remove(p)
                changed = True
                break
    nonlinear = [p for p in psyms if p not in linear]
    return [str(p) for p in linear], [str(p) for p in nonlinear]


def varpro_fit(exprs, var_order, F, Y, pnames, init=None, n_restarts=8, seed=0, ridge=1e-10):
    """exprs: {var: sympy expr in data symbols + params}; F: {symbol: 1-D array}; Y: list of targets
    (one per var in var_order, already scaled). Returns (param dict, rel residual)."""
    lin, nonlin = split_params(exprs, pnames)
    data_names = list(F)
    syms_data = [sp.Symbol(n) for n in data_names]
    syms_q = [sp.Symbol(q) for q in nonlin]
    scale = [np.std(y) + 1e-12 for y in Y]
    # e = g0(x; q) + sum_i p_i g_i(x; q)
    parts = {}
    for v in var_order:
        e = sp.expand(exprs[v])
        gi = [sp.diff(e, sp.Symbol(p)) for p in lin]
        g0 = sp.simplify(e - sum(sp.Symbol(p) * g for p, g in zip(lin, gi))) if lin else e
        parts[v] = (sp.lambdify(syms_data + syms_q, g0, "numpy"),
                    [sp.lambdify(syms_data + syms_q, g, "numpy") for g in gi])
    args_data = [F[n] for n in data_names]
    n = len(Y[0])

    def design(q):
        rows_A, rows_b = [], []
        with np.errstate(all="ignore"):
            for k, v in enumerate(var_order):
                g0f, gfs = parts[v]
                g0 = np.broadcast_to(np.asarray(g0f(*args_data, *q), float), (n,))
                A = np.stack([np.broadcast_to(np.asarray(g(*args_data, *q), float), (n,)) for g in gfs], 1) \
                    if gfs else np.zeros((n, 0))
                rows_A.append(A / scale[k])
                rows_b.append((Y[k] - g0) / scale[k])
        return np.concatenate(rows_A), np.concatenate(rows_b)

    def solve_linear(q):
        A, b = design(q)
        if not (np.all(np.isfinite(A)) and np.all(np.isfinite(b))):
            return None, np.full(b.shape, 1e6)
        if A.shape[1] == 0:
            return np.zeros(0), -b
        w = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ b)
        return w, A @ w - b

    rng = np.random.default_rng(seed)
    if nonlin:
        starts = ([np.asarray(init, float)[[pnames.index(q) for q in nonlin]]] if init is not None else [])
        starts += [np.ones(len(nonlin))] + [rng.choice([-1, 1], len(nonlin)) * 10 ** rng.uniform(-1.5, 1, len(nonlin))
                                            for _ in range(n_restarts)]
        best = None
        for q0 in starts:
            try:
                r = least_squares(lambda q: solve_linear(q)[1], q0, method="trf", max_nfev=600)
            except Exception:  # noqa: BLE001
                continue
            if best is None or r.cost < best.cost:
                best = r
        if best is None:
            return {}, None
        q = best.x
    else:
        q = np.zeros(0)
    w, res = solve_linear(q)
    if w is None:
        return {}, None
    vals = {p: float(f"{x:.6g}") for p, x in zip(lin, w)}
    vals.update({p: float(f"{x:.6g}") for p, x in zip(nonlin, q)})
    return vals, float(np.sqrt(np.mean(res ** 2)))
