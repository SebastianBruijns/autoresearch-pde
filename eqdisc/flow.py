"""Flow-map (integral / shooting) fitting for coarsely sampled ODE data.

When samples are too far apart for derivatives (e.g. a satellite sampled hourly with a 3.8 h orbit), fit a proposed
structure by integrating it from each observed state over one sampling interval and matching the next observed
state:  min_p  sum_i || Phi_dt(x_i; p) - x_{i+1} ||^2   (all arcs integrated together, vectorised).

    fit_flow(meta, data, {"vx": "-p0*x/r**3 + ...", ...}, max_pairs=1500)
"""
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from .solvers import parse


def _pairs(data, max_pairs, seed=0, stride=1):
    U = data["U"]
    X0, X1 = [], []
    for j in range(U.shape[0]):
        X0.append(U[j, :-stride])
        X1.append(U[j, stride:])
    X0, X1 = np.concatenate(X0), np.concatenate(X1)
    idx = np.random.default_rng(seed).choice(len(X0), min(max_pairs, len(X0)), replace=False)
    return X0[idx], X1[idx]


def make_flow_fn(variables, rhs_with_params, pnames):
    names = list(variables) + ["t"] + pnames
    exprs = [parse(rhs_with_params.get(v, "0"), names) for v in variables]
    f = sp.lambdify([sp.Symbol(n) for n in names], exprs, "numpy")

    def rhs(X, p):
        cols = [X[:, i] for i in range(X.shape[1])]
        out = f(*cols, np.zeros(len(X)), *p)
        return np.stack([np.broadcast_to(np.asarray(o, float), (len(X),)) for o in out], 1)
    return rhs


def propagate(rhs, p, X0, dt, rtol=1e-10, atol=1e-12):
    n, q = X0.shape
    sol = solve_ivp(lambda t, y: rhs(y.reshape(n, q), p).ravel(), (0, dt), X0.ravel(), rtol=rtol, atol=atol,
                    method="DOP853")
    return sol.y[:, -1].reshape(n, q)


def fit_flow(meta, data, rhs_with_params, max_pairs=1500, init=None, stride=1, seed=0):
    """Fit p0, p1, ... in rhs_with_params by one-interval shooting. Returns fitted rhs, params with 1-sigma errors
    (from the Gauss-Newton covariance) and the relative one-step prediction error (train / held-out pairs)."""
    variables = list(meta["variables"])
    allp = sorted({str(s) for e in rhs_with_params.values() for s in
                   parse(e, variables + ["t"] + [f"p{i}" for i in range(20)]).free_symbols
                   if str(s).startswith("p") and str(s)[1:].isdigit()}, key=lambda s: int(s[1:]))
    rhs = make_flow_fn(variables, rhs_with_params, allp)
    dt = meta["dt"] * stride
    X0, X1 = _pairs(data, 2 * max_pairs, seed, stride)
    tr, va = slice(0, max_pairs), slice(max_pairs, None)
    scale = X1.std(0) + 1e-30

    def resid(p, sl=tr):
        return ((propagate(rhs, p, X0[sl], dt) - X1[sl]) / scale).ravel()
    p0 = np.asarray(init if init is not None else np.ones(len(allp)), float)
    r = least_squares(resid, p0, method="lm", x_scale="jac") if allp else None
    p = r.x if r is not None else np.zeros(0)
    err = {}
    if r is not None:
        J = r.jac
        dof = max(len(r.fun) - len(p), 1)
        s2 = float(r.fun @ r.fun) / dof
        try:
            cov = np.linalg.inv(J.T @ J) * s2
            err = {k: float(np.sqrt(max(cov[i, i], 0))) for i, k in enumerate(allp)}
        except np.linalg.LinAlgError:
            err = {}
    rel = lambda sl: float(np.sqrt(np.mean(resid(p, sl) ** 2)))
    fitted = {v: str(parse(e, variables + ["t"] + allp).subs({sp.Symbol(k): float(x) for k, x in zip(allp, p)}))
              for v, e in rhs_with_params.items()}
    for v in variables:
        fitted.setdefault(v, "0")
    return {"rhs": fitted, "params": {k: float(x) for k, x in zip(allp, p)}, "param_sigma": err,
            "one_step_rel_err_train": rel(tr), "one_step_rel_err_heldout": rel(va), "n_pairs": max_pairs,
            "sampling_interval": dt}
