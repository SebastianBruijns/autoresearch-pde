"""Seed discovery program for evolution (a plain SINDy pipeline).

Contract: discover(meta, data) -> {"rhs": {var: "sympy expression"}}
  meta : dict from meta.json (kind, variables, dt, L, nx, allowed_symbols, ...)
  data : {"t": (nt,), "U": (n_traj, nt, n_vars) | (n_traj, nt, nx, n_fields), "x": (nx,) for PDEs}
Expressions may only use meta["allowed_symbols"] (PDE derivatives as u_x, u_xx, ...).
"""
import numpy as np

from eqdisc.baselines import lowpass, pde_library, select, time_derivative, to_expr
from eqdisc.solvers import spectral_derivs


def poly_library(X, names, degree=3):
    import itertools
    cols, feats = [np.ones(len(X))], ["1"]
    for deg in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(range(X.shape[1]), deg):
            cols.append(np.prod(X[:, combo], axis=1))
            feats.append("*".join(names[i] for i in combo))
    return np.stack(cols, 1), feats


def discover(meta, data):
    U, dt, names = data["U"], meta["dt"], meta["variables"]
    if meta["kind"] == "ode":
        Us, dUdt = time_derivative(U, dt)
        X, Y = Us.reshape(-1, U.shape[-1]), dUdt.reshape(-1, U.shape[-1])
        Theta, feats = poly_library(X, names, 3)
        n_val = U.shape[1]
        th = [1e-3, 3e-3, 1e-2, 3e-2, 0.1]
        return {"rhs": {v: to_expr(select(Theta, Y[:, i], feats, th, n_val), feats)
                        for i, v in enumerate(names)}}

    Us, Ut = time_derivative(U, dt, window=7)
    Us, Ut = lowpass(Us, 0.3), lowpass(Ut, 0.3)
    D = {f: [d[:, 3:-3, ..., i] for d in spectral_derivs(Us, meta["L"])] for i, f in enumerate(names)}
    Ut = Ut[:, 3:-3]
    Theta, feats = pde_library(names, D)
    rng = np.random.default_rng(0)
    n, per = Theta.shape[0], Theta.shape[0] // U.shape[0]
    idx = np.concatenate([rng.choice(n - per, min(40000, n - per), replace=False),
                          rng.choice(np.arange(n - per, n), min(10000, per), replace=False)])
    th = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1]
    return {"rhs": {f: to_expr(select(Theta[idx], Ut[..., i].ravel()[idx], feats, th, min(10000, per)), feats)
                    for i, f in enumerate(names)}}
