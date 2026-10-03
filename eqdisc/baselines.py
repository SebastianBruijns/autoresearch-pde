"""Baseline discovery methods, all returning {"rhs": {var: expr_string}}.

    sindy  - ODE: PySINDy (polynomial library, STLSQ, threshold sweep)
             PDE: hand-built PDE library + STLSQ on smoothed spectral derivatives
    pysr   - ODE only: PySR symbolic regression of smoothed derivatives

    python -m eqdisc.baselines datasets/lorenz_n0.01_dt1_s0 --method sindy --eval
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter

from .evaluate import evaluate, load
from .solvers import spectral_derivs


# ----------------------------------------------------------------------------- shared helpers
def time_derivative(U, dt, window=9, order=3):
    """Savitzky-Golay smoothed value and d/dt along axis 1 (time)."""
    nt = U.shape[1]
    w = min(window, nt - (1 - nt % 2))
    w = w if w % 2 else w - 1
    if w <= order:
        return U, np.gradient(U, dt, axis=1)
    return (savgol_filter(U, w, order, axis=1),
            savgol_filter(U, w, order, deriv=1, delta=dt, axis=1))


def lowpass(U, frac=0.25, axis=-2):
    """Smooth periodic data by keeping the lowest `frac` of Fourier modes (Gaussian taper)."""
    U = np.moveaxis(U, axis, -1)
    Uh = np.fft.rfft(U, axis=-1)
    n = np.arange(Uh.shape[-1])
    kc = max(frac * Uh.shape[-1], 2)
    Uh *= np.exp(-(n / kc) ** 4)
    return np.moveaxis(np.fft.irfft(Uh, n=U.shape[-1], axis=-1), -1, axis)


def stlsq(Theta, y, threshold, ridge=1e-6, iters=20):
    """Sequentially thresholded ridge regression on column-normalised Theta."""
    norms = np.linalg.norm(Theta, axis=0) + 1e-12
    A = Theta / norms
    active = np.ones(A.shape[1], bool)
    xi = np.zeros(A.shape[1])
    for _ in range(iters):
        if not active.any():
            break
        As = A[:, active]
        xi_a = np.linalg.solve(As.T @ As + ridge * np.eye(As.shape[1]), As.T @ y)
        xi = np.zeros(A.shape[1])
        xi[active] = xi_a
        coef = xi / norms
        # threshold relative to the size of each term's contribution
        contrib = np.abs(xi) / (np.linalg.norm(y) + 1e-12)
        new = active & (contrib >= threshold)
        if (new == active).all():
            break
        active = new
    return xi / norms


def to_expr(coefs, names, precision=4):
    parts = [f"{c:+.{precision}g}*{n}" if n != "1" else f"{c:+.{precision}g}"
             for c, n in zip(coefs, names) if c != 0]
    return " ".join(parts).lstrip("+") if parts else "0"


def select(Theta, y, names, thresholds, n_val):
    """Fit on all but the last n_val rows, pick the sparsest model whose validation
    error is within 5% of the best, then refit on everything."""
    tr, va = slice(None, -n_val), slice(-n_val, None)
    best = []
    for th in thresholds:
        c = stlsq(Theta[tr], y[tr], th)
        err = np.linalg.norm(Theta[va] @ c - y[va]) / (np.linalg.norm(y[va]) + 1e-12)
        best.append((err, int((c != 0).sum()), th))
    emin = min(b[0] for b in best)
    ok = [b for b in best if b[0] <= 1.05 * emin + 1e-4]
    th = min(ok, key=lambda b: (b[1], b[0]))[2]
    return stlsq(Theta, y, th)


# ----------------------------------------------------------------------------- SINDy
def sindy_ode(meta, data, degree=3, thresholds=(0.01, 0.03, 0.05, 0.1, 0.2, 0.4, 0.8)):
    import pysindy as ps
    U, dt, names = data["U"], meta["dt"], meta["variables"]
    Us, dUdt = time_derivative(U, dt)
    X = Us.reshape(-1, U.shape[-1])
    Y = dUdt.reshape(-1, U.shape[-1])
    lib = ps.PolynomialLibrary(degree=degree)
    lib.fit(X)
    feats = [f.replace("^", "**").replace(" ", "*") for f in lib.get_feature_names(names)]
    Theta = lib.transform(X)
    Theta = np.asarray(Theta)
    n_val = U.shape[1]  # last trajectory as validation
    th_abs = [t * 1.0 for t in thresholds]
    rhs = {}
    for i, v in enumerate(names):
        c = select(Theta, Y[:, i], feats, [t / 10 for t in th_abs], n_val)
        rhs[v] = to_expr(c, feats)
    return {"rhs": rhs, "method": "sindy_ode"}


def pde_library(fields, D, max_poly=3, max_deriv=4):
    """D[f][k] = k-th x-derivative of field f, arrays of identical shape.
    Library: products  (monomial in fields, degree<=max_poly) x (1 or one derivative)."""
    cols, names = [], []
    monos = [()]
    for deg in range(1, max_poly + 1):
        monos += list(itertools.combinations_with_replacement(range(len(fields)), deg))
    derivs = [None] + [(i, k) for i in range(len(fields)) for k in range(1, max_deriv + 1)]
    for m in monos:
        for d in derivs:
            if d is not None and len(m) > 2:
                continue
            col = np.ones_like(D[fields[0]][0])
            name = []
            for i in m:
                col = col * D[fields[i]][0]
                name.append(fields[i])
            if d is not None:
                col = col * D[fields[d[0]]][d[1]]
                name.append(f"{fields[d[0]]}_{'x' * d[1]}")
            cols.append(col.ravel())
            names.append("*".join(name) if name else "1")
    return np.stack(cols, axis=1), names


def sindy_pde(meta, data, max_points=40000, seed=0,
              thresholds=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1)):
    U, dt, fields, L = data["U"], meta["dt"], meta["variables"], meta["L"]
    Us, Ut = time_derivative(U, dt, window=7)
    Us = lowpass(Us, 0.3)
    Ut = lowpass(Ut, 0.3)
    D = {f: [d[..., i] for d in spectral_derivs(Us, L)] for i, f in enumerate(fields)}
    # trim time edges where Savitzky-Golay is least accurate
    sl = (slice(None), slice(3, -3))
    D = {f: [d[sl] for d in v] for f, v in D.items()}
    Ut = Ut[sl]
    Theta, names = pde_library(fields, D)
    rng = np.random.default_rng(seed)
    n = Theta.shape[0]
    # validation = last trajectory; training rows sub-sampled from the others
    per = n // U.shape[0]
    tr_idx = rng.choice(n - per, size=min(max_points, n - per), replace=False)
    va_idx = rng.choice(np.arange(n - per, n), size=min(max_points // 4, per), replace=False)
    idx = np.concatenate([tr_idx, va_idx])
    rhs = {}
    for i, f in enumerate(fields):
        c = select(Theta[idx], Ut[..., i].ravel()[idx], names, thresholds, len(va_idx))
        rhs[f] = to_expr(c, names)
    return {"rhs": rhs, "method": "sindy_pde"}


def sindy(meta, data):
    return sindy_ode(meta, data) if meta["kind"] == "ode" else sindy_pde(meta, data)


# ----------------------------------------------------------------------------- PySR
def pysr_ode(meta, data, niterations=40, maxsize=20, timeout=120):
    from pysr import PySRRegressor
    U, dt, names = data["U"], meta["dt"], meta["variables"]
    Us, dUdt = time_derivative(U, dt)
    X = Us.reshape(-1, U.shape[-1])
    Y = dUdt.reshape(-1, U.shape[-1])
    sub = np.random.default_rng(0).choice(len(X), size=min(2000, len(X)), replace=False)
    safe = [f"v{i}" for i in range(len(names))]          # avoid clashes with Julia/sympy names
    rhs = {}
    for i, v in enumerate(names):
        model = PySRRegressor(niterations=niterations, maxsize=maxsize,
                              binary_operators=["+", "-", "*", "/"],
                              unary_operators=["sin", "cos", "exp"],
                              model_selection="best", timeout_in_seconds=timeout,
                              verbosity=0, progress=False, random_state=0,
                              deterministic=True, parallelism="serial")
        model.fit(X[sub], Y[sub, i], variable_names=safe)
        e = str(model.sympy())
        for j in reversed(range(len(names))):              # v10 before v1
            e = e.replace(f"v{j}", f"({names[j]})")
        rhs[v] = e
    return {"rhs": rhs, "method": "pysr_ode"}


METHODS = {"sindy": sindy, "pysr": pysr_ode}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset")
    p.add_argument("--method", choices=sorted(METHODS), default="sindy")
    p.add_argument("--out", help="write candidate JSON here")
    p.add_argument("--eval", action="store_true", help="score against hidden test (with truth reveal)")
    a = p.parse_args()
    meta, data = load(a.dataset)
    cand = METHODS[a.method](meta, data)
    if a.out:
        Path(a.out).write_text(json.dumps(cand, indent=2))
    print(json.dumps(cand, indent=2))
    if a.eval:
        r = evaluate(a.dataset, cand, reveal=True)
        print(json.dumps({k: r.get(k) for k in ["score", "vf_nrmse", "rollout_nrmse", "valid_frac",
                                                "n_terms", "f1", "coef_rel_err", "truth"]}, indent=2))


if __name__ == "__main__":
    main()
