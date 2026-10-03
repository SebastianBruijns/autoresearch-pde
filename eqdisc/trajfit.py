"""Fit and judge models by FORWARD SIMULATION instead of estimated derivatives.

Derivative regression (SINDy, fit_skeleton) and the weak form both read the dynamics off the samples, so they fail
when the sampling is coarse compared with the dynamics (e.g. stiff u_xxxx terms with dt = 0.5: fast modes change
a lot between frames). Here the model fills in between frames instead:

    for many start frames n:  simulate the model from U[n] for `horizon` sampling intervals  ->  compare with U[n+h]

Periodic PDEs use a batched ETDRK4 integrator whose stiff linear part (each field's own constant-coefficient
derivative terms, possibly with free parameters, e.g. p1*u_xx) is integrated exactly per Fourier mode; only modes
above the noise floor are compared. ODEs use batched fixed-step RK4. Short horizons keep chaotic systems tractable
(small coefficient changes -> small, smooth changes in the prediction).

    fit_trajectories(meta, data, {"u": "p0*u*u_x + p1*u_xx + p2*u_xxxx"})  -> fitted rhs, params, one-step skill
    one_step_error(meta, data, rhs)                                        -> held-out one-step error of a model
"""
import time

import numpy as np
import sympy as sp
from scipy.optimize import least_squares

from . import solvers
from .solvers import MAX_DERIV, derivative_suffixes, derivative_symbols, parse


# ----------------------------------------------------------------------------- data split / modes
def _transitions(data, horizon, max_train, max_test, seed=0):
    """(train, test) lists of (traj, start index); held out = last trajectory, or last 25% of time if only one."""
    U = data["U"]
    ntr, nt = U.shape[0], U.shape[1]
    rng = np.random.default_rng(seed)
    if ntr > 1:
        tr = [(i, n) for i in range(ntr - 1) for n in range(nt - horizon)]
        te = [(ntr - 1, n) for n in range(nt - horizon)]
    else:
        cut = int(0.75 * nt)
        tr = [(0, n) for n in range(cut - horizon)]
        te = [(0, n) for n in range(cut, nt - horizon)]

    def pick(lst, k):
        if len(lst) <= k:
            return lst
        idx = np.sort(rng.choice(len(lst), k, replace=False))
        return [lst[i] for i in idx]
    return pick(tr, max_train), pick(te, max_test)


def _pde_grid(meta):
    lay = solvers.pde_layout(meta)
    if lay["boundary"] != "periodic":
        raise ValueError("fit_trajectories supports ODEs and PERIODIC PDEs (non-periodic PDEs: not yet)")
    dims = lay["spatial_dims"]
    nd = len(dims)
    ks, dmask, shape = [], np.ones((), bool), []
    for j, d in enumerate(dims):
        n, dx = lay["grid"][d]["n"], lay["grid"][d]["dx"]
        last = j == nd - 1
        k, _ = solvers._spectral_k(n, dx, last)
        sh = [1] * nd
        sh[j] = k.size
        ks.append(k.reshape(sh))
        idx = np.arange(k.size) if last else np.abs(np.fft.fftfreq(n) * n)
        dmask = dmask & (idx < n / 3).reshape(sh)
        shape.append(n)
    return lay, dims, ks, dmask, tuple(shape)


def _signal_mask(data, dims, ks, dmask, frac=10.0):
    """Fourier modes whose mean power is > frac x the noise floor (median power of the top quarter of |k|)."""
    U = data["U"]
    nd = len(dims)
    axes = tuple(range(U.ndim - 1 - nd, U.ndim - 1))
    P = (np.abs(np.fft.rfftn(U, axes=axes)) ** 2).mean(axis=(0, 1, -1))
    kmag = np.sqrt(sum(np.broadcast_to(k, P.shape) ** 2 for k in ks))
    kmax = kmag.max()
    floor = np.median(P[kmag > 0.75 * kmax]) + 1e-30
    sig = (P > frac * floor) & np.broadcast_to(dmask, P.shape)
    kc = kmag[sig].max() if sig.any() else 0.25 * kmax
    return (kmag <= kc) & np.broadcast_to(dmask, P.shape), float(kc)


# ----------------------------------------------------------------------------- symbolic preparation
def _params(rhs_with_params, names):
    ps = sorted({str(s) for e in rhs_with_params.values() for s in parse(e, names + [f"p{i}" for i in range(40)]).free_symbols
                 if str(s).startswith("p") and str(s)[1:].isdigit()}, key=lambda s: int(s[1:]))
    return ps


def _prep_pde(meta, rhs_with_params, pnames):
    """Split each field's rhs into  sum_s c_s(p) * f_s  (own derivatives; linear, integrated exactly) + N(fields, p)."""
    fields = meta["variables"]
    lay = solvers.pde_layout(meta)
    dims = lay["spatial_dims"]
    names = derivative_symbols(fields, dims, MAX_DERIV)
    psyms = [sp.Symbol(p) for p in pnames]
    field_syms = {sp.Symbol(n) for n in names}
    lin, nl, used = {}, {}, set()
    for f in fields:
        e = sp.expand(parse(rhs_with_params.get(f, "0"), names + pnames))
        own = {sp.Symbol(f if not s else f"{f}_{s}"): s for s in derivative_suffixes(dims, MAX_DERIV)}
        coeffs, rest = {}, sp.Integer(0)
        for term in sp.Add.make_args(e):
            c, m = term.as_independent(*field_syms, as_Add=False)
            if m in own:
                coeffs[own[m]] = coeffs.get(own[m], 0) + c
            else:
                rest += term
        lin[f] = {s: sp.lambdify(psyms, c, "numpy") for s, c in coeffs.items()}
        used |= {str(s) for s in rest.free_symbols} & set(names)
        nl[f] = rest
    used = sorted(used)
    nl_fun = {f: sp.lambdify([sp.Symbol(n) for n in used] + psyms, nl[f], "numpy") for f in fields}
    return lin, nl_fun, used


def _prep_ode(meta, rhs_with_params, pnames):
    xs = meta["variables"]
    names = xs + ["t"]
    return {x: sp.lambdify([sp.Symbol(n) for n in names] + [sp.Symbol(p) for p in pnames],
                           parse(rhs_with_params.get(x, "0"), names + pnames), "numpy") for x in xs}


# ----------------------------------------------------------------------------- batched integrators
def _pde_stepper(meta, prep, grid, h, nsub):
    lin, nl_fun, used = prep
    lay, dims, ks, dmask, shape = grid
    fields = meta["variables"]
    nd = len(dims)

    def run(V0, p, horizon):
        """V0: (B, *shape, nf) physical -> (B, *shape, nf) after `horizon` intervals of h*nsub."""
        axes = tuple(range(1, 1 + nd))
        Lop = []
        for f in fields:
            Lf = np.zeros(dmask.shape, complex)
            for s, cf in lin[f].items():
                term = complex(cf(*p))
                for j, d in enumerate(dims):
                    term = term * (1j * ks[j]) ** s.count(d)
                Lf = Lf + term
            Lop.append(np.broadcast_to(Lf, dmask.shape))
        Lop = np.stack(Lop, -1)[None]
        dealias = dmask[None, ..., None]
        E, E2, Q, f1, f2, f3 = solvers._etdrk4_coeffs(Lop, h)

        def N(vh):
            U = np.fft.irfftn(vh, s=shape, axes=axes)
            feats = solvers.derivative_features(U, lay, fields, only=set(used), coords=True) if used else {}
            out = np.stack([np.broadcast_to(nl_fun[f](*[feats[n] for n in used], *p), U.shape[:-1]) for f in fields], -1)
            return dealias * np.fft.rfftn(out, axes=axes)

        v = np.fft.rfftn(V0, axes=axes)
        with np.errstate(all="ignore"):
            for _ in range(horizon * nsub):
                Nv = N(v)
                a = E2 * v + Q * Nv
                Na = N(a)
                b = E2 * v + Q * Na
                Nb = N(b)
                c = E2 * a + Q * (2 * Nb - Nv)
                Nc = N(c)
                v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        return v                                      # Fourier coefficients (compared on signal modes)
    return run


def _ode_stepper(meta, funs, h, nsub):
    xs = meta["variables"]

    def f(X, t, p):
        return np.stack([np.broadcast_to(funs[x](*[X[:, i] for i in range(len(xs))], t, *p), X.shape[:1])
                         for x in xs], 1)

    def run(X0, t0, p, horizon):
        X, t = X0.copy(), t0.copy()
        with np.errstate(all="ignore"):
            for _ in range(horizon * nsub):
                k1 = f(X, t, p)
                k2 = f(X + h / 2 * k1, t + h / 2, p)
                k3 = f(X + h / 2 * k2, t + h / 2, p)
                k4 = f(X + h * k3, t + h, p)
                X = X + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                t = t + h
        return X
    return run


# ----------------------------------------------------------------------------- core
def _auto_horizon(meta, data):
    """ODEs: enough sampling steps that the state changes ~20% (well above typical noise) over one simulation."""
    from .toolbox import diagnose
    step = max(diagnose(meta, data).get("mean_change_per_step_rel", {}).values() or [0.2])
    return int(np.clip(round(0.2 / max(step, 1e-6)), 1, max(1, data["U"].shape[1] // 4)))


def _smoothed_states(data, window=9):
    """Savitzky-Golay smoothed trajectories: simulation START states (noisy starts would reward models that merely
    denoise, e.g. a huge decay rate snapping a variable onto its trend). Targets stay raw (unbiased)."""
    from scipy.signal import savgol_filter
    U = data["U"]
    w = min(window, U.shape[1] - (1 - U.shape[1] % 2))
    w = w if w % 2 else w - 1
    return savgol_filter(U, w, 3, axis=1) if w > 3 else U


class _Problem:
    def __init__(self, meta, data, rhs_with_params, horizon=None, nsub=None, max_train=None, max_test=None, seed=0):
        self.pde = meta["kind"] == "pde"
        if horizon is None:
            horizon = 1 if self.pde else _auto_horizon(meta, data)
        self.meta, self.data, self.horizon = meta, data, int(horizon)
        self.starts = data["U"] if self.pde else _smoothed_states(data)
        names = derivative_symbols(meta["variables"], solvers.pde_layout(meta)["spatial_dims"]) if self.pde \
            else meta["variables"] + ["t"]
        self.pnames = _params(rhs_with_params, names)
        U, t = data["U"], data["t"]
        dt = float(t[1] - t[0])
        self.nsub = int(nsub or (5 if self.pde else 10))
        h = dt / self.nsub
        if self.pde:          # keep each residual evaluation affordable: ~1e6 grid values per batch
            npts = int(np.prod(U.shape[2:]))
            budget = int(np.clip(1e6 / npts, 10, 150))
            mt, mte = max_train or budget, max_test or max(10, budget // 2)
        else:
            mt, mte = max_train or 600, max_test or 300
        self.train, self.test = _transitions(data, self.horizon, mt, mte, seed)
        if self.pde:
            self.grid = _pde_grid(meta)
            lay, dims, ks, dmask, shape = self.grid
            self.mask, self.kc = _signal_mask(data, dims, ks, dmask)
            self.axes = tuple(range(1, 1 + len(dims)))
            self.run = _pde_stepper(meta, _prep_pde(meta, rhs_with_params, self.pnames), self.grid, h, self.nsub)
            self.make = lambda nsub: _pde_stepper(meta, _prep_pde(meta, rhs_with_params, self.pnames), self.grid,
                                                  dt / nsub, nsub)
        else:
            self.run = _ode_stepper(meta, _prep_ode(meta, rhs_with_params, self.pnames), h, self.nsub)
            self.make = lambda nsub: _ode_stepper(meta, _prep_ode(meta, rhs_with_params, self.pnames), dt / nsub, nsub)
        self.scale = np.array([U[..., i].std() + 1e-12 for i in range(U.shape[-1])])

    def _batch(self, pairs):
        U, t = self.data["U"], self.data["t"]
        X0 = np.stack([self.starts[i, n] for i, n in pairs])
        X1 = np.stack([U[i, n + self.horizon] for i, n in pairs])
        T0 = np.array([t[n] for _, n in pairs], float)
        return X0, X1, T0

    def residual(self, p, pairs, run=None):
        run = run or self.run
        X0, X1, T0 = self._batch(pairs)
        if self.pde:
            V = run(X0, p, self.horizon)
            V1 = np.fft.rfftn(X1, axes=self.axes)
            V0 = np.fft.rfftn(X0, axes=self.axes)
            m = self.mask[None, ..., None]
            r, base = (V - V1) / self.scale, (V0 - V1) / self.scale
            r, base = r[np.broadcast_to(m, r.shape)], base[np.broadcast_to(m, base.shape)]
            r = np.concatenate([r.real, r.imag])
            base = np.concatenate([base.real, base.imag])
        else:
            Y = run(X0, T0, p, self.horizon)
            r, base = ((Y - X1) / self.scale).ravel(), ((X0 - X1) / self.scale).ravel()
        r = np.where(np.isfinite(r), r, 1e3)
        return r, base

    def skill(self, p, pairs, run=None):
        r, base = self.residual(p, pairs, run)
        e = float(np.sqrt(np.mean(r ** 2)) / (np.sqrt(np.mean(base ** 2)) + 1e-30))
        return e


def _numeric_rhs(rhs_with_params, pnames, vals, names):
    sub = {sp.Symbol(p): float(f"{v:.6g}") for p, v in zip(pnames, vals)}
    return {k: str(parse(e, names + pnames).xreplace(sub)) for k, e in rhs_with_params.items()}


def fit_trajectories(meta, data, rhs_with_params, init=None, horizon=None, substeps=None, max_nfev=80,
                     max_train=None, max_test=None, seed=0):
    """Fit free parameters p0, p1, ... of a structure by matching `horizon`-step forward simulations to the data.
    init: list of starting values (default: fit_skeleton on derivatives, else ones). horizon: sampling steps per
    simulation (default: 1 for PDEs; for ODEs enough steps for a ~20% change, starting from smoothed states)."""
    t0 = time.time()
    prob = _Problem(meta, data, rhs_with_params, horizon, substeps, max_train, max_test, seed)
    names = derivative_symbols(meta["variables"], solvers.pde_layout(meta)["spatial_dims"]) if prob.pde \
        else meta["variables"] + ["t"]
    if not prob.pnames:
        return {"error": "no free parameters p0, p1, ... in rhs_with_params; use one_step_error to judge a fixed model"}
    if init is None:
        try:
            from .toolbox import fit_skeleton
            sk = fit_skeleton(meta, data, rhs_with_params)
            init = [sk["params"].get(p, 1.0) for p in prob.pnames]
        except Exception:  # noqa: BLE001
            init = [1.0] * len(prob.pnames)
    init = np.asarray(init, float)
    sol = least_squares(lambda p: prob.residual(p, prob.train)[0], init, method="trf", x_scale="jac",
                        max_nfev=max_nfev, diff_step=1e-4)
    p = sol.x
    err_tr, err_te = prob.skill(p, prob.train), prob.skill(p, prob.test)
    err_init = prob.skill(init, prob.test)
    fine = prob.make(prob.nsub * 2)               # integrator check: halve the sub-step
    err_fine = prob.skill(p, prob.test, fine)
    rhs = _numeric_rhs(rhs_with_params, prob.pnames, p, names)
    out = {"rhs": rhs, "params": {k: float(f"{v:.6g}") for k, v in zip(prob.pnames, p)},
           "init_params": {k: float(f"{v:.6g}") for k, v in zip(prob.pnames, init)},
           "one_step_rel_err_train": round(err_tr, 4), "one_step_rel_err_heldout": round(err_te, 4),
           "heldout_err_at_init": round(err_init, 4),
           "integrator_check": {"heldout_err_half_substep": round(err_fine, 4),
                                "ok": bool(abs(err_fine - err_te) <= 0.02 * max(err_te, 1e-3) + 1e-4)},
           "horizon_steps": prob.horizon, "substeps": prob.nsub, "n_train": len(prob.train), "n_test": len(prob.test),
           "converged": bool(sol.success), "nfev": int(sol.nfev), "seconds": round(time.time() - t0, 1),
           "note": ("one_step_rel_err = error of the simulated next frame / error of 'no change'; 0 = perfect, "
                    "1 = no better than persistence. Compared on modes above the noise floor" if prob.pde else
                    "one_step_rel_err = error of the simulated next sample / error of 'no change'")}
    if prob.pde:
        out["compared_up_to_k"] = round(prob.kc, 4)
    return out


def one_step_error(meta, data, rhs, horizon=None, substeps=None, max_test=None, seed=0):
    """Held-out one-step (or `horizon`-step) prediction error of a fixed model, relative to 'no change'."""
    prob = _Problem(meta, data, rhs, horizon, substeps, max_train=1, max_test=max_test, seed=seed)
    return prob.skill(np.zeros(0), prob.test)


def coarse_sampling(meta, data):
    """True when derivative-based statistics are unreliable: the state changes a lot between samples."""
    from .toolbox import diagnose
    d = diagnose(meta, data)
    return max(d.get("mean_change_per_step_rel", {}).values() or [0]) > 0.15
