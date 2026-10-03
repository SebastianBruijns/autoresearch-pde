"""Integrators shared by the data generator and the evaluator.

The same code path integrates ground truth and candidate models, so candidates
are scored under exactly the numerics that produced the data.

PDE meta conventions (meta.json of a dataset, also accepted by every PDE helper here)
-----------------------------------------------------------------------------------
    meta["spatial_dims"]  ["x"] (default) or ["x", "y"]. Data U has shape
                          (n_traj, nt, n_x[, n_y], n_fields); spatial axes come in this order.
    meta["boundary"]      "periodic" (default) | "dirichlet" | "neumann" | "unknown"  (all sides).
                          periodic  : grid x = x0 + i*L/n, i < n (right end excluded)
                          otherwise : grid x = x0 + i*L/(n-1), i <= n-1 (both ends included)
    meta["grid"]          {"x": {"n": .., "L": .., "dx": .., "x0": ..}, "y": {...}}.  For backward
                          compatibility 1-D datasets also carry meta["L"], meta["nx"] (and legacy
                          meta.json files that only have L/nx are understood: periodic, x0 = 0).
    meta["allowed_symbols"] = derivative_symbols(fields, spatial_dims, 4): each field followed by
                          its partial derivatives up to total order 4, letters sorted
                          (u, u_x, u_y, u_xx, u_xy, u_yy, u_xxx, u_xxy, ...), then the coordinates.
                          In 1-D this is exactly the legacy list u, u_x, u_xx, u_xxx, u_xxxx, x.

Numerics
    derivatives(U, meta)          spectral for periodic grids (1-D legacy spectral_derivs, n-D FFT),
                                  high-order finite differences (one-sided near edges) otherwise.
    integrate_pde_general(...)    1-D periodic -> legacy ETDRK4 (integrate_pde, bit-identical);
                                  n-D periodic -> ETDRK4 with n-D wavenumbers;
                                  non-periodic -> method of lines (BDF, sparse FD Jacobian).
    Boundary values in the method of lines: if `boundary_data` (an array shaped like the output,
    e.g. an observed trajectory) is given, the outermost grid layer is driven by it (Dirichlet from
    data, spline-interpolated in time) whatever meta["boundary"] says. Without data: "dirichlet" /
    "unknown" hold the boundary layer at its initial values, "neumann" imposes zero normal
    derivative (boundary node solved algebraically from a one-sided stencil).
    All integrators have a wall-clock cap (max_seconds); failure/timeout/blow-up -> NaN padding.
"""
import itertools
import time
from functools import lru_cache

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations

MAX_DERIV = 4


# ----------------------------------------------------------------------------- parsing
def deriv_name(f, k):
    return f if k == 0 else f"{f}_{'x' * k}"


def pde_symbols(fields):
    return [deriv_name(f, k) for f in fields for k in range(MAX_DERIV + 1)] + ["x"]


def parse(expr, names):
    """Parse an expression string, treating every name in `names` as a plain symbol
    (so 'S', 'I', 'E', 'gamma', ... are not hijacked by sympy builtins)."""
    if isinstance(expr, sp.Basic):
        return expr
    local = {n: sp.Symbol(n) for n in names}
    return parse_expr(str(expr), local_dict=local, transformations=standard_transformations)


def _lambdify(names, exprs):
    syms = [sp.Symbol(n) for n in names]
    fns = [sp.lambdify(syms, e, modules="numpy") for e in exprs]

    def f(*args):
        shape = np.broadcast(*args).shape if args else ()
        return [np.broadcast_to(np.asarray(fn(*args), dtype=float), shape) for fn in fns]
    return f


# ----------------------------------------------------------------------------- ODEs
def make_ode_rhs(variables, rhs):
    """Return vectorised f(X) with X shape (..., n_vars) -> dX/dt of same shape."""
    names = list(variables) + ["t"]
    exprs = [parse(rhs[v], names) for v in variables]
    fn = _lambdify(names, exprs)

    def f(X, t=0.0):
        X = np.asarray(X, float)
        out = fn(*[X[..., i] for i in range(X.shape[-1])], np.broadcast_to(t, X.shape[:-1]))
        return np.stack(out, axis=-1)
    return f


class _Timeout(Exception):
    pass


def integrate_ode(variables, rhs, x0, t_eval, blowup=1e6, rtol=1e-9, atol=1e-9, max_seconds=10.0):
    """Integrate an ODE; returns (len(t_eval), n_vars), NaN-padded after blow-up, failure or timeout.
    The wall-clock cap matters: candidate models with singularities (e.g. 1/cos(x)) can make the
    adaptive solver take ever smaller steps and never finish."""
    f = make_ode_rhs(variables, rhs)
    out = np.full((len(t_eval), len(variables)), np.nan)
    ev = lambda t, y: blowup - np.max(np.abs(y))
    ev.terminal = True
    t_start = time.time()
    last = {"t": t_eval[0], "y": np.asarray(x0, float)}

    def fun(t, y):
        if time.time() - t_start > max_seconds:
            raise _Timeout
        return f(y, t)

    with np.errstate(all="ignore"):
        try:
            sol = solve_ivp(fun, (t_eval[0], t_eval[-1]), np.asarray(x0, float),
                            t_eval=t_eval, method="LSODA", rtol=rtol, atol=atol, events=ev)
            out[: sol.y.shape[1]] = sol.y.T
        except Exception:  # noqa: BLE001  (_Timeout or solver failure -> NaN = failed rollout)
            pass
    return out


# ----------------------------------------------------------------------------- PDEs
def wavenumbers(nx, L):
    return 2 * np.pi * np.fft.rfftfreq(nx, d=L / nx)


def spectral_derivs(U, L, kmax=MAX_DERIV, axis=-2):
    """Spectral x-derivatives of periodic data. Returns list [U, U_x, ..., U_x^kmax]."""
    U = np.moveaxis(np.asarray(U, float), axis, -1)
    nx = U.shape[-1]
    k = wavenumbers(nx, L)
    Uh = np.fft.rfft(U, axis=-1)
    out = [np.moveaxis(U, -1, axis)]
    for n in range(1, kmax + 1):
        m = (1j * k) ** n
        if n % 2 == 1 and nx % 2 == 0:
            m[-1] = 0.0                      # drop Nyquist mode for odd derivatives
        out.append(np.moveaxis(np.fft.irfft(m * Uh, n=nx, axis=-1), -1, axis))
    return out


def make_pde_rhs(fields, rhs, L):
    """Return f(U, x) with U shape (..., nx, n_fields) -> dU/dt (spectral derivatives)."""
    names = pde_symbols(fields)
    fn = _lambdify(names, [parse(rhs[f], names) for f in fields])

    def f(U, x):
        D = spectral_derivs(U, L)                         # each (..., nx, nf)
        args = [D[k][..., i] for i in range(len(fields)) for k in range(MAX_DERIV + 1)]
        args.append(np.broadcast_to(x, D[0].shape[:-1]))
        return np.stack(fn(*args), axis=-1)
    return f


def split_linear(fields, rhs):
    """Split rhs[f] = sum_k c_k * d^k f/dx^k  +  N(...), c_k numeric constants.
    The diagonal constant-coefficient linear part is treated exactly by ETDRK4."""
    names = pde_symbols(fields)
    lin, nonlin = {}, {}
    for f in fields:
        e = sp.expand(parse(rhs[f], names))
        own = {sp.Symbol(deriv_name(f, k)): k for k in range(MAX_DERIV + 1)}
        coeffs = np.zeros(MAX_DERIV + 1)
        rest = sp.Integer(0)
        for term in sp.Add.make_args(e):
            c, m = term.as_coeff_Mul()
            if m in own and c.is_number:
                coeffs[own[m]] += float(c)
            else:
                rest += term
        lin[f], nonlin[f] = coeffs, rest
    return lin, nonlin


def _etdrk4_coeffs(Lop, h, M=32):
    r = np.exp(2j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = h * Lop[..., None] + r
    eLR = np.exp(LR)
    Q = h * np.mean((np.exp(LR / 2) - 1) / LR, axis=-1)
    f1 = h * np.mean((-4 - LR + eLR * (4 - 3 * LR + LR ** 2)) / LR ** 3, axis=-1)
    f2 = h * np.mean((2 + LR + eLR * (-2 + LR)) / LR ** 3, axis=-1)
    f3 = h * np.mean((-4 - 3 * LR - LR ** 2 + eLR * (4 - LR)) / LR ** 3, axis=-1)
    return np.exp(h * Lop), np.exp(h * Lop / 2), Q, f1, f2, f3


def integrate_pde(fields, rhs, L, U0, t_eval, dt_sim, blowup=1e6, max_seconds=300.0, x0=0.0):
    """ETDRK4 on a periodic grid with 2/3 dealiasing.
    U0: (nx, n_fields). Returns (len(t_eval), nx, n_fields), NaN-padded after blow-up or timeout."""
    U0 = np.asarray(U0, float)
    nx, nf = U0.shape
    x = x0 + np.arange(nx) * L / nx
    t_start = time.time()
    k = wavenumbers(nx, L)
    lin, nonlin = split_linear(fields, rhs)
    Lop = np.stack([sum(lin[f][n] * (1j * k) ** n for n in range(MAX_DERIV + 1)) for f in fields], -1)
    Lop = Lop.astype(complex)
    dealias = (np.arange(k.size) < nx / 3)[:, None]
    Nfun = make_pde_rhs(fields, nonlin, L)

    def N(vh):
        return dealias * np.fft.rfft(Nfun(np.fft.irfft(vh, n=nx, axis=0), x), axis=0)

    out = np.full((len(t_eval), nx, nf), np.nan)
    out[0] = U0
    steps = np.rint(np.diff(t_eval) / dt_sim).astype(int)
    h = float(np.diff(t_eval)[0] / steps[0]) if len(steps) else dt_sim
    with np.errstate(all="ignore"):
        E, E2, Q, f1, f2, f3 = _etdrk4_coeffs(Lop, h)
        v = np.fft.rfft(U0, axis=0)
        try:
            for i, ns in enumerate(steps):
                for _ in range(ns):
                    if max_seconds is not None and time.time() - t_start > max_seconds:
                        raise _Timeout
                    Nv = N(v)
                    a = E2 * v + Q * Nv
                    Na = N(a)
                    b = E2 * v + Q * Na
                    Nb = N(b)
                    c = E2 * a + Q * (2 * Nb - Nv)
                    Nc = N(c)
                    v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
                U = np.fft.irfft(v, n=nx, axis=0)
                if not np.all(np.isfinite(U)) or np.abs(U).max() > blowup:
                    break
                out[i + 1] = U
        except Exception:
            pass
    return out


# ============================================================================= general PDE support
# 2-D grids, non-periodic boundaries. See the module docstring for the meta conventions.
BOUNDARY_TYPES = ("periodic", "dirichlet", "neumann", "unknown")


def _combos(ndim, max_order):
    """Multi-indices as sorted tuples of axis numbers, by total order: (), (0,), (1,), (0,0), (0,1), ..."""
    return [c for k in range(max_order + 1) for c in itertools.combinations_with_replacement(range(ndim), k)]


def derivative_suffixes(spatial_dims=("x",), max_order=MAX_DERIV):
    """'' (no derivative), 'x', 'y', 'xx', 'xy', 'yy', ... up to total order max_order."""
    dims = list(spatial_dims)
    return ["".join(dims[i] for i in c) for c in _combos(len(dims), max_order)]


def derivative_symbols(fields, spatial_dims=("x",), max_order=MAX_DERIV, coords=True):
    """Allowed PDE symbols: every field and its partial derivatives up to total order max_order
    (letters sorted, e.g. u_xxy), followed by the coordinate names. 1-D: identical to pde_symbols."""
    out = [f if not s else f"{f}_{s}" for f in fields for s in derivative_suffixes(spatial_dims, max_order)]
    return out + (list(spatial_dims) if coords else [])


def pde_layout(meta):
    """Normalise meta (new-style or legacy L/nx) to {"spatial_dims", "boundary", "grid"}.
    grid[d] = {"n", "L", "dx", "x0"}. Idempotent."""
    dims = list(meta.get("spatial_dims") or ["x"])
    boundary = meta.get("boundary") or "periodic"
    if boundary not in BOUNDARY_TYPES:
        raise ValueError(f"unknown boundary {boundary!r}; choose from {BOUNDARY_TYPES}")
    g = meta.get("grid") or {}
    grid = {}
    for d in dims:
        gd = dict(g.get(d) or {})
        if not gd:                                   # legacy 1-D meta: L, nx
            gd = {"n": meta["nx"], "L": meta["L"]}
        n, L = int(gd["n"]), float(gd["L"])
        dx = gd.get("dx") or (L / n if boundary == "periodic" else L / (n - 1))
        grid[d] = {"n": n, "L": L, "dx": float(dx), "x0": float(gd.get("x0", 0.0))}
    return {"spatial_dims": dims, "boundary": boundary, "grid": grid}


def is_legacy_pde(meta):
    """1-D periodic grid starting at 0: handled by the original spectral / ETDRK4 code paths."""
    lay = pde_layout(meta)
    return (len(lay["spatial_dims"]) == 1 and lay["boundary"] == "periodic"
            and lay["grid"][lay["spatial_dims"][0]]["x0"] == 0.0)


def grid_coords(meta):
    """List of 1-D coordinate arrays, one per spatial dim."""
    lay = pde_layout(meta)
    per = lay["boundary"] == "periodic"
    out = []
    for d in lay["spatial_dims"]:
        g = lay["grid"][d]
        out.append(g["x0"] + np.arange(g["n"]) * g["L"] / (g["n"] if per else g["n"] - 1))
    return out


def coordinate_arrays(meta, shape):
    """{coord name: array broadcast to `shape` (= U.shape[:-1])}."""
    lay = pde_layout(meta)
    dims = lay["spatial_dims"]
    nd = len(dims)
    out = {}
    for j, (d, c) in enumerate(zip(dims, grid_coords(lay))):
        s = [1] * nd
        s[j] = c.size
        out[d] = np.broadcast_to(c.reshape(s), shape)
    return out


# ----------------------------------------------------------------------------- finite differences
def fornberg_weights(z, x, m):
    """Fornberg (1988): weights c[j, k] for the k-th derivative (k <= m) at z using nodes x[j]."""
    n = len(x)
    c = np.zeros((n, m + 1))
    c1, c4 = 1.0, x[0] - z
    c[0, 0] = 1.0
    for i in range(1, n):
        mn = min(i, m)
        c2, c5, c4 = 1.0, c4, x[i] - z
        for j in range(i):
            c3 = x[i] - x[j]
            c2 *= c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    c[i, k] = c1 * (k * c[i - 1, k - 1] - c5 * c[i - 1, k]) / c2
                c[i, 0] = -c1 * c5 * c[i - 1, 0] / c2
            for k in range(mn, 0, -1):
                c[j, k] = (c4 * c[j, k] - k * c[j, k - 1]) / c3
            c[j, 0] = c4 * c[j, 0] / c3
        c1 = c2
    return c


def _stencil_size(k, acc):
    p = k + acc - 1
    return p if p % 2 else p + 1


@lru_cache(maxsize=256)
def fd_matrix(n, dx, k, acc=4):
    """Sparse (n, n) matrix of the k-th derivative on a uniform non-periodic grid: centred
    stencils in the interior, same-width one-sided stencils near the edges (accuracy ~acc)."""
    from scipy import sparse
    if k == 0:
        return sparse.identity(n, format="csr")
    p = min(_stencil_size(k, acc), n)
    rows, cols, vals = [], [], []
    for i in range(n):
        s = min(max(i - p // 2, 0), n - p)
        idx = np.arange(s, s + p)
        w = fornberg_weights(float(i), idx.astype(float), k)[:, k] / dx ** k
        rows += [i] * p
        cols += list(idx)
        vals += list(w)
    return sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))


def _apply_axis(M, A, axis):
    A = np.moveaxis(A, axis, 0)
    sh = A.shape
    B = (M @ A.reshape(sh[0], -1)).reshape((M.shape[0],) + sh[1:])
    return np.moveaxis(B, 0, axis)


# ----------------------------------------------------------------------------- derivatives
def _spectral_k(n, dx, last):
    k = 2 * np.pi * (np.fft.rfftfreq(n, d=dx) if last else np.fft.fftfreq(n, d=dx))
    nyq = (k.size - 1 if last else n // 2) if n % 2 == 0 else None
    return k, nyq


def derivatives(U, meta, max_order=MAX_DERIV, only=None, acc=4):
    """Spatial derivatives of U (..., n_x[, n_y], n_fields).
    Returns {suffix: array like U} for suffix in derivative_suffixes(dims, max_order) ('' = U itself),
    restricted to `only` if given. Periodic -> spectral (1-D: legacy spectral_derivs, bit-identical);
    otherwise finite differences of accuracy ~acc, one-sided near the edges."""
    lay = pde_layout(meta)
    U = np.asarray(U, float)
    dims = lay["spatial_dims"]
    nd = len(dims)
    axes = tuple(range(U.ndim - 1 - nd, U.ndim - 1))
    sufs = [s for s in derivative_suffixes(dims, max_order) if only is None or s in only]
    out = {}
    if lay["boundary"] == "periodic" and nd == 1:
        g = lay["grid"][dims[0]]
        kmax = max([len(s) for s in sufs], default=0)
        D = spectral_derivs(U, g["L"], kmax=kmax, axis=-2)
        return {s: D[len(s)] for s in sufs}
    if lay["boundary"] == "periodic":
        shape = [U.shape[a] for a in axes]
        Uh = np.fft.rfftn(U, axes=axes)
        ks = []
        for j, d in enumerate(dims):
            k, nyq = _spectral_k(lay["grid"][d]["n"], lay["grid"][d]["dx"], j == nd - 1)
            ks.append((k, nyq))
        for s in sufs:
            if not s:
                out[s] = U
                continue
            mult = 1.0
            for j, d in enumerate(dims):
                c = s.count(d)
                if c == 0:
                    continue
                k, nyq = ks[j]
                mj = (1j * k) ** c
                if c % 2 == 1 and nyq is not None:
                    mj = mj.copy()
                    mj[nyq] = 0.0
                sh = [1] * U.ndim
                sh[axes[j]] = mj.size
                mult = mult * mj.reshape(sh)
            out[s] = np.fft.irfftn(mult * Uh, s=shape, axes=axes)
        return out
    for s in sufs:
        A = U
        for j, d in enumerate(dims):
            c = s.count(d)
            if c:
                g = lay["grid"][d]
                A = _apply_axis(fd_matrix(g["n"], g["dx"], c, acc), A, axes[j])
        out[s] = A
    return out


def derivative_features(U, meta, fields, max_order=MAX_DERIV, only=None, coords=True):
    """{symbol name: array of shape U.shape[:-1]} for every derivative symbol (+ coordinates).
    `only`: optional set of symbol names to compute (others are skipped)."""
    lay = pde_layout(meta)
    dims = lay["spatial_dims"]
    sufs = None
    if only is not None:
        sufs = set()
        for name in only:
            for f in fields:
                if name == f:
                    sufs.add("")
                elif name.startswith(f + "_"):
                    sufs.add(name[len(f) + 1:])
    D = derivatives(U, lay, max_order, only=sufs)
    out = {}
    for i, f in enumerate(fields):
        for s, A in D.items():
            out[f if not s else f"{f}_{s}"] = A[..., i]
    if coords:
        out.update(coordinate_arrays(lay, np.shape(U)[:-1]))
    return out


def make_pde_rhs_general(fields, rhs, meta, max_order=MAX_DERIV):
    """Return f(U) with U (..., n_x[, n_y], n_fields) -> dU/dt, for any grid / boundary type.
    Only the derivatives actually used by the expressions are computed."""
    lay = pde_layout(meta)
    names = derivative_symbols(fields, lay["spatial_dims"], max_order)
    exprs = [parse(rhs.get(f, "0") if isinstance(rhs, dict) else rhs[f], names) for f in fields]
    used = {str(s) for e in exprs for s in e.free_symbols} & set(names)
    fn = _lambdify(names, exprs)

    def f(U):
        U = np.asarray(U, float)
        shape = U.shape[:-1]
        feats = derivative_features(U, lay, fields, max_order, only=used)
        zero = np.broadcast_to(0.0, shape)
        args = [feats[n] if n in used else zero for n in names]
        return np.stack([np.broadcast_to(a, shape) for a in fn(*args)], axis=-1)
    f.used_symbols = used
    return f


def split_linear_general(fields, rhs, spatial_dims=("x",), max_order=MAX_DERIV):
    """rhs[f] = sum_s c_s * f_s (own derivatives, numeric c_s) + N. Returns ({f: {suffix: c}}, {f: N})."""
    names = derivative_symbols(fields, spatial_dims, max_order)
    lin, nonlin = {}, {}
    for f in fields:
        e = sp.expand(parse(rhs[f], names))
        own = {sp.Symbol(f if not s else f"{f}_{s}"): s for s in derivative_suffixes(spatial_dims, max_order)}
        coeffs, rest = {}, sp.Integer(0)
        for term in sp.Add.make_args(e):
            c, m = term.as_coeff_Mul()
            if m in own and c.is_number:
                coeffs[own[m]] = coeffs.get(own[m], 0.0) + float(c)
            else:
                rest += term
        lin[f], nonlin[f] = coeffs, rest
    return lin, nonlin


# ----------------------------------------------------------------------------- integrators
def default_dt_sim(meta, U0, dt_obs):
    lay = pde_layout(meta)
    dx = min(g["dx"] for g in lay["grid"].values())
    h = min(dt_obs, 0.2 * dx / (np.nanmax(np.abs(U0)) + 1e-9))
    return dt_obs / max(1, int(np.ceil(dt_obs / h - 1e-9)))


def integrate_pde_spectral(fields, rhs, meta, U0, t_eval, dt_sim, blowup=1e6, max_seconds=120.0):
    """ETDRK4 on a periodic n-D grid with 2/3 dealiasing in every direction.
    U0: (n_x[, n_y], n_fields). Returns (len(t_eval), *U0.shape), NaN-padded on blow-up/timeout."""
    lay = pde_layout(meta)
    U0 = np.asarray(U0, float)
    dims = lay["spatial_dims"]
    nd = len(dims)
    shape, nf = U0.shape[:-1], U0.shape[-1]
    axes = tuple(range(nd))
    out = np.full((len(t_eval),) + U0.shape, np.nan)
    out[0] = U0
    if len(t_eval) < 2:
        return out
    t_start = time.time()
    with np.errstate(all="ignore"):
        try:
            ks, dmask = [], np.ones((), bool)
            for j, d in enumerate(dims):
                n, dx = lay["grid"][d]["n"], lay["grid"][d]["dx"]
                last = j == nd - 1
                k, _ = _spectral_k(n, dx, last)
                sh = [1] * nd
                sh[j] = k.size
                ks.append(k.reshape(sh))
                idx = np.arange(k.size) if last else np.abs(np.fft.fftfreq(n) * n)
                dmask = dmask & (idx < n / 3).reshape(sh)
            lin, nonlin = split_linear_general(fields, rhs, dims)
            Lop = []
            for f in fields:
                Lf = np.zeros(dmask.shape, complex)
                for s, c in lin[f].items():
                    term = c * np.ones((), complex)
                    for j, d in enumerate(dims):
                        term = term * (1j * ks[j]) ** s.count(d)
                    Lf = Lf + term
                Lop.append(np.broadcast_to(Lf, dmask.shape))
            Lop = np.stack(Lop, -1)
            dealias = dmask[..., None]
            Nfun = make_pde_rhs_general(fields, {f: str(nonlin[f]) for f in fields}, lay)

            def N(vh):
                return dealias * np.fft.rfftn(Nfun(np.fft.irfftn(vh, s=shape, axes=axes)), axes=axes)

            steps = np.maximum(np.rint(np.diff(t_eval) / dt_sim).astype(int), 1)
            h = float(np.diff(t_eval)[0] / steps[0])
            E, E2, Q, f1, f2, f3 = _etdrk4_coeffs(Lop, h)
            v = np.fft.rfftn(U0, axes=axes)
            for i, ns in enumerate(steps):
                for _ in range(ns):
                    if time.time() - t_start > max_seconds:
                        raise _Timeout
                    Nv = N(v)
                    a = E2 * v + Q * Nv
                    Na = N(a)
                    b = E2 * v + Q * Na
                    Nb = N(b)
                    c = E2 * a + Q * (2 * Nb - Nv)
                    Nc = N(c)
                    v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
                U = np.fft.irfftn(v, s=shape, axes=axes)
                if not np.all(np.isfinite(U)) or np.abs(U).max() > blowup:
                    break
                out[i + 1] = U
        except Exception:  # noqa: BLE001
            pass
    return out


def boundary_mask(meta):
    """Boolean array over the spatial grid: True on the outermost layer of every non-periodic dim."""
    lay = pde_layout(meta)
    shape = tuple(lay["grid"][d]["n"] for d in lay["spatial_dims"])
    m = np.zeros(shape, bool)
    if lay["boundary"] == "periodic":
        return m
    for j in range(len(shape)):
        sl = [slice(None)] * len(shape)
        sl[j] = 0
        m[tuple(sl)] = True
        sl[j] = -1
        m[tuple(sl)] = True
    return m


def _neumann_weights(n, acc):
    """One-sided first-derivative weights at node 0 using nodes 0..p (p = min(acc, n-1))."""
    p = min(acc, n - 1)
    return fornberg_weights(0.0, np.arange(p + 1, dtype=float), 1)[:, 1]


def integrate_pde_mol(fields, rhs, meta, U0, t_eval, boundary_data=None, rtol=1e-7, atol=1e-9,
                      method="BDF", blowup=1e6, max_seconds=60.0, acc=4):
    """Method of lines for non-periodic grids: high-order FD in space, implicit BDF in time with a
    sparse finite-difference Jacobian (handles stiff diffusion). Boundary handling: see module doc.
    U0: (n_x[, n_y], n_fields); boundary_data: None or array (len(t_eval), *U0.shape) whose
    outermost layer is imposed (Dirichlet from data). Returns (len(t_eval), *U0.shape), NaN on failure."""
    from scipy import sparse
    lay = pde_layout(meta)
    U0 = np.asarray(U0, float)
    t_eval = np.asarray(t_eval, float)
    dims = lay["spatial_dims"]
    nd = len(dims)
    shape = U0.shape
    out = np.full((len(t_eval),) + shape, np.nan)
    t_start = time.time()
    with np.errstate(all="ignore"):
        try:
            f_rhs = make_pde_rhs_general(fields, rhs, lay)
            mask = boundary_mask(lay)
            mode = lay["boundary"]
            if boundary_data is not None:
                B = np.asarray(boundary_data, float)[:len(t_eval)][:, mask]      # (nt, nb, nf)
                if B.shape[0] != len(t_eval) or not np.all(np.isfinite(B)):
                    raise ValueError("boundary_data must cover t_eval with finite values")
                if len(t_eval) >= 4:
                    from scipy.interpolate import CubicSpline
                    cs = CubicSpline(t_eval, B, axis=0)
                    g, gp = cs, cs.derivative()
                elif len(t_eval) >= 2:
                    slope = (B[-1] - B[0]) / (t_eval[-1] - t_eval[0])
                    g = lambda t: B[0] + (t - t_eval[0]) * slope
                    gp = lambda t: slope
                else:
                    g, gp = (lambda t: B[0]), (lambda t: 0.0 * B[0])
                mode = "data"
            elif mode in ("dirichlet", "unknown"):
                B0 = U0[mask].copy()
                g, gp = (lambda t: B0), (lambda t: 0.0 * B0)
                mode = "data"
            wn = {d: _neumann_weights(lay["grid"][d]["n"], acc) for d in dims}

            def impose(U, t):
                U = np.array(U, float)
                if mode == "data":
                    U[mask] = g(t)
                elif mode == "neumann":
                    for j, d in enumerate(dims):
                        w = wn[d]
                        p = len(w) - 1
                        A = np.moveaxis(U, j, 0)
                        A[0] = -np.tensordot(w[1:], A[1:p + 1], axes=(0, 0)) / w[0]
                        A[-1] = -np.tensordot(w[1:], A[-2:-p - 2:-1], axes=(0, 0)) / w[0]
                return U

            # Jacobian sparsity: kron over dims of the 1-D stencil patterns, times full field coupling
            used = f_rhs.used_symbols
            pats = []
            for j, d in enumerate(dims):
                n, dx = lay["grid"][d]["n"], lay["grid"][d]["dx"]
                kmax = 0
                for name in used:
                    for f in fields:
                        if name.startswith(f + "_"):
                            kmax = max(kmax, name[len(f) + 1:].count(d))
                P = sparse.identity(n, format="csr")
                for k in range(1, kmax + 1):
                    P = P + abs(fd_matrix(n, dx, k, acc))
                if mode == "neumann":
                    p = len(wn[d]) - 1
                    Bm = sparse.lil_matrix(sparse.identity(n))
                    Bm[0, :p + 1] = 1
                    Bm[n - 1, n - p - 1:] = 1
                    P = abs(P) @ abs(Bm.tocsr())
                pats.append((abs(P) > 0).astype(float))
            S = sparse.csr_matrix(np.ones((shape[-1], shape[-1])))
            for P in reversed(pats):
                S = sparse.kron(P, S, format="csr")
            S = (S > 0).astype(float)

            def fun(t, y):
                if time.time() - t_start > max_seconds:
                    raise _Timeout
                U = impose(y.reshape(shape), t)
                F = np.array(f_rhs(U))
                if mode == "data":
                    F[mask] = gp(t)
                elif mode == "neumann":
                    F[mask] = 0.0
                return F.ravel()

            ev = lambda t, y: blowup - np.max(np.abs(y))
            ev.terminal = True
            y0 = impose(U0, t_eval[0])
            out[0] = y0
            if len(t_eval) < 2:
                return out
            kw = {"jac_sparsity": S} if method in ("BDF", "Radau") else {}
            sol = solve_ivp(fun, (t_eval[0], t_eval[-1]), y0.ravel(), t_eval=t_eval, method=method,
                            rtol=rtol, atol=atol, events=ev, **kw)
            for i in range(sol.y.shape[1]):
                Ui = impose(sol.y[:, i].reshape(shape), sol.t[i])
                if not np.all(np.isfinite(Ui)):
                    break
                out[i] = Ui
        except Exception:  # noqa: BLE001  (_Timeout or solver failure -> NaN = failed rollout)
            pass
    return out


def integrate_pde_general(fields, rhs, meta, U0, t_eval, dt_sim=None, boundary_data=None,
                          max_seconds=120.0, **kw):
    """Dispatching PDE integrator. U0: (n_x[, n_y], n_fields). Returns (len(t_eval), *U0.shape).
    1-D periodic -> integrate_pde (legacy ETDRK4); n-D periodic -> integrate_pde_spectral;
    otherwise -> integrate_pde_mol (boundary_data drives the boundary layer if given).
    Never raises: failures / timeouts give NaN-padded output."""
    U0 = np.asarray(U0, float)
    t_eval = np.asarray(t_eval, float)
    try:
        lay = pde_layout(meta)
        if lay["boundary"] == "periodic":
            if dt_sim is None:
                dt_sim = default_dt_sim(lay, U0, float(np.diff(t_eval)[0]) if len(t_eval) > 1 else 1.0)
            if len(lay["spatial_dims"]) == 1:
                g = lay["grid"][lay["spatial_dims"][0]]
                return integrate_pde(fields, rhs, g["L"], U0, t_eval, dt_sim, max_seconds=max_seconds,
                                     x0=g["x0"], **kw)
            return integrate_pde_spectral(fields, rhs, lay, U0, t_eval, dt_sim, max_seconds=max_seconds, **kw)
        return integrate_pde_mol(fields, rhs, lay, U0, t_eval, boundary_data=boundary_data,
                                 max_seconds=max_seconds, **kw)
    except Exception:  # noqa: BLE001
        out = np.full((len(t_eval),) + U0.shape, np.nan)
        if len(t_eval):
            out[0] = U0
        return out


def smooth_space(U, meta, frac=0.3):
    """Spatial smoothing of data U (..., n_x[, n_y], n_fields).
    Periodic: keep ~frac of the Fourier modes in every direction (Gaussian taper, as baselines.lowpass).
    Non-periodic: same taper applied to the odd extension of U minus the straight line through its end
    values (a sine-series low-pass: no wrap-around jump, end values kept)."""
    lay = pde_layout(meta)
    U = np.asarray(U, float)
    nd = len(lay["spatial_dims"])
    axes = range(U.ndim - 1 - nd, U.ndim - 1)
    for ax in axes:
        n = U.shape[ax]
        if lay["boundary"] == "periodic":
            A = np.moveaxis(U, ax, -1)
            Ah = np.fft.rfft(A, axis=-1)
            m = np.arange(Ah.shape[-1])
            Ah *= np.exp(-(m / max(frac * Ah.shape[-1], 2)) ** 4)
            U = np.moveaxis(np.fft.irfft(Ah, n=n, axis=-1), -1, ax)
        else:
            A = np.moveaxis(U, ax, -1)
            xi = np.linspace(0.0, 1.0, n)
            lin = A[..., :1] + (A[..., -1:] - A[..., :1]) * xi
            R = A - lin
            ext = np.concatenate([R, -R[..., -2:0:-1]], axis=-1)          # odd, period 2(n-1)
            Eh = np.fft.rfft(ext, axis=-1)
            m = np.arange(Eh.shape[-1])
            Eh *= np.exp(-(m / max(frac * n, 2)) ** 4)
            U = np.moveaxis(np.fft.irfft(Eh, n=ext.shape[-1], axis=-1)[..., :n] + lin, -1, ax)
    return U
