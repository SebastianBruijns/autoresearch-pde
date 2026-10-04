"""Residual decomposition: does the model leave only noise in its residual?

    audit(meta, data, rhs) -> list[Finding]     (stage "model"; rhs coefficients are used as given)

Residual (weak form in time, strong in space; see _prepare). For a smooth compact test function phi(t) of half-width
m samples, r(t_c) = [-sum phi'(t - t_c) U(t) - sum phi(t - t_c) f(U(t))] / sum phi, so the time derivative of the
noisy data is never taken and both sides see the same time filter (no Savitzky-Golay bias). For periodic 1-D PDEs the
same is done in x with a bump whose width follows the resolved scale (weakform's corner wavenumber); spatial
derivatives are spectral on the low-passed field as in uq._prepare. Rows with any non-finite value are dropped
(NaN gaps remove the test functions that touch them). Records shorter than 12 steps use the strong form (uq._prepare).

Every decomposition is a PARTIAL regression: the residual is first regressed on a modest candidate library L
(toolbox.build_library: state polynomials, and for PDEs their products with u_x..u_xxxx, plus the model's own terms,
all filtered like the residual), then we ask how much of what L leaves is explained by a basis B:
    partial R^2(B | L) = (RSS_L - RSS_[L,B]) / RSS_L
minus the same quantity on a pure-noise surrogate residual (white noise of the estimated level pushed through the
same pipeline: same rows, same L and B). A missing field term or a wrong coefficient is explained by L and leaves
nothing for B: this is the tie-break against the "missing term" confound. (Requiring the time-only fit to beat the
library on its own fails when differentiation bias dominates the residual: Burgers / KS forcing were missed.)

    residual_time_only   B = cubic B-splines in t shared by all trajectories (and all x for PDEs): forcing is
                         usually common to every run and every point. A per-trajectory spline basis is also tested
                         (details["per_variable"][v]["per_traj_excess"]) at a higher threshold.
    residual_space_only  (1-D PDE) B = periodic Fourier modes in x (B-splines on non-periodic grids), shared by all
                         trajectories and times.
    residual_amplitude   residual mean square by decile of |v| relative to the noise surrogate in the same decile;
                         top decile / median deciles; scope = range of v where the residual stays near the floor.
    residual_white       the model's residual over the predicted measurement-noise floor in weakform.weak_sindy's
                         system (as used by assess on noisy PDEs), max over variables; falls back to the strong-form
                         surrogate ratio (mean nrmse / mean floor, as uq.compare_models) if the weak system fails.

Why this residual and not the plain strong form: on 12 dev seeds per case the strong form missed spatial sources on
KS (0/10 at threshold 0.10) and half of the KdV ones, fired residual_amplitude on 6/10 clean KdV sets, and its
white ratio was 9-40x the floor on clean KdV / KS (coarse dt). weakform.weak_sindy's own system cannot be used for
the t / x bases because it does not expose the test function centres.
"""
import numpy as np
from scipy.interpolate import BSpline
from scipy.signal import savgol_coeffs, savgol_filter

from . import finding, threshold

MAX_ROWS = 60000


# ----------------------------------------------------------------------------- residual on the fitting grid
def _bump(m, p=4):
    """Test function phi(s) = (1 - s^2)^p on 2m+1 samples, and d phi / d(sample index)."""
    s = np.arange(-m, m + 1) / m
    phi = (1 - s ** 2) ** p
    dphi = -2 * p * s * (1 - s ** 2) ** (p - 1) / m
    return phi, dphi


def _tfilter(A, w):
    """Valid-mode correlation of A (n_traj, nt, ...) with weights w along axis 1 -> (n_traj, nt - len(w) + 1, ...)."""
    m = len(w)
    nt = A.shape[1]
    out = np.zeros((A.shape[0], nt - m + 1) + A.shape[2:])
    for k in range(m):
        out += w[k] * A[:, k:k + nt - m + 1]
    return out


def _prepare(meta, data, rhs):
    """Residual, library and noise-surrogate residual on the fitting grid.

    Weak form in time, strong in space. With a smooth compactly supported test function phi of half-width m samples,
    r(t_c) = [-sum phi' U - sum phi f(U)] / sum phi, i.e. the time derivative is moved onto phi and both sides see the
    same time filter, so no smoothing bias enters the residual; spatial derivatives are spectral on the low-passed
    field (as in uq._prepare). Records shorter than 12 steps: uq._prepare (Savitzky-Golay derivative minus f)."""
    from .. import uq
    from ..baselines import lowpass
    from ..solvers import is_legacy_pde, smooth_space
    from ..toolbox import build_library, feature_arrays, smooth_and_differentiate, symbols
    variables = list(meta["variables"])
    nv = len(variables)
    U = np.asarray(data["U"], float)
    t = np.asarray(data["t"], float)
    pde = meta["kind"] == "pde"
    struct = uq._structure(meta, rhs)
    names = symbols(meta)
    rng = np.random.default_rng(0)
    lp_frac = 0.3

    def lp(A):
        if not pde:
            return A
        return lowpass(A, lp_frac) if is_legacy_pde(meta) else smooth_space(A, meta, lp_frac)

    # noise level (as uq.noise_floor: residual of a 7-point cubic Savitzky-Golay fit), NaN-safe
    flat = U.reshape(-1, nv)
    sd = np.nanstd(flat, axis=0) + 1e-12
    nt = U.shape[1]
    if nt > 12:
        h0 = savgol_coeffs(7, 3)[3]
        res = (U - savgol_filter(U, 7, 3, axis=1))[:, 3:-3].reshape(-1, nv)
        nrel = np.nanstd(res, axis=0) / np.sqrt(1 - h0) / sd
    else:
        nrel = np.full(nv, 0.01)
    if pde and U.ndim == 4 and (meta.get("boundary") or "periodic") == "periodic":
        # spatial spectrum tail (as weakform._noise_std): robust to coarse time sampling
        for i in range(nv):
            F = U[..., i].reshape(-1, U.shape[2])
            F = F[np.isfinite(F).all(1)]
            if len(F):
                E = (np.abs(np.fft.rfft(F, axis=-1)) ** 2).mean(0)
                nrel[i] = np.sqrt(np.median(E[int(0.75 * len(E)):]) / F.shape[1]) / sd[i]
    nrel = np.where(np.isfinite(nrel), nrel, 0.0)
    noise = rng.standard_normal(U.shape) * (nrel * sd)
    noise[~np.isfinite(U)] = np.nan

    def columns(feats, grid, terms_list):
        from ..toolbox import eval_exprs
        return [np.broadcast_to(c, grid).astype(float) for c in eval_exprs(terms_list, feats, names)]

    deg = 3 if (not pde and nv <= 3) or (pde and nv == 1) else 2
    basis = "strong" if nt < 12 else "weak"
    if basis == "strong":
        prep = uq._prepare(meta, data)
        grid = prep["grid"]
        cols = uq._structure_columns(prep, struct)
        N = prep["Y"].shape[0]
        pred = uq._predict(cols, struct, {v: [c for _, c in struct[v]] for v in variables}, variables, N)
        R = prep["Y"] - pred
        Y = prep["Y"]
        Us = prep["Us"].reshape(-1, nv)
        te = prep["te"]
        try:
            terms, lib = build_library(meta, prep["feats"], poly_degree=deg, max_deriv=4)
            lib = [np.broadcast_to(c, grid).ravel().astype(float) for c in lib]
        except Exception:  # noqa: BLE001
            terms, lib = ["1"], [np.ones(N)]
        for v in variables:
            for j, (tm, _) in enumerate(struct[v]):
                if tm not in terms:
                    terms.append(tm)
                    lib.append(cols[v][:, j])
        _, dN = smooth_and_differentiate(meta, noise, prep["window"], prep["order"], lp_frac if pde else None,
                                         prep["diff_method"])
        if nt > 12:
            dN = dN[:, 3:-3]
        if pde:
            dN = lowpass(dN, lp_frac) if np.isfinite(dN).all() else dN
        RN = dN.reshape(-1, nv)
    else:
        m = int(np.clip(nt // 40, 4, 8))
        phi, dphi = _bump(m)
        w, wd = phi / phi.sum(), dphi / phi.sum() / float(t[1] - t[0])
        Ul = lp(U)
        Nl = lp(noise)
        te = t[m:nt - m]
        feats = feature_arrays(meta, Ul, t)
        featsN = feature_arrays(meta, Ul + Nl, t)
        full = Ul.shape[:-1]
        model_terms_l = sorted({tm for v in variables for tm, _ in struct[v]})
        mcols = dict(zip(model_terms_l, columns(feats, full, model_terms_l)))
        mcolsN = dict(zip(model_terms_l, columns(featsN, full, model_terms_l)))
        F = np.stack([sum((c * mcols[tm] for tm, c in struct[v]), np.zeros(full)) for v in variables], -1)
        FN = np.stack([sum((c * mcolsN[tm] for tm, c in struct[v]), np.zeros(full)) for v in variables], -1)
        dU = -_tfilter(Ul, wd)
        Rg = dU - _tfilter(F, w)
        RNg = -_tfilter(Nl, wd) - _tfilter(FN - F, w)
        Usg = _tfilter(Ul, w)
        try:
            terms, lib = build_library(meta, feats, poly_degree=deg, max_deriv=4)
            lib = [np.broadcast_to(c, full).astype(float) for c in lib]
        except Exception:  # noqa: BLE001
            terms, lib = ["1"], [np.ones(full)]
        for tm in model_terms_l:
            if tm not in terms:
                terms.append(tm)
                lib.append(mcols[tm])
        lib = [_tfilter(c[..., None], w)[..., 0] for c in lib]
        if pde:
            Rg, RNg, dU = lp(Rg), lp(RNg), lp(dU)
            lib = [lp(c[..., None])[..., 0] for c in lib]
            if U.ndim == 4 and (meta.get("boundary") or "periodic") == "periodic":
                # weak form in x as well: average everything (consistently) against a periodic bump in x whose
                # half-width follows the resolved scale of the data (weakform's corner wavenumber), which pushes
                # the noise floor down without touching linear relations between residual and library
                from ..solvers import pde_layout
                from ..weakform import _auto_halfwidth, _corner_k
                g = pde_layout(meta)["grid"][(meta.get("spatial_dims") or ["x"])[0]]
                Uf = U[np.isfinite(U).all(axis=(2, 3))][None] if not np.isfinite(U).all() else U
                kx = _corner_k(Uf[..., 0], 2, g["dx"], True) if Uf.size else np.pi / g["dx"]
                mx = _auto_halfwidth(kx, g["dx"], 4, U.shape[2], 2, 0.06, 1.0)
                phx, _ = _bump(mx)
                phx = phx / phx.sum()

                def xf(A, ax=2):
                    return sum(wk * np.roll(A, k - mx, axis=ax) for k, wk in enumerate(phx))
                Rg, RNg, dU, Usg = xf(Rg), xf(RNg), xf(dU), xf(np.abs(Usg))
                lib = [xf(c) for c in lib]
        grid = Rg.shape[:-1]
        R, RN, Y, Us = Rg.reshape(-1, nv), RNg.reshape(-1, nv), dU.reshape(-1, nv), Usg.reshape(-1, nv)
        lib = [c.ravel() for c in lib]
        N = R.shape[0]
    Lib = np.stack(lib, 1)
    model_terms = {tm for v in variables for tm, _ in struct[v]}

    # rows: finite everywhere; interior only on non-periodic grids
    ok = np.isfinite(R).all(1) & np.isfinite(Lib).all(1) & np.isfinite(RN).all(1) & np.isfinite(Y).all(1)
    ix = np.indices(grid).reshape(len(grid), -1)
    if pde and (meta.get("boundary") or "periodic") != "periodic":
        for a in range(2, len(grid)):
            n = grid[a]
            mm = max(2, int(0.05 * n))
            ok &= (ix[a] >= mm) & (ix[a] < n - mm)
    rows = np.where(ok)[0]
    if rows.size > MAX_ROWS:
        rows = np.sort(rng.choice(rows, MAX_ROWS, replace=False))
    return {"meta": meta, "variables": variables, "R": R[rows], "RN": RN[rows], "Y": Y[rows],
            "Lib": Lib[rows], "terms": terms, "model_terms": model_terms, "struct": struct,
            "traj": ix[0][rows], "ti": ix[1][rows], "xi": ix[2][rows] if pde and len(grid) == 3 else None,
            "te": te, "grid": grid, "Us": Us[rows], "nrel": nrel, "sd": sd, "n_rows_total": int(N),
            "n_rows_used": int(rows.size), "names": names, "basis": basis,
            "ms": np.mean(Y[rows] ** 2, axis=0) + 1e-30}


# ----------------------------------------------------------------------------- regression helpers
def _orth(A):
    """Orthonormal basis of the column space of A (columns scaled first; rank-revealing via SVD)."""
    A = A / (np.linalg.norm(A, axis=0) + 1e-300)
    Uq, s, _ = np.linalg.svd(A, full_matrices=False)
    keep = s > s[0] * 1e-8 if s.size else s > 0
    return Uq[:, keep]


def _basis_beyond(QL, B):
    """Orthonormal basis of the part of B's column space that the library L does not already explain."""
    return _orth(B - QL @ (QL.T @ B))


def _partial(QL, QB, y):
    """(R2 of L, partial R2 of B given L, fitted B-part on the rows) for a single response y; QB = _basis_beyond."""
    yy = float(y @ y) + 1e-300
    e = y - QL @ (QL.T @ y)
    rss_l = float(e @ e)
    fit_b = QB @ (QB.T @ e)
    return 1 - rss_l / yy, float(fit_b @ fit_b) / (rss_l + 1e-300), fit_b


def _bspline_basis(x, lo, hi, n_int):
    k = 3
    inner = np.linspace(lo, hi, n_int + 2)
    knots = np.r_[[lo] * k, inner, [hi] * k]
    xc = np.clip(x, lo, hi - 1e-12 * max(1.0, abs(hi)))
    return BSpline.design_matrix(xc, knots, k).toarray()


def _fourier_basis(x, L, kmax):
    cols = []
    for k in range(1, kmax + 1):
        cols += [np.cos(2 * np.pi * k * x / L), np.sin(2 * np.pi * k * x / L)]
    return np.stack(cols, 1)


def _downsample(xs, ys, n=50):
    order = np.argsort(xs)
    xs, ys = np.asarray(xs)[order], np.asarray(ys)[order]
    if xs.size > n:
        idx = np.linspace(0, xs.size - 1, n).astype(int)
        xs, ys = xs[idx], ys[idx]
    return [[float(f"{a:.5g}"), float(f"{b:.4g}")] for a, b in zip(xs, ys)]


def _profile(keys, fit, coord):
    """Average the fitted B-part over rows sharing each key (time or x index) -> (coord values, profile)."""
    u, inv = np.unique(keys, return_inverse=True)
    s = np.bincount(inv, weights=fit, minlength=u.size)
    c = np.bincount(inv, minlength=u.size)
    return coord[u], s / np.maximum(c, 1)


# ----------------------------------------------------------------------------- the four statistics
def _time_stats(P):
    te, ti, traj = P["te"], P["ti"], P["traj"]
    nt = te.size
    n_int = int(np.clip(nt // 15, 6, 22))
    B_shared = _bspline_basis(te[ti], te[0], te[-1], n_int)
    n_traj = int(traj.max()) + 1
    n_int_pt = int(np.clip(nt // 30, 4, 12))
    Bt = _bspline_basis(te[ti], te[0], te[-1], n_int_pt)
    B_pt = np.concatenate([Bt * (traj == j)[:, None] for j in range(n_traj)], 1) if n_traj > 1 else None
    QL = P["QL"]
    QB = _basis_beyond(QL, B_shared)
    QB_pt = _basis_beyond(QL, B_pt) if B_pt is not None else None
    out = {}
    for i, v in enumerate(P["variables"]):
        r, rn = P["R"][:, i], P["RN"][:, i]
        r2_lib, part, fit = _partial(QL, QB, r)
        _, part_null, _ = _partial(QL, QB, rn)
        rec = {"partial_r2": part, "partial_r2_null": part_null, "excess": part - part_null, "r2_lib": r2_lib,
               "fit": fit}
        if QB_pt is not None:
            _, ppt, _ = _partial(QL, QB_pt, r)
            _, ppt0, _ = _partial(QL, QB_pt, rn)
            rec["per_traj_excess"] = ppt - ppt0
        out[v] = rec
    return out, n_int


def _space_stats(P):
    meta = P["meta"]
    from ..solvers import pde_layout
    lay = pde_layout(meta)
    g = lay["grid"][lay["spatial_dims"][0]]
    x = g["x0"] + np.arange(g["n"]) * g["dx"]
    kmax = int(np.clip(g["n"] // 16, 3, 8))
    xr = x[P["xi"]]
    if lay["boundary"] == "periodic":
        B = _fourier_basis(xr - g["x0"], g["L"], kmax)
    else:
        B = _bspline_basis(xr, x[0], x[-1], 2 * kmax)
    QL = P["QL"]
    QB = _basis_beyond(QL, B)
    out = {}
    for i, v in enumerate(P["variables"]):
        r, rn = P["R"][:, i], P["RN"][:, i]
        r2_lib, part, fit = _partial(QL, QB, r)
        _, part_null, _ = _partial(QL, QB, rn)
        out[v] = {"partial_r2": part, "partial_r2_null": part_null, "excess": part - part_null, "r2_lib": r2_lib,
                  "fit": fit}
    return out, x, kmax


def _amp_stats(P, n_bins=10):
    """Residual mean square by decile of |v| (own variable), each relative to the noise surrogate in the same decile
    (the noise floor is heteroscedastic, e.g. larger near steep fronts); statistic = top decile / median deciles.
    The model's coefficients are used as given: a biased coefficient also makes the residual grow with amplitude
    (and residual_white fires too). Re-estimating the model's own coefficients on low-amplitude rows was tried and
    rejected: on fields that are ~0 over most of the domain (KdV solitons) it fits noise (errors in variables) and
    explodes at the top decile; re-estimating on all rows absorbs most of a missing cubic term."""
    out = {}
    for i, v in enumerate(P["variables"]):
        a = np.abs(P["Us"][:, i])
        r = P["R"][:, i]
        rn = P["RN"][:, i]
        edges = np.quantile(a, np.linspace(0, 1, n_bins + 1))
        b = np.clip(np.searchsorted(edges, a, side="right") - 1, 0, n_bins - 1)
        ms = np.array([np.mean(r[b == k] ** 2) if np.any(b == k) else np.nan for k in range(n_bins)])
        msn = np.array([np.mean(rn[b == k] ** 2) if np.any(b == k) else np.nan for k in range(n_bins)])
        floor = float(np.nanmean(msn))
        med = float(np.nanmean(ms[n_bins // 2 - 1:n_bins // 2 + 1]))
        ref = max(med, floor, 1e-300)
        ratio_raw = float(ms[-1] / ref)
        rel = ms / np.maximum(msn, 1e-300)            # residual relative to the noise floor of the same decile
        ratio = float(rel[-1] / max(np.nanmean(rel[n_bins // 2 - 1:n_bins // 2 + 1]), 1.0))
        z = (r - r.mean()) / (r.std() + 1e-300)
        out[v] = {"ratio": ratio, "ratio_raw": ratio_raw, "ms_by_decile": ms, "floor_ms": floor, "edges": edges,
                  "rel_floor_by_decile": rel,
                  "excess_kurtosis": float(np.mean(z ** 4) - 3.0)}
    return out


def _white_stats(P):
    rms = np.sqrt(np.mean(P["R"] ** 2, axis=0))
    nrmse = rms / np.sqrt(P["ms"])
    floor = np.sqrt(np.mean(P["RN"] ** 2, axis=0)) / np.sqrt(P["ms"])
    return {"deriv_nrmse": dict(zip(P["variables"], map(float, nrmse))),
            "floor": dict(zip(P["variables"], map(float, floor))),
            "ratio": float(nrmse.mean() / max(floor.mean(), 1e-12)),
            "strong_ratio": float(nrmse.mean() / max(floor.mean(), 1e-12))}


def _weak_white(meta, data, rhs):
    """Residual of the GIVEN model in the weak-form system of weakform.weak_sindy (time and space integrated against
    test functions; no noisy derivatives), divided by that module's predicted measurement-noise floor, per variable
    (held-out test functions). ~1 when the model leaves only noise."""
    import sympy as sp
    from .. import repair
    from ..assess import _norm
    from ..solvers import parse
    from ..toolbox import symbols
    from ..weakform import weak_sindy
    names = symbols(meta)
    mt = {v: [str(m) for m in repair._terms(parse(rhs.get(v, "0"), names))] for v in meta["variables"]}
    allt = sorted({t for ts in mt.values() for t in ts if t != "1"})
    S = weak_sindy(meta, data, poly_degree=2 if meta["kind"] == "ode" else 3, custom_terms=allt,
                   thresholds=(0.1,), backward=False, return_system=True, seed=0)
    Th, lhs = np.asarray(S["Theta"]), np.asarray(S["lhs"])
    lib = {_norm(t, names): j for j, t in enumerate(S["terms"])}
    va = S["val_rows"]
    out = {}
    for i, v in enumerate(meta["variables"]):
        c = np.zeros(Th.shape[1])
        for tm in sp.Add.make_args(sp.expand(parse(rhs.get(v, "0"), names))):
            if tm == 0:
                continue
            co, m = tm.as_coeff_Mul()
            c[lib[_norm(str(m), names)]] += float(co)
        r = Th[va] @ c - lhs[va, i]
        fl = S["noise_floor"](c, i)
        out[v] = float(np.linalg.norm(r) / fl) if fl > 0 else float("nan")
    if not all(np.isfinite(list(out.values()))):
        raise ValueError("weak-form noise floor unavailable")
    return out


def stats(meta, data, rhs):
    P = _prepare(meta, data, rhs)
    P["QL"] = _orth(P["Lib"])                 # library basis, shared by the time and space decompositions
    out = {"P": P, "time": _time_stats(P), "amp": _amp_stats(P), "white": _white_stats(P)}
    if meta["kind"] == "pde" and P["xi"] is not None:
        out["space"] = _space_stats(P)
    return out


# ----------------------------------------------------------------------------- findings
def audit(meta, data, rhs):
    S = stats(meta, data, rhs)
    P = S["P"]
    pde = meta["kind"] == "pde"
    out = []
    common = {"n_rows_used": P["n_rows_used"], "n_rows_total": P["n_rows_total"], "basis": P["basis"]}

    # 1. time only
    T, n_int = S["time"]
    thr_t = threshold("residual_time_only", 0.06)
    thr_tc = threshold("residual_time_only_critical", 0.40)
    thr_pt = threshold("residual_time_only_per_traj", 0.30)
    best = max(T, key=lambda v: T[v]["excess"])
    rec = T[best]
    shared_ok = rec["excess"] > thr_t
    pt_v = max(T, key=lambda v: T[v].get("per_traj_excess", -1))
    pt_ok = T[pt_v].get("per_traj_excess", -1) > thr_pt
    fired = shared_ok or pt_ok
    prof_t, prof = _profile(P["ti"], rec["fit"], P["te"])
    out.append(finding(
        "residual_time_only", "model", rec["excess"], thr_t, fired,
        severity="critical" if fired and rec["excess"] > thr_tc else "warn",
        response="widen" if fired else None,
        fix={"tool": "add_forcing", "args": {"basis": "time", "variable": best, "n_knots": n_int,
                                             "profile": _downsample(prof_t, prof)}} if fired else None,
        message="residual follows a function of time only: unmodelled external forcing or drift" if fired else
        "residual has no component that depends on time only",
        details={**common, "variable": best, "n_interior_knots": n_int, "shared_across": "trajectories" +
                 (" and space" if pde else ""),
                 "per_variable": {v: {k: float(f"{T[v][k]:.4g}") for k in T[v] if k != "fit"} for v in T},
                 "fired_by": ("shared" if shared_ok else "") + (" per_trajectory" if pt_ok else ""),
                 "per_traj_threshold": thr_pt, "profile": _downsample(prof_t, prof)}))

    # 2. space only
    if "space" in S:
        X, x, kmax = S["space"]
        thr_s = threshold("residual_space_only", 0.08)
        thr_sc = threshold("residual_space_only_critical", 0.40)
        best = max(X, key=lambda v: X[v]["excess"])
        rec = X[best]
        fired = rec["excess"] > thr_s
        px, prof = _profile(P["xi"], rec["fit"], x)
        out.append(finding(
            "residual_space_only", "model", rec["excess"], thr_s, fired,
            severity="critical" if fired and rec["excess"] > thr_sc else "warn",
            response="widen" if fired else None,
            fix={"tool": "add_forcing", "args": {"basis": "space", "variable": best, "n_modes": kmax,
                                                 "profile": _downsample(px, prof)}} if fired else None,
            message=(f"residual follows a function of position only (in d{best}/dt): spatial source term or "
                     "uneven medium") if fired else "residual has no component that depends on position only",
            details={**common, "variable": best, "n_modes": kmax,
                     "per_variable": {v: {k: float(f"{X[v][k]:.4g}") for k in X[v] if k != "fit"} for v in X},
                     "profile": _downsample(px, prof)}))

    # 3. amplitude
    A = S["amp"]
    thr_a = threshold("residual_amplitude", 2.5)
    best = max(A, key=lambda v: A[v]["ratio"])
    rec = A[best]
    fired = rec["ratio"] > thr_a
    scope = None
    if fired:
        rel, edges = rec["rel_floor_by_decile"], rec["edges"]
        ref = 2.0 * max(float(np.nanmean(rel[4:6])), 1.0)
        k = 5
        while k + 1 < len(rel) and rel[k + 1] <= ref:
            k += 1
        hi = float(f"{edges[k + 1]:.4g}")
        i = P["variables"].index(best)
        lo_u = float(np.min(P["Us"][:, i]))
        rng_u = [float(f"{edges[0]:.4g}"), hi] if lo_u >= 0 else [-hi, hi]
        scope = {"variable": best, "range": rng_u}
    out.append(finding(
        "residual_amplitude", "model", rec["ratio"], thr_a, fired, severity="warn",
        response="scope" if fired else None, scope=scope,
        message=(f"residual grows at large |{best}| (top amplitude decile {rec['ratio']:.1f}x further above the noise "
                 f"floor than the median deciles): the model misses behaviour at extreme amplitudes; trust it only "
                 f"for {best} in {scope['range']}") if fired else "residual size does not depend on amplitude",
        details={**common, "variable": best,
                 "per_variable": {v: {"ratio": float(f"{A[v]['ratio']:.4g}"),
                                      "excess_kurtosis": float(f"{A[v]['excess_kurtosis']:.4g}"),
                                      "ratio_raw": float(f"{A[v]['ratio_raw']:.4g}"),
                                      "ms_rel_floor_by_decile": [float(f"{m:.4g}") for m in A[v]["rel_floor_by_decile"]],
                                      "decile_edges": [float(f"{e:.4g}") for e in A[v]["edges"]]} for v in A}}))

    # 4. white (noise-floor ratio, as in assess.grade: weak-form floor when available, else the strong-form one)
    W = S["white"]
    try:
        ww = _weak_white(meta, data, rhs)
        W = {**W, "weak_ratio": ww, "ratio": max(ww.values()), "source": "weak form (weakform.weak_sindy floor)"}
    except Exception as e:  # noqa: BLE001
        W = {**W, "source": f"strong form surrogate ({type(e).__name__})"}
    thr_w = threshold("residual_white", 1.5)
    fired = W["ratio"] >= thr_w
    out.append(finding(
        "residual_white", "model", W["ratio"], thr_w, fired, severity="warn",
        response="widen" if fired else None,
        message=(f"residual error is {W['ratio']:.1f}x the noise floor: systematic misfit remains" if fired else
                 f"residual error is at the noise floor (x{W['ratio']:.2f})"),
        details={**common, "deriv_nrmse": W["deriv_nrmse"], "noise_floor": W["floor"], "source": W["source"],
                 "weak_ratio_per_variable": W.get("weak_ratio"), "strong_ratio": W.get("strong_ratio"),
                 "noise_rel_estimate": dict(zip(P["variables"], map(float, P["nrel"])))}))
    return out
