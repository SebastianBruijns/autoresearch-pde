"""Slice consistency of a fitted model (WS2): a correct equation has the same coefficients on every slice of the data.

    audit(meta, data, rhs) -> list[Finding]     ids: slice_trajectory, slice_time, slice_space, slice_amplitude
    combined_intervals(findings) -> {"var:term": [lo, hi]}   widest random-effects interval over the slice findings

The STRUCTURE of `rhs` is kept fixed; its linear coefficients are refitted separately on each slice of the data, in
the weak form (integrals of the data against compactly supported test functions, reusing eqdisc/weakform.py), so no
noisy derivative is ever taken and every test function belongs to exactly one slice:

    trajectory   one slice per trajectory                                   (>= 2 trajectories; >= 3 to be meaningful)
    time         contiguous time blocks (3, or 2 if short)                  (test-function support inside the block)
    space        left / right half of the domain (1-D PDEs)                 (support inside the half, also if periodic)
    amplitude    terciles of the local amplitude (weighted mean of |u| over the test-function support); every
                 |variable| (and, with >1 variable, the RMS-scaled state norm) is tried, Bonferroni-corrected

Per slice and coefficient: least-squares estimate and a standard error = max(cluster-robust CR1, delete-one-cluster
jackknife), clusters = blocks of test functions one support-width long in t (and x), so overlapping test functions
are not counted as independent and a few high-leverage clusters (a short transient carrying all the information,
e.g. Hopf relaxing onto its limit cycle) do not make the se over-confident. Polynomial columns are evaluated with
Hermite-debiased powers (E[He_k(u + noise)] = u^k). A slice is skipped for a coefficient when it cannot identify it:
LHS signal below slice_min_snr x its white-noise level (saturated, near-stationary phases), variance inflation factor
above slice_max_vif, or predicted errors-in-variables bias above slice_max_eiv.
Across slices: Cochran's Q, I^2 and the DerSimonian-Laird tau^2. A slicing fires when, for some coefficient,
    I^2 > slice_i2  AND  Bonferroni-adjusted p(Q) < slice_alpha  AND  tau/|pooled| > slice_rel_tau
(the last gate keeps large data sets from flagging immaterial, numerically induced differences of ~1%).
Severity is critical when a firing coefficient has I^2 > slice_i2_critical and its slice estimates change sign or
span more than slice_rel_critical of its value.

Every finding carries details["re_intervals"] = {"var:term": [lo, hi]} (term naming as assess.py / uq._structure):
the random-effects 90% interval  b_full +- 1.645 sqrt(se_full^2 + tau^2), with b_full, se_full the whole-data fit
and cluster-robust se, and tau^2 counted only when the heterogeneity is significant (DL tau^2 is noise under the
null); on clean data it matches the bootstrap intervals. details["i2"] holds I^2 per coefficient.
"""
import math

import numpy as np
import sympy as sp
from scipy import stats

from . import finding, threshold
from .. import toolbox as tb
from .. import weakform as wf
from ..solvers import parse
from ..uq import _structure

Z90 = 1.6448536269514722
JACKKNIFE = True
X_MIN_DIV = 16             # spatial half-width >= nx/16: narrow test functions on noise-free data give
                           # amplitude-dependent quadrature bias in high-order derivative terms
CAUSES = {
    "slice_trajectory": "coefficients differ between runs: a hidden parameter varies between trajectories",
    "slice_time": "coefficients drift over time: forcing, drift or a missing variable",
    "slice_space": "coefficients depend on position, or a boundary artefact",
    "slice_amplitude": "coefficients change at large amplitude: a missing nonlinearity at the extremes",
}


def _thr(fid=None):
    rel = threshold("slice_amplitude_rel_tau", 0.02) if fid == "slice_amplitude" else threshold("slice_rel_tau", 0.02)
    return {"i2": threshold("slice_i2", 0.75), "alpha": threshold("slice_alpha", 0.01),
            "rel_tau": rel, "i2_crit": threshold("slice_i2_critical", 0.9),
            "rel_crit": threshold("slice_rel_critical", 0.3), "min_clusters": threshold("slice_min_clusters", 4)}


# ----------------------------------------------------------------------------- weak-form system with row metadata
def _fill_nan(U):
    """Linear interpolation of NaNs along time (only used for derivative/smoothing evaluation; rows whose support
    touches a NaN are dropped anyway)."""
    if np.isfinite(U).all():
        return U
    V = np.moveaxis(U, 1, -1).copy()
    flat = V.reshape(-1, V.shape[-1])
    tt = np.arange(flat.shape[1])
    for r in range(flat.shape[0]):
        ok = np.isfinite(flat[r])
        if ok.all():
            continue
        flat[r] = np.interp(tt, tt[ok], flat[r][ok]) if ok.any() else 0.0
    return np.moveaxis(flat.reshape(V.shape), -1, 1)


def _hermite(z, k, var):
    """He_k(z; var): E[He_k(u + n)] = u^k for n ~ N(0, var) (probabilists' Hermite polynomials, scaled)."""
    h0, h1 = np.ones_like(z), z
    if k == 0:
        return h0
    for j in range(1, k):
        h0, h1 = h1, z * h1 - j * var * h0
    return h1


def _debiased(expr, fields, arrays, var, coords):
    """Unbiased value of a polynomial (in the fields) term evaluated on noisy data: every power u^k is replaced by
    He_k(u; sigma_u^2), so white measurement noise does not bias nonlinear columns (errors-in-variables, e.g.
    E[(x+n)^3] = x^3 + 3 sigma^2 x, which is collinear with x on a limit cycle). None if not polynomial."""
    fs = [sp.Symbol(f) for f in fields if sp.Symbol(f) in expr.free_symbols]
    if not fs:
        return None
    try:
        P = sp.Poly(expr, *fs)
    except sp.PolynomialError:
        return None
    if any(fs_ in c.free_symbols for c in P.coeffs() for fs_ in fs):
        return None
    out = 0.0
    for mon, c in P.terms():
        term = np.ones(1)
        for f_, k in zip(fs, mon):
            if k:
                term = term * _hermite(arrays[str(f_)], k, var[str(f_)])
        if c.free_symbols:
            fn = sp.lambdify(sorted(c.free_symbols, key=str), c, "numpy")
            cv = fn(*[coords[str(z)] for z in sorted(c.free_symbols, key=str)])
        else:
            cv = float(c)
        out = out + cv * term
    return out


def _centres(n, m, periodic, stride):
    if periodic:
        return np.arange(0, n, stride)
    c = np.arange(m, n - m, stride)
    return c if c.size else np.array([n // 2])


def weak_system(meta, data, struct, p_time=4, max_deriv=4, t_div=None, x_div=12, wf_t=None, debias=True):
    """Weak-form rows for every term of `struct` on a full strided grid of test-function centres.
    Returns dict(lhs (R, nv), cols {var: (R, k)}, traj, tc, xc (R,) centre indices, amp {name: (R,)}, m_t, m_x, ...)
    with non-finite rows removed."""
    U_raw = np.asarray(data["U"], float)
    t = np.asarray(data["t"], float)
    fields = list(meta["variables"])
    names = tb.symbols(meta)
    pde = meta["kind"] == "pde"
    dims = wf._spatial_dims(meta) if pde else []
    if len(dims) > 1:
        raise NotImplementedError("slice audit supports ODEs and 1-D PDEs")
    nsp = len(dims)
    n_traj, nt = U_raw.shape[0], U_raw.shape[1]
    bad = ~np.isfinite(U_raw).all(axis=-1)                   # (n_traj, nt, [nx])
    U = _fill_nan(U_raw)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else float(meta.get("dt", 1.0))
    # --- widths: weak_sindy's automatic choice, capped so that each slice holds several independent windows
    # PDEs: wider time windows than weak_sindy (factor 6, not 3): under-resolved fast features (KdV solitons at
    # dt = 0.1) otherwise bias coefficients by a few % in an amplitude-dependent way. ODEs: weak_sindy's choice
    # (local amplitude must stay local).
    wf_t = wf_t or (6.0 if pde else 1.0)
    t_div = t_div or (8 if pde else 16)
    kt = wf._corner_k(U, 1, dt, False)
    m_t = wf._auto_halfwidth(kt, dt, p_time, nt, 3, 0.1, wf_t)
    m_t = int(max(3, min(m_t, nt // t_div)))
    if nt < 2 * m_t + 3:
        m_t = max(1, (nt - 3) // 2)
    p_x = max_deriv + 2
    if pde:
        nx = U.shape[2]
        x, dx = wf._grid(meta, data, dims[0], nx, 0)
        periodic = wf._periodic(meta, dims[0])
        kx = wf._corner_k(U, 2, dx, periodic)
        m_x = wf._auto_halfwidth(kx, dx, p_x, nx, max(4, p_x), 0.15, 3.0)
        m_x = int(max(4, nx // X_MIN_DIV, min(m_x, nx // x_div)))
    st = max(1, m_t // 2)
    c_t = _centres(nt, m_t, False, st)
    Ms_cache = {}

    def mats(order_t, alpha):
        key = (order_t, tuple(alpha))
        if key not in Ms_cache:
            Ms = [wf._axis_matrix(nt, c_t, m_t, dt, order_t, p_time, False)]
            if pde:
                Ms.append(wf._axis_matrix(nx, c_x, m_x, dx, alpha[0], p_x, periodic))
            Ms_cache[key] = Ms
        return Ms_cache[key]

    if pde:
        c_x = _centres(nx, m_x, periodic, max(1, m_x // 2))

    def integ(G, order_t=0, alpha=(0,), squared=False):
        Ms = mats(order_t, alpha)
        if squared:                     # variance of the integral of white noise weighted by G
            Ms = [M ** 2 for M in Ms]
        G = np.broadcast_to(G, U.shape[:-1])
        return np.concatenate([wf._project(np.asarray(G[j], float), Ms).ravel() for j in range(n_traj)])

    # row metadata
    if pde:
        TC, XC = np.meshgrid(c_t, c_x, indexing="ij")
        tc = np.tile(TC.ravel(), n_traj)
        xc = np.tile(XC.ravel(), n_traj)
        per = TC.size
    else:
        tc = np.tile(c_t, n_traj)
        xc = np.full(tc.shape, -1)
        per = len(c_t)
    traj = np.repeat(np.arange(n_traj), per)
    # rows whose support touches a missing sample
    Sm = [(np.abs(M) > 0).astype(float) for M in mats(0, (0,))]
    touched = np.concatenate([wf._project(bad[j].astype(float), Sm).ravel() for j in range(n_traj)]) > 0
    wsum = integ(np.ones(U.shape[:-1]))

    raw = {f: U[..., i] for i, f in enumerate(fields)}
    if pde:
        coords = {dims[0]: np.broadcast_to(x.reshape(1, 1, -1), U.shape[:-1]),
                  "t": np.broadcast_to(t.reshape(1, -1, 1), U.shape[:-1])}
        axes_info = [(1, dx, periodic)]
    else:
        coords = {"t": np.broadcast_to(t[None, :], U.shape[:-1])}
    smooth_cache = {}

    def smoothed(f):
        if f not in smooth_cache:
            smooth_cache[f] = wf._smooth(raw[f], axes_info, 0.5) if pde else raw[f]
        return smooth_cache[f]

    def dfield(F, alpha):
        return wf._diff(F, 1, alpha[0], dx, periodic) if alpha[0] else F

    def evalg(g, src):
        syms = sorted(g.free_symbols, key=str)
        if not syms:
            return np.full(U.shape[:-1], float(g))
        fn = sp.lambdify(syms, g, "numpy")
        args = [src(str(s)) if str(s) in fields else coords[str(s)] for s in syms]
        with np.errstate(all="ignore"):
            return np.broadcast_to(np.asarray(fn(*args), float), U.shape[:-1])

    lhs = np.stack([-integ(raw[f], order_t=1) for f in fields], 1)
    # rms of the LHS that white measurement noise alone would produce (slices with no signal above it are skipped:
    # near-stationary data identify only coefficient ratios, and errors-in-variables shrink their scale)
    wnorm = float(np.prod([np.linalg.norm(M[len(M) // 2]) for M in mats(1, (0,))]))
    sig = np.array([wf._noise_std(U_raw[..., i] if np.isfinite(U_raw[..., i]).all() else raw[f],
                                  periodic if pde else None) for i, f in enumerate(fields)])
    lhs_noise = sig * wnorm
    all_terms = []
    for v in fields:
        for tm, _ in struct.get(v, []):
            if tm not in all_terms:
                all_terms.append(tm)
    col, cnoise = {}, {}
    var = {f: float(sig[i]) ** 2 if debias else 0.0 for i, f in enumerate(fields)}
    var_n = {f: float(sig[i]) ** 2 for i, f in enumerate(fields)}

    def pointwise_noise(expr, alpha=(0,)):
        """Per-row variance that measurement noise adds to the integral of expr(fields) (linearised)."""
        tot = 0.0
        for f in fields:
            if sp.Symbol(f) in expr.free_symbols:
                dg = evalg(sp.diff(expr, sp.Symbol(f)), lambda f_: raw[f_])
                tot = tot + var_n[f] * integ(dg ** 2, alpha=alpha, squared=True)
        return tot

    if not pde:
        feats = tb.feature_arrays(meta, U, t)
        for tm in all_terms:
            cnoise[tm] = pointwise_noise(parse(tm, names))
            d = _debiased(parse(tm, names), fields, raw, var, coords)
            col[tm] = integ(np.asarray(d if d is not None else tb.eval_exprs([tm], feats, names)[0], float))
    else:
        gfeats = None
        for tm in all_terms:
            kind = wf._classify(parse(tm, names), fields, dims, dims)
            if kind[0] == "mono":
                cnoise[tm] = pointwise_noise(kind[1])
                d = _debiased(kind[1], fields, raw, var, coords)
                col[tm] = integ(d if d is not None else evalg(kind[1], lambda f: raw[f]))
                continue
            if kind[0] == "generic":
                if gfeats is None:
                    gfeats = wf._generic_feats(smoothed, fields, dims, dfield, coords, max_deriv)
                col[tm] = integ(np.asarray(tb.eval_exprs([str(kind[1])], gfeats, list(gfeats))[0], float))
                continue
            _, g, fi, alpha = kind
            f = fields[fi]
            order = sum(alpha)
            gsyms = {str(s_) for s_ in g.free_symbols}
            if not gsyms:
                col[tm] = (-1) ** order * float(g) * integ(raw[f], alpha=alpha)
                cnoise[tm] = var_n[f] * float(g) ** 2 * integ(np.ones(U.shape[:-1]), alpha=alpha, squared=True)
                continue
            gp = sp.Poly(g, sp.Symbol(f)) if gsyms == {f} else None
            if order == 1 and gp is not None and len(gp.terms()) == 1:
                (pw,), cf = gp.terms()[0]
                col[tm] = -integ(float(cf) * _hermite(raw[f], pw + 1, var[f]) / (pw + 1), alpha=alpha)
                cnoise[tm] = var_n[f] * integ((float(cf) * raw[f] ** pw) ** 2, alpha=alpha, squared=True)
                continue
            beta = wf._split_alpha(alpha, order // 2)
            gS = evalg(g, smoothed)
            rest = dfield(smoothed(f), tuple(a - b for a, b in zip(alpha, beta)))
            acc = 0.0
            for gam in [(k,) for k in range(beta[0] + 1)]:
                coef = math.comb(beta[0], gam[0])
                acc = acc + coef * integ(dfield(gS, gam) * rest, alpha=(beta[0] - gam[0],))
            col[tm] = (-1) ** sum(beta) * acc
    cols = {v: (np.stack([col[tm] for tm, _ in struct.get(v, [])], 1) if struct.get(v) else
                np.zeros((lhs.shape[0], 0))) for v in fields}
    # noise variance per row of each column (0 where not modelled: terms evaluated on smoothed fields)
    ncols = {v: (np.stack([np.broadcast_to(cnoise.get(tm, 0.0), lhs.shape[:1]) for tm, _ in struct.get(v, [])], 1)
                 if struct.get(v) else np.zeros((lhs.shape[0], 0))) for v in fields}
    # local amplitude measures (weighted means over each test function's support)
    amp = {}
    absU = np.abs(U)
    for i, f in enumerate(fields):
        amp[f"|{f}|"] = integ(absU[..., i]) / wsum
    if len(fields) > 1:
        rms = np.sqrt(np.nanmean(U_raw.reshape(-1, len(fields)) ** 2, axis=0)) + 1e-300
        amp["state_norm"] = integ(np.sqrt(((U / rms) ** 2).sum(-1))) / wsum
    ok = ~touched & np.isfinite(lhs).all(1)
    for v in fields:
        ok &= np.isfinite(cols[v]).all(1)
    out = {"lhs": lhs[ok], "cols": {v: c[ok] for v, c in cols.items()}, "ncols": {v: c[ok] for v, c in ncols.items()}, "traj": traj[ok], "tc": tc[ok],
           "xc": xc[ok], "amp": {k: a[ok] for k, a in amp.items()}, "m_t": m_t, "nt": nt, "n_traj": n_traj,
           "n_rows_total": int(ok.size), "n_rows_dropped": int((~ok).sum()),
           "lhs_noise": lhs_noise}
    if pde:
        out.update(m_x=m_x, nx=nx, periodic=periodic)
    # clusters: blocks of ~one support width in time (and space) -> nearly independent residuals
    tb_ = out["tc"] // max(1, 2 * m_t)
    xb_ = (out["xc"] // max(1, 2 * m_x)) if pde else np.zeros_like(tb_)
    out["cluster"] = (out["traj"] * 100000 + tb_ * 1000 + xb_).astype(np.int64)
    return out


# ----------------------------------------------------------------------------- per-slice fits and meta-analysis
def _fit(A, y, cl, Nv=None):
    """OLS with cluster-robust (CR1) standard errors. Returns (coef, se, n_clusters, vif, eiv) where eiv_j = noise
    variance of column j / its variance not explained by the other columns (errors-in-variables bias ~ eiv)."""
    k = A.shape[1]
    if k == 0 or A.shape[0] <= k:
        return np.full(k, np.nan), np.full(k, np.nan), 0, np.full(k, np.inf), np.full(k, np.inf)
    nrm = np.linalg.norm(A, axis=0) + 1e-300
    An = A / nrm
    G = An.T @ An
    try:
        Gi = np.linalg.inv(G)
    except np.linalg.LinAlgError:
        Gi = np.linalg.pinv(G)
    c = Gi @ (An.T @ y)
    e = y - An @ c
    ids, inv = np.unique(cl, return_inverse=True)
    g = len(ids)
    S = np.zeros((g, k))
    np.add.at(S, inv, An * e[:, None])
    meat = S.T @ S
    n = len(y)
    corr = g / max(g - 1, 1) * (n - 1) / max(n - k, 1)
    V = Gi @ meat @ Gi * corr
    se = np.sqrt(np.maximum(np.diag(V), 0))
    if JACKKNIFE and g >= 3:
        # delete-one-cluster jackknife: robust to a few high-leverage clusters (e.g. a short transient carrying all
        # the information about a coefficient), where CR1 is badly anti-conservative
        XtX = np.zeros((g, k, k))
        np.add.at(XtX, inv, An[:, :, None] * An[:, None, :])
        Xty = np.zeros((g, k))
        np.add.at(Xty, inv, An * y[:, None])
        tot_b = An.T @ y
        Bj = np.empty((g, k))
        for j in range(g):
            try:
                Bj[j] = np.linalg.solve(G - XtX[j], tot_b - Xty[j])
            except np.linalg.LinAlgError:
                Bj[j] = np.nan
        Bj = Bj[np.isfinite(Bj).all(1)]
        if len(Bj) >= 3:
            se = np.maximum(se, np.sqrt((len(Bj) - 1) / len(Bj) * ((Bj - Bj.mean(0)) ** 2).sum(0)))
    se = se / nrm
    vif = np.diag(Gi)
    eiv = (Nv.sum(0) / nrm ** 2 * vif) if Nv is not None else np.zeros(k)
    return c / nrm, se, g, vif, eiv


def _hetero(b, se):
    """Cochran's Q, I^2, DL tau^2, fixed and random-effects pooled estimates, for one coefficient."""
    ok = np.isfinite(b) & np.isfinite(se) & (se > 0)
    b, se = b[ok], se[ok]
    S = len(b)
    if S < 2:
        return None
    w = 1 / se ** 2
    mu = float((w * b).sum() / w.sum())
    Q = float((w * (b - mu) ** 2).sum())
    df = S - 1
    i2 = max(0.0, (Q - df) / Q) if Q > 0 else 0.0
    tau2 = max(0.0, (Q - df) / (w.sum() - (w ** 2).sum() / w.sum()))
    ws = 1 / (se ** 2 + tau2)
    mu_re = float((ws * b).sum() / ws.sum())
    se_re = float(np.sqrt(1 / ws.sum()))
    half = Z90 * math.sqrt(se_re ** 2 + tau2)
    return {"Q": Q, "df": df, "p": float(stats.chi2.sf(Q, df)), "I2": i2, "tau2": tau2, "tau": math.sqrt(tau2),
            "pooled_fe": mu, "se_fe": float(np.sqrt(1 / w.sum())), "pooled": mu_re, "se_pooled": se_re,
            "re_interval": [mu_re - half, mu_re + half],
            "rel_tau": math.sqrt(tau2) / (abs(mu_re) + 1e-300),
            "rel_range": float((b.max() - b.min()) / (abs(mu_re) + 1e-300)),
            "sign_change": bool(np.any(b - 2 * se > 0) and np.any(b + 2 * se < 0))}


def _slice_stats(sysw, struct, labels, slice_names, min_rows=None):
    """Fit every variable on each slice (labels: (R,) ints, -1 = unused). Returns per-coefficient results."""
    variables = list(sysw["cols"])
    res = {}
    for v_i, v in enumerate(variables):
        terms = [tm for tm, _ in struct.get(v, [])]
        if not terms:
            continue
        k = len(terms)
        B = np.full((len(slice_names), k), np.nan)
        SE = np.full((len(slice_names), k), np.nan)
        NC = np.zeros(len(slice_names), int)
        SNR = np.zeros(len(slice_names))
        VIF = np.full((len(slice_names), k), np.inf)
        EIV = np.full((len(slice_names), k), np.inf)
        for s in range(len(slice_names)):
            rows = np.where(labels == s)[0]
            if rows.size < max(min_rows or 0, 3 * k + 3):
                continue
            SNR[s] = np.sqrt(np.mean(sysw["lhs"][rows, v_i] ** 2)) / (sysw["lhs_noise"][v_i] + 1e-300)
            if SNR[s] < threshold("slice_min_snr", 3.0):
                continue
            c, se, g, vif, eiv = _fit(sysw["cols"][v][rows], sysw["lhs"][rows, v_i], sysw["cluster"][rows],
                                      sysw["ncols"][v][rows])
            # a coefficient whose column is (nearly) explained by the others inside the slice (e.g. x and
            # x*(x^2+y^2) on a limit cycle) is identified there only through noise: errors-in-variables bias its
            # estimate by ~eiv, so the slice is skipped for it
            bad = (vif > threshold("slice_max_vif", 100.0)) | (eiv > threshold("slice_max_eiv", 0.02))
            B[s], SE[s], NC[s], VIF[s], EIV[s] = np.where(bad, np.nan, c), np.where(bad, np.nan, se), g, vif, eiv
        for j, tm in enumerate(terms):
            res[f"{v}:{tm}"] = {"b": B[:, j], "se": SE[:, j], "n_clusters": NC, "snr": SNR, "vif": VIF[:, j], "eiv": EIV[:, j]}
    return res


def _evaluate(fid, stats_, slice_names, full, n_extra_tests=1, extra=None):
    """Fire decision + Finding for one slicing. `full`: whole-data fit (per coefficient b, se) that centres the
    random-effects interval: full estimate +- z sqrt(se_full^2 + tau^2)."""
    th = _thr(fid)
    rows, ntest = {}, 0
    for key, r in stats_.items():
        ok = r["n_clusters"] >= th["min_clusters"]
        h = _hetero(r["b"][ok], r["se"][ok])
        if h is None:
            continue
        ntest += 1
        rows[key] = (h, r)
    ntest = max(ntest * n_extra_tests, 1)
    worst, worst_key, fired_keys = None, None, []
    for key, (h, r) in rows.items():
        h["p_adj"] = min(1.0, h["p"] * ntest)
        # widen only by between-slice variance that is statistically established (DL tau^2 is noise under the null)
        h["tau2_used"] = h["tau2"] if h["p_adj"] < th["alpha"] else 0.0
        fb, fse = full.get(key, {}).get("b", [np.nan])[0], full.get(key, {}).get("se", [np.nan])[0]
        if np.isfinite(fb) and np.isfinite(fse):
            half = Z90 * math.sqrt(fse ** 2 + h["tau2_used"])
            h.update(full_fit=float(fb), se_full=float(fse), re_interval=[fb - half, fb + half])
        else:
            half = Z90 * math.sqrt(h["se_pooled"] ** 2 + h["tau2_used"])
            h["re_interval"] = [h["pooled"] - half, h["pooled"] + half]
        h["fires"] = bool(h["I2"] > th["i2"] and h["p_adj"] < th["alpha"] and h["rel_tau"] > th["rel_tau"])
        if h["fires"]:
            fired_keys.append(key)
        score = (h["fires"], h["I2"] if h["p_adj"] < th["alpha"] else 0.0, -h["p_adj"])
        if worst is None or score > worst:
            worst, worst_key = score, key
    fired = bool(fired_keys)
    stat = rows[worst_key][0]["I2"] if worst_key else 0.0
    crit = [k for k in fired_keys if rows[k][0]["I2"] > th["i2_crit"]
            and (rows[k][0]["sign_change"] or rows[k][0]["rel_range"] > th["rel_crit"])]
    severity = "critical" if crit else "warn"
    details = {
        "slices": slice_names,
        "re_intervals": {k: [float(f"{x:.6g}") for x in h["re_interval"]] for k, (h, _) in rows.items()},
        "i2": {k: round(h["I2"], 4) for k, (h, _) in rows.items()},
        "per_coefficient": {k: {"full_fit": h.get("full_fit"), "se_full": h.get("se_full"), "pooled": h["pooled"], "se_pooled": h["se_pooled"], "tau": h["tau"],
                                "Q": h["Q"], "df": h["df"], "p_adj": h["p_adj"], "rel_tau": h["rel_tau"],
                                "per_slice": [None if not np.isfinite(b) else float(f"{b:.6g}") for b in r["b"]],
                                "per_slice_se": [None if not np.isfinite(s) else float(f"{s:.4g}") for s in r["se"]],
                                "per_slice_snr": [round(float(x), 2) for x in r["snr"]],
                                "per_slice_vif": [round(float(x), 1) if np.isfinite(x) else None for x in r["vif"]],
                                "per_slice_eiv": [float(f"{x:.3g}") if np.isfinite(x) else None for x in r["eiv"]]}
                            for k, (h, r) in rows.items()},
        "inconsistent_coefficients": fired_keys, "n_tests": ntest, "thresholds": th,
    }
    if extra:
        details.update(extra)
    if fired:
        names = ", ".join(fired_keys[:4])
        msg = f"{CAUSES[fid]} (inconsistent across {len(slice_names)} slices: {names}; I2 up to {stat:.2f})."
    else:
        msg = f"coefficients are consistent across {len(slice_names)} {fid.split('_', 1)[1]} slices."
    return fired, stat, severity, details, msg, rows


# ----------------------------------------------------------------------------- slicings
def _trajectory(sysw, struct):
    n = sysw["n_traj"]
    if n < 2:
        return None
    names = [f"traj{j}" for j in range(n)]
    st = _slice_stats(sysw, struct, sysw["traj"], names)
    fired, stat, sev, det, msg, rows = _evaluate("slice_trajectory", st, names, sysw["full"])
    det["per_trajectory"] = {k: d["per_slice"] for k, d in det["per_coefficient"].items()}
    det["spread"] = {k: {"rel_tau": round(h["rel_tau"], 4), "rel_range": round(h["rel_range"], 4)}
                     for k, (h, _) in rows.items()}
    if n < 3:
        det["note"] = "only 2 trajectories: weak test"
    return finding("slice_trajectory", "model", stat, _thr()["i2"], fired, sev if fired else "info",
                   response="repair" if fired else None,
                   fix={"tool": "per_trajectory", "args": {}} if fired else None, message=msg, details=det)


def _time(sysw, struct):
    nt, m = sysw["nt"], sysw["m_t"]
    nb = 3 if nt >= 3 * (6 * m) else 2
    edges = np.linspace(0, nt, nb + 1).astype(int)
    lab = np.full(len(sysw["tc"]), -1)
    for b in range(nb):
        inside = (sysw["tc"] - m >= edges[b]) & (sysw["tc"] + m < edges[b + 1])
        lab[inside] = b
    names = [f"t[{a}:{b}]" for a, b in zip(edges[:-1], edges[1:])]
    if (np.bincount(lab[lab >= 0], minlength=nb) == 0).any():
        return None
    st = _slice_stats(sysw, struct, lab, names)
    fired, stat, sev, det, msg, _ = _evaluate("slice_time", st, names, sysw["full"])
    return finding("slice_time", "model", stat, _thr()["i2"], fired, sev if fired else "info",
                   response="widen" if fired else None, message=msg, details=det)


def _space(sysw, struct):
    if "nx" not in sysw:
        return None
    nx, m = sysw["nx"], sysw["m_x"]
    half = nx // 2
    lab = np.full(len(sysw["xc"]), -1)
    lab[(sysw["xc"] - m >= 0) & (sysw["xc"] + m < half)] = 0
    lab[(sysw["xc"] - m >= half) & (sysw["xc"] + m < nx)] = 1
    if (lab == 0).sum() == 0 or (lab == 1).sum() == 0:
        return None
    names = ["left half", "right half"]
    st = _slice_stats(sysw, struct, lab, names)
    fired, stat, sev, det, msg, _ = _evaluate("slice_space", st, names, sysw["full"],
                                              extra={"periodic": bool(sysw["periodic"])})
    return finding("slice_space", "model", stat, _thr()["i2"], fired, sev if fired else "info",
                   response="widen" if fired else None, message=msg, details=det)


def _amplitude(sysw, struct, n_bins=3):
    cands = sysw["amp"]
    results = []
    for name, a in cands.items():
        qs = np.quantile(a, np.linspace(0, 1, n_bins + 1))
        lab = np.clip(np.searchsorted(qs[1:-1], a, side="right"), 0, n_bins - 1)
        names = [f"{name} in [{qs[b]:.4g}, {qs[b + 1]:.4g}]" for b in range(n_bins)]
        st = _slice_stats(sysw, struct, lab, names)
        ev = _evaluate("slice_amplitude", st, names, sysw["full"], n_extra_tests=len(cands))
        results.append((name, qs, st, ev))
    if not results:
        return None
    # the amplitude measure with the strongest evidence
    def key(r):
        fired, stat, _, det, _, rows = r[3]
        pmin = min((h["p_adj"] for h, _ in rows.values()), default=1.0)
        return (fired, stat if pmin < _thr()["alpha"] else 0.0, -pmin)
    name, qs, st, (fired, stat, sev, det, msg, rows) = max(results, key=key)
    # scope: grow from the low-amplitude bin while coefficients stay consistent
    hi_bin = n_bins - 1
    if fired:
        hi_bin = 0
        for b in range(1, n_bins):
            okb = True
            for k in det["inconsistent_coefficients"]:
                r = st[k]
                h = _hetero(r["b"][:b + 1], r["se"][:b + 1])
                if h and h["p"] * max(det["n_tests"], 1) < det["thresholds"]["alpha"] and \
                        h["rel_tau"] > det["thresholds"]["rel_tau"]:
                    okb = False
            if not okb:
                break
            hi_bin = b
    scope = {"variable": name, "range": [float(f"{qs[0]:.4g}"), float(f"{qs[hi_bin + 1]:.4g}")]}
    det["amplitude_variable"] = name
    det["observed_range"] = [float(f"{qs[0]:.4g}"), float(f"{qs[-1]:.4g}")]
    det["candidates"] = {r[0]: {"fired": r[3][0], "max_I2": round(r[3][1], 4)} for r in results}
    if fired:
        msg = msg[:-1] + f"; consistent for {name} up to {scope['range'][1]:.4g} (observed up to {qs[-1]:.4g})."
    return finding("slice_amplitude", "model", stat, _thr()["i2"], fired, sev if fired else "info",
                   response="widen" if fired else None, scope=scope, message=msg, details=det)


# ----------------------------------------------------------------------------- entry points
def audit(meta, data, rhs=None, slicings=("trajectory", "time", "space", "amplitude")):
    """Slice-consistency findings for the fixed structure of `rhs` (dict var -> expression)."""
    if not rhs:
        return []
    rhs = rhs.get("rhs", rhs)
    struct = _structure(meta, rhs)
    if not any(struct.values()):
        return []
    sysw = weak_system(meta, data, struct)
    sysw["full"] = _slice_stats(sysw, struct, np.zeros(len(sysw["lhs"]), int), ["all"])
    fns = {"trajectory": _trajectory, "time": _time, "space": _space, "amplitude": _amplitude}
    out = []
    for s in slicings:
        f = fns[s](sysw, struct)
        if f is not None:
            f["details"]["n_rows"] = int(len(sysw["lhs"]))
            f["details"]["n_rows_dropped_nan"] = sysw["n_rows_dropped"]
            out.append(f)
    return out


def combined_intervals(findings):
    """Widest random-effects 90% interval per coefficient over the slice findings (for the grade)."""
    out = {}
    for f in findings:
        for k, (lo, hi) in (f.get("details") or {}).get("re_intervals", {}).items():
            if k not in out or hi - lo > out[k][1] - out[k][0]:
                out[k] = [lo, hi]
    return out
