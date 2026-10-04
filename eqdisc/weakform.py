"""Weak-form SINDy (WSINDy; Messenger & Bortz 2021, Reinbold, Gurevich & Grigoriev 2020).

    weak_sindy(meta, data, ...) -> {"rhs", "sparsity_path", "library_size", "method", "validation"}

Same output shape as toolbox.run_sindy, so it is a drop-in agent tool. Uses PUBLIC data only.

Method
------
Test functions are separable, compactly supported polynomial bumps
    phi(t, x_1..x_d) = a((t - t_c)/h_t) * prod_i b_i((x_i - c_i)/h_i),   a(s) = (1 - s^2)^p
with analytic derivatives (numpy polynomial). Each test function gives one row of
    -∫ phi_t U  =  sum_k c_k ∫ phi Theta_k(U)
(time derivative moved onto phi, so no numerical d/dt of noisy data is ever taken).
Spatial derivatives are moved onto phi by integration by parts, per library term
    g(fields, x) * ∂^alpha f:
  * g == 1                 : ∫ phi ∂^a f = (-1)^|a| ∫ (∂^a phi) f                      exact, raw data
  * g == f^p, alpha = e_i  : f^p f_xi = (f^{p+1}/(p+1))_xi -> -∫ phi_xi f^{p+1}/(p+1)  exact, raw data
  * otherwise              : move beta = floor(|alpha|/2) derivatives onto (phi g) (Leibniz):
                             (-1)^|b| sum_{g<=b} C(b,g) ∫ ∂^{b-g}phi ∂^g g ∂^{a-b} f
                             with the remaining (lower-order) derivatives of g and f taken
                             spectrally (periodic axes) / by central differences (other axes)
                             on a lightly low-passed field.
  * pure monomials g(U)    : ∫ phi g(U) on raw data.
Non-library custom terms (anything not of the form g * one derivative symbol) are evaluated
pointwise on the smoothed field and integrated against phi (no integration by parts).
Quadrature is a plain Riemann sum (= trapezoid, since phi vanishes at the support ends).
Integrals for all centres on a strided grid are computed as separable matrix products,
then n_test_functions centres per trajectory are drawn at random (seed).

Widths ("auto"): the corner wavenumber k_c of each axis is estimated from the data spectrum
(last mode more than 10x above the noise floor); the half-width is chosen so the test
function's (approximately Gaussian) spectrum is ~e^{-4.5} at k_c: h = width_factor sqrt(2p) / k_c
(width_factor 3 for PDEs, 1 for ODEs), clamped to [max(4, p), max_width_frac * n] samples.
Wide windows on saturating data (Fisher-KPP) lose the transient information, hence the clamp.

Regression / model selection:
  * candidate supports = column-normalised STLSQ (baselines.stlsq) over a threshold sweep
    + greedy backward elimination from the full library, each size refined by 1-swap local
    search on training RSS (fixes STLSQ dropping a small true term for collinear impostors,
    e.g. Lorenz -y vs y*z, x*z*z);
  * every candidate is fit on training test functions (all but the last trajectory; last 25%
    of time if there is one trajectory) and scored on the held-out ones;
  * accepted if val_err <= (1 + selection_tolerance) * min val_err, OR val_err <=
    noise_floor_factor * (its predicted noise floor). The noise floor is the expected residual
    from white measurement noise (sigma estimated from the flat high-k spectrum / third
    differences), propagated through phi_t on the LHS and linearised through every library
    term (exact for linear/IBP terms, d g/du on the smoothed field for nonlinear ones).
    Without it, dense models fit the noise that appears on both sides (errors-in-variables)
    and win on validation (e.g. FitzHugh-Nagumo);
  * the sparsest accepted model wins (ties: lower val_err); final LS refit on all rows on that
    support.

Extension hooks: meta["spatial_dims"] (default ["x"]), data shape
(n_traj, nt, *spatial, n_fields), meta["boundary"] ('periodic' default, anything else ->
test functions kept inside the domain and FD derivatives). Derivative symbols are
f + "_" + letters, e.g. u_xxy (letters ordered as in spatial_dims). The library is filtered
to meta["allowed_symbols"].
"""
import itertools
import math

import numpy as np
import sympy as sp
from numpy.polynomial import Polynomial

from .baselines import stlsq, to_expr
from .solvers import parse
from . import toolbox

DEFAULT_THRESHOLDS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3)


# ----------------------------------------------------------------------------- geometry helpers
def _spatial_dims(meta):
    return list(meta.get("spatial_dims") or ["x"])


def _periodic(meta, dim):
    b = meta.get("boundary", "periodic")
    if isinstance(b, dict):
        b = b.get(dim, "periodic")
    return str(b).lower().startswith("periodic")


def _grid(meta, data, dim, n, i):
    """Coordinates and spacing of spatial axis `dim` (index i among spatial dims)."""
    if dim in data and np.ndim(data[dim]) == 1 and len(data[dim]) == n:
        x = np.asarray(data[dim], float)
        return x, float(x[1] - x[0])
    L = meta.get("L")
    if isinstance(L, (list, tuple)):
        L = L[i]
    L = meta.get(f"L{dim}", L) if i > 0 else L
    d = float(L) / n
    return np.arange(n) * d, d


def _deriv_name(f, alpha, dims):
    letters = "".join(c * k for c, k in zip(dims, alpha))
    return f if not letters else f"{f}_{letters}"


def _parse_deriv(name, fields, dims):
    """'u_xxy' -> (field_index, alpha) or None."""
    if "_" not in name:
        return None
    f, _, letters = name.rpartition("_")
    if f not in fields or not letters or any(c not in dims for c in letters):
        return None
    return fields.index(f), tuple(letters.count(c) for c in dims)


# ----------------------------------------------------------------------------- test functions
def _bump_poly(p):
    return Polynomial([1.0, 0.0, -1.0]) ** p


def _axis_matrix(n, centres, m, d, order, p, periodic):
    """Rows: test function centred at sample `c` with half-width m samples, differentiated
    `order` times (physical units) and multiplied by the quadrature weight d."""
    P = _bump_poly(p).deriv(order) if order else _bump_poly(p)
    off = np.arange(-m, m + 1)
    w = P(off / m) / (m * d) ** order * d
    M = np.zeros((len(centres), n))
    for r, c in enumerate(centres):
        idx = c + off
        if periodic:
            np.add.at(M[r], idx % n, w)
        else:
            M[r, idx] = w
    return M


def _project(G, mats):
    """G: (nt, n1, ..., nd); mats[j]: (c_j, n_j) for axis j. Returns (c_0, c_1, ..., c_d)."""
    out = G
    for ax, M in enumerate(mats):
        out = np.moveaxis(np.tensordot(M, out, axes=([1], [ax])), 0, ax)
    return out


def _corner_k(U, axis, d, periodic):
    """Angular wavenumber/frequency where the averaged power spectrum meets the noise floor."""
    X = np.moveaxis(U, axis, -1).reshape(-1, U.shape[axis])
    if X.shape[0] > 2000:
        X = X[np.random.default_rng(0).choice(X.shape[0], 2000, replace=False)]
    if not periodic:
        n = X.shape[1]
        tt = np.arange(n)
        A = np.vstack([tt, np.ones(n)]).T
        coef = np.linalg.lstsq(A, X.T, rcond=None)[0]
        X = (X - (A @ coef).T) * np.hanning(n)
    E = (np.abs(np.fft.rfft(X, axis=1)) ** 2).mean(0)
    nk = len(E)
    floor = np.median(E[int(0.75 * nk):]) + 1e-300
    above = np.where(E > 10 * floor)[0]
    kidx = int(above.max()) if above.size else nk - 1
    kidx = max(kidx, 1)
    return 2 * np.pi * kidx / (X.shape[1] * d)


# ----------------------------------------------------------------------------- field derivatives
def _diff(F, axis, order, d, periodic):
    if order == 0:
        return F
    if periodic:
        n = F.shape[axis]
        k = 2 * np.pi * np.fft.rfftfreq(n, d=d)
        m = (1j * k) ** order
        if order % 2 == 1 and n % 2 == 0:
            m[-1] = 0.0
        shape = [1] * F.ndim
        shape[axis] = len(k)
        return np.fft.irfft(np.fft.rfft(F, axis=axis) * m.reshape(shape), n=n, axis=axis)
    for _ in range(order):
        F = np.gradient(F, d, axis=axis)
    return F


def _smooth(F, axes_info, frac):
    """Light Gaussian-tapered spectral low-pass (periodic axes) / Savitzky-Golay (others)."""
    from scipy.signal import savgol_filter
    for ax, d, per in axes_info:
        n = F.shape[ax]
        if per:
            Fh = np.fft.rfft(F, axis=ax)
            j = np.arange(Fh.shape[ax])
            kc = max(frac * Fh.shape[ax], 2)
            shape = [1] * F.ndim
            shape[ax] = len(j)
            F = np.fft.irfft(Fh * np.exp(-(j / kc) ** 4).reshape(shape), n=n, axis=ax)
        elif n > 7:
            F = savgol_filter(F, 7, 3, axis=ax)
    return F


# ----------------------------------------------------------------------------- term classification
def _classify(expr, fields, dims, coords):
    """Return ('mono', g_expr) | ('deriv', g_expr, field_idx, alpha) | ('generic', expr)."""
    expr = sp.sympify(expr)
    c, rest = expr.as_coeff_Mul()
    if not c.is_number:
        return ("generic", expr)
    factors = sp.Mul.make_args(rest)
    derivs, others = [], []
    for fct in factors:
        base, ex = fct.as_base_exp()
        if isinstance(base, sp.Symbol) and _parse_deriv(str(base), fields, dims):
            derivs.append((base, ex))
        elif fct.free_symbols <= {sp.Symbol(n) for n in fields + coords}:
            others.append(fct)
        else:
            return ("generic", expr)
    g = c * sp.Mul(*others)
    if not derivs:
        return ("mono", g)
    if len(derivs) == 1 and derivs[0][1] == 1:
        fi, alpha = _parse_deriv(str(derivs[0][0]), fields, dims)
        return ("deriv", g, fi, alpha)
    return ("generic", expr)


# ----------------------------------------------------------------------------- library
def _pde_library_terms(meta, fields, dims, poly_degree, max_deriv):
    """Same naming as toolbox.build_library in 1-D; generalised to mixed derivatives in n-D."""
    monos = [()]
    for deg in range(1, poly_degree + 1):
        monos += list(itertools.combinations_with_replacement(fields, deg))
    alphas = []
    for order in range(1, max_deriv + 1):
        for a in itertools.product(range(order + 1), repeat=len(dims)):
            if sum(a) == order:
                alphas.append(a)
    alphas.sort(key=lambda a: (sum(a), [-x for x in a]))
    derivs = [None] + [_deriv_name(f, a, dims) for f in fields for a in alphas]
    terms = []
    for m in monos:
        for dv in derivs:
            if dv is not None and len(m) > 2:
                continue
            parts = list(m) + ([dv] if dv else [])
            terms.append("*".join(parts) if parts else "1")
    return terms


def _finalise_terms(terms, custom_terms, exclude_terms, names):
    terms = list(terms) + list(custom_terms)
    allowed = {sp.Symbol(n) for n in names}
    ex = {str(sp.expand(parse(e, names))) for e in exclude_terms}
    seen, final = set(), []
    for tm in terms:
        e = parse(tm, names)
        key = str(sp.expand(e))
        if key in seen or key in ex or not e.free_symbols <= allowed:
            continue
        seen.add(key)
        final.append(tm)
    return final


# ----------------------------------------------------------------------------- weak system assembly
def _choose_centres(n, m, periodic, stride, rng):
    if periodic:
        start = int(rng.integers(stride))
        return np.arange(start, n, stride)
    lo, hi = m, n - 1 - m
    if hi < lo:
        return np.array([], int)
    start = lo + int(rng.integers(stride))
    c = np.arange(start, hi + 1, stride)
    return c if c.size else np.array([lo])


def _auto_halfwidth(k_c, d, p, n, lo, hi_frac, factor):
    h = factor * math.sqrt(2 * p) / max(k_c, 1e-12)
    m = int(round(h / d))
    return int(np.clip(m, lo, max(lo, int(hi_frac * n))))


def weak_sindy(meta, data, poly_degree=3, max_deriv=4, custom_terms=(), exclude_terms=(),
               thresholds=DEFAULT_THRESHOLDS, n_test_functions=600, test_fn_width="auto",
               ridge=1e-6, selection_tolerance=0.05, targets=None, seed=0,
               include_trig=False, library_vars=None, p_time=4, p_space=None,
               width_factor=None, smooth_frac=0.5, selection="val",
               backward=True, noise_floor_factor=1.1, max_width_frac=(0.1, 0.15),
               return_system=False):
    """Weak-form SINDy. Returns the same structure as toolbox.run_sindy.

    n_test_functions : test functions per trajectory (random subset of a strided centre grid).
    test_fn_width    : 'auto' or half-widths in SAMPLES: int (time) for ODE,
                       (m_t, m_x[, m_y ...]) for PDE; None entries are auto.
    p_time / p_space : bump exponents (phi = (1-s^2)^p); p_space default max_deriv + 2.
    width_factor     : auto width h = width_factor * sqrt(2p) / k_c (larger = smoother).
    width_factor     : auto width h = width_factor * sqrt(2p) / k_c; scalar or (time, space).
                       Default 3 (PDE) / 1 (ODE).
    max_width_frac   : (time, space) cap on auto half-widths as a fraction of the axis length.
    smooth_frac      : fraction of Fourier modes kept when evaluating derivatives of non-exact
                       terms (only those terms see smoothed data).
    selection        : 'val' (default, see module doc) or 'mstls' (Messenger-Bortz loss).
    backward         : add backward-elimination + swap-refined supports to the candidate set.
    noise_floor_factor: accept a candidate whose held-out residual is within this factor of its
                       predicted measurement-noise floor (0 disables).
    include_trig, library_vars : as in toolbox.run_sindy.
    return_system    : debug/advanced: return Theta, lhs, terms, row splits, sigma instead.
    Extra output key "weak_info": chosen half-widths, #test functions, noise std estimate,
    per-variable noise floor and how each term was integrated (ibp-exact / ibp-flux / ...).
    """
    rng = np.random.default_rng(seed)
    U = np.asarray(data["U"], float)
    t = np.asarray(data["t"], float)
    fields = list(meta["variables"])
    names = toolbox.symbols(meta)
    n_traj, nt = U.shape[0], U.shape[1]
    dt = float(meta.get("dt", t[1] - t[0]))
    is_pde = meta["kind"] == "pde"
    dims = _spatial_dims(meta) if is_pde else []
    nsp = len(dims)
    sp_shape = U.shape[2:2 + nsp]
    grids = [_grid(meta, data, dim, sp_shape[i], i) for i, dim in enumerate(dims)]
    periodic = [_periodic(meta, dim) for dim in dims]

    # --- library
    if is_pde:
        lib_fields = list(library_vars or fields)
        terms = _pde_library_terms(meta, lib_fields, dims, poly_degree, max_deriv)
        terms = _finalise_terms(terms, custom_terms, exclude_terms, names)
    else:
        terms = None  # built by toolbox.build_library below (identical naming)
    p_x = p_space or (max_deriv + 2)

    # --- widths (half-widths in samples)
    if isinstance(test_fn_width, (int, float)) and not isinstance(test_fn_width, bool):
        widths = [int(test_fn_width)] + [None] * nsp
    elif isinstance(test_fn_width, (list, tuple)):
        widths = list(test_fn_width) + [None] * (1 + nsp - len(test_fn_width))
    else:
        widths = [None] * (1 + nsp)
    if width_factor is None:
        width_factor = 3.0 if is_pde else 1.0
    wf_t, wf_x = (width_factor, width_factor) if np.isscalar(width_factor) else tuple(width_factor)
    if widths[0] is None:
        kt = _corner_k(U, 1, dt, False)
        widths[0] = _auto_halfwidth(kt, dt, p_time, nt, 3, max_width_frac[0], wf_t)
    for i in range(nsp):
        if widths[1 + i] is None:
            kx = _corner_k(U, 2 + i, grids[i][1], periodic[i])
            widths[1 + i] = _auto_halfwidth(kx, grids[i][1], p_x, sp_shape[i], max(4, p_x), max_width_frac[1], wf_x)
    widths = [int(w) for w in widths]
    m_t, m_sp = widths[0], widths[1:]
    if nt < 2 * m_t + 2:
        m_t = max(2, (nt - 2) // 2)

    # --- centres (strided grid, random offset), then random subset per trajectory
    c_t = _choose_centres(nt, m_t, False, max(1, m_t // 2), rng)
    c_sp = [_choose_centres(sp_shape[i], m_sp[i], periodic[i], max(1, m_sp[i] // 2), rng) for i in range(nsp)]
    grid_size = len(c_t) * int(np.prod([len(c) for c in c_sp])) if nsp else len(c_t)
    sel = [np.sort(rng.choice(grid_size, min(n_test_functions, grid_size), replace=False))
           for _ in range(n_traj)]

    mat_cache = {}

    def mats(order_t, alpha):
        key = (order_t, tuple(alpha))
        if key not in mat_cache:
            Ms = [_axis_matrix(nt, c_t, m_t, dt, order_t, p_time, False)]
            for i in range(nsp):
                Ms.append(_axis_matrix(sp_shape[i], c_sp[i], m_sp[i], grids[i][1], alpha[i], p_x, periodic[i]))
            mat_cache[key] = Ms
        return mat_cache[key]

    def integrate(G, order_t=0, alpha=None):
        """G: (n_traj, nt, *spatial). Returns stacked selected rows (sum over traj of n_sel)."""
        alpha = alpha or (0,) * nsp
        Ms = mats(order_t, alpha)
        out = []
        for j in range(n_traj):
            out.append(_project(G[j], Ms).ravel()[sel[j]])
        return np.concatenate(out)

    # --- fields: raw, and smoothed (only used for derivative evaluation of non-exact terms)
    raw = {f: U[..., i] for i, f in enumerate(fields)}
    coords = {}
    if is_pde:
        for i, dim in enumerate(dims):
            shp = [1] * (2 + nsp)
            shp[2 + i] = sp_shape[i]
            coords[dim] = np.broadcast_to(grids[i][0].reshape(shp), U.shape[:-1])
    else:
        coords["t"] = np.broadcast_to(t[None, :], U.shape[:-1])
    # a single field array is (n_traj, nt, *space): spatial axis i is array axis 2 + i (was 1 + i, i.e. time)
    axes_info = [(2 + i, grids[i][1], periodic[i]) for i in range(nsp)]
    smooth_cache = {}

    def smoothed(f):
        if f not in smooth_cache:
            smooth_cache[f] = _smooth(raw[f], axes_info, smooth_frac) if is_pde else raw[f]
        return smooth_cache[f]

    def dfield(F, alpha):
        for i, k in enumerate(alpha):
            F = _diff(F, 2 + i, k, grids[i][1], periodic[i])   # spatial axis i of (n_traj, nt, *space)
        return F

    def evalg(g, src):
        syms = sorted(g.free_symbols, key=str)
        if not syms:
            return np.full(U.shape[:-1], float(g))
        fn = sp.lambdify(syms, g, "numpy")
        args = [src(str(s)) if str(s) in fields else coords[str(s)] for s in syms]
        with np.errstate(all="ignore"):
            return np.broadcast_to(np.asarray(fn(*args), float), U.shape[:-1])

    # --- LHS: -∫ phi_t u
    lhs = np.stack([-integrate(raw[f], order_t=1) for f in fields], 1)
    # noise floor of the LHS: white noise sigma integrated against phi_t (per-row weight norms)
    sigma = np.array([_noise_std(raw[f], periodic[0] if nsp else None) for f in fields])

    # --- library columns.  ncomp[k]: how measurement noise enters column k, linearised:
    #     list of (field_idx, factor, alpha, weight-field or None) meaning
    #     factor * ∫ (d^alpha phi) * weight * noise_field   (used only for the noise-floor estimate)
    def pointwise_comps(expr):
        out = []
        for s_ in expr.free_symbols:
            if str(s_) not in fields:
                continue
            dg = sp.diff(expr, s_)
            if dg.is_number:
                out.append((fields.index(str(s_)), float(dg), (0,) * nsp, None))
            else:
                out.append((fields.index(str(s_)), 1.0, (0,) * nsp, evalg(dg, smoothed)))
        return out

    cols, how, ncomp = [], [], []
    if not is_pde:
        feats = toolbox.feature_arrays(meta, U, t)
        terms, vals = toolbox.build_library(meta, feats, poly_degree, max_deriv, include_trig, custom_terms,
                                            exclude_terms, library_vars)
        for tm, v in zip(terms, vals):
            cols.append(integrate(np.asarray(v, float)))
            how.append("pointwise")
            try:
                ncomp.append(pointwise_comps(parse(tm, names)))
            except Exception:  # noqa: BLE001
                ncomp.append([])
    else:
        for tm in terms:
            kind = _classify(parse(tm, names), fields, dims, dims)
            if kind[0] == "mono":
                cols.append(integrate(evalg(kind[1], lambda f: raw[f])))
                how.append("pointwise")
                ncomp.append(pointwise_comps(kind[1]))
                continue
            if kind[0] == "generic":
                feats = _generic_feats(smoothed, fields, dims, dfield, coords, max_deriv)
                cols.append(integrate(np.asarray(toolbox.eval_exprs([str(kind[1])], feats, list(feats))[0], float)))
                how.append("pointwise-smoothed")
                ncomp.append([])
                continue
            _, g, fi, alpha = kind
            f = fields[fi]
            order = sum(alpha)
            gsyms = {str(s_) for s_ in g.free_symbols}
            if not gsyms:
                cols.append((-1) ** order * float(g) * integrate(raw[f], alpha=alpha))
                how.append("ibp-exact")
                ncomp.append([(fi, (-1) ** order * float(g), tuple(alpha), None)])
                continue
            gp = sp.Poly(g, sp.Symbol(f)) if gsyms == {f} else None
            if order == 1 and gp is not None and len(gp.terms()) == 1:
                (pw,), cf = gp.terms()[0]
                flux = float(cf) * raw[f] ** (pw + 1) / (pw + 1)
                cols.append(-integrate(flux, alpha=alpha))
                how.append("ibp-flux")
                ncomp.append([(fi, -float(cf), tuple(alpha), smoothed(f) ** pw)])
                continue
            # general: move beta = floor(order/2) derivatives onto (phi * g)
            beta = _split_alpha(alpha, order // 2)
            gS = evalg(g, smoothed)
            rest = dfield(smoothed(f), tuple(a - b for a, b in zip(alpha, beta)))
            acc = 0.0
            for gam in itertools.product(*[range(b + 1) for b in beta]):
                coef = np.prod([math.comb(b, c) for b, c in zip(beta, gam)])
                acc = acc + coef * integrate(dfield(gS, gam) * rest, alpha=tuple(b - c for b, c in zip(beta, gam)))
            cols.append((-1) ** sum(beta) * acc)
            how.append(f"ibp-partial({sum(beta)})")
            ncomp.append([])   # noise of these (smoothed) terms is not modelled

    Theta = np.stack(cols, 1)
    ok = np.all(np.isfinite(Theta), axis=0) & (np.linalg.norm(Theta, axis=0) > 0)
    terms = [tm for tm, o in zip(terms, ok) if o]
    how = [h for h, o in zip(how, ok) if o]
    ncomp = [h for h, o in zip(ncomp, ok) if o]
    Theta = Theta[:, ok]

    # --- train / validation split over test functions
    counts = [len(s) for s in sel]
    owner = np.repeat(np.arange(n_traj), counts)
    if n_traj >= 2:
        va = np.where(owner == n_traj - 1)[0]
        tr = np.where(owner != n_traj - 1)[0]
    else:
        tc_of_row = sel[0] // (grid_size // len(c_t))
        cut = c_t[tc_of_row] >= int(0.75 * nt)
        va, tr = np.where(cut)[0], np.where(~cut)[0]
        if va.size == 0 or tr.size == 0:
            idx = rng.permutation(len(owner))
            va, tr = idx[: len(idx) // 4], idx[len(idx) // 4:]

    # --- noise Gram matrices on validation rows: floor^2 = sum_f sigma_f^2 c_hat^T N_f c_hat,
    #     c_hat = [1 (LHS, only for the target field), c_1..c_J]
    va_traj = sorted(set(owner[va].tolist()))
    va_mask = np.zeros(len(owner), bool)
    va_mask[va] = True

    def cross(p, q):
        (fa, wa, aa, Ga, ta), (fb, wb, ab, Gb, tb) = p, q
        Ma, Mb = mats(ta, aa), mats(tb, ab)
        Ms = [P * Q for P, Q in zip(Ma, Mb)]
        G = (Ga if Ga is not None else 1.0) * (Gb if Gb is not None else 1.0)
        tot, off = 0.0, 0
        for j in range(n_traj):
            if j in va_traj:
                Gj = np.broadcast_to(G, U.shape[:-1])[j] if np.ndim(G) else np.ones((nt,) + tuple(sp_shape))
                tot += float(_project(Gj, Ms).ravel()[sel[j]][va_mask[off:off + len(sel[j])]].sum())
            off += len(sel[j])
        return wa * wb * tot

    Ngram = []
    if noise_floor_factor:
        for fj in range(len(fields)):
            comps = [[(fj, 1.0, (0,) * nsp, None, 1)]] + \
                    [[(a, w, al, G, 0) for (a, w, al, G) in nc if a == fj] for nc in ncomp]
            J1 = len(comps)
            N = np.zeros((J1, J1))
            for a_ in range(J1):
                for b_ in range(a_, J1):
                    if comps[a_] and comps[b_]:
                        N[a_, b_] = N[b_, a_] = sum(cross(p, q) for p in comps[a_] for q in comps[b_])
            Ngram.append(N)

    def noise_floor(c, i):
        tot = 0.0
        for fj, N in enumerate(Ngram):
            ch = np.concatenate([[1.0 if fj == i else 0.0], c])
            tot += sigma[fj] ** 2 * max(float(ch @ N @ ch), 0.0)
        return math.sqrt(tot)

    rhs, path, floors = {}, {}, {}
    for i, v in enumerate(fields):
        if targets and v not in targets:
            continue
        y = lhs[:, i]
        res, coefs = [], []
        for th in thresholds:
            c = stlsq(Theta[tr], y[tr], th, ridge)
            coefs.append(c)
            e = np.linalg.norm(Theta[va] @ c - y[va]) / (np.linalg.norm(y[va]) + 1e-12)
            res.append({"threshold": th, "n_terms": int((c != 0).sum()), "val_err": float(e),
                        "model": to_expr(c, terms, 4)})
        if backward:
            # greedy backward elimination from the full library: STLSQ's hard thresholds can drop a
            # small-but-true term in favour of collinear impostors (e.g. Lorenz -y vs y*z, x*z*z)
            for S in _backward_path(Theta[tr], y[tr], ridge):
                c = _ls_support(Theta[tr], y[tr], S, ridge)
                if any(np.array_equal(c != 0, cc != 0) for cc in coefs):
                    continue
                coefs.append(c)
                e = np.linalg.norm(Theta[va] @ c - y[va]) / (np.linalg.norm(y[va]) + 1e-12)
                res.append({"threshold": None, "n_terms": int((c != 0).sum()), "val_err": float(e),
                            "model": to_expr(c, terms, 4)})
        if selection == "mstls":
            # Messenger & Bortz MSTLS loss, evaluated on held-out rows:
            # ||Theta (c - c_LS)|| / ||Theta c_LS|| + n_terms / J
            c0 = stlsq(Theta[tr], y[tr], 0.0, ridge)
            ref = np.linalg.norm(Theta[va] @ c0) + 1e-12
            for r, cc in zip(res, coefs):
                r["loss"] = float(np.linalg.norm(Theta[va] @ (cc - c0)) / ref + r["n_terms"] / Theta.shape[1])
            best = min(res, key=lambda r: (r["loss"], r["n_terms"]))
        else:
            emin = min(r["val_err"] for r in res)
            ny = np.linalg.norm(y[va]) + 1e-12
            for r, cc in zip(res, coefs):
                r["noise_floor"] = float(noise_floor(cc, i) / ny) if noise_floor_factor else 0.0
            floors[v] = min(r["noise_floor"] for r in res)
            best = min([r for r in res if r["val_err"] <= (1 + selection_tolerance) * emin + 1e-4
                        or r["val_err"] <= noise_floor_factor * r["noise_floor"]],
                       key=lambda r: (r["n_terms"], r["val_err"]))
        # refit on all rows, keeping the selected support (an STLSQ re-run could change it)
        c = _ls_support(Theta, y, coefs[res.index(best)] != 0, ridge)
        rhs[v] = to_expr(c, terms, 5)
        path[v] = [{k: r.get(k) for k in ("threshold", "n_terms", "val_err", "noise_floor")} | {"model": r["model"][:300]}
                   for r in res]
    full = {v: rhs.get(v, "0") for v in fields}
    if return_system:
        return {"Theta": Theta, "lhs": lhs, "terms": terms, "train_rows": tr, "val_rows": va,
                "sigma": sigma, "noise_floor": noise_floor, "rhs": full, "sparsity_path": path}
    return {"rhs": full, "library_size": len(terms), "sparsity_path": path, "method": "weak_sindy",
            "weak_info": {"halfwidth_samples": {"t": m_t, **{d: m for d, m in zip(dims, m_sp)}},
                          "n_test_functions": int(len(owner)),
                          "noise_std_estimate": dict(zip(fields, map(float, sigma))),
                          "val_noise_floor": floors,
                          "term_treatment": dict(zip(terms, how))},
            "validation": toolbox.validate(meta, data, full)}


def _noise_std(F, periodic_x):
    """White measurement-noise std of one field. F: (n_traj, nt, *spatial).
    Periodic PDE: from the flat high-wavenumber tail of the x-spectrum (E|U_k|^2 = n sigma^2).
    Otherwise: robust std of third differences in time (var = 20 sigma^2 for white noise)."""
    if periodic_x:
        n = F.shape[2]
        E = (np.abs(np.fft.rfft(np.moveaxis(F, 2, -1), axis=-1)) ** 2).reshape(-1, n // 2 + 1).mean(0)
        return float(np.sqrt(np.median(E[int(0.75 * len(E)):]) / n))
    d = np.diff(F, n=3, axis=1).ravel()
    return float(1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(20.0))


def _ls_support(Theta, y, support, ridge):
    support = np.asarray(support)
    if support.dtype != bool:
        m = np.zeros(Theta.shape[1], bool)
        m[support] = True
        support = m
    c = np.zeros(Theta.shape[1])
    if support.any():
        norms = np.linalg.norm(Theta[:, support], axis=0) + 1e-12
        A = Theta[:, support] / norms
        c[support] = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ y) / norms
    return c


def _backward_path(Theta, y, ridge, max_terms=None, refine_upto=8):
    """Supports of every size from greedy backward elimination (least residual increase)."""
    A = Theta / (np.linalg.norm(Theta, axis=0) + 1e-12)
    G, b = A.T @ A, A.T @ y
    yy = float(y @ y)
    S = list(range(A.shape[1]))
    out = []

    def rss(idx):
        Gi = G[np.ix_(idx, idx)] + ridge * np.eye(len(idx))
        x = np.linalg.solve(Gi, b[idx])
        return yy - float(b[idx] @ x)

    def swap_refine(S, max_sweeps=5):
        """1-swap local search: replace a term by an excluded one while training RSS drops."""
        S = list(S)
        cur = rss(S)
        for _ in range(max_sweeps):
            improved = False
            for k in range(len(S)):
                for j in range(A.shape[1]):
                    if j in S:
                        continue
                    T = S[:k] + [j] + S[k + 1:]
                    r = rss(T)
                    if r < cur * (1 - 1e-9):
                        S, cur, improved = T, r, True
            if not improved:
                break
        return sorted(S)

    while len(S) > 1:
        scores = [rss(S[:k] + S[k + 1:]) for k in range(len(S))]
        S = S[:int(np.argmin(scores))] + S[int(np.argmin(scores)) + 1:]
        if max_terms is None or len(S) <= max_terms:
            out.append(list(S))
            if refine_upto and len(S) <= refine_upto:
                R = swap_refine(S)
                if R != sorted(S):
                    out.append(R)
    return out


def _split_alpha(alpha, j):
    """Greedy beta <= alpha with |beta| = j, taking from the largest components first."""
    beta = [0] * len(alpha)
    rem = list(alpha)
    for _ in range(j):
        k = int(np.argmax(rem))
        beta[k] += 1
        rem[k] -= 1
    return tuple(beta)


def _generic_feats(smoothed, fields, dims, dfield, coords, max_deriv):
    feats = dict(coords)
    for f in fields:
        S = smoothed(f)
        for order in range(0, max_deriv + 1):
            for a in itertools.product(range(order + 1), repeat=len(dims)):
                if sum(a) == order:
                    feats[_deriv_name(f, a, dims)] = dfield(S, a)
    return feats
