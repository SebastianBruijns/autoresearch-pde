"""Uncertainty quantification and ground-truth-free model assessment (PUBLIC data only).

On real data there is no hidden test set, so the agent must judge its own models:

    ensemble_sindy(meta, data, ...)             E-SINDy (Fasel et al. 2022): bootstrap-bagged STLSQ ->
                                                term inclusion probabilities, coefficient intervals,
                                                consensus model (refit on all data) + validation
    cross_validate(meta, data, rhs_or_fitter)   held-out-fold scoring of a fixed rhs, or refit-per-fold
                                                of a fitter callable with term-set stability
    compare_models(meta, data, {name: rhs})     CV error + AICc/BIC + rollout valid time + noise floor
                                                -> ranking and a short textual verdict
    coefficient_uncertainty(meta, data, rhs)    block-bootstrap intervals for a fixed structure's
                                                linear coefficients

Everything works on (meta, data) as loaded from data.npz; nothing here reads hidden/ files.
Outputs are JSON-serialisable, compact, with floats rounded to 4 significant figures.

Conventions
-----------
* Derivatives are Savitzky-Golay (+ spectral low-pass for PDEs), edge-trimmed, exactly as in
  toolbox.run_sindy. "deriv_nrmse" here is the MEAN over variables of per-variable
  RMS(f - dU/dt) / RMS(dU/dt) (scale-invariant; toolbox.validate pools variables instead).
* Rows are strongly correlated in time (smoothing window) and space (low-pass), so all
  bootstraps resample contiguous blocks, and information criteria use an effective sample size.
"""
import itertools
from typing import Callable

import numpy as np
import sympy as sp

from .baselines import lowpass, stlsq, to_expr
from .solvers import integrate_ode, integrate_pde, parse
from .toolbox import (_pde_step, build_library, diagnose, eval_exprs, feature_arrays, smooth_and_differentiate,
                      split_rows, symbols, validate)

DEFAULT_THRESHOLDS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3)


# ----------------------------------------------------------------------------- small helpers
def _r(x, sig=4):
    """Round floats (recursively) to `sig` significant figures for compact JSON."""
    if isinstance(x, dict):
        return {k: _r(v, sig) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_r(v, sig) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        if not np.isfinite(x):
            return None
        return float(f"{x:.{sig}g}")
    return x


def _prepare(meta, data, window=9, order=3, lowpass_frac=0.3, diff_method="savgol"):
    """Smoothed state + derivative on the edge-trimmed grid, as in toolbox.run_sindy."""
    U, t = data["U"], data["t"]
    pde = meta["kind"] == "pde"
    Us, dUdt = smooth_and_differentiate(meta, U, window, order, lowpass_frac if pde else None, diff_method)
    nt = U.shape[1]
    edge = slice(3, -3) if nt > 12 else slice(None)
    Us, dUdt = Us[:, edge], dUdt[:, edge]
    if pde and lowpass_frac:
        dUdt = lowpass(dUdt, lowpass_frac)
    te = t[edge]
    grid = Us.shape[:-1]
    feats = feature_arrays(meta, Us, te)
    Y = dUdt.reshape(-1, dUdt.shape[-1])
    off = 3 if nt > 12 else 0
    return {"meta": meta, "U": U, "t": t, "Us": Us, "dUdt": dUdt, "te": te, "grid": grid, "feats": feats,
            "Y": Y, "idx": np.arange(int(np.prod(grid))).reshape(grid), "t_offset": off,
            "window": window, "order": order, "lowpass_frac": lowpass_frac, "diff_method": diff_method,
            "ms": np.mean(Y ** 2, axis=0) + 1e-30}


def _blocks(prep, block_len=None, x_chunks=4):
    """Contiguous row blocks: (trajectory, time window[, x chunk]). Block length > smoothing window."""
    grid, idx = prep["grid"], prep["idx"]
    n_traj, nt = grid[0], grid[1]
    bl = block_len or max(2 * prep["window"], nt // 25, 4)
    bl = min(bl, nt)
    tb = [(a, min(a + bl, nt)) for a in range(0, nt, bl)]
    if tb and tb[-1][1] - tb[-1][0] < bl // 2 and len(tb) > 1:      # merge short tail
        tb[-2] = (tb[-2][0], tb[-1][1])
        tb.pop()
    out = []
    for j in range(n_traj):
        for a, b in tb:
            if len(grid) == 2:
                out.append(idx[j, a:b].ravel())
            else:
                nx = grid[2]
                xs = np.array_split(np.arange(nx), x_chunks)
                out += [idx[j, a:b][:, xc].ravel() for xc in xs]
    return out


def _structure(meta, rhs):
    """{var: [(term_str, coef), ...]} from the expanded expression of each rhs component."""
    names = symbols(meta)
    out = {}
    for v in meta["variables"]:
        e = sp.expand(parse(rhs.get(v, "0"), names))
        bad = e.free_symbols - {sp.Symbol(n) for n in names}
        if bad:
            raise ValueError(f"unknown symbols {sorted(map(str, bad))}; allowed {names}")
        terms = {}
        for tm in sp.Add.make_args(e):
            if tm == 0:
                continue
            c, m = tm.as_coeff_Mul()
            key = str(m)
            terms[key] = terms.get(key, 0.0) + float(c)
        out[v] = [(k, c) for k, c in terms.items() if c != 0]
    return out


def _structure_columns(prep, struct):
    """Evaluate each term of the structure on the prepared grid -> {var: Theta (N, k)}."""
    names = symbols(prep["meta"])
    N = prep["Y"].shape[0]
    cache, out = {}, {}
    for v, terms in struct.items():
        cols = []
        for tm, _ in terms:
            if tm not in cache:
                c = eval_exprs([tm], prep["feats"], names)[0]
                cache[tm] = np.broadcast_to(c, prep["grid"]).ravel().astype(float)
            cols.append(cache[tm])
        out[v] = np.stack(cols, 1) if cols else np.zeros((N, 0))
    return out


def _lstsq(A, y):
    if A.shape[1] == 0:
        return np.zeros(0)
    A = np.where(np.isfinite(A), A, 0.0)
    norms = np.linalg.norm(A, axis=0) + 1e-12
    c = np.linalg.lstsq(A / norms, y, rcond=None)[0]
    return c / norms


def _rhs_from(struct, coefs=None, prec=6):
    out = {}
    for v, terms in struct.items():
        cs = coefs[v] if coefs is not None else [c for _, c in terms]
        parts = [f"{c:+.{prec}g}" + ("" if tm == "1" else f"*({tm})") for (tm, _), c in zip(terms, cs) if c != 0]
        out[v] = " ".join(parts).lstrip("+") if parts else "0"
    return out


def _sub(rows, n, rng):
    return rows if rows.size <= n else rng.choice(rows, n, replace=False)


# ----------------------------------------------------------------------------- folds + rollout
def _folds(prep, folds="auto", n_blocks=4):
    """List of (name, heldout_rows, segments). segments = [(traj, a, b)] in trimmed time index."""
    n_traj, nt = prep["grid"][0], prep["grid"][1]
    if folds == "auto":
        folds = "trajectory" if n_traj >= 3 else "time_blocks"
    if folds == "trajectory" and n_traj < 2:
        folds = "time_blocks"
    out = []
    if folds == "trajectory":
        for j in range(n_traj):
            out.append((f"traj{j}", prep["idx"][j].ravel(), [(j, 0, nt)]))
    elif folds == "time_blocks":
        edges = np.linspace(0, nt, n_blocks + 1).astype(int)
        for k in range(n_blocks):
            a, b = edges[k], edges[k + 1]
            out.append((f"block{k}", prep["idx"][:, a:b].ravel(), [(j, a, b) for j in range(n_traj)]))
    else:
        raise ValueError("folds must be 'trajectory', 'time_blocks' or 'auto'")
    return folds, out


def _rollout(prep, rhs, seg, max_rollout_steps=3000):
    """Integrate rhs from the smoothed first frame of a segment; valid time = error < 0.3."""
    meta = prep["meta"]
    j, a, b = seg
    te, Us = prep["te"], prep["Us"]
    tt = te[a:b]
    if len(tt) < 2:
        return None
    if meta["kind"] == "ode":
        roll = integrate_ode(meta["variables"], rhs, Us[j, a], tt, max_seconds=5.0)
        ref = Us[j, a:b]
    else:
        h = _pde_step(meta, prep["U"])
        n_obs = max(2, min(len(tt), int(max_rollout_steps * h / meta["dt"]) + 1))
        tt = tt[:n_obs]
        sub = max(1, int(round(meta["dt"] / h)))
        roll = integrate_pde(meta["variables"], rhs, meta["L"], Us[j, a], tt, meta["dt"] / sub)
        ref = Us[j, a:a + n_obs]
    red = tuple(range(1, roll.ndim))
    with np.errstate(all="ignore"):
        e_t = np.sqrt(np.mean((roll - ref) ** 2, axis=red)) / (np.sqrt(np.mean(ref ** 2, axis=red)) + 1e-12)
    e_t = np.where(np.isfinite(e_t), e_t, np.inf)
    bad = np.where(e_t > 0.3)[0]
    horizon = float(tt[-1] - tt[0])
    vt = float(tt[bad[0]] - tt[0]) if bad.size else horizon
    return {"valid_time": vt, "horizon": horizon, "valid_frac": vt / horizon if horizon > 0 else 0.0,
            "blew_up": bool(np.isinf(e_t).any())}


def _deriv_err(prep, pred, rows):
    """Per-variable nrmse on `rows` given predictions pred (N, nv) (NaN-safe, capped at 10)."""
    Y = prep["Y"][rows]
    P = pred[rows]
    with np.errstate(all="ignore"):
        e = np.sqrt(np.nanmean((P - Y) ** 2, axis=0)) / (np.sqrt(np.mean(Y ** 2, axis=0)) + 1e-12)
    return np.where(np.isfinite(e), np.minimum(e, 10.0), 10.0)


def _predict(cols, struct, coefs, variables, N):
    P = np.zeros((N, len(variables)))
    for i, v in enumerate(variables):
        if cols[v].shape[1]:
            with np.errstate(all="ignore"):
                P[:, i] = cols[v] @ np.asarray(coefs[v], float)
    return P


def _cv_structure(prep, struct, fold_list, refit=True, max_rollouts_per_fold=2, max_rollout_steps=3000,
                  max_train_rows=40000, seed=0, rollouts=True):
    """CV of a fixed structure. refit=True: linear coefficients refit on training folds (true out-of-
    sample); refit=False: the given coefficients are just scored on each fold.
    Returns (summary, out_of_fold predictions (N, nv))."""
    meta = prep["meta"]
    variables = meta["variables"]
    N = prep["Y"].shape[0]
    rng = np.random.default_rng(seed)
    cols = _structure_columns(prep, struct)
    oof = np.full((N, len(variables)), np.nan)
    per_fold = []
    all_rows = np.arange(N)
    for name, held, segs in fold_list:
        if refit:
            train = np.setdiff1d(all_rows, held, assume_unique=True)
            train = _sub(train, max_train_rows, rng)
            coefs = {v: _lstsq(cols[v][train], prep["Y"][train, i]) for i, v in enumerate(variables)}
        else:
            coefs = {v: [c for _, c in struct[v]] for v in variables}
        P = _predict(cols, struct, coefs, variables, N)
        oof[held] = P[held]
        e = _deriv_err(prep, P, held)
        rec = {"fold": name, "deriv_nrmse": float(e.mean())}
        if rollouts:
            rhs = _rhs_from(struct, coefs)
            rs = [r for r in (_rollout(prep, rhs, s, max_rollout_steps) for s in segs[:max_rollouts_per_fold]) if r]
            if rs:
                rec["rollout_valid_time"] = float(np.mean([r["valid_time"] for r in rs]))
                rec["rollout_valid_frac"] = float(np.mean([r["valid_frac"] for r in rs]))
                rec["rollout_blew_up"] = bool(any(r["blew_up"] for r in rs))
        per_fold.append(rec)
    return per_fold, oof


def _summarise_folds(per_fold):
    d = np.array([f["deriv_nrmse"] for f in per_fold])
    out = {"deriv_nrmse_mean": float(d.mean()), "deriv_nrmse_std": float(d.std(ddof=1)) if d.size > 1 else 0.0}
    vt = [f["rollout_valid_time"] for f in per_fold if "rollout_valid_time" in f]
    if vt:
        out["rollout_valid_time_mean"] = float(np.mean(vt))
        out["rollout_valid_frac_mean"] = float(np.mean([f["rollout_valid_frac"] for f in per_fold
                                                        if "rollout_valid_frac" in f]))
        out["any_blow_up"] = bool(any(f.get("rollout_blew_up") for f in per_fold))
    return out


# ----------------------------------------------------------------------------- noise floor
def noise_floor(meta, data, window=9, order=3, lowpass_frac=0.3, diff_method="savgol", prep=None, seed=0):
    """Derivative nrmse that pure measurement noise alone produces through the SAME smoothing /
    differentiation pipeline. Noise level from toolbox.diagnose (noise_rel_estimate, assumes white
    noise; red/correlated noise is underestimated). A model whose derivative error is ~ this floor
    is as good as these data (with this differentiator) allow."""
    prep = prep or _prepare(meta, data, window, order, lowpass_frac, diff_method)
    U = data["U"]
    rep = diagnose(meta, data)
    nrel_diag = rep["noise_rel_estimate"]
    flat = U.reshape(-1, U.shape[-1])
    # calibrated version of diagnose's estimate: residual of a 7-point cubic Savitzky-Golay fit has
    # std sigma*sqrt(1-h0) for white noise (h0 = centre weight); diagnose uses a fixed factor 1.6.
    from scipy.signal import savgol_coeffs, savgol_filter
    nrel = dict(nrel_diag)
    if U.shape[1] > 12:
        h0 = savgol_coeffs(7, 3)[3]
        res = (U - savgol_filter(U, 7, 3, axis=1))[:, 3:-3].reshape(-1, U.shape[-1])
        nrel = {v: float(res[:, i].std() / np.sqrt(1 - h0) / (flat[:, i].std() + 1e-12))
                for i, v in enumerate(meta["variables"])}
    sig = np.array([nrel[v] * flat[:, i].std() for i, v in enumerate(meta["variables"])])
    rng = np.random.default_rng(seed)
    Usub = U[: min(U.shape[0], 2)]
    noise = rng.standard_normal(Usub.shape) * sig
    pde = meta["kind"] == "pde"
    _, dN = smooth_and_differentiate(meta, noise, window, order, lowpass_frac if pde else None, diff_method)
    if U.shape[1] > 12:
        dN = dN[:, 3:-3]
    if pde and lowpass_frac:
        dN = lowpass(dN, lowpass_frac)
    dN = dN.reshape(-1, dN.shape[-1])
    floor = np.sqrt(np.mean(dN ** 2, axis=0)) / np.sqrt(prep["ms"])
    return {"noise_rel_estimate": {v: float(nrel[v]) for v in meta["variables"]},
            "noise_rel_diagnose": {v: float(nrel_diag[v]) for v in meta["variables"]},
            "deriv_nrmse_floor": {v: float(f) for v, f in zip(meta["variables"], floor)},
            "deriv_nrmse_floor_mean": float(floor.mean())}


# ----------------------------------------------------------------------------- 1. ensemble SINDy
def _select_thresholds(Theta, Y, tr, va, thresholds, ridge, selection_tolerance):
    """Per-variable threshold chosen exactly as toolbox.run_sindy (sparsest within tol of best val err)."""
    ths = []
    for i in range(Y.shape[1]):
        y = Y[:, i]
        res = []
        for th in thresholds:
            c = stlsq(Theta[tr], y[tr], th, ridge)
            e = np.linalg.norm(Theta[va] @ c - y[va]) / (np.linalg.norm(y[va]) + 1e-12)
            res.append((th, int((c != 0).sum()), float(e)))
        emin = min(r[2] for r in res)
        best = min([r for r in res if r[2] <= (1 + selection_tolerance) * emin + 1e-4], key=lambda r: (r[1], r[2]))
        ths.append(best[0])
    return ths


def _backward_path(G, b, ridge=1e-9):
    """Greedy backward elimination on normal equations (columns pre-normalised).
    At each step drop the term whose removal increases the residual least: c_j^2 / (G_S^-1)_jj
    (i.e. the smallest |t|-statistic). Unlike magnitude thresholding this is not fooled by large
    mutually-cancelling coefficients of collinear library terms. Returns [(active_idx, coef)] for
    model sizes M, M-1, ..., 1."""
    M = G.shape[0]
    S = list(range(M))
    path = []
    while S:
        Gs = G[np.ix_(S, S)] + ridge * np.trace(G) / M * np.eye(len(S))
        try:
            Gi = np.linalg.inv(Gs)
        except np.linalg.LinAlgError:
            Gi = np.linalg.pinv(Gs)
        c = Gi @ b[S]
        path.append((list(S), c))
        if len(S) == 1:
            break
        drop = int(np.argmin(c ** 2 / np.maximum(np.diag(Gi), 1e-300)))
        S.pop(drop)
    return path


def _solve_sub(G, b, S, ridge=1e-9):
    Gs = G[np.ix_(S, S)] + ridge * np.trace(G) / G.shape[0] * np.eye(len(S))
    try:
        return np.linalg.solve(Gs, b[S])
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(Gs, b[S], rcond=None)[0]


def _forward_path(G, b, kmax):
    """Greedy forward selection (add the term that most reduces the residual) up to kmax terms.
    Complements backward elimination when exact collinearity (e.g. a conservation law making
    1, S, I, R linearly dependent) lets backward elimination drop the 'right' member of a group."""
    M = G.shape[0]
    S, path = [], []
    for _ in range(min(kmax, M)):
        best, bj, bc = -np.inf, None, None
        for j in range(M):
            if j in S:
                continue
            T = S + [j]
            c = _solve_sub(G, b, T)
            gain = float(b[T] @ c)            # = |y|^2 - RSS
            if gain > best:
                best, bj, bc = gain, j, c
        S = S + [bj]
        path.append((list(S), bc))
    return path


def _select_sparse(A_tr, y_tr, A_va, y_va, tol, kmax_forward=12):
    """Candidate supports of every size from backward elimination and forward selection (the one
    with lower training residual wins at each size); then the sparsest size whose held-out error is
    within (1+tol) of the best."""
    norms = np.linalg.norm(A_tr, axis=0) + 1e-12
    At, Av = A_tr / norms, A_va / norms
    G, bb = At.T @ At, At.T @ y_tr
    cands = {}
    for S, c in _backward_path(G, bb) + _forward_path(G, bb, kmax_forward):
        gain = float(bb[S] @ c)
        if len(S) not in cands or gain > cands[len(S)][2]:
            cands[len(S)] = (S, c, gain)
    sizes = sorted(cands)
    ny = np.linalg.norm(y_va) + 1e-12
    errs = {k: np.linalg.norm(Av[:, cands[k][0]] @ cands[k][1] - y_va) / ny for k in sizes}
    emin = min(errs.values())
    k = min(k for k in sizes if errs[k] <= (1 + tol) * emin + 1e-4)
    S, c, _ = cands[k]
    out = np.zeros(A_tr.shape[1])
    out[S] = c
    return out / norms


def ensemble_sindy(meta, data, n_models=50, bagging="time_blocks", library_bagging=0.0, inclusion_threshold=0.6,
                   method="stepwise", threshold=None, poly_degree=3, max_deriv=4, include_trig=False,
                   custom_terms=(), exclude_terms=(),
                   thresholds=DEFAULT_THRESHOLDS, ridge=1e-6, window=9, order=3, lowpass_frac=0.3,
                   diff_method="savgol", selection_tolerance=0.05, targets=None, library_vars=None,
                   max_rows=20000, block_len=None, report_min_inclusion=0.1, seed=0, validate_consensus=True):
    """E-SINDy: bootstrap-aggregated sparse regression on a library built once.

    bagging: 'time_blocks' (default; resample contiguous (traj, time[, x]) blocks - honest for
             correlated rows), 'trajectory' (resample whole trajectories; needs n_traj >= 3,
             else falls back to time_blocks), 'rows' (iid rows; intervals too narrow).
    library_bagging: probability of dropping each library term in a given model (0 = off);
             inclusion probability is then computed over the models in which the term was available.
    method: 'stepwise' (default) - per bag, candidate supports of every size from backward
             elimination (drop smallest |t|) and forward selection; the sparsest one within
             selection_tolerance of the best OUT-OF-BAG error is kept. Robust to collinear libraries
             (cubic terms in Lorenz, conservation laws in SIR) where magnitude thresholding keeps
             large cancelling spurious terms.
             'stlsq' - toolbox-style STLSQ with a fixed threshold per variable.
    threshold: (stlsq only) relative-contribution threshold. None -> per-variable threshold picked
             as in toolbox.run_sindy on the full data, then held fixed across bags.
    Library options (poly_degree, include_trig, custom_terms, ...) are the same as toolbox.run_sindy.
    Returns per-variable term table {term: [inclusion_prob, median_coef, q05, q95]} (coefficient
    stats over models that included the term), the consensus model (inclusion >= inclusion_threshold,
    coefficients refit by least squares on all data) and its toolbox.validate result."""
    rng = np.random.default_rng(seed)
    prep = _prepare(meta, data, window, order, lowpass_frac, diff_method)
    terms, cols = build_library(meta, prep["feats"], poly_degree, max_deriv, include_trig, custom_terms,
                                exclude_terms, library_vars)
    Theta = np.stack([np.broadcast_to(c, prep["grid"]).ravel() for c in cols], 1)
    ok = np.all(np.isfinite(Theta), axis=0)
    terms = [tm for tm, o in zip(terms, ok) if o]
    Theta = Theta[:, ok]
    Y = prep["Y"]
    variables = meta["variables"]
    tvars = [v for v in variables if not targets or v in targets]
    vidx = [variables.index(v) for v in tvars]
    N, M = Theta.shape

    if method == "stepwise":
        ths = [None] * len(tvars)
    elif threshold is None:
        tr, va = split_rows(meta, prep["grid"])
        ths = _select_thresholds(Theta, Y[:, vidx], tr, va, thresholds, ridge, selection_tolerance)
    else:
        ths = [float(threshold)] * len(tvars)

    if bagging == "trajectory" and prep["grid"][0] < 3:
        bagging = "time_blocks"
    if bagging == "time_blocks":
        blocks = _blocks(prep, block_len)
    elif bagging == "trajectory":
        blocks = [prep["idx"][j].ravel() for j in range(prep["grid"][0])]
    elif bagging != "rows":
        raise ValueError("bagging must be 'trajectory', 'rows' or 'time_blocks'")

    C = np.zeros((n_models, len(tvars), M))
    avail = np.ones((n_models, M), bool)
    for b in range(n_models):
        if bagging == "rows":
            rows = rng.integers(0, N, min(N, max_rows))
            oob = np.setdiff1d(np.arange(N), rows)
        else:
            pick = rng.integers(0, len(blocks), len(blocks))
            rows = np.concatenate([blocks[k] for k in pick])
            left = np.setdiff1d(np.arange(len(blocks)), pick)
            oob = np.concatenate([blocks[k] for k in left]) if left.size else np.zeros(0, int)
            if rows.size > max_rows:
                rows = rng.choice(rows, max_rows, replace=False)
        if oob.size < 20:                       # no out-of-bag rows -> validate on a random 20% of the bag
            oob = rng.choice(rows, max(rows.size // 5, 1), replace=False)
        oob = _sub(oob, max_rows // 2, rng)
        if library_bagging > 0:
            avail[b] = rng.random(M) >= library_bagging
            if not avail[b].any():
                avail[b, rng.integers(M)] = True
        A = Theta[rows][:, avail[b]]
        for k, i in enumerate(vidx):
            if method == "stepwise":
                C[b, k, avail[b]] = _select_sparse(A, Y[rows, i], Theta[oob][:, avail[b]], Y[oob, i],
                                                   selection_tolerance)
            else:
                C[b, k, avail[b]] = stlsq(A, Y[rows, i], ths[k], ridge)

    incl = (C != 0).sum(0) / np.maximum(avail.sum(0), 1)[None, :]       # (nv, M)
    table, consensus_terms, unstable = {}, {}, {}
    for k, v in enumerate(tvars):
        rowsv = {}
        order_ = np.argsort(-incl[k])
        for m in order_:
            p = incl[k, m]
            if p < report_min_inclusion:
                break
            nz = C[:, k, m][C[:, k, m] != 0]
            q05, q50, q95 = np.percentile(nz, [5, 50, 95])
            rowsv[terms[m]] = _r([p, q50, q05, q95])
        table[v] = rowsv
        consensus_terms[v] = [terms[m] for m in range(M) if incl[k, m] >= inclusion_threshold]
        unstable[v] = [terms[m] for m in order_ if 0.2 <= incl[k, m] < inclusion_threshold]

    # consensus model: least squares on the selected columns using all (subsampled) data
    rows_all = _sub(np.arange(N), 4 * max_rows, rng)
    rhs = {}
    for k, v in enumerate(tvars):
        sel = [terms.index(tm) for tm in consensus_terms[v]]
        c = np.zeros(M)
        if sel:
            c[sel] = _lstsq(Theta[rows_all][:, sel], Y[rows_all, vidx[k]])
        rhs[v] = to_expr(c, terms, 5)
    full = {v: rhs.get(v, "0") for v in variables}
    out = {"n_models": n_models, "bagging": bagging, "method": method, "library_size": M,
           "inclusion_threshold": inclusion_threshold,
           "terms": table,
           "terms_doc": "term: [inclusion_prob, median_coef, q05, q95] (coef stats over models including the term)",
           "unstable_terms": {v: u for v, u in unstable.items() if u},
           "consensus_rhs": full}
    if method == "stlsq":
        out["thresholds"] = dict(zip(tvars, ths))
    if validate_consensus:
        val = validate(meta, data, full)
        out["validation"] = {k: val[k] for k in val if k != "rhs"}
    return _r(out)


# ----------------------------------------------------------------------------- 2. cross-validation
def _fold_train_data(meta, data, prep, fold_name, folds_kind, n_blocks):
    """Public-data subset for refitting a fitter on the training part of a fold."""
    U, t = data["U"], data["t"]
    if folds_kind == "trajectory":
        j = int(fold_name[4:])
        keep = [i for i in range(U.shape[0]) if i != j]
        Ut, tt = U[keep], t
    else:
        # raw-time blocks aligned with the trimmed-grid fold edges; the other blocks become
        # pseudo-trajectories of equal length (autonomous systems assumed: t is reset per block)
        k = int(fold_name[5:])
        nt_e = prep["grid"][1]
        e = np.linspace(0, nt_e, n_blocks + 1).astype(int) + prep["t_offset"]
        e[0], e[-1] = 0, U.shape[1]
        segs = [(e[i], e[i + 1]) for i in range(n_blocks) if i != k]
        Lb = min(b - a for a, b in segs)
        Ut = np.concatenate([U[:, a:a + Lb] for a, b in segs], axis=0)
        tt = t[:Lb]
    m = dict(meta)
    m["n_traj"] = int(Ut.shape[0])
    m["shape"] = list(Ut.shape)
    d = {"U": Ut, "t": tt}
    if "x" in data:
        d["x"] = data["x"]
    return m, d


def _term_sets(meta, rhs):
    try:
        return {v: {tm for tm, _ in tl} for v, tl in _structure(meta, rhs).items()}
    except Exception:  # noqa: BLE001
        return {v: set() for v in meta["variables"]}


def cross_validate(meta, data, rhs_or_fitter, folds="auto", n_blocks=4, refit_coefficients=False,
                   max_rollouts_per_fold=2, max_rollout_steps=3000, window=9, order=3, lowpass_frac=0.3,
                   diff_method="savgol", seed=0, rollouts=True):
    """K-fold assessment on public data.

    rhs_or_fitter:
      * dict {var: expr} (or {'rhs': {...}}): score the model on each held-out fold - derivative
        nrmse on the fold's rows + rollout valid time from the start of the fold's segment(s).
        refit_coefficients=True refits the structure's LINEAR coefficients on the other folds first
        (genuine out-of-sample test of the structure; nonlinear inner constants stay fixed).
      * callable fitter(meta, data) -> {'rhs': {...}}: refit on the training folds, score on the
        held-out fold, and report term-set stability across folds.
    folds: 'trajectory' (leave-one-trajectory-out), 'time_blocks' (n_blocks contiguous time blocks
        across all trajectories) or 'auto' (trajectory if n_traj >= 3 else time_blocks)."""
    prep = _prepare(meta, data, window, order, lowpass_frac, diff_method)
    kind, fold_list = _folds(prep, folds, n_blocks)
    variables = meta["variables"]
    out = {"folds": kind, "n_folds": len(fold_list)}
    if not callable(rhs_or_fitter):
        rhs = rhs_or_fitter.get("rhs", rhs_or_fitter)
        try:
            struct = _structure(meta, rhs)
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}
        per_fold, _ = _cv_structure(prep, struct, fold_list, refit_coefficients, max_rollouts_per_fold,
                                    max_rollout_steps, seed=seed, rollouts=rollouts)
        out.update({"mode": "fixed_rhs" + ("+refit" if refit_coefficients else ""),
                    "n_terms": sum(len(v) for v in struct.values())})
        out.update(_summarise_folds(per_fold))
        out["per_fold"] = per_fold
        return _r(out)

    # ---- fitter mode
    N = prep["Y"].shape[0]
    per_fold, sets, models = [], [], []
    for name, held, segs in fold_list:
        mt, dt_ = _fold_train_data(meta, data, prep, name, kind, n_blocks)
        rec = {"fold": name}
        try:
            res = rhs_or_fitter(mt, dt_)
            rhs = {v: str(res.get("rhs", res).get(v, "0")) for v in variables}
            struct = _structure(meta, rhs)
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"[:200]
            per_fold.append(rec)
            continue
        cols = _structure_columns(prep, struct)
        P = _predict(cols, struct, {v: [c for _, c in struct[v]] for v in variables}, variables, N)
        rec["deriv_nrmse"] = float(_deriv_err(prep, P, held).mean())
        rec["n_terms"] = sum(len(v) for v in struct.values())
        if rollouts:
            rs = [r for r in (_rollout(prep, rhs, s, max_rollout_steps) for s in segs[:max_rollouts_per_fold]) if r]
            if rs:
                rec["rollout_valid_time"] = float(np.mean([r["valid_time"] for r in rs]))
                rec["rollout_valid_frac"] = float(np.mean([r["valid_frac"] for r in rs]))
                rec["rollout_blew_up"] = bool(any(r["blew_up"] for r in rs))
        per_fold.append(rec)
        sets.append({v: {tm for tm, _ in struct[v]} for v in variables})
        models.append(struct)
    ok = [f for f in per_fold if "deriv_nrmse" in f]
    out["mode"] = "fitter"
    if ok:
        out.update(_summarise_folds(ok))
    # term-set stability
    if sets:
        freq, coef = {}, {}
        for v in variables:
            cnt = {}
            for s, st in zip(sets, models):
                for tm, c in st[v]:
                    cnt[tm] = cnt.get(tm, 0) + 1
                    coef.setdefault((v, tm), []).append(c)
            freq[v] = {tm: [n / len(sets), float(np.median(coef[(v, tm)]))]
                       for tm, n in sorted(cnt.items(), key=lambda kv: -kv[1])}
        jac = []
        for a, b in itertools.combinations(range(len(sets)), 2):
            ua = set().union(*[{(v, t) for t in sets[a][v]} for v in variables])
            ub = set().union(*[{(v, t) for t in sets[b][v]} for v in variables])
            jac.append(len(ua & ub) / max(len(ua | ub), 1))
        out["term_stability"] = {
            "mean_pairwise_jaccard": float(np.mean(jac)) if jac else 1.0,
            "identical_structure_all_folds": bool(all(s == sets[0] for s in sets)),
            "term_freq": freq,
            "term_freq_doc": "term: [fraction of folds selecting it, median coef]"}
    out["per_fold"] = per_fold
    return _r(out)


# ----------------------------------------------------------------------------- 3. model comparison
def _paired_block_test(prep, oof_a, oof_b, blocks):
    """Paired block comparison of normalised squared out-of-fold residuals. Returns
    (mean diff a-b, z). Positive diff => a has larger error."""
    Y, ms = prep["Y"], prep["ms"]
    with np.errstate(all="ignore"):
        ea = np.nanmean(np.minimum((oof_a - Y) ** 2 / ms, 100.0), axis=1)
        eb = np.nanmean(np.minimum((oof_b - Y) ** 2 / ms, 100.0), axis=1)
    d = np.array([np.nanmean(ea[b] - eb[b]) for b in blocks])
    d = d[np.isfinite(d)]
    if d.size < 2:
        return 0.0, 0.0
    se = d.std(ddof=1) / np.sqrt(d.size)
    return float(d.mean()), float(d.mean() / se) if se > 0 else (np.inf if d.mean() != 0 else 0.0)


def compare_models(meta, data, candidates: dict, folds="auto", n_blocks=4, refit=True, rel_tol=0.05, z_tol=2.0,
                   max_rollouts_per_fold=2, max_rollout_steps=3000, window=9, order=3, lowpass_frac=0.3,
                   diff_method="savgol", seed=0, rollouts=True):
    """Rank candidate models {name: rhs} on public data.

    For each candidate: K-fold CV derivative error (refit=True refits the structure's linear
    coefficients on the training folds, so over-fitted structures are penalised out-of-sample),
    rollout valid time per fold of the candidate AS GIVEN (its own coefficients), n_terms, and AICc / BIC from in-sample derivative residuals with
    n_terms parameters and an effective sample size n_eff = rows / correlation length (smoothing
    window, x low-pass) - plain row counts would make any extra term look significant.
    A model is 'indistinguishable' from the best-CV one if a paired block test of out-of-fold
    squared residuals gives |z| < z_tol, or if it is within rel_tol (relative) AND has a BIC no worse
    than the best (significant-but-tiny differences, e.g. fitting smoothing bias). Among the models indistinguishable from
    the best, the one with fewest terms is preferred (ties -> BIC).
    A noise floor (see noise_floor) says whether the remaining error is at the level of the data."""
    prep = _prepare(meta, data, window, order, lowpass_frac, diff_method)
    kind, fold_list = _folds(prep, folds, n_blocks)
    variables = meta["variables"]
    N, nv = prep["Y"].shape
    blocks = _blocks(prep)
    corr = window * (max(1, int(round(1 / lowpass_frac))) if meta["kind"] == "pde" and lowpass_frac else 1)
    n_eff = max(N / corr, 10.0)
    rng = np.random.default_rng(seed)
    rows_fit = _sub(np.arange(N), 40000, rng)

    res, oofs = {}, {}
    for name, cand in candidates.items():
        rhs = cand.get("rhs", cand) if isinstance(cand, dict) else cand
        try:
            struct = _structure(meta, rhs)
        except Exception as e:  # noqa: BLE001
            res[name] = {"error": str(e)[:200]}
            continue
        k = sum(len(v) for v in struct.values())
        per_fold, oof = _cv_structure(prep, struct, fold_list, refit, seed=seed, rollouts=False)
        if rollouts:   # roll out the candidate AS GIVEN (its own coefficients) from each fold's start
            rhs_g = _rhs_from(struct)
            for rec, (_, _, segs) in zip(per_fold, fold_list):
                rs = [x for x in (_rollout(prep, rhs_g, sg, max_rollout_steps) for sg in segs[:max_rollouts_per_fold])
                      if x]
                if rs:
                    rec["rollout_valid_time"] = float(np.mean([x["valid_time"] for x in rs]))
                    rec["rollout_valid_frac"] = float(np.mean([x["valid_frac"] for x in rs]))
                    rec["rollout_blew_up"] = bool(any(x["blew_up"] for x in rs))
        cols = _structure_columns(prep, struct)
        coefs = ({v: _lstsq(cols[v][rows_fit], prep["Y"][rows_fit, i]) for i, v in enumerate(variables)} if refit
                 else {v: [c for _, c in struct[v]] for v in variables})
        P = _predict(cols, struct, coefs, variables, N)
        with np.errstate(all="ignore"):
            mse = np.nanmean((P - prep["Y"]) ** 2, axis=0) / prep["ms"]
        mse = np.where(np.isfinite(mse), np.maximum(mse, 1e-30), 1e6)
        ll = n_eff * float(np.sum(np.log(mse)))
        n_tot = n_eff * nv
        aicc = ll + 2 * k + (2 * k * (k + 1) / (n_tot - k - 1) if n_tot - k - 1 > 0 else np.inf)
        bic = ll + k * np.log(n_tot)
        oof_err = _deriv_err(prep, oof, np.arange(N))
        r = {"n_terms": k, "cv_deriv_nrmse": float(oof_err.mean()),
             "cv_deriv_nrmse_per_var": {v: float(e) for v, e in zip(variables, oof_err)},
             "insample_deriv_nrmse": float(np.sqrt(mse).mean()), "aicc": aicc, "bic": bic}
        s = _summarise_folds(per_fold)
        r["cv_fold_std"] = s["deriv_nrmse_std"]
        for key in ("rollout_valid_time_mean", "rollout_valid_frac_mean", "any_blow_up"):
            if key in s:
                r[key] = s[key]
        res[name] = r
        oofs[name] = oof

    good = [n for n in res if "error" not in res[n]]
    if not good:
        return {"error": "no valid candidates", "candidates": res}
    for key in ("aicc", "bic"):
        mn = min(res[n][key] for n in good)
        for n in good:
            res[n]["d" + key] = res[n].pop(key) - mn

    best = min(good, key=lambda n: res[n]["cv_deriv_nrmse"])
    e_best = res[best]["cv_deriv_nrmse"]
    equiv = []
    for n in good:
        rel = res[n]["cv_deriv_nrmse"] / max(e_best, 1e-12) - 1
        _, z = (0.0, 0.0) if n == best else _paired_block_test(prep, oofs[n], oofs[best], blocks)
        res[n]["rel_err_vs_best"] = rel
        res[n]["z_vs_best"] = z
        if n == best or abs(z) < z_tol:
            equiv.append(n)
        elif rel < rel_tol and res[n]["dbic"] <= res[best]["dbic"]:
            equiv.append(n)                       # significant but tiny, and BIC favours it
            res[n]["practically_equivalent"] = True
    pick = min(equiv, key=lambda n: (res[n]["n_terms"], res[n]["dbic"]))
    ranking = sorted(good, key=lambda n: (n not in equiv, res[n]["n_terms"] if n in equiv else 0,
                                          res[n]["cv_deriv_nrmse"]))

    nf = noise_floor(meta, data, window, order, lowpass_frac, diff_method, prep=prep)
    floor = nf["deriv_nrmse_floor_mean"]
    ratio = res[pick]["cv_deriv_nrmse"] / max(floor, 1e-12)

    # verdict text
    fmt = lambda n: f"{n} ({res[n]['cv_deriv_nrmse']:.3g}, {res[n]['n_terms']} terms)"
    parts = [f"Lowest CV derivative error: {fmt(best)}."]
    others_eq = [n for n in equiv if n != best]
    if others_eq:
        parts.append("Indistinguishable from it: " + ", ".join(
            f"{n} ({100 * res[n]['rel_err_vs_best']:+.1f}%, z={res[n]['z_vs_best']:.1f}"
            + (", <" + f"{100 * rel_tol:.0f}% and lower BIC" if res[n].get("practically_equivalent") else "") + ")"
            for n in others_eq) + ".")
    worse = [n for n in good if n not in equiv]
    if worse:
        parts.append("Significantly worse: " + ", ".join(
            f"{n} ({100 * res[n]['rel_err_vs_best']:+.1f}%, z={res[n]['z_vs_best']:.1f}, dBIC={res[n]['dbic']:.0f})"
            for n in worse) + ".")
    if pick != best:
        parts.append(f"{pick} is simpler ({res[pick]['n_terms']} vs {res[best]['n_terms']} terms) -> prefer {pick}.")
    else:
        parts.append(f"Prefer {pick}.")
    if rollouts and "rollout_valid_frac_mean" in res[pick]:
        vf_best = max(res[n].get("rollout_valid_frac_mean", 0) for n in good)
        vf = res[pick]["rollout_valid_frac_mean"]
        if vf < vf_best - 0.25:
            better = max(good, key=lambda n: res[n].get("rollout_valid_frac_mean", 0))
            parts.append(f"CAUTION: {pick} rolls out much worse (valid frac {vf:.2f} vs {vf_best:.2f} for {better}); "
                         "derivative-based and rollout-based rankings disagree - derivative estimates may be "
                         "biased (coarse dt / heavy smoothing), so trust the rollout or use weak-form/trajectory "
                         "fitting before deciding.")
        else:
            parts.append(f"Rollout valid frac {vf:.2f} (best {vf_best:.2f}).")
    if ratio < 1.3:
        parts.append(f"Its error is AT the estimated noise floor ({floor:.3g}, ratio {ratio:.2f}): "
                     "the model is about as good as these data allow.")
    elif ratio < 2.0:
        parts.append(f"Its error is NEAR the noise floor ({floor:.3g}, ratio {ratio:.2f}): little left to gain.")
    else:
        parts.append(f"Its error is {ratio:.1f}x the noise floor ({floor:.3g}): systematic error remains "
                     "(missing/wrong terms, or differentiation bias / correlated noise).")
    out = {"preferred": pick, "best_cv": best, "indistinguishable_from_best": equiv, "ranking": ranking,
           "verdict": " ".join(parts), "noise_floor": nf, "error_to_floor_ratio": ratio,
           "folds": kind, "n_eff": n_eff, "refit_per_fold": refit, "candidates": res}
    return _r(out)


# ----------------------------------------------------------------------------- 4. coefficient bootstrap
def coefficient_uncertainty(meta, data, rhs, n_boot=100, block_len=None, window=9, order=3, lowpass_frac=0.3,
                            diff_method="savgol", max_rows=40000, seed=0):
    """Block-bootstrap intervals for the linear coefficients of a FIXED structure (terms of the
    expanded rhs). Each replicate resamples contiguous (traj, time[, x]) blocks with replacement
    and refits by least squares. Returns per variable {term: {given, fit, median, ci90, rel_ci_halfwidth,
    sig}} where sig = 0 not inside the 90% interval. Nonlinear inner constants (e.g. K in x/(K+x))
    are held fixed."""
    rhs = rhs.get("rhs", rhs)
    prep = _prepare(meta, data, window, order, lowpass_frac, diff_method)
    struct = _structure(meta, rhs)
    cols = _structure_columns(prep, struct)
    rng = np.random.default_rng(seed)
    N = prep["Y"].shape[0]
    blocks = _blocks(prep, block_len)
    variables = meta["variables"]
    fit_rows = _sub(np.arange(N), max_rows, rng)
    fit = {v: _lstsq(cols[v][fit_rows], prep["Y"][fit_rows, i]) for i, v in enumerate(variables)}
    B = {v: np.zeros((n_boot, len(struct[v]))) for v in variables}
    for b in range(n_boot):
        rows = np.concatenate([blocks[k] for k in rng.integers(0, len(blocks), len(blocks))])
        rows = _sub(rows, max_rows, rng)
        for i, v in enumerate(variables):
            if struct[v]:
                B[v][b] = _lstsq(cols[v][rows], prep["Y"][rows, i])
    out = {}
    for v in variables:
        tv = {}
        for j, (tm, c0) in enumerate(struct[v]):
            q05, q50, q95 = np.percentile(B[v][:, j], [5, 50, 95])
            tv[tm] = {"given": c0, "fit": fit[v][j], "median": q50, "ci90": [q05, q95],
                      "rel_ci_halfwidth": (q95 - q05) / 2 / (abs(q50) + 1e-12), "sig": bool(q05 > 0 or q95 < 0)}
        out[v] = tv
    insig = {v: [tm for tm, d in out[v].items() if not d["sig"]] for v in variables}
    return _r({"n_boot": n_boot, "n_blocks": len(blocks), "coefs": out,
               "not_significant": {v: t for v, t in insig.items() if t},
               "fit_rhs": _rhs_from(struct, fit, 5)})
