"""Data audit (pre-fit): guards the data before any model is fitted. Deterministic numpy/scipy, no imputation.

    audit(meta, data, rhs=None) -> list[Finding]        (stage "data")

Detectors
    outliers              isolated spikes: leave-one-out cubic residual, robust (MAD) z, fraction with |z| > 6
    gaps                  missing samples (NaN) or non-uniform time stamps
    gaps_state_dependent  is missingness predictable from the state? (censored tails)
    coarse_sampling       signal changes a lot per time step              (re-expressed from assess.data_advice)
    single_trajectory     fewer than 3 independent runs                   (re-expressed from assess.data_advice)
    grid_scale_signal     PDE spectrum reaches the grid scale              (re-expressed from assess.data_advice)

Repair tools (called by the integration layer through Finding["fix"])
    despike(meta, data, **args)        -> (data_clean, n_replaced)
    split_at_gaps(meta, data, min_len) -> (meta2, data2)
"""
import copy
import warnings

import numpy as np
from scipy import ndimage, stats

from . import _safe, finding, threshold

STAGE = "data"
Z_SPIKE = 6.0       # |z| beyond which a sample counts as a spike
K_LOO = 3           # leave-one-out cubic prediction from the +-K_LOO neighbours
HALF = 7            # half-width of the local (Hampel) window
LOCAL_DIV = 2.0     # local scale only counts where it exceeds the global one by this factor
AXIS_RATIO = 2.0    # PDE: test along every axis whose residual scale is within this factor of the best one
FLOOR_REL = 1e-3    # residual scale floor (x field std): spikes below this never matter


def _loo_weights(k):
    x = np.array([i for i in range(-k, k + 1) if i])
    return dict(zip(x.tolist(), np.linalg.pinv(np.vander(x, 4, increasing=True))[0]))


# ----------------------------------------------------------------------------------------------- helpers
def _spatial_axes(U):
    return tuple(range(2, U.ndim - 1))


def _periodic(meta):
    return meta.get("kind") == "pde" and meta.get("boundary", "periodic") == "periodic"


def _shift(A, k, axis, periodic):
    """S[i] = A[i + k] along `axis` (NaN beyond the ends unless periodic)."""
    if periodic:
        return np.roll(A, -k, axis=axis)
    S = np.full(A.shape, np.nan)
    n = A.shape[axis]
    src = [slice(None)] * A.ndim
    dst = [slice(None)] * A.ndim
    if k >= 0:
        src[axis], dst[axis] = slice(k, n), slice(0, n - k)
    else:
        src[axis], dst[axis] = slice(0, n + k), slice(-k, n)
    S[tuple(dst)] = A[tuple(src)]
    return S


def _robust_sigma(r):
    r = r[np.isfinite(r)]
    if r.size == 0:
        return np.nan
    return 1.4826 * np.median(np.abs(r - np.median(r)))


def _local_median(A, ax, half, periodic):
    stack = np.stack([_shift(A, k, ax, periodic) for k in range(-half, half + 1)], 0)
    med = np.median(stack, axis=0)          # fast path; nanmedian only where a NaN is in the window
    bad = np.isnan(med)
    if bad.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med[bad] = np.nanmedian(stack[:, bad], axis=0)
    return med


def _spike_z(meta, U, half=None, k=None):
    """Robust (Hampel) z of every sample against a leave-one-out cubic prediction from its neighbours.

    r = u - (least-squares cubic through the +-k neighbours, centre excluded, evaluated at the centre).
    z = (r - local median of r) / scale, scale = max(global MAD of r, local MAD of r over +-`half` / LOCAL_DIV):
    the local term stops smooth curvature error (noise-free or sharp data) from looking like spikes, the global
    term (floored at FLOOR_REL x the field std) stops quiet stretches from doing so.
    Computed along time and (PDE) every spatial axis. Axes whose residual scale is > AXIS_RATIO x the best axis are
    dominated by signal curvature (coarse sampling) and are ignored; a sample must stand out along every remaining axis
    (min |z|; NaN if untestable along any of them), so shocks and fronts are not taken for spikes.
    Returns z (U.shape, NaN where undefined), the axes used, the best axis, and the per-field noise std."""
    U = np.asarray(U, float)
    half, k = half or HALF, k or K_LOO
    w = _loo_weights(k)
    gain = np.sqrt(1 + sum(v ** 2 for v in w.values()))      # std of r / noise std (white noise)
    axes = (1,) + (_spatial_axes(U) if meta.get("kind") == "pde" else ())
    nf = U.shape[-1]
    sd = np.nanstd(U.reshape(-1, nf), axis=0)
    floor = threshold("outliers_floor_rel", FLOOR_REL) * sd     # noise-free data
    zs, sig = {}, {}
    for ax in axes:
        per = _periodic(meta) and ax >= 2
        if U.shape[ax] < 2 * max(half, k) + 1:
            continue
        r = U - sum(c * _shift(U, j, ax, per) for j, c in w.items())
        sig[ax] = np.maximum([_robust_sigma(r[..., i]) for i in range(nf)], floor)
        zs[ax] = r
    if not zs:
        return np.full(U.shape, np.nan), (), 1, np.full(nf, np.nan)
    rel = {ax: np.nanmedian(sig[ax] / (sd + 1e-300)) for ax in zs}
    best = min(rel, key=lambda a: rel[a] if np.isfinite(rel[a]) else np.inf)
    used = tuple(ax for ax in zs if np.isfinite(rel[ax]) and rel[ax] <= AXIS_RATIO * rel[best])
    if rel[best] <= 3 * threshold("outliers_floor_rel", FLOOR_REL):
        used = tuple(zs)    # (nearly) noise-free: every axis is curvature-limited, all of them must agree
    z = None
    for ax in used:
        per = _periodic(meta) and ax >= 2
        rc = zs[ax] - _local_median(zs[ax], ax, half, per)
        loc = 1.4826 * _local_median(np.abs(rc), ax, half, per)
        a_ = np.abs(rc / np.maximum(np.where(sig[ax] > 0, sig[ax], np.nan), loc / LOCAL_DIV))
        z = a_ if z is None else np.minimum(z, a_)
    return z, used, best, sig[best] / gain


# ---------------------------------------------------------------------------------------------- outliers
def _outliers(meta, data):
    U = np.asarray(data["U"], float)
    z, used, _, sig = _spike_z(meta, U)
    ok = np.isfinite(z)
    n = int(ok.sum())
    if n == 0:
        return []
    flag = ok & (z > Z_SPIKE)
    frac = float(flag.sum() / n)
    expected = float(2 * stats.norm.sf(Z_SPIKE))
    thr = threshold("outliers", 2e-4)
    crit = threshold("outliers_critical", 0.02)
    fired = frac > thr
    per_var = {v: float(flag[..., i].sum() / max(ok[..., i].sum(), 1)) for i, v in enumerate(meta["variables"])}
    sev = "critical" if frac > crit else "warn"
    worst = max(per_var, key=per_var.get)
    msg = (f"{frac:.2%} of samples are isolated spikes (|z| > {Z_SPIKE:g} against their neighbours; "
           f"Gaussian noise would give {expected:.0e}), worst in {worst} ({per_var[worst]:.2%}): despike before "
           f"fitting." if fired else
           f"No excess of isolated spikes ({frac:.3%} of samples with |z| > {Z_SPIKE:g}).")
    return [finding("outliers", STAGE, frac, thr, fired, sev if fired else "info", "repair" if fired else None,
                    fix={"tool": "despike", "args": {"z": 4.5}} if fired else None, message=msg,
                    details={"n_flagged": int(flag.sum()), "n_tested": n, "gaussian_expectation": expected,
                             "excess_ratio": frac / expected, "per_variable": per_var,
                             "axes_used": ["t" if a == 1 else f"space{a - 2}" for a in used],
                             "noise_sigma": dict(zip(meta["variables"], np.asarray(sig, float).tolist()))})]


def despike(meta, data, z=4.5, max_iter=3):
    """Hampel-style despiking: samples whose robust leave-one-out z exceeds `z` are replaced by the median of
    their 4 nearest neighbours along the best-resolved axis (time for ODEs). Only isolated spikes are touched;
    NaNs are left as they are (no imputation). Returns (data_clean, n_replaced)."""
    U = np.array(data["U"], float, copy=True)
    n_rep = 0
    for _ in range(max_iter):
        zz, _, ax, _ = _spike_z(meta, U)
        flag = np.isfinite(zz) & (zz > z)
        if not flag.any():
            break
        per = _periodic(meta) and ax >= 2
        nb = np.stack([_shift(U, k, ax, per) for k in (-2, -1, 1, 2)], 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(nb, axis=0)
        flag &= np.isfinite(med)
        U[flag] = med[flag]
        n_rep += int(flag.sum())
    out = dict(data)
    out["U"] = U
    return out, n_rep


# -------------------------------------------------------------------------------------------------- gaps
def _row_missing(U):
    """(n_traj, nt) bool: time rows with any NaN."""
    return ~np.isfinite(U).reshape(U.shape[0], U.shape[1], -1).all(-1)


def _runs(b):
    """Contiguous True runs of a 1-D bool array -> list of (start, stop)."""
    d = np.diff(np.concatenate([[0], b.astype(int), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _gaps(meta, data):
    U, t = np.asarray(data["U"], float), np.asarray(data["t"], float)
    miss = ~np.isfinite(U)
    frac = float(miss.mean())
    rows = _row_missing(U)
    full_rows = miss.reshape(U.shape[0], U.shape[1], -1).all(-1)
    dt = np.diff(t)
    nonuni = float(np.std(dt) / (abs(np.mean(dt)) + 1e-300)) if dt.size > 1 else 0.0
    windows = [(j, float(t[a]), float(t[b - 1])) for j in range(U.shape[0]) for a, b in _runs(rows[j])]
    thr = threshold("gaps", 0.0)
    fired = frac > thr or nonuni > 1e-6
    if not fired:
        return [finding("gaps", STAGE, frac, thr, False, "info", message="No missing samples; uniform sampling.",
                        details={"nan_fraction": 0.0, "dt_rel_variation": nonuni})]
    min_len = _default_min_len(meta)
    plan = _plan_split(rows, min_len) if frac > thr else None
    kept = float(len(plan[1]) * plan[0] / rows.size) if plan else 0.0
    parts = []
    if frac > thr:
        partial = float(rows.mean() - full_rows.mean())
        if partial > 0.01:
            where = (f"NaNs touch {rows.mean():.0%} of time rows ({partial:.0%} only partly missing, i.e. holes in "
                     f"space)")
        else:
            ex = "; ".join(f"traj {j} t=[{a:g}, {b:g}]" for j, a, b in windows[:4])
            more = f" and {len(windows) - 4} more" if len(windows) > 4 else ""
            where = f"in {len(windows)} time window(s) ({ex}{more})"
        parts.append(f"{frac:.1%} of samples are missing (NaN): {where}.")
        if plan:
            parts.append(f"Fit on NaN-free segments (split_at_gaps keeps {kept:.0%} of the time rows in "
                         f"{len(plan[1])} windows of {plan[0]} steps) or with the weak form; no imputation.")
        else:
            parts.append("Too few NaN-free time rows to split into segments; only a fit that masks missing "
                         "samples can use these data (no imputation).")
    if nonuni > 1e-6:
        parts.append(f"Time stamps are non-uniform (dt varies by {nonuni:.1%}).")
    return [finding("gaps", STAGE, frac, thr, True, "warn", "repair" if plan else "scope",
                    fix={"tool": "split_at_gaps", "args": {"min_len": min_len}} if plan else None,
                    message=" ".join(parts),
                    details={"nan_fraction": frac, "rows_with_nan_fraction": float(rows.mean()),
                             "rows_partly_missing_fraction": float(rows.mean() - full_rows.mean()),
                             "windows": windows[:50], "n_windows": len(windows), "dt_rel_variation": nonuni,
                             "split_L": plan[0] if plan else None, "split_n_windows": len(plan[1]) if plan else 0,
                             "split_kept_row_fraction": kept})]


def _plan_split(rows, min_len):
    """(L, [(traj, start), ...]) for split_at_gaps, or None if no NaN-free run has >= 8 rows."""
    runs = [(j, a, b) for j in range(rows.shape[0]) for a, b in _runs(~rows[j])]
    lens = np.array([b - a for _, a, b in runs])
    if lens.size == 0 or lens.max() < 8:
        return None
    min_len = min(int(min_len), int(lens.max()))
    cand = np.arange(min_len, lens.max() + 1)
    target = max(min_len, lens.max() / 2)      # favour long pieces (rollout / validation horizons)
    usable = np.array([(lens // L).sum() * (L - EDGE_LOSS) * min(1.0, L / target) for L in cand])
    L = int(cand[np.argmax(usable)])
    starts = []
    for j, a, b in runs:
        k = (b - a) // L
        s0 = a + ((b - a) - k * L) // 2
        starts += [(j, int(s0 + i * L)) for i in range(k)]
    return L, starts


def _default_min_len(meta):
    """ODE: 4x the default 9-point smoothing window. PDE: every snapshot carries a whole field, so shorter windows
    (2x the window + 2) still hold plenty of rows; demanding 36 would throw most of a 100-200 step record away."""
    return 20 if meta.get("kind") == "pde" else 36


EDGE_LOSS = 8   # rows per window lost to differentiation / test-function support at the two ends


def split_at_gaps(meta, data, min_len=None):
    """Cut every trajectory at time rows containing a NaN into contiguous NaN-free runs, tile each run into
    windows of a common length L and stack the windows as new trajectories (no imputation).

    L >= min_len maximises usable rows x min(1, L / L_target): usable rows = sum over windows of (L - EDGE_LOSS),
    L_target = half the longest NaN-free run. Long pieces keep rollout / validation horizons, so runs shorter than
    L are dropped rather than shrinking L to fit them (e.g. a 1000-step record with a gap at steps 100-140 gives
    one ~860-step piece per trajectory, not many short ones). If no run reaches min_len, L = the longest run
    (>= 8 rows, else ValueError).
    Every piece restarts time: t = t[:L] (starting at the dataset's t0). This assumes autonomous dynamics; with
    explicit time dependence (forcing) the true start times matter: they are in meta["segment_t0"] (one per piece,
    original time of its first row) and meta["split_at_gaps"]["source"] = [traj, start_index].
    Returns (meta2, data2); meta2 has updated n_traj / shape. PDE rows with any NaN are dropped whole."""
    U, t = np.asarray(data["U"], float), np.asarray(data["t"], float)
    rows = _row_missing(U)
    if not rows.any():
        return copy.deepcopy(meta), dict(data)
    plan = _plan_split(rows, min_len or _default_min_len(meta))
    if plan is None:
        raise ValueError("no NaN-free segment of at least 8 time rows")
    L, starts = plan
    wins = [U[j, s:s + L] for j, s in starts]
    U2 = np.stack(wins, 0)
    meta2 = copy.deepcopy(meta)
    meta2["n_traj"] = int(U2.shape[0])
    meta2["shape"] = list(U2.shape)
    meta2["segment_t0"] = [float(t[s]) for _, s in starts]
    meta2["split_at_gaps"] = {"L": L, "n_windows": len(wins),
                              "kept_fraction": float(U2.size / max(np.isfinite(U).sum(), 1)),
                              "source": [list(s) for s in starts], "t0": [float(t[s]) for _, s in starts],
                              "note": "windows share t[:L]; assumes autonomous dynamics"}
    data2 = dict(data)
    data2["U"] = U2
    data2["t"] = t[:L].copy()
    return meta2, data2


# ---------------------------------------------------------------------------------- gaps_state_dependent
def _edge_stat(A, mask, conn_axes):
    """Where do observed samples adjacent to missing regions sit in the amplitude distribution?

    Every connected missing region (connected along `conn_axes`, never across trajectories) contributes one
    value: the mean percentile rank (among all observed samples) of A on its observed neighbours. Under
    missingness unrelated to A these are means of U(0,1) percentiles; Fisher's method combines them into a one-sided
    z (large = gap edges sit in the upper tail, i.e. the missing samples are the high-A ones)."""
    obs = ~mask & np.isfinite(A)
    if not mask.any() or obs.sum() < 20:
        return None
    p = np.full(A.shape, np.nan)
    p[obs] = stats.rankdata(A[obs]) / (obs.sum() + 1)
    st = np.zeros((3,) * mask.ndim, bool)
    c = (1,) * mask.ndim
    st[c] = True
    for ax in conn_axes:
        for d in (0, 2):
            idx = list(c)
            idx[ax] = d
            st[tuple(idx)] = True
    lab, nreg = ndimage.label(mask, structure=st)
    sums, cnts = np.zeros(nreg + 1), np.zeros(nreg + 1)
    edge_A = []
    for ax in conn_axes:
        for k in (1, -1):
            nb = np.nan_to_num(_shift(lab.astype(float), k, ax, False)).astype(int)
            sel = obs & (nb > 0)
            if sel.any():
                sums += np.bincount(nb[sel], weights=p[sel], minlength=nreg + 1)
                cnts += np.bincount(nb[sel], minlength=nreg + 1)
                edge_A.append(A[sel])
    good = cnts[1:] > 0
    n = int(good.sum())
    if n == 0:
        return None
    m = sums[1:][good] / cnts[1:][good]
    # Fisher combination of the per-region upper-tail values. A mean of U(0,1) percentiles is smaller than a single
    # U(0,1) in convex order, so treating each region value as uniform is conservative for large regions.
    X = 2 * np.sum(-np.log1p(-np.clip(m, 0, 1 - 1e-12)))
    z = float(-stats.norm.ppf(np.clip(stats.chi2.sf(X, 2 * n), 1e-300, 1)))
    return {"z": min(z, 40.0), "z_mean": float((m.mean() - 0.5) * np.sqrt(12 * n)), "mean_pct": float(m.mean()),
            "n_regions": n,
            "edge_level": float(np.median(np.concatenate(edge_A)))}


def _gaps_state_dependent(meta, data):
    U = np.asarray(data["U"], float)
    if np.isfinite(U).all():
        return []
    names = meta["variables"]
    pde = meta.get("kind") == "pde"
    sp = _spatial_axes(U)
    conn = (1,) + (sp if pde else ())
    cands = []      # (stats, variable, transform, scope-range builder)
    flat = U.reshape(-1, U.shape[-1])
    for i, v in enumerate(names):
        F = U[..., i]
        mask = ~np.isfinite(F)
        if not mask.any():
            continue
        med = float(np.nanmedian(F))
        lo, hi = float(np.nanmin(F)), float(np.nanmax(F))
        for kind, A in (("high", F), ("low", -F), ("abs", np.abs(F - med))):
            s = _edge_stat(A, mask, conn)
            if s:
                c = s["edge_level"]
                rng = {"high": [lo, c], "low": [-c, hi], "abs": [med - c, med + c]}[kind]
                cands.append((s, v, kind, rng))
        if pde:     # whole snapshots missing: test a per-snapshot amplitude
            row_mask = mask.reshape(U.shape[0], U.shape[1], -1).all(-1)
            if row_mask.any():
                Fr = F.reshape(U.shape[0], U.shape[1], -1)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    rmax = np.nanmax(np.abs(Fr - med), -1)
                    rrms = np.sqrt(np.nanmean((Fr - med) ** 2, -1))
                for kind, A in (("row_max", rmax), ("row_rms", rrms)):
                    s = _edge_stat(np.where(row_mask, np.nan, A), row_mask, (1,))
                    if s:
                        c = s["edge_level"]
                        cands.append((s, v, kind, [med - c, med + c]))
    if not pde and U.shape[-1] > 1:     # joint amplitude of the state
        mask = ~np.isfinite(U).all(-1)
        med = np.nanmedian(flat, 0)
        sc = np.array([_robust_sigma(flat[:, i]) for i in range(U.shape[-1])])
        sc = np.where(sc > 0, sc, 1.0)
        A = np.sqrt(np.sum(((U - med) / sc) ** 2, -1))
        s = _edge_stat(np.where(mask, np.nan, A), mask, (1,))
        if s:
            cands.append((s, "|state|", "norm", [0.0, s["edge_level"]]))
    if not cands:
        return []
    thr = threshold("gaps_state_dependent", 3.0)
    min_pct = threshold("gaps_state_dependent_min_pct", 0.6)
    hit = [c for c in cands if c[0]["z"] > thr and c[0]["mean_pct"] > min_pct]
    s, v, kind, rng = max(hit or cands, key=lambda c: c[0]["z"])
    fired = bool(hit)
    what = {"high": f"{v} is high", "low": f"{v} is low", "abs": f"|{v}| is large",
            "row_max": f"max |{v}| over space is large", "row_rms": f"the rms of {v} is large",
            "norm": "the state amplitude is large"}[kind]
    details = {"z": s["z"], "mean_edge_percentile": s["mean_pct"], "n_regions": s["n_regions"],
               "variable": v, "transform": kind, "edge_level": s["edge_level"], "min_mean_percentile": min_pct,
               "candidates": [{"variable": c[1], "transform": c[2], "z": c[0]["z"], "mean_pct": c[0]["mean_pct"],
                               "n_regions": c[0]["n_regions"]} for c in cands]}
    if not fired:
        return [finding("gaps_state_dependent", STAGE, s["z"], thr, False, "info",
                        message="Missing data look unrelated to the state (gap edges sit at typical amplitudes).",
                        details=details)]
    scope_var = v if kind in ("high", "low", "abs", "row_max") else (f"rms({v})" if kind == "row_rms" else v)
    tail = f"low-{v} behaviour" if kind == "low" else "high-amplitude behaviour"
    msg = (f"Missing data are concentrated where {what}: observed samples next to the {s['n_regions']} gaps sit at "
           f"percentile {s['mean_pct']:.2f} of the observed values on average (z = {s['z']:.1f}), so {tail} is "
           f"under-sampled (censored tail); trust the model only for {scope_var} in [{rng[0]:.3g}, {rng[1]:.3g}] "
           f"and collect data {'at lower values' if kind == 'low' else 'at larger amplitudes'}.")
    return [finding("gaps_state_dependent", STAGE, s["z"], thr, True, "critical", "scope",
                    scope={"variable": scope_var, "range": [float(rng[0]), float(rng[1])]},
                    message=msg, details=details)]


# ------------------------------------------------------------------- re-expressed signals (assess.data_advice)
def _coarse_sampling(meta, data):
    U = np.asarray(data["U"], float)
    nf = U.shape[-1]
    flat = U.reshape(-1, nf)
    with np.errstate(all="ignore"):
        step = np.nanmean(np.abs(np.diff(U, axis=1)).reshape(-1, nf), 0) / (np.nanstd(flat, 0) + 1e-12)
    step = np.nan_to_num(step)
    s = float(step.max()) if step.size else 0.0
    thr = threshold("coarse_sampling", 0.15)
    fired = s > thr
    v = meta["variables"][int(np.argmax(step))]
    msg = (f"Sampling is coarse ({v} changes {s:.0%} of its std per step): sample at least "
           f"{int(np.ceil(s / 0.05))}x faster, or rely on weak-form / rollout-based fitting." if fired else
           f"Sampling is fine enough ({s:.0%} of the std per step).")
    return [finding("coarse_sampling", STAGE, s, thr, fired, "warn" if fired else "info", "scope" if fired else None,
                    message=msg, details={"mean_change_per_step_rel": dict(zip(meta["variables"], step.tolist()))})]


def _single_trajectory(meta, data):
    n = int(np.asarray(data["U"]).shape[0])
    thr = threshold("single_trajectory", 3)
    fired = n < thr
    msg = (f"Only {n} trajectory(ies): independent runs from different initial conditions are the cheapest way "
           f"to separate competing models." if fired else f"{n} independent trajectories.")
    return [finding("single_trajectory", STAGE, n, thr, fired, "warn" if fired else "info",
                    "scope" if fired else None, message=msg, details={"n_traj": n})]


def _grid_scale_signal(meta, data):
    if meta.get("kind") != "pde":
        return []
    from .. import toolbox as tb
    U = np.asarray(data["U"], float)
    ok = ~_row_missing(U)
    if ok.sum() < 8:
        return []
    Uf = U[ok][None]       # NaN-free snapshots only (the spectrum is computed per snapshot)
    d = tb.diagnose(meta, {"U": Uf, "t": np.arange(Uf.shape[1]) * meta["dt"]})
    per = {}
    for f, s in (d.get("spectrum") or {}).items():
        if isinstance(s, dict) and s.get("modes_above_noise_floor") and s.get("n_modes"):
            per[f] = (s["modes_above_noise_floor"], s["n_modes"])
    if not per:
        return []
    frac = {f: a / n for f, (a, n) in per.items()}
    f = max(frac, key=frac.get)
    thr = threshold("grid_scale_signal", 0.6)
    fired = frac[f] > thr
    msg = (f"{f}: signal reaches the grid scale (resolved modes {per[f][0]}/{per[f][1]}): increase spatial "
           f"resolution; high spatial derivatives are unreliable." if fired else
           f"Spatial spectrum decays before the grid scale ({per[f][0]}/{per[f][1]} modes above the noise floor).")
    return [finding("grid_scale_signal", STAGE, frac[f], thr, fired, "warn" if fired else "info",
                    "scope" if fired else None, message=msg,
                    details={"modes_above_noise_floor": {k: list(v) for k, v in per.items()}})]


DETECTORS = (("outliers", _outliers), ("gaps", _gaps), ("gaps_state_dependent", _gaps_state_dependent),
             ("coarse_sampling", _coarse_sampling), ("single_trajectory", _single_trajectory),
             ("grid_scale_signal", _grid_scale_signal))


def audit(meta, data, rhs=None):
    """All data checks; each detector is isolated so one failure cannot hide the others."""
    out = []
    for name, fn in DETECTORS:
        out += _safe(STAGE, name, fn, meta, data)
    return out
