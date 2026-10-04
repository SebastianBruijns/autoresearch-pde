"""Data audit (pre-fit): deterministic checks on the data before any model is fitted. No imputation, no row deletion
except at missing values.

    audit(meta, data) -> list[Finding]                     (stage "data")
        gaps                  missing samples (NaN) or non-uniform time stamps
        gaps_state_dependent  is missingness predictable from the state? (censored tails)
        coarse_sampling       signal changes a lot per time step
        single_trajectory     fewer than 3 independent runs
        grid_scale_signal     PDE spectrum reaches the grid scale

    split_at_gaps(meta, data)  -> (meta2, data2, kept)     NaN-free windows; kept = fraction of finite rows kept
    glitches(meta, data)       -> (Finding, data2 | None)  isolated single-sample glitches, clipped only if that
                                                           does not change the fitted structure

Glitch rule. A PDE solution is smooth in space at every instant, so a measurement glitch is a sample that disagrees
with its spatial neighbours at one time step (ODE: with its time neighbours) and nowhere else. Real events, however
large, are spatially coherent and never flagged. Flagged samples are clipped to CLIP noise sd of the neighbour
prediction (winsorized), then the cheap non-LLM fit (autobase.auto_fit) is run on raw and clipped data:
    same terms      -> the glitches are immaterial: use the clipped data (info)
    different terms -> the result depends on these samples: keep the raw data, critical finding (no confident verdict)
"""
import copy
import warnings

import numpy as np
from scipy import ndimage, stats

from . import _safe, finding, threshold

STAGE = "data"
K = 3               # leave-one-out cubic through the +-K neighbours
HALF = 7            # half-width of the local median / MAD window
LOCAL_DIV = 2.0     # local scale counts where it exceeds the global one by this factor (curvature, fronts)
FLOOR_REL = 1e-3    # residual scale floor (x field std): differences below this never matter
CLIP = 3.0          # clipped residual, in noise sd
MIN_ROWS = 8        # shortest NaN-free window worth fitting


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
    return 1.4826 * float(np.median(np.abs(r - np.median(r)))) if r.size else np.nan


def _local_median(A, ax, periodic):
    stack = np.stack([_shift(A, k, ax, periodic) for k in range(-HALF, HALF + 1)], 0)
    if np.isfinite(stack).all():                    # periodic axes: no edge NaN; np.median is ~2x faster
        return np.median(stack, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(stack, axis=0)


def _row_missing(U):
    """(n_traj, nt) bool: time rows with any NaN."""
    return ~np.isfinite(U).reshape(U.shape[0], U.shape[1], -1).all(-1)


def _runs(b):
    """Contiguous True runs of a 1-D bool array -> list of (start, stop)."""
    d = np.diff(np.concatenate([[0], b.astype(int), [0]]))
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


# --------------------------------------------------------------------------------------------- glitches
_X = np.array([i for i in range(-K, K + 1) if i])
LOO = dict(zip(_X.tolist(), np.linalg.pinv(np.vander(_X, 4, increasing=True))[0]))   # cubic, centre left out


def glitch_z(meta, U):
    """(z, excess, scale): robust |z| of every sample against a leave-one-out cubic through its +-K neighbours, and the
    residual beyond the local median (signed). Along every spatial axis for PDEs (a sample must stand out along each),
    along time for ODEs. Scale = max(global MAD, local MAD / LOCAL_DIV, FLOOR_REL x field std): the local term keeps
    fronts and sharp curvature from looking like glitches."""
    U = np.asarray(U, float)
    pde = meta.get("kind") == "pde"
    axes = _spatial_axes(U) if pde else (1,)
    per = _periodic(meta)
    floor = threshold("glitch_floor_rel", FLOOR_REL) * np.nanstd(U.reshape(-1, U.shape[-1]), axis=0)
    z = exc = None
    for ax in axes:
        if U.shape[ax] < 2 * HALF + 1:
            return np.zeros(U.shape), np.zeros(U.shape), np.ones(U.shape)
        p = per and ax >= 2
        r = U - sum(c * _shift(U, j, ax, p) for j, c in LOO.items())
        rc = r - _local_median(r, ax, p)
        sig = np.maximum([_robust_sigma(rc[..., i]) for i in range(U.shape[-1])], floor)
        scale = np.maximum(sig, 1.4826 * _local_median(np.abs(rc), ax, p) / LOCAL_DIV)
        za = np.nan_to_num(np.abs(rc) / scale)
        if z is None:
            z, exc, sc = za, rc, scale
        else:
            z = np.minimum(z, za)
    return z, exc, sc


def _find(meta, U, zmax):
    """Glitch mask: |z| > zmax, the largest |z| within +-K along the test axis (a glitch also inflates its
    neighbours' residuals), and (PDE) the same place one step earlier and later looks normal (|z| < zmax / 2
    within +-1 grid point): a glitch is one sample, while a sharp real feature persists from step to step."""
    z, exc, sc = glitch_z(meta, U)
    hit = z > zmax
    if not hit.any():
        return hit, exc, sc
    ax = _spatial_axes(U)[0] if meta.get("kind") == "pde" else 1
    size = [1] * U.ndim
    size[ax] = 2 * K + 1
    mode = "wrap" if (_periodic(meta) and ax >= 2) else "nearest"
    hit &= z >= ndimage.maximum_filter(z, size=size, mode=mode)
    if meta.get("kind") == "pde":
        hit &= _isolated_in_time(z > zmax / 2, ax)
    return hit, exc, sc


def _isolated_in_time(flag, ax):
    """True where no sample within +-1 grid point is flagged at t-1 or t+1."""
    size = [1] * flag.ndim
    size[ax] = 3
    sp = ndimage.maximum_filter(flag.astype(np.uint8), size=size, mode="constant") > 0
    prev = np.zeros_like(sp)
    nxt = np.zeros_like(sp)
    prev[:, 1:] = sp[:, :-1]
    nxt[:, :-1] = sp[:, 1:]
    return ~(prev | nxt)


def clip_glitches(meta, data, zmax=None):
    """Winsorize isolated glitches: residual clipped to CLIP x scale. Returns (data2, mask of clipped samples)."""
    zmax = threshold("glitch_z", 6.0) if zmax is None else zmax
    U = np.array(data["U"], float, copy=True)
    done = np.zeros(U.shape, bool)
    for _ in range(3):                   # a clipped glitch no longer hides a smaller neighbour
        hit, exc, sc = _find(meta, U, zmax)
        hit &= ~done
        if not hit.any():
            break
        U[hit] -= (exc[hit] - np.sign(exc[hit]) * CLIP * sc[hit])
        done |= hit
    return dict(data, U=U), done


def _terms(meta, data):
    from ..autobase import auto_fit
    from ..uq import _term_sets
    rhs = auto_fit(meta, data)["rhs"]
    return rhs, _term_sets(meta, rhs)


def glitches(meta, data):
    """(Finding, clipped data or None). See the module doc for the rule."""
    U = np.asarray(data["U"], float)
    if not np.isfinite(U).all():
        return finding("glitches", STAGE, None, None, False, "info",
                       message="glitch check skipped: data contain NaN"), None
    zmax = threshold("glitch_z", 6.0)
    d2, mask = clip_glitches(meta, data, zmax)
    n, frac = int(mask.sum()), float(mask.mean())
    if not n:
        return finding("glitches", STAGE, 0, zmax, False, "info", message="No isolated glitches."), None
    where = [[int(i) for i in ix] for ix in np.argwhere(mask)[:50]]
    raw_rhs, raw = _terms(meta, data)
    clip_rhs, clipped = _terms(meta, d2)
    det = {"n_clipped": n, "fraction": frac, "z_threshold": zmax, "clip_sd": CLIP, "samples": where,
           "fit_raw": raw_rhs, "fit_clipped": clip_rhs}
    what = f"{n} isolated single-sample glitch(es) ({frac:.2%} of samples)"
    if raw == clipped:
        return finding("glitches", STAGE, frac, 0, True, "info", "repair", fix={"tool": "clip_glitches"},
                       message=f"{what} clipped to {CLIP:g} noise sd; the cheap fit finds the same terms with and "
                               f"without clipping, so they do not affect the result.", details=det), d2
    return finding("glitches", STAGE, frac, 0, True, "critical", "scope",
                   message=f"{what}: the cheap fit finds different terms with and without them (raw "
                           f"{raw_rhs}, clipped {clip_rhs}). Data kept unmodified; the result depends on these "
                           f"samples, so it cannot be confident. Check them at the source.", details=det), None


# -------------------------------------------------------------------------------------------------- gaps
def _gaps(meta, data):
    U, t = np.asarray(data["U"], float), np.asarray(data["t"], float)
    miss = ~np.isfinite(U)
    frac = float(miss.mean())
    rows = _row_missing(U)
    dt = np.diff(t)
    nonuni = float(np.std(dt) / (abs(np.mean(dt)) + 1e-300)) if dt.size > 1 else 0.0
    windows = [(j, float(t[a]), float(t[b - 1])) for j in range(U.shape[0]) for a, b in _runs(rows[j])]
    if frac == 0 and nonuni <= 1e-6:
        return [finding("gaps", STAGE, 0.0, 0.0, False, "info", message="No missing samples; uniform sampling.",
                        details={"nan_fraction": 0.0, "dt_rel_variation": nonuni})]
    parts = []
    if frac:
        ex = "; ".join(f"traj {j} t=[{a:g}, {b:g}]" for j, a, b in windows[:4])
        parts.append(f"{frac:.1%} of samples are missing (NaN), touching {rows.mean():.0%} of time rows "
                     f"({ex}{' ...' if len(windows) > 4 else ''}). The record is cut at those rows into NaN-free "
                     f"windows; no imputation.")
    if nonuni > 1e-6:
        parts.append(f"Time stamps are non-uniform (dt varies by {nonuni:.1%}).")
    return [finding("gaps", STAGE, frac, 0.0, True, "warn", "repair" if frac else "scope",
                    fix={"tool": "split_at_gaps"} if frac else None, message=" ".join(parts),
                    details={"nan_fraction": frac, "rows_with_nan_fraction": float(rows.mean()),
                             "windows": windows[:50], "n_windows": len(windows), "dt_rel_variation": nonuni})]


def split_at_gaps(meta, data):
    """Cut each trajectory at time rows containing a NaN and tile the NaN-free runs into windows of one common length
    L (the data format needs it), L chosen to keep the most rows. Returns (meta2, data2, kept), kept = kept rows /
    NaN-free rows. Windows share t[:L]; meta2["segment_t0"] holds each window's true start time."""
    U, t = np.asarray(data["U"], float), np.asarray(data["t"], float)
    rows = _row_missing(U)
    if not rows.any():
        return copy.deepcopy(meta), dict(data), 1.0
    runs = [(j, a, b) for j in range(U.shape[0]) for a, b in _runs(~rows[j])]
    lens = np.array([b - a for _, a, b in runs])
    if lens.size == 0 or lens.max() < MIN_ROWS:
        raise ValueError(f"no NaN-free stretch of at least {MIN_ROWS} time rows")
    lo = min(20 if meta.get("kind") == "pde" else 36, int(lens.max()))        # preferred shortest window
    cand = np.arange(lo, lens.max() + 1)
    kept_rows = np.array([(lens // L).sum() * L for L in cand])
    L = int(cand[len(cand) - 1 - np.argmax(kept_rows[::-1])])          # most rows kept; longest L on ties
    starts = [(j, int(a + i * L)) for j, a, b in runs for i in range((b - a) // L)]
    U2 = np.stack([U[j, s:s + L] for j, s in starts], 0)
    kept = float(len(starts) * L / max((~rows).sum(), 1))
    meta2 = copy.deepcopy(meta)
    meta2.update(n_traj=int(U2.shape[0]), shape=list(U2.shape), segment_t0=[float(t[s]) for _, s in starts],
                 split_at_gaps={"L": L, "n_windows": len(starts), "kept_fraction": kept,
                                "source": [list(s) for s in starts]})
    return meta2, dict(data, U=U2, t=t[:L].copy()), kept


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
    return {"z": min(z, 40.0), "mean_pct": float(m.mean()),
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
        flat = U.reshape(-1, U.shape[-1])
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
    scope_var = f"rms({v})" if kind == "row_rms" else v
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


DETECTORS = (("gaps", _gaps), ("gaps_state_dependent", _gaps_state_dependent),
             ("coarse_sampling", _coarse_sampling), ("single_trajectory", _single_trajectory),
             ("grid_scale_signal", _grid_scale_signal))


def audit(meta, data):
    """All data checks; each detector is isolated so one failure cannot hide the others."""
    out = []
    for name, fn in DETECTORS:
        out += _safe(STAGE, name, fn, meta, data)
    return out
