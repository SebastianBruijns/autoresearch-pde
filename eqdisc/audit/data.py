"""Data audit (pre-fit): guards the data before any model is fitted. Deterministic numpy/scipy, no imputation.

    audit(meta, data, rhs=None) -> list[Finding]        (stage "data")

Detectors
    outliers              measurement glitches: events of <= 2 samples that leave no trace (leave-one-out cubic
                          residual, robust MAD z); fires on one strong glitch or on a glitch fraction
    gaps                  missing samples (NaN) or non-uniform time stamps
    gaps_state_dependent  is missingness predictable from the state? (censored tails)
    coarse_sampling       signal changes a lot per time step              (re-expressed from assess.data_advice)
    single_trajectory     fewer than 3 independent runs                   (re-expressed from assess.data_advice)
    grid_scale_signal     PDE spectrum reaches the grid scale              (re-expressed from assess.data_advice)

    external_shock        a sudden lasting change in the state (a kick): kept, record split there
    extreme_event         a large excursion the record recovers from: kept (info; checked against the model later)

`outliers`, `external_shock` and `extreme_event` come from one event detection (detect_events): flagged samples are
grouped into events and classified glitch / external_shock / extreme_event by whether the system remembers them.

Repair tools (called by the integration layer through Finding["fix"])
    despike(meta, data, **args)               -> (data_clean, n_replaced)   replaces exactly the glitch samples
    split_at_gaps(meta, data, min_len)        -> (meta2, data2)
    split_at_events(meta, data, cuts, ...)    -> (meta2, data2)             new pieces start after each kick
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


def _spike_z(meta, U, half=None, k=None, parts=None):
    """Robust (Hampel) z of every sample against a leave-one-out cubic prediction from its neighbours.

    r = u - (least-squares cubic through the +-k neighbours, centre excluded, evaluated at the centre).
    z = (r - local median of r) / scale, scale = max(global MAD of r, local MAD of r over +-`half` / LOCAL_DIV):
    the local term stops smooth curvature error (noise-free or sharp data) from looking like spikes, the global
    term (floored at FLOOR_REL x the field std) stops quiet stretches from doing so.
    Computed along time and (PDE) every spatial axis. Axes whose residual scale is > AXIS_RATIO x the best axis are
    dominated by signal curvature (coarse sampling) and are ignored; a sample must stand out along every remaining axis
    (min |z|; NaN if untestable along any of them), so shocks and fronts are not taken for spikes.
    Returns z (U.shape, NaN where undefined), the axes used, the best axis, and the per-field noise std.
    If `parts` is a dict, parts[axis] = (r, local median of r, scale) is filled for every used axis and for time
    (axis 1) whenever it could be computed, so per-sample z along that axis is |r - med| / scale."""
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
    extra = (1,) if parts is not None and 1 in zs and 1 not in used else ()
    for ax in used + extra:
        per = _periodic(meta) and ax >= 2
        med = _local_median(zs[ax], ax, half, per)
        rc = zs[ax] - med
        loc = 1.4826 * _local_median(np.abs(rc), ax, half, per)
        scale = np.maximum(np.where(sig[ax] > 0, sig[ax], np.nan), loc / LOCAL_DIV)
        if parts is not None:
            parts[ax] = (zs[ax], med, scale)
        if ax not in used:
            continue
        a_ = np.abs(rc / scale)
        z = a_ if z is None else np.minimum(z, a_)
    return z, used, best, sig[best] / gain


# ------------------------------------------------------------------------------ events: glitch or real?
# A flagged sample is only a symptom. Flagged samples are grouped into events (connected in time, and in time x space
# for fields; never across trajectories or variables) and each event is classified by asking whether the system
# remembers it:
#   glitch          <= 2 samples (<= 2 grid points), its neighbours are normal once it is left out, and the series
#                   continues where it would have gone without it            -> remove exactly those samples
#   external_shock  the series after the event is shifted from where the series before it was heading (a lasting
#                   offset beyond noise): a kick                              -> keep, split the record there
#   extreme_event   anything else (multi-sample excursion that the record recovers from)  -> keep, check the model
# When unsure an event is NOT a glitch: deleting real tail data is worse than leaving one glitch in.
Z_LO = 4.5          # suspect sample: part of an event
Z_CAND = threshold("events_z_cand", 5.5)        # an event needs at least one sample beyond this (clean dev data: max |z| < 5)
Z_OK = 4.5          # glitch test: every neighbour of the left-out samples must be below this
GLITCH_MAX = 2      # samples in time (and grid points in each spatial direction) for a glitch
K_FIT = 3           # masked leave-one-out stencil: this many valid neighbours on each side ...
K_REACH = 8         # ... found within this distance
K_STEP = 8          # samples on each side for the lasting-offset (step) test
CORE_ANY_AXIS = True
GAIN0 = float(np.sqrt(1 + sum(v ** 2 for v in _loo_weights(K_LOO).values())))


def _cubic_w(off):
    """Least-squares cubic through samples at integer offsets `off` (centre excluded), evaluated at 0."""
    off = np.asarray(off, float)
    deg = min(3, len(off) - 2)
    s = max(np.abs(off).max(), 1.0)
    return np.linalg.pinv(np.vander(off / s, deg + 1, increasing=True))[0]


def _masked_z(y, q, excl, med, scale):
    """|z| of y[q] against a cubic through its nearest K_FIT valid, non-excluded neighbours on each side (NaN if a
    side has < 2), on the scale of the standard leave-one-out residual (`med`, `scale` from _spike_z)."""
    n = len(y)
    left = [j for j in range(q - 1, max(q - K_REACH, 0) - 1, -1) if np.isfinite(y[j]) and not excl[j]][:K_FIT]
    right = [j for j in range(q + 1, min(q + K_REACH, n - 1) + 1) if np.isfinite(y[j]) and not excl[j]][:K_FIT]
    if len(left) < 2 or len(right) < 2 or not (np.isfinite(scale[q]) and scale[q] > 0):
        return np.nan
    idx = np.array(left[::-1] + right)
    w = _cubic_w(idx - q)
    r = y[q] - w @ y[idx]
    gain = np.sqrt(1 + np.sum(w ** 2))
    m = med[q] if np.isfinite(med[q]) else 0.0
    return float(abs(r - m) / (scale[q] * gain / GAIN0))


def _step(y, a, b, excl, sigma, k=K_STEP):
    """Lasting offset across samples [a, b] (left out): cubic trend + step fitted on the nearest k valid samples
    before a and after b. Returns (offset, |offset| / se, misfit = fit rms / noise); NaN if a side has < 4 samples.
    se uses max(noise, fit rms), so strong curvature (coarse sampling) cannot pass for a step."""
    n = len(y)
    pre = [j for j in range(a - 1, max(a - 3 * k, 0) - 1, -1) if np.isfinite(y[j]) and not excl[j]][:k]
    post = [j for j in range(b + 1, min(b + 3 * k, n - 1) + 1) if np.isfinite(y[j]) and not excl[j]][:k]
    if len(pre) < 4 or len(post) < 4 or not (np.isfinite(sigma) and sigma > 0):
        return np.nan, np.nan, np.nan
    idx = np.array(pre[::-1] + post)
    c = 0.5 * (a + b)
    tau = (idx - c) / max(c - idx[0], idx[-1] - c, 1.0)
    deg = 3 if len(idx) >= 10 else 2
    X = np.column_stack([np.vander(tau, deg + 1, increasing=True), (idx > b).astype(float)])
    coef, *_ = np.linalg.lstsq(X, y[idx], rcond=None)
    dof = len(idx) - X.shape[1]
    fit_rms = float(np.sqrt(np.sum((y[idx] - X @ coef) ** 2) / max(dof, 1)))
    cov = np.linalg.pinv(X.T @ X)
    se = max(sigma, fit_rms) * np.sqrt(max(cov[-1, -1], 1e-300))
    return float(coef[-1]), float(abs(coef[-1]) / se), fit_rms / sigma


def _excursion(y, a, b, excl, sigma, k=K_STEP):
    """Shape of the excursion over [a, b] against a cubic trend through k samples on each side (no step):
    (peak / noise, samples beyond 3 noise sd)."""
    n = len(y)
    pre = [j for j in range(a - 1, max(a - 3 * k, 0) - 1, -1) if np.isfinite(y[j]) and not excl[j]][:k]
    post = [j for j in range(b + 1, min(b + 3 * k, n - 1) + 1) if np.isfinite(y[j]) and not excl[j]][:k]
    if len(pre) + len(post) < 6 or not (np.isfinite(sigma) and sigma > 0):
        return np.nan, 0
    idx = np.array(pre[::-1] + post)
    c = 0.5 * (a + b)
    s = max(np.abs(idx - c).max(), 1.0)
    p = np.polyfit((idx - c) / s, y[idx], min(3, len(idx) - 3))
    seg = np.arange(a, b + 1)
    exc = (y[seg] - np.polyval(p, (seg - c) / s)) / sigma
    exc = exc[np.isfinite(exc)]
    if exc.size == 0:
        return np.nan, 0
    return float(np.abs(exc).max()), int((np.abs(exc) > 3).sum())


def detect_events(meta, U, z_cand=None):
    """Group flagged samples into events and classify each (see the block comment above). Cheap (vectorised z,
    then a few tiny least-squares fits per event). Returns a dict:
        events   list of {kind, traj, variable, var, t_start, t_end, i_start, i_end, duration, extent (grid points,
                 PDE), x_start/x_end (PDE), peak_z, step, step_z, samples ([index tuples], glitches only), ...}
        n_tested number of samples with a defined z;  noise_sigma per field;  axes_used
    """
    U = np.asarray(U, float)
    z_cand = Z_CAND if z_cand is None else float(z_cand)
    parts = {}
    z, used, best, sig = _spike_z(meta, U, parts=parts)
    ok = np.isfinite(z)
    out = {"events": [], "n_tested": int(ok.sum()), "noise_sigma": np.asarray(sig, float).tolist(),
           "axes_used": ["t" if a == 1 else f"space{a - 2}" for a in used], "z_max": float(np.nanmax(z)) if ok.any()
           else float("nan")}
    if not ok.any():
        return out
    zf = np.where(ok, z, 0.0)
    suspect = zf > Z_LO
    # event core: stands out beyond z_cand along at least one tested axis while beyond Z_LO along all of them
    # (the min over axes alone is too strict for single grid points on fields with spatial curvature)
    zax = [np.nan_to_num(np.abs(parts[a][0] - parts[a][1]) / parts[a][2]) for a in used if a in parts]
    zcore = np.maximum.reduce(zax) if (CORE_ANY_AXIS and len(zax) > 1) else zf
    zcore = np.where(suspect, zcore, 0.0)
    if not (zcore > z_cand).any():
        return out
    pde = meta.get("kind") == "pde"
    st = np.zeros((3,) * U.ndim, bool)
    sl = [slice(1, 2)] + [slice(None)] * (U.ndim - 2) + [slice(1, 2)]
    st[tuple(sl)] = True                      # connected in time (and space), never across trajectories / fields
    lab, n = ndimage.label(suspect, structure=st)
    peaks = ndimage.maximum(zcore, lab, index=np.arange(1, n + 1))
    keep = np.where(peaks > z_cand)[0] + 1
    t = np.arange(U.shape[1]) * float(meta.get("dt", 1.0))
    names = list(meta.get("variables") or [f"u{i}" for i in range(U.shape[-1])])
    tp = parts.get(1)
    if tp is not None:
        with np.errstate(all="ignore"):
            zt = np.nan_to_num(np.abs(tp[0] - tp[1]) / tp[2])
        tp = tp + (zt,)
        suspect_t = suspect | (zt > Z_LO)      # also stands out along time alone (PDE: other spikes in a column)
    else:
        suspect_t = suspect
    boxes = ndimage.find_objects(lab)
    for lb in keep:
        box = boxes[lb - 1]
        m = lab[box] == lb
        j, v = box[0].start, box[-1].start
        a, b = box[1].start, box[1].stop - 1
        sp_ext = [bx.stop - bx.start for bx in box[2:-1]]
        loc = np.unravel_index(np.argmax(np.where(m, zf[box], -1)), m.shape)
        peak_idx = tuple(bx.start + i for bx, i in zip(box, loc))
        ev = {"traj": int(j), "var": int(v), "variable": names[v] if v < len(names) else str(v),
              "i_start": int(a), "i_end": int(b), "t_start": float(t[a]), "t_end": float(t[b]),
              "duration": int(b - a + 1), "n_samples": int(m.sum()), "peak_z": float(zf[peak_idx]),
              "peak_index": [int(i) for i in peak_idx]}
        if pde:
            ev["extent"] = int(max(sp_ext)) if sp_ext else 1
            ev["x_index"] = [[bx.start, bx.stop - 1] for bx in box[2:-1]]
        # columns: every spatial point touched by the event (ODE: one), each with its time rows inside the event
        cols = {}
        for idx in zip(*np.nonzero(m)):
            full = tuple(bx.start + i for bx, i in zip(box, idx))
            cols.setdefault(full[2:-1], []).append(full[1])
        ev.update(_classify(U, j, v, cols, suspect_t, tp, pde, ev))
        if ev["kind"] != "glitch" and ev["peak_z"] <= z_cand:
            continue        # only the any-axis core stood out: not strong enough to report as a real event
        out["events"].append(ev)
    # a glitch candidate next to a real event in the same trajectory (any variable) is part of it: a kick or burst
    # moves coupled variables together. Conservative: never delete samples there.
    real = [e for e in out["events"] if e["kind"] == "external_shock" or
            (e["kind"] == "extreme_event" and e["duration"] >= 3)]
    for e in out["events"]:
        if e["kind"] != "glitch":
            continue
        for r in real:
            if r["traj"] == e["traj"] and r["i_start"] - K_STEP <= e["i_end"] and e["i_start"] <= r["i_end"] + K_STEP:
                e["kind"] = r["kind"] if r["kind"] == "external_shock" else "extreme_event"
                e["reclassified"] = "next to a real event"
                e.setdefault("step_sd", r.get("step_sd", float("nan")))
                e.pop("samples", None)
                break
    return out


def _series(A, j, col, v):
    return A[(j, slice(None)) + tuple(col) + (v,)]


def _classify(U, j, v, cols, susp_all, tp, pde, ev):
    """kind + test statistics for one event. tp = (r, med, scale, |z|) along time (None if time is untestable).
    `susp_all` marks suspect samples (left out of every fit)."""
    rows_all = sorted({i for rows in cols.values() for i in rows})
    small = (len(rows_all) <= GLITCH_MAX and rows_all[-1] - rows_all[0] < GLITCH_MAX
             and (not pde or ev["extent"] <= GLITCH_MAX))
    res = {"glitch_test": None}
    if tp is None:      # cannot look along time: never call it a glitch
        res.update(kind="extreme_event", step=float("nan"), step_z=float("nan"))
        return res
    r_t, med_t, sc_t, zt = tp
    # ---- glitch hypothesis: leave out 1-2 samples per column; everything around must look normal
    if not pde or small:
        tests = []
        for col, rows in cols.items():
            y = _series(U, j, col, v)
            med, sc = _series(med_t, j, col, v), _series(sc_t, j, col, v)
            zc = _series(zt, j, col, v)
            susp = _series(susp_all, j, col, v)
            if pde:
                hyps = [sorted(rows)]
            else:
                p = max(rows, key=lambda i: zc[i])
                hyps = [[p], [p - 1, p], [p, p + 1]]
                # a cluster of separate spikes (e.g. 1% outliers): every local maximum of |z| is its own glitch
                peaks = [i for i in rows if zc[i] > Z_LO and zc[i] >= zc[max(i - 1, 0)] and
                         zc[i] >= zc[min(i + 1, len(zc) - 1)]]
                if len(peaks) >= 2 and np.all(np.diff(peaks) >= 2):
                    hyps.append(peaks)
            best = None
            for G in hyps:
                if min(G) < 0 or max(G) >= len(y):
                    continue
                best = _glitch_test(y, G, rows, susp, med, sc)
                if best is not None:
                    break
            tests.append((col, best))
        if tests and all(b is not None for _, b in tests):
            samples = [[j, g] + list(col) + [v] for col, b in tests for g in b["G"]]
            res.update(kind="glitch", samples=samples,
                       step=float(np.nanmax([b["step"] for _, b in tests] + [np.nan])),
                       step_z=float(np.nanmax([b["step_z"] for _, b in tests] + [np.nan])),
                       glitch_test={"neighbours_max_z": max(b["neighbours_max_z"] for _, b in tests),
                                    "z_left_out": [x for _, b in tests for x in b["z_left_out"]]})
            return res
    # ---- real event: does the series after it continue where it was heading (extreme) or stay shifted (shock)?
    steps, szs, mis, exc = [], [], [], []
    for col, rows in cols.items():
        y = _series(U, j, col, v)
        cr = _series(susp_all, j, col, v).copy()
        a, b = min(rows), max(rows)
        cr[a:b + 1] = True
        sigma = float(np.nanmedian(_series(sc_t, j, col, v)[max(a - K_STEP, 0):b + K_STEP + 1])) / GAIN0
        s_, z_, m_ = _step(y, a, b, cr, sigma)
        if np.isfinite(z_):
            steps.append(s_ / sigma)
            szs.append(z_)
            mis.append(m_)
        exc.append(_excursion(y, a, b, cr, sigma))
    step_z = float(np.median(szs)) if szs else float("nan")
    step_sd = float(np.median(np.abs(steps))) if steps else float("nan")
    shock = (np.isfinite(step_z) and step_z > threshold("events_step_z", 6.0)
             and step_sd > threshold("events_step_min_sd", 3.0))
    pk = [e[0] for e in exc if np.isfinite(e[0])]
    res.update(kind="external_shock" if shock else "extreme_event", step=float(np.median(steps)) if steps else
               float("nan"), step_z=step_z, step_sd=step_sd, fit_misfit=float(np.median(mis)) if mis else float("nan"),
               excursion_peak_sd=float(max(pk)) if pk else float("nan"),
               excursion_samples=int(max(e[1] for e in exc)) if exc else 0)
    return res


def _glitch_test(y, G, rows, susp, med, sc):
    """Leave out G (and every other suspect sample) and refit: G must stand out, and the event's other samples and
    G's neighbours (+-K_LOO) must look normal; no lasting offset across G. Returns the test record or None."""
    excl = susp.copy()
    excl[G] = True
    zg = [_masked_z(y, g, excl, med, sc) for g in G]
    if not np.all(np.isfinite(zg)) or max(zg) <= Z_CAND - 1.0 or min(zg) <= Z_LO - 1.0:
        return None
    Gs = set(G)
    near = {g + d for g in G for d in (-1, 1)}
    lo, hi = min(min(rows), G[0]) - K_LOO, max(max(rows), G[-1]) + K_LOO
    # test: event samples, G's direct neighbours, and every non-suspect sample around; other suspects (separate
    # spikes nearby) are only left out of the fits
    test = [q for q in range(max(lo, 0), min(hi, len(y) - 1) + 1)
            if q not in Gs and (q in rows or q in near or not susp[q])]
    nb = [_masked_z(y, q, excl, med, sc) for q in test]
    if not nb or np.sum(np.isfinite(nb)) < min(4, len(nb)):
        return None
    nbm = float(np.nanmax(nb))
    if nbm >= Z_OK or any(not np.isfinite(x) for q, x in zip(test, nb) if q in rows or q in near):
        return None
    sigma = sc[G[0]] / GAIN0
    step, sz, _ = _step(y, G[0], G[-1], excl, sigma)
    # near the ends of the record the step test may lack samples; the neighbour test (>= 2 valid samples on each
    # side, one cubic through both sides) already showed the series is continuous there
    if np.isfinite(sz) and sz >= threshold("events_step_z", 6.0):
        return None
    return {"G": list(G), "z_left_out": [float(x) for x in zg], "neighbours_max_z": nbm, "step": step,
            "step_z": sz}


def _event_rows(evs, kind):
    return [e for e in evs if e["kind"] == kind]


def _outliers(meta, data):
    """outliers (glitch events), external_shock and extreme_event findings from one event detection."""
    U = np.asarray(data["U"], float)
    det = detect_events(meta, U)
    n = det["n_tested"]
    if n == 0:
        return []
    evs = det["events"]
    gl = _event_rows(evs, "glitch")
    n_gl = int(sum(len(e["samples"]) for e in gl))
    frac = float(n_gl / n)
    zmax = float(max((e["peak_z"] for e in gl), default=0.0))
    thr = threshold("outliers", 2e-4)
    thr1 = threshold("outliers_single_z", 7.0)       # one glitch this strong is enough
    crit = threshold("outliers_critical", 0.02)
    fired = frac > thr or zmax > thr1
    per_var = {}
    for e in gl:
        per_var[e["variable"]] = per_var.get(e["variable"], 0) + len(e["samples"])
    worst = max(per_var, key=per_var.get) if per_var else None
    expected = float(2 * stats.norm.sf(Z_SPIKE))
    if fired:
        when = "; ".join(f"traj {e['traj']} t={e['t_start']:g} ({e['variable']}, |z|={e['peak_z']:.0f})"
                         for e in sorted(gl, key=lambda e: -e["peak_z"])[:3])
        msg = (f"{len(gl)} isolated glitch(es) ({n_gl} sample(s), {frac:.2%} of samples; worst in {worst}): single "
               f"samples that stand out from their neighbours and leave no trace afterwards, e.g. {when}. "
               f"Despike removes exactly these samples before fitting.")
    else:
        msg = (f"No measurement glitches ({len(gl)} weak candidate(s), max |z| {zmax:.1f}; "
               f"{len(evs) - len(gl)} real event(s) kept)." if evs else "No measurement glitches (isolated spikes).")
    sev = "critical" if frac > crit else "warn"
    common = {"n_tested": n, "axes_used": det["axes_used"],
              "noise_sigma": dict(zip(meta["variables"], det["noise_sigma"]))}
    out = [finding("outliers", STAGE, frac, thr, fired, sev if fired else "info", "repair" if fired else None,
                   fix={"tool": "despike", "args": {"z": Z_CAND}} if fired else None, message=msg,
                   details={**common, "n_glitches": len(gl), "n_flagged": n_gl, "max_glitch_z": zmax,
                            "single_event_threshold": thr1, "gaussian_expectation": expected,
                            "per_variable": {k: v / n for k, v in per_var.items()},
                            "glitches": [[e["traj"], e["t_start"], e["variable"], round(e["peak_z"], 2)]
                                         for e in gl[:50]]})]
    # ---- external shocks: lasting change in the state -> split the record there (new initial condition)
    sh = _event_rows(evs, "external_shock")
    cuts = {}
    for e in sh:
        cuts.setdefault(e["traj"], []).append([e["i_start"], e["i_end"]])
    cuts = [[j, a, b] for j, ab in sorted(cuts.items()) for a, b in _merge_spans(ab)]     # as shock_cuts
    if sh:
        when = ", ".join(sorted({f"{e['t_start']:g}" for e in sh}, key=float)[:5])
        msg = (f"a sudden lasting change in the state at t={when} (trajectories "
               f"{sorted({e['traj'] for e in sh})}): possible external kick; kept, not removed. Fit the pieces "
               f"before and after it as separate runs (split_at_events).")
    else:
        msg = "No sudden lasting changes in the state (no external kicks)."
    out.append(finding("external_shock", STAGE, max((e["step_z"] for e in sh), default=0.0),
                       threshold("events_step_z", 6.0), bool(sh), "warn" if sh else "info",
                       "repair" if sh else None,
                       fix={"tool": "split_at_events", "args": {"cuts": cuts}} if sh else None, message=msg,
                       details={"events": [[e["traj"], e["t_start"], e["t_end"], e["variable"], round(e["peak_z"], 2),
                                            round(e["step_sd"], 2)] for e in sh[:50]], "cuts": cuts,
                                "columns": ["traj", "t_start", "t_end", "variable", "peak_z", "offset_sd"]}))
    # ---- extreme events: kept as real dynamics; the model check (audit/events.py) asks if the model produces them
    ex = _event_rows(evs, "extreme_event")
    if ex:
        when = ", ".join(f"{e['t_start']:g}" for e in sorted(ex, key=lambda e: -e["peak_z"])[:5])
        msg = f"large excursions that the record recovers from (t={when}); kept as real dynamics."
    else:
        msg = "No large excursions beyond noise."
    out.append(finding("extreme_event", STAGE, len(ex), 0, bool(ex), "info", None, message=msg,
                       details={"events": [[e["traj"], e["t_start"], e["t_end"], e["variable"], round(e["peak_z"], 2)]
                                           for e in sorted(ex, key=lambda e: -e["peak_z"])[:50]],
                                "columns": ["traj", "t_start", "t_end", "variable", "peak"], "n_events": len(ex)}))
    return out


def shock_cuts(meta, data):
    """[[traj, i_start, i_end], ...] of the external shocks in `data` (what the external_shock fix splits at)."""
    evs = detect_events(meta, np.asarray(data["U"], float))["events"]
    cuts = {}
    for e in evs:
        if e["kind"] == "external_shock":
            cuts.setdefault(e["traj"], []).append([e["i_start"], e["i_end"]])
    return [[j, a, b] for j, ab in sorted(cuts.items()) for a, b in _merge_spans(ab)]


def _merge_spans(spans, gap=K_LOO):
    out = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def despike(meta, data, z=None, max_iter=3):
    """Replace exactly the samples of glitch events (see detect_events), each by the median of its nearest
    non-flagged neighbours (2 on each side along the best-resolved axis: time for ODEs). Real events (shocks,
    extreme events) are never touched; NaNs are left as they are (no imputation). Returns (data_clean, n_replaced)."""
    U = np.array(data["U"], float, copy=True)
    n_rep = 0
    for _ in range(max_iter):
        det = detect_events(meta, U, z_cand=z)
        gl = [s for e in det["events"] if e["kind"] == "glitch" for s in e["samples"]]
        if not gl:
            break
        zz, _, ax, _ = _spike_z(meta, U)
        bad = np.isfinite(zz) & (zz > Z_LO)
        for s in gl:
            bad[tuple(s)] = True
        new = {}
        for s in gl:
            s = tuple(s)
            vals = []
            for d in (-1, 1):
                k, got = 1, 0
                while got < 2 and k <= K_REACH:
                    q = list(s)
                    q[ax] = s[ax] + d * k
                    if _periodic(meta) and ax >= 2:
                        q[ax] %= U.shape[ax]
                    if 0 <= q[ax] < U.shape[ax] and np.isfinite(U[tuple(q)]) and not bad[tuple(q)]:
                        vals.append(U[tuple(q)])
                        got += 1
                    k += 1
            if len(vals) >= 2:
                new[s] = float(np.median(vals))
        for s, val in new.items():
            U[s] = val
        n_rep += len(new)
        if not new:
            break
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
    U = np.asarray(data["U"], float)
    rows = _row_missing(U)
    if not rows.any():
        return copy.deepcopy(meta), dict(data)
    return _split_rows(meta, data, rows, min_len, "split_at_gaps", "no NaN-free segment of at least 8 time rows")


def split_at_events(meta, data, cuts=(), min_len=None, margin=None):
    """Split the record at external shocks (kicks) so every piece starts from a fresh initial condition.

    cuts = [[traj, i_start, i_end], ...] (time indices of the shock events, inclusive; from the external_shock
    finding). Rows i_start - margin .. i_end + margin (margin = K_LOO, the reach of the spike stencil) and rows with
    NaN are cut out; the remaining runs are tiled into windows of a common length exactly as split_at_gaps does
    (same planner, meta["segment_t0"] holds each piece's original start time). Returns (meta2, data2)."""
    U = np.asarray(data["U"], float)
    rows = _row_missing(U)
    m = K_LOO if margin is None else int(margin)
    for c in cuts or ():
        j, a, b = int(c[0]), int(c[1]), int(c[-1])
        if 0 <= j < rows.shape[0]:
            rows[j, max(a - m, 0):min(b + m + 1, rows.shape[1])] = True
    if not rows.any():
        return copy.deepcopy(meta), dict(data)
    meta2, data2 = _split_rows(meta, data, rows, min_len, "split_at_events",
                               "no shock-free, NaN-free segment of at least 8 time rows")
    meta2["split_at_events"]["cuts"] = [list(map(int, c)) for c in cuts or ()]
    return meta2, data2


def _split_rows(meta, data, rows, min_len, key, err):
    """Shared by split_at_gaps / split_at_events: tile the runs of rows == False into windows of length L."""
    U, t = np.asarray(data["U"], float), np.asarray(data["t"], float)
    plan = _plan_split(rows, min_len or _default_min_len(meta))
    if plan is None:
        raise ValueError(err)
    L, starts = plan
    wins = [U[j, s:s + L] for j, s in starts]
    U2 = np.stack(wins, 0)
    meta2 = copy.deepcopy(meta)
    meta2["n_traj"] = int(U2.shape[0])
    meta2["shape"] = list(U2.shape)
    t0 = [float(t[s]) for _, s in starts]
    prev = meta.get("segment_t0")
    if key != "split_at_gaps" and isinstance(prev, list) and len(prev) == U.shape[0]:
        t0 = [float(prev[j]) + float(t[s] - t[0]) for j, s in starts]      # already split once: keep true times
    meta2["segment_t0"] = t0
    meta2[key] = {"L": L, "n_windows": len(wins), "kept_fraction": float(U2.size / max(np.isfinite(U).sum(), 1)),
                  "source": [list(s) for s in starts], "t0": t0,
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
