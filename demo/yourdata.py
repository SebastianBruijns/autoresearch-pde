"""Your Data extras: a forecast check with a video, and what the found law means. GUI only.

The discovery framework (eqdisc) is called, never changed.
- Forecast check: the last part of every run is held back before discovery. The agents see only the rest. The found
  law then forecasts the held-back part from the last state the agents saw, and a "Forecast vs reality" video shows
  it, like the example pages. Time series (ODE) data only.
- What the equation means: a rule-based reading of every term of the found law (exact numbers, time scales), which
  Claude rewrites in plain language when an API key is available. Rehearsal mode never calls the API.
"""
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

GREEN = "#16a34a"            # "Discovered law" in every video
THRESHOLD = 0.5              # a forecast is useful while its error stays below half the data's typical spread
VIDEO_FS = 18                # one text size in the videos, as in eqdisc/oos.py
EXPLAIN_MODEL = "claude-opus-5-5"
FUNCS = {"sin", "cos", "tan", "exp", "log", "sqrt", "tanh", "sinh", "cosh", "atan", "asin", "acos", "Abs", "sign",
         "pi", "Heaviside", "Max", "Min", "re", "im"}


def _g(x):
    return f"{x:.3g}"


# ----------------------------------------------------------------------------- .npz uploads
_NPZ_CACHE = {}
COORD_KEYS = {"t", "time", "times", "tt", "x", "y", "xx", "yy", "names", "variables", "vars", "labels"}


def read_npz(raw, name, root):
    """Uploaded .npz -> what the page needs, read by the framework's own ingester (local, no API).

    Returns {"path", "kind", "variables", "shape", "summary", "df"}: for a time series, df is the same table a CSV
    upload gives (traj, t, one column per variable), so the rest of the page works unchanged; for a field over space,
    df is None and `path` (the saved .npz) goes to discovery as it is. On failure: {"error": message}."""
    import hashlib
    from eqdisc.evaluate import load
    from eqdisc.ingest import ingest
    key = hashlib.sha1(raw).hexdigest()[:16]
    if key in _NPZ_CACHE:
        return _NPZ_CACHE[key]
    d = Path(root) / "_uploads" / key
    d.mkdir(parents=True, exist_ok=True)
    path = d / (re.sub(r"\W+", "_", Path(name).stem).strip("_") + ".npz")
    path.write_bytes(raw)
    try:
        ds, card = ingest(path, out_dir=str(d / "ingest"))
        meta, data = load(ds)
    except Exception as e:  # noqa: BLE001
        return {"error": f"could not read this .npz: {e}"[:400]}
    out = {"path": str(path), "kind": meta["kind"], "variables": meta["variables"], "shape": meta["shape"],
           "summary": (card or {}).get("summary", ""), "df": None}
    with np.load(path) as z:                      # several per-variable arrays read as more variables than arrays
        named = [k for k in z.files if k.lower() not in COORD_KEYS and z[k].ndim >= 1 and z[k].dtype.kind in "fi"]
    if meta["kind"] == "ode" and len(named) > 1 and len(meta["variables"]) > len(named):
        out["warning"] = (f"This file has {len(named)} data arrays ({', '.join(named)}) but was read as "
                          f"{len(meta['variables'])} variables, so runs and variables are probably mixed up. Separate "
                          "arrays per variable work for a single run. For several runs, save one array U with shape "
                          "(runs, times, variables) plus names=['a', 'b', ...], e.g. "
                          "np.savez('f.npz', t=t, U=U, names=names).")
    if meta["kind"] == "ode":
        t, U, names = data["t"], data["U"], meta["variables"]
        tcol = "t" if "t" not in names else "time"
        frames = []
        for j in range(U.shape[0]):
            f = pd.DataFrame(U[j], columns=names)
            f.insert(0, tcol, t)
            f.insert(0, "traj", j)
            for k, p in enumerate(meta.get("parameters") or []):
                f[p] = data["params"][j, k]
            frames.append(f.dropna(subset=names, how="all"))
        out["df"] = pd.concat(frames, ignore_index=True)
    _NPZ_CACHE[key] = out
    return out


# ----------------------------------------------------------------------------- forecast check
def split_holdout(src_path, run_dir, frac):
    """Ingest the full file, keep it for scoring, and write a training file with the last `frac` of every run cut off:
    a CSV for time series, an .npz (t, x[, y], U) for fields."""
    from eqdisc.evaluate import load
    from eqdisc.ingest import ingest
    run_dir = Path(run_dir)
    ds, _ = ingest(src_path, out_dir=str(run_dir / "full"))
    meta, data = load(ds)
    t, U, names = data["t"], data["U"], meta["variables"]
    n_hold = int(round(frac * len(t)))
    n_train = len(t) - n_hold
    if n_hold < 5 or n_train < 30:
        return {"skipped": f"too few time samples ({len(t)}) to hold some back"}
    out = run_dir / "train"
    out.mkdir(parents=True, exist_ok=True)
    if meta["kind"] != "ode":
        train = out / (Path(src_path).stem + ".npz")
        extra = {} if names == [f"u{i}" for i in range(len(names))] else {"names": np.array(names)}
        coords = {k: data[k] for k in ("x", "y") if k in data}
        np.savez(train, t=t[:n_train], U=U[:, :n_train], **coords, **extra)
    else:
        tcol = "t" if "t" not in names else "time"
        pnames, params = meta.get("parameters") or [], data.get("params")
        frames = []
        for j in range(U.shape[0]):
            d = pd.DataFrame(U[j, :n_train], columns=names)
            d.insert(0, tcol, t[:n_train])
            d.insert(0, "traj", j)
            for k, p in enumerate(pnames):
                d[p] = params[j, k]
            frames.append(d.dropna(subset=names, how="all"))
        train = out / (Path(src_path).stem + ".csv")
        pd.concat(frames).to_csv(train, index=False)
    return {"train_path": str(train), "full_ds": str(ds), "n_train": n_train, "n_hold": n_hold, "frac": frac,
            "nt": len(t), "n_runs": int(U.shape[0]), "kind": meta["kind"]}


def _noise_rel(U, scale):
    """Rough measurement-noise level, as a share of the typical spread (residual of a short smoothing window)."""
    from scipy.signal import savgol_filter
    if U.shape[1] < 11 or not np.all(np.isfinite(U)):
        return None
    r = (U - savgol_filter(U, 11, 3, axis=1)) / scale
    return float(np.sqrt(np.mean(r ** 2)))


def forecast(final_model, split, out_dir):
    """Run the found law forward from the last sample the agents saw, over the held-back part of every run."""
    from eqdisc.evaluate import load
    from eqdisc.solvers import integrate_ode, integrate_pde_general, pde_layout
    meta, data = load(split["full_ds"])
    names = meta["variables"]
    if set(names) - set(final_model or {}):
        return {"skipped": "the law does not cover every variable of this file"}
    t, U, n = data["t"], data["U"].astype(float), split["n_train"]
    tf = t[n - 1:]
    field = U.ndim >= 4                                                    # (runs, times, x[, y], fields)
    scale = np.nanstd(U[:, :n].reshape(-1, len(names)), axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    rhs = {v: final_model[v] for v in names}
    lay = pde_layout(meta) if field else None
    P = np.full(U[:, n - 1:].shape, np.nan)
    for j in range(U.shape[0]):
        if np.all(np.isfinite(U[j, n - 1])):
            P[j] = (integrate_pde_general(names, rhs, lay, U[j, n - 1], tf, max_seconds=90.0) if field else
                    integrate_ode(names, rhs, U[j, n - 1], tf, max_seconds=20.0))
    coords = None
    if field:
        P, coords = _retry_in_log(names, rhs, lay, U, n, tf, P)
    T = U[:, n - 1:]
    ax = tuple(range(2, U.ndim))
    with np.errstate(all="ignore"):
        err = np.sqrt(np.mean(((P - T) / scale) ** 2, axis=ax))            # (n_runs, n_forecast)
    err[~np.isfinite(err) & np.isfinite(T).all(ax)] = np.inf              # law blew up or failed: never useful
    runs = []
    for j in range(U.shape[0]):
        bad = np.where(err[j] > THRESHOLD)[0]
        useful = float(tf[bad[0]] - tf[0]) if bad.size else None
        fin = err[j][np.isfinite(err[j])]
        inf = np.where(np.isinf(err[j]))[0]
        runs.append({"run": j, "useful_for": useful, "blew_up": bool(inf.size),
                     "blew_up_at": float(tf[inf[0]] - tf[0]) if inf.size else None,
                     "err_end": float(err[j, -1]) if np.isfinite(err[j, -1]) else None,
                     "err_median": float(np.median(fin)) if fin.size else None})
    horizon = float(tf[-1] - tf[0])
    noise = _noise_rel(U[:, :n], scale)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / "forecast.npz"
    np.savez_compressed(npz, t=t, U=U, P=P, err=err, scale=scale, n_train=n, names=np.array(names),
                        noise_rel=np.nan if noise is None else noise,
                        coords=np.array(f"simulated in log({', '.join(coords['log_of'])}): the same law, rewritten exactly"
                                        if coords else ""),
                        **{k: data[k] for k in ("x", "y") if field and k in data})
    full = [r for r in runs if r["useful_for"] is None and not r["blew_up"]]
    score = [np.nanmean(np.where(np.isinf(e), 9.0, e)) if np.isfinite(e).any() or np.isinf(e).any() else 99.0
             for e in err]
    return {"npz": str(npz), "variables": names, "frac": split["frac"], "n_train": n, "n_hold": split["n_hold"],
            "t_split": float(tf[0]), "horizon": horizon, "runs": runs, "noise_rel": noise,
            "field": field, "field2d": U.ndim == 5, "coords": coords, "n_useful_whole": len(full), "video_run": int(np.argsort(score)[(len(score) - 1) // 2])}   # median run, not the best


NN_COLOR = "#7c3aed"          # "Neural network" in the example videos
NN_MINUTES = 3.0              # training cap for the baseline


def _mlp_step(runs, steps=4000, seed=0, max_minutes=NN_MINUTES):
    """Plain neural baseline for time series (as eqdisc.oos.train_mlp_step, any number of variables): an MLP maps the
    state to its next increment, trained on consecutive samples of the training runs, rolled out autoregressively."""
    import time
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(seed)
    A = np.concatenate([r[:-1] for r in runs])
    B = np.concatenate([r[1:] - r[:-1] for r in runs])
    ok = np.isfinite(A).all(1) & np.isfinite(B).all(1)
    A, B = A[ok], B[ok]
    mu_a, sd_a, mu_b, sd_b = A.mean(0), A.std(0) + 1e-12, B.mean(0), B.std(0) + 1e-12
    xa = torch.tensor((A - mu_a) / sd_a, dtype=torch.float32, device=dev)
    xb = torch.tensor((B - mu_b) / sd_b, dtype=torch.float32, device=dev)
    d, nn = A.shape[1], torch.nn
    net = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(),
                        nn.Linear(256, d)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0 = time.time()
    for _ in range(steps):
        i = torch.randint(0, len(xa), (min(512, len(xa)),), device=dev)
        loss = torch.mean((net(xa[i]) - xb[i]) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if time.time() - t0 > max_minutes * 60:
            break
    net.eval()

    def roll(x0, n_steps):
        out, x = [x0], x0.copy()
        with torch.no_grad():
            for _ in range(n_steps):
                zz = torch.tensor(((x - mu_a) / sd_a)[None], dtype=torch.float32, device=dev)
                x = x + net(zz).cpu().numpy()[0] * sd_b + mu_b
                if not np.all(np.isfinite(x)) or np.abs(x).max() > 1e6:
                    out += [np.full_like(x, np.nan)] * (n_steps + 1 - len(out))
                    break
                out.append(x.copy())
        return np.array(out)
    return roll


def nn_baseline(fc, minutes=NN_MINUTES):
    """Train a plain neural network on the same training data the agents saw (MLP step model for time series, the
    framework's FNO for fields), forecast the held-back part from the same last state, and add it to forecast.npz.
    Nothing is tuned on the held-back data. Returns a summary like fc['runs'], or {'skipped': reason}."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return {"skipped": "PyTorch is not installed (pip install torch) - no neural-network comparison"}
    z = dict(np.load(fc["npz"]))
    U, n, t, scale = z["U"].astype(float), int(z["n_train"]), z["t"], z["scale"]
    nfc = U.shape[1] - n + 1
    PN = np.full(U[:, n - 1:].shape, np.nan)
    if U.ndim == 3:
        roll = _mlp_step([U[j, :n] for j in range(U.shape[0])], max_minutes=minutes)
        kind = "MLP step model"
        for j in range(U.shape[0]):
            if np.all(np.isfinite(U[j, n - 1])):
                PN[j] = roll(U[j, n - 1], nfc - 1)
    else:
        from eqdisc.fno import rollout_fno, train_fno
        grid = U.shape[2:-1]
        modes = int(max(4, min(16, min(grid) // 4)))
        model = train_fno(U[:, :n], epochs=300, modes=modes, width=32, max_minutes=minutes)
        kind = "FNO"
        for j in range(U.shape[0]):
            if np.all(np.isfinite(U[j, n - 1])):
                with np.errstate(all="ignore"):
                    Y = rollout_fno(model, U[j, n - 1], nfc - 1)
                Y[~np.isfinite(Y).reshape(len(Y), -1).all(1)] = np.nan
                PN[j] = Y
    T = U[:, n - 1:]
    ax = tuple(range(2, U.ndim))
    with np.errstate(all="ignore"):
        errN = np.sqrt(np.mean(((PN - T) / scale) ** 2, axis=ax))
    errN[~np.isfinite(errN) & np.isfinite(T).all(ax)] = np.inf
    z.update(PN=PN, errN=errN, nn_kind=np.array(kind))
    np.savez_compressed(fc["npz"], **z)
    tf = t[n - 1:]
    runs = []
    for j in range(U.shape[0]):
        bad = np.where(errN[j] > THRESHOLD)[0]
        runs.append({"run": j, "useful_for": float(tf[bad[0]] - tf[0]) if bad.size else None})
    return {"kind": kind, "runs": runs, "minutes_cap": minutes}


def _useful_text(rs, horizon):
    u = [r["useful_for"] for r in rs]
    if all(x is None for x in u):
        return f"the whole window ({_g(horizon)})"
    vals = sorted(horizon if x is None else x for x in u)
    return f"{_g(vals[len(vals) // 2])} time units" + (" (median run)" if len(vals) > 1 else "")


def nn_sentence(fc):
    nn = fc.get("nn") or {}
    if nn.get("skipped"):
        return nn["skipped"] + "."
    if not nn.get("runs"):
        return ""
    return (f"A plain neural network ({nn['kind']}) trained on the same data stays useful for "
            f"{_useful_text(nn['runs'], fc['horizon'])}; the discovered law for {_useful_text(fc['runs'], fc['horizon'])}.")


def _finite_steps(P):
    return np.isfinite(P).reshape(P.shape[0], P.shape[1], -1).all(-1).sum(1)          # per run


def _retry_in_log(names, rhs, lay, U, n, tf, P):
    """If the law breaks down numerically, simulate the SAME law rewritten exactly in s = log(v) for every field v that
    is positive everywhere in the data and that the law takes the log of or divides by. The rewrite is the framework's
    exact chain-rule map (eqdisc.pde_coords); nothing is clipped, filtered or refitted. Used only when it runs longer."""
    from eqdisc import pde_coords
    from eqdisc.solvers import integrate_pde_general
    if (_finite_steps(P) == P.shape[1]).all():
        return P, None
    text = " ".join(rhs.values())
    pos = [v for i, v in enumerate(names) if np.nanmin(U[..., i]) > 0
           and re.search(rf"log\({re.escape(v)}\)|/\s*\(?{re.escape(v)}\b", text)]
    if not pos:
        return P, None
    new = {v: (f"s{v}" if f"s{v}" not in names else f"log{v}") for v in pos}
    try:
        zs = [new.get(v, v) for v in names]
        co = pde_coords.make_coords({**{"variables": zs}, **{k: v for k, v in lay.items() if k != "variables"}},
                                    {v: (f"exp({new[v]})" if v in new else v) for v in names},
                                    inverse={new.get(v, v): (f"log({v})" if v in new else v) for v in names})
        rz = pde_coords.map_back(rhs, co)
    except Exception:  # noqa: BLE001
        return P, None
    Q = np.full_like(P, np.nan)
    idx = [names.index(v) for v in pos]
    for j in range(U.shape[0]):
        Z0 = U[j, n - 1].copy()
        Z0[..., idx] = np.log(Z0[..., idx])
        if np.all(np.isfinite(Z0)):
            Q[j] = integrate_pde_general(zs, rz, lay, Z0, tf, max_seconds=90.0)
    Q[..., idx] = np.exp(Q[..., idx])
    if _finite_steps(Q).sum() <= _finite_steps(P).sum():
        return P, None
    raw = _finite_steps(P)
    return Q, {"log_of": pos, "raw_breakdown_at": [float(tf[min(k, len(tf) - 1)] - tf[0]) for k in raw],
               "rhs": {str(k): str(v) for k, v in rz.items()}}


def coords_note(fc):
    """Plain disclosure when the forecast was simulated in log coordinates."""
    c = fc.get("coords")
    if not c:
        return None
    v = ", ".join(c["log_of"])
    b = min(c["raw_breakdown_at"])
    return (f"Simulated in log({v}): the same law, rewritten exactly with the chain rule (nothing clipped, filtered "
            f"or refitted). Simulated directly in {v}, it breaks down numerically after {_g(b)} time units, because "
            f"one small step pushes {v} below zero where log({v}) is undefined.")


def outcome(fc):
    """One plain sentence: how long the law stays useful on the held-back data."""
    n, k = len(fc["runs"]), fc["n_useful_whole"]
    b = sorted(r["blew_up_at"] for r in fc["runs"] if r.get("blew_up_at") is not None)
    if b and len(b) == n:
        when = f"{_g(b[0])}" if b[0] == b[-1] else f"{_g(b[0])}–{_g(b[-1])}"
        return (f"The law breaks down {when} time units into the forecast"
                + ("" if n == 1 else f" in all {n} runs")
                + f" (of {_g(fc['horizon'])}): its values become infinite or invalid, so it cannot forecast this data.")
    if k == n:
        return (f"The law stays close to reality over the whole held-back window in "
                f"{'the run' if n == 1 else f'all {n} runs'} ({_g(fc['horizon'])} time units).")
    u = sorted(r["useful_for"] for r in fc["runs"] if r["useful_for"] is not None)
    if len(u) == 1:
        return (f"It drifts off after {_g(u[0])} of {_g(fc['horizon'])} time units"
                + (f" in 1 of {n} runs; the others stay close throughout." if n > 1 else "."))
    return (f"It stays close over the whole window in {k} of {n} runs; in the others it drifts off after "
            f"{_g(u[0])}–{_g(u[-1])} time units (window {_g(fc['horizon'])}).")


def blew_up_early(fc):
    """True when the law fails within the first few forecast steps in every run: the video has almost nothing to show."""
    b = [r.get("blew_up_at") for r in fc.get("runs") or []]
    return bool(b) and all(x is not None and x <= 0.05 * (fc.get("horizon") or 1) for x in b)


def caption(fc):
    n = len(fc["runs"])
    note = coords_note(fc)
    return (f"The agents never saw the last {fc['frac']:.0%} of {'each run' if n > 1 else 'the data'}. {outcome(fc)} "
            + (note + " " if note else "") + (nn_sentence(fc) + " " if nn_sentence(fc) else "") +
            f"Error = distance from reality as a share of each variable's typical spread; useful while below "
            f"{THRESHOLD:.0%}." + (f" The video shows run {fc['video_run'] + 1} of {n}, the one with the median error." if n > 1 else ""))


def _blow_banner(fig, P, tt):
    """Banner shown from the frame where the law's forecast stops being finite; returns the per-frame updater."""
    ok = np.isfinite(P).reshape(len(P), -1).all(1)
    k_b = int(np.argmin(ok)) if not ok.all() else None
    txt = fig.text(0.5, 0.905, "", ha="center", va="center", color="white", fontsize=VIDEO_FS,
                   bbox=dict(boxstyle="round,pad=0.4", fc="#dc2626", ec="none"), visible=False)

    def upd(k):
        if k_b is not None and k >= k_b:
            txt.set_text(f"The discovered law broke down {_g(tt[k_b])} time units in: its values became "
                         "infinite or invalid")
            txt.set_visible(True)
    return upd


def render_video(npz_path, out_path, run=0):
    """Forecast vs reality video (mp4, 1500×840) in the style of the example videos."""
    import matplotlib
    from matplotlib.animation import FFMpegWriter
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import FuncFormatter
    if not shutil.which(matplotlib.rcParams["animation.ffmpeg_path"]):
        raise RuntimeError("ffmpeg not found; install it (brew install ffmpeg) for the forecast video")
    z = np.load(npz_path)
    names = [str(s) for s in z["names"]]
    n, t = int(z["n_train"]), z["t"]
    U, P, err = z["U"][run], z["P"][run], z["err"][run]
    PN = z["PN"][run] if "PN" in z.files else None
    errN = z["errN"][run] if "errN" in z.files else None
    noise = float(z["noise_rel"])
    tf, T = t[n - 1:], U[n - 1:]
    tt = tf - tf[0]
    t_lo = max(t[0], tf[0] - 1.5 * (tf[-1] - tf[0]))
    win = (t >= t_lo) & (np.arange(len(t)) < n)
    nv = min(len(names), 3)
    rc = {"font.family": "cmb10", "mathtext.fontset": "cm", "axes.unicode_minus": False, "font.size": VIDEO_FS,
          "axes.titlesize": VIDEO_FS, "axes.labelsize": VIDEO_FS, "xtick.labelsize": VIDEO_FS,
          "ytick.labelsize": VIDEO_FS, "legend.fontsize": VIDEO_FS, "axes.titleweight": "normal"}
    cross = np.where(err > THRESHOLD)[0]
    k_cross = int(cross[0]) if cross.size else None
    with matplotlib.rc_context(rc):
        fig = Figure(figsize=(15, 8.4), dpi=100)
        FigureCanvasAgg(fig)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1], left=0.09, right=0.97, top=0.85, bottom=0.11, wspace=0.3)
        gl = gs[0, 0].subgridspec(nv, 1, hspace=0.3)
        gr = gs[0, 1].subgridspec(2, 1, hspace=0.62) if len(names) >= 2 else gs[0, 1].subgridspec(1, 1)
        series = []
        for i in range(nv):
            ax = fig.add_subplot(gl[i, 0])
            ax.axvspan(tf[0], tf[-1], color="#f3f4f6", zorder=0)
            ax.axvline(tf[0], color="#6b7280", ls="--", lw=1.2)
            ax.plot(t[win], U[win, i], color="#9ca3af", lw=1.8)
            ref = np.concatenate([U[win, i], T[:, i]])
            lo, hi = np.nanmin(ref), np.nanmax(ref)
            pad = 0.15 * (hi - lo or 1.0)
            ax.set(xlim=(t_lo, tf[-1]), ylim=(lo - pad, hi + pad))
            ax.set_ylabel(names[i], rotation=0 if len(names[i]) <= 3 else 90, ha="right" if len(names[i]) <= 3 else "center")
            if i == 0:
                ax.text(0.01, 1.04, "seen by the agents", transform=ax.transAxes, color="#6b7280", va="bottom")
                ax.text((tf[0] - t_lo) / (tf[-1] - t_lo) + 0.01, 1.04, "held back", transform=ax.transAxes,
                        color="#111827", va="bottom")
            if i == nv - 1:
                ax.set_xlabel("time")
            else:
                ax.tick_params(labelbottom=False)
            real, = ax.plot([], [], color="black", lw=2.2)
            law, = ax.plot([], [], color=GREEN, lw=2.8)
            rdot, = ax.plot([], [], "+", color="black", ms=18, mew=3)
            ldot, = ax.plot([], [], "o", color=GREEN, ms=11)
            nnl, = ax.plot([], [], color=NN_COLOR, lw=2.0)
            nnd, = ax.plot([], [], "o", color=NN_COLOR, ms=9)
            series.append((i, real, law, rdot, ldot, nnl, nnd))
        phase = None
        if len(names) >= 2:
            ap = fig.add_subplot(gr[0, 0])
            ap.plot(U[win, 0], U[win, 1], color="#d1d5db", lw=1.2)
            ref0, ref1 = np.concatenate([U[win, 0], T[:, 0]]), np.concatenate([U[win, 1], T[:, 1]])
            for setter, r in ((ap.set_xlim, ref0), (ap.set_ylim, ref1)):
                lo, hi = np.nanmin(r), np.nanmax(r)
                pad = 0.12 * (hi - lo or 1.0)
                setter(lo - pad, hi + pad)
            ap.set(xlabel=names[0], ylabel=names[1])
            ap.set_title(f"{names[1]} against {names[0]}")
            phase = (ap.plot([], [], color="black", lw=2.0)[0], ap.plot([], [], color=GREEN, lw=2.4)[0],
                     ap.plot([], [], "+", color="black", ms=18, mew=3)[0], ap.plot([], [], "o", color=GREEN, ms=11)[0],
                     ap.plot([], [], color=NN_COLOR, lw=1.8)[0], ap.plot([], [], "o", color=NN_COLOR, ms=9)[0])
            ae = fig.add_subplot(gr[1, 0])
        else:
            ae = fig.add_subplot(gr[0, 0])
        fin = np.concatenate([err[np.isfinite(err)]] + ([errN[np.isfinite(errN)]] if errN is not None else []))
        ae.set(xlim=(0, tt[-1] or 1.0), ylim=(0, max(1.0, min(3.0, 1.1 * (fin.max() if fin.size else 1)))))
        refs = [ae.axhline(THRESHOLD, color="#6b7280", ls="--", lw=1.4, label="useful limit")]
        if np.isfinite(noise) and noise < THRESHOLD:
            refs.append(ae.axhline(noise, color="#9ca3af", ls=":", lw=1.8, label="noise level"))
        ae.legend(handles=refs, loc="upper right", bbox_to_anchor=(1.0, 1.0), frameon=False, handlelength=1.0,
                  labelspacing=0.2, borderaxespad=0.2)
        ae.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{100 * y:.0f}%"))
        ae.set_title("Distance from reality")
        ae.set_xlabel("time since the last sample seen")
        eline, = ae.plot([], [], color=GREEN, lw=2.6)
        etxt = ae.text(0.02, 0.97, "", transform=ae.transAxes, va="top", color=GREEN)
        nline, = ae.plot([], [], color=NN_COLOR, lw=2.2)
        ntxt = ae.text(0.02, 0.86, "", transform=ae.transAxes, va="top", color=NN_COLOR)
        mark = ae.axvline(np.nan, color="black", ls="--", lw=1.4)
        mtxt = ae.text(0, 0, "", va="bottom", ha="left")
        stamp = fig.text(0.02, 0.965, "", va="top")
        blow = _blow_banner(fig, P, tt)
        if "coords" in z.files and str(z["coords"]):
            fig.text(0.02, 0.925, str(z["coords"]), va="top", color="#6b7280")
        from matplotlib.lines import Line2D
        fig.legend(handles=[Line2D([], [], color="black", lw=2.2, marker="+", ms=16, mew=3, label="Reality"),
                            Line2D([], [], color=GREEN, lw=2.8, marker="o", ms=10, label="Discovered law")]
                   + ([Line2D([], [], color=NN_COLOR, lw=2.2, marker="o", ms=9, label="Neural network")]
                      if PN is not None else []),
                   loc="upper right", bbox_to_anchor=(0.985, 0.995), ncol=3, frameon=False, columnspacing=1.4)

        def upd(k):
            s = slice(0, k + 1)
            for i, real, law, rdot, ldot, nnl, nnd in series:
                real.set_data(tf[s], T[s, i])
                law.set_data(tf[s], P[s, i])
                rdot.set_data([tf[k]], [T[k, i]])
                ldot.set_data([tf[k]], [P[k, i]])
                if PN is not None:
                    nnl.set_data(tf[s], PN[s, i])
                    nnd.set_data([tf[k]], [PN[k, i]])
            if phase:
                phase[0].set_data(T[s, 0], T[s, 1])
                phase[1].set_data(P[s, 0], P[s, 1])
                phase[2].set_data([T[k, 0]], [T[k, 1]])
                phase[3].set_data([P[k, 0]], [P[k, 1]])
                if PN is not None:
                    phase[4].set_data(PN[s, 0], PN[s, 1])
                    phase[5].set_data([PN[k, 0]], [PN[k, 1]])
            e = np.where(np.isfinite(err[s]), err[s], np.nan)
            eline.set_data(tt[s], e)
            etxt.set_text("Discovered law: " + (f"{100 * err[k]:.1f}%" if np.isfinite(err[k]) else "blew up"))
            if errN is not None:
                nline.set_data(tt[s], np.where(np.isfinite(errN[s]), errN[s], np.nan))
                ntxt.set_text("Neural network: " + (f"{100 * errN[k]:.1f}%" if np.isfinite(errN[k]) else "blew up"))
            if k_cross is not None and k >= k_cross:
                mark.set_xdata([tt[k_cross], tt[k_cross]])
                mtxt.set_position((tt[k_cross], 0.03 * ae.get_ylim()[1]))
                mtxt.set_text(f" useful until here ({_g(tt[k_cross])})")
            stamp.set_text(f"Forecast: {_g(tt[k])} time units after the last sample the agents saw")
            blow(k)

        idx = np.unique(np.linspace(0, len(tf) - 1, min(len(tf), 220)).round().astype(int))
        fps = int(np.clip(len(idx) / 9, 8, 25))
        writer = FFMpegWriter(fps=fps, codec="h264", bitrate=2400, extra_args=["-movflags", "+faststart"])
        with writer.saving(fig, str(out_path), dpi=100):
            for k in idx:
                upd(int(k))
                writer.grab_frame()
            for _ in range(int(1.5 * fps)):                       # hold the last frame
                writer.grab_frame()
    return str(out_path)


def render_field_video(npz_path, out_path, run=0):
    """Forecast vs reality for a field along one space axis: animated profiles per field (left), space-time of the
    first field filling in for reality and law (right), and the distance from reality."""
    import matplotlib
    from matplotlib.animation import FFMpegWriter
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter
    if not shutil.which(matplotlib.rcParams["animation.ffmpeg_path"]):
        raise RuntimeError("ffmpeg not found; install it (brew install ffmpeg) for the forecast video")
    z = np.load(npz_path)
    names = [str(s) for s in z["names"]]
    n, t, x = int(z["n_train"]), z["t"], z["x"]
    T, P, err = z["U"][run, n - 1:], z["P"][run], z["err"][run]          # (nfc, nx, nf)
    PN = z["PN"][run] if "PN" in z.files else None
    errN = z["errN"][run] if "errN" in z.files else None
    noise = float(z["noise_rel"])
    tt = t[n - 1:] - t[n - 1]
    nf = min(len(names), 2)
    rc = {"font.family": "cmb10", "mathtext.fontset": "cm", "axes.unicode_minus": False, "font.size": VIDEO_FS,
          "axes.titlesize": VIDEO_FS, "axes.labelsize": VIDEO_FS, "xtick.labelsize": VIDEO_FS,
          "ytick.labelsize": VIDEO_FS, "legend.fontsize": VIDEO_FS, "axes.titleweight": "normal"}
    cross = np.where(err > THRESHOLD)[0]
    k_cross = int(cross[0]) if cross.size else None
    with matplotlib.rc_context(rc):
        fig = Figure(figsize=(15, 8.4), dpi=100)
        FigureCanvasAgg(fig)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], left=0.08, right=0.97, top=0.85, bottom=0.1, wspace=0.28)
        gl = gs[0, 0].subgridspec(nf, 1, hspace=0.35)
        gr = gs[0, 1].subgridspec(3, 1, hspace=0.75, height_ratios=[1, 1, 1.3])
        prof = []
        for i in range(nf):
            ax = fig.add_subplot(gl[i, 0])
            lo, hi = np.nanmin(T[..., i]), np.nanmax(T[..., i])
            pad = 0.12 * (hi - lo or 1.0)
            ax.set(xlim=(x[0], x[-1]), ylim=(lo - pad, hi + pad))
            ax.set_ylabel(names[i], rotation=0 if len(names[i]) <= 3 else 90, ha="right" if len(names[i]) <= 3 else "center")
            if i == nf - 1:
                ax.set_xlabel("x")
            else:
                ax.tick_params(labelbottom=False)
            prof.append((i, ax.plot(x, T[0, :, i], color="black", lw=2.2)[0],
                         ax.plot(x, P[0, :, i], color=GREEN, lw=2.6)[0],
                         ax.plot(x, PN[0, :, i] if PN is not None else np.full_like(x, np.nan), color=NN_COLOR,
                                 lw=2.0)[0]))
        vmin, vmax = np.nanmin(T[..., 0]), np.nanmax(T[..., 0])
        maps = []
        for r, (lab, A) in enumerate((("Reality", T[..., 0]), ("Discovered law", P[..., 0]))):
            am = fig.add_subplot(gr[r, 0])
            im = am.imshow(np.full_like(A.T, np.nan), aspect="auto", origin="lower", cmap="RdBu_r", vmin=vmin,
                           vmax=vmax, extent=[0, tt[-1] or 1.0, x[0], x[-1]], interpolation="nearest")
            am.set_title(f"{lab}: {names[0]}")
            am.set_ylabel("x")
            am.tick_params(labelbottom=False)
            maps.append((im, A))
        ae = fig.add_subplot(gr[2, 0])
        fin = np.concatenate([err[np.isfinite(err)]] + ([errN[np.isfinite(errN)]] if errN is not None else []))
        ae.set(xlim=(0, tt[-1] or 1.0), ylim=(0, max(1.0, min(3.0, 1.1 * (fin.max() if fin.size else 1)))))
        refs = [ae.axhline(THRESHOLD, color="#6b7280", ls="--", lw=1.4, label="useful limit")]
        if np.isfinite(noise) and noise < THRESHOLD:
            refs.append(ae.axhline(noise, color="#9ca3af", ls=":", lw=1.8, label="noise level"))
        ae.legend(handles=refs, loc="upper right", bbox_to_anchor=(1.0, 1.0), frameon=False, handlelength=1.0,
                  labelspacing=0.2, borderaxespad=0.2)
        ae.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{100 * y:.0f}%"))
        ae.set_title("Distance from reality")
        ae.set_xlabel("time since the last snapshot seen")
        eline, = ae.plot([], [], color=GREEN, lw=2.6)
        etxt = ae.text(0.02, 0.97, "", transform=ae.transAxes, va="top", color=GREEN)
        nline, = ae.plot([], [], color=NN_COLOR, lw=2.2)
        ntxt = ae.text(0.02, 0.84, "", transform=ae.transAxes, va="top", color=NN_COLOR)
        mark = ae.axvline(np.nan, color="black", ls="--", lw=1.4)
        stamp = fig.text(0.02, 0.965, "", va="top")
        blow = _blow_banner(fig, P, tt)
        if "coords" in z.files and str(z["coords"]):
            fig.text(0.02, 0.925, str(z["coords"]), va="top", color="#6b7280")
        fig.legend(handles=[Line2D([], [], color="black", lw=2.2, label="Reality"),
                            Line2D([], [], color=GREEN, lw=2.8, label="Discovered law")]
                   + ([Line2D([], [], color=NN_COLOR, lw=2.2, label="Neural network")] if PN is not None else []),
                   loc="upper right", bbox_to_anchor=(0.985, 0.995), ncol=3, frameon=False, columnspacing=1.4)

        def upd(k):
            for i, real, law, nnl in prof:
                real.set_ydata(T[k, :, i])
                law.set_ydata(P[k, :, i] if np.all(np.isfinite(P[k, :, i])) else np.full_like(x, np.nan))
                if PN is not None:
                    nnl.set_ydata(PN[k, :, i] if np.all(np.isfinite(PN[k, :, i])) else np.full_like(x, np.nan))
            for im, A in maps:
                D = np.full_like(A, np.nan)
                D[: k + 1] = A[: k + 1]
                im.set_data(D.T)
            s = slice(0, k + 1)
            eline.set_data(tt[s], np.where(np.isfinite(err[s]), err[s], np.nan))
            etxt.set_text("Discovered law: " + (f"{100 * err[k]:.1f}%" if np.isfinite(err[k]) else "blew up"))
            if errN is not None:
                nline.set_data(tt[s], np.where(np.isfinite(errN[s]), errN[s], np.nan))
                ntxt.set_text("Neural network: " + (f"{100 * errN[k]:.1f}%" if np.isfinite(errN[k]) else "blew up"))
            if k_cross is not None and k >= k_cross:
                mark.set_xdata([tt[k_cross], tt[k_cross]])
            stamp.set_text(f"Forecast: {_g(tt[k])} time units after the last snapshot the agents saw")
            blow(k)

        idx = np.unique(np.linspace(0, len(tt) - 1, min(len(tt), 220)).round().astype(int))
        fps = int(np.clip(len(idx) / 9, 8, 25))
        writer = FFMpegWriter(fps=fps, codec="h264", bitrate=2400, extra_args=["-movflags", "+faststart"])
        with writer.saving(fig, str(out_path), dpi=100):
            for k in idx:
                upd(int(k))
                writer.grab_frame()
            for _ in range(int(1.5 * fps)):
                writer.grab_frame()
    return str(out_path)


def render_field2d_video(npz_path, out_path, run=0):
    """Forecast vs reality for a field on a 2-D grid: reality and law side by side per field (left), the distance
    from reality (right), like the Gray-Scott example video."""
    import matplotlib
    from matplotlib.animation import FFMpegWriter
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import FuncFormatter
    if not shutil.which(matplotlib.rcParams["animation.ffmpeg_path"]):
        raise RuntimeError("ffmpeg not found; install it (brew install ffmpeg) for the forecast video")
    z = np.load(npz_path)
    names = [str(s) for s in z["names"]]
    n, t = int(z["n_train"]), z["t"]
    T, P, err = z["U"][run, n - 1:], z["P"][run], z["err"][run]          # (nfc, nx, ny, nf)
    PN = z["PN"][run] if "PN" in z.files else None
    errN = z["errN"][run] if "errN" in z.files else None
    x = z["x"]
    y = z["y"] if "y" in z.files else np.arange(T.shape[2])
    noise = float(z["noise_rel"])
    tt = t[n - 1:] - t[n - 1]
    nf = min(len(names), 2)
    rc = {"font.family": "cmb10", "mathtext.fontset": "cm", "axes.unicode_minus": False, "font.size": VIDEO_FS,
          "axes.titlesize": VIDEO_FS, "axes.labelsize": VIDEO_FS, "xtick.labelsize": VIDEO_FS,
          "ytick.labelsize": VIDEO_FS, "legend.fontsize": VIDEO_FS, "axes.titleweight": "normal"}
    cross = np.where(err > THRESHOLD)[0]
    k_cross = int(cross[0]) if cross.size else None
    ext = [x[0], x[-1], y[0], y[-1]]
    with matplotlib.rc_context(rc):
        fig = Figure(figsize=(15, 8.4), dpi=100)
        FigureCanvasAgg(fig)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], left=0.04, right=0.97, top=0.86, bottom=0.1, wspace=0.22)
        cols = [("Reality", T, "black"), ("Discovered law", P, GREEN)] + ([("Neural network", PN, NN_COLOR)]
                                                                          if PN is not None else [])
        gl = gs[0, 0].subgridspec(nf, len(cols), hspace=0.3, wspace=0.08)
        ims = []
        for i in range(nf):
            vmin, vmax = np.nanmin(T[..., i]), np.nanmax(T[..., i])
            for c, (lab, A, colr) in enumerate((lab_, B[..., i], col_) for lab_, B, col_ in cols):
                am = fig.add_subplot(gl[i, c])
                im = am.imshow(A[0].T, origin="lower", cmap="RdBu_r", vmin=vmin, vmax=vmax, extent=ext,
                               interpolation="nearest")
                am.set_title(f"{lab}: {names[i]}" if len(cols) < 3 else f"{lab}\n{names[i]}", color=colr)
                am.set_xticks([])
                am.set_yticks([])
                ims.append((im, A))
        ae = fig.add_subplot(gs[0, 1])
        fin = np.concatenate([err[np.isfinite(err)]] + ([errN[np.isfinite(errN)]] if errN is not None else []))
        ae.set(xlim=(0, tt[-1] or 1.0), ylim=(0, max(1.0, min(3.0, 1.1 * (fin.max() if fin.size else 1)))))
        refs = [ae.axhline(THRESHOLD, color="#6b7280", ls="--", lw=1.4, label="useful limit")]
        if np.isfinite(noise) and noise < THRESHOLD:
            refs.append(ae.axhline(noise, color="#9ca3af", ls=":", lw=1.8, label="noise level"))
        ae.legend(handles=refs, loc="upper right", bbox_to_anchor=(1.0, 1.0), frameon=False, handlelength=1.0,
                  labelspacing=0.2, borderaxespad=0.2)
        ae.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
        ae.set_title("Distance from reality")
        ae.set_xlabel("time since the last snapshot seen")
        eline, = ae.plot([], [], color=GREEN, lw=2.6)
        etxt = ae.text(0.02, 0.97, "", transform=ae.transAxes, va="top", color=GREEN)
        nline, = ae.plot([], [], color=NN_COLOR, lw=2.2)
        ntxt = ae.text(0.02, 0.84, "", transform=ae.transAxes, va="top", color=NN_COLOR)
        mark = ae.axvline(np.nan, color="black", ls="--", lw=1.4)
        mtxt = ae.text(0, 0, "", va="bottom", ha="left")
        stamp = fig.text(0.02, 0.965, "", va="top")
        blow = _blow_banner(fig, P, tt)
        if "coords" in z.files and str(z["coords"]):
            fig.text(0.02, 0.925, str(z["coords"]), va="top", color="#6b7280")

        def upd(k):
            for im, A in ims:
                im.set_data((A[k] if np.all(np.isfinite(A[k])) else np.full_like(A[k], np.nan)).T)
            s = slice(0, k + 1)
            eline.set_data(tt[s], np.where(np.isfinite(err[s]), err[s], np.nan))
            etxt.set_text("Discovered law: " + (f"{100 * err[k]:.1f}%" if np.isfinite(err[k]) else "blew up"))
            if errN is not None:
                nline.set_data(tt[s], np.where(np.isfinite(errN[s]), errN[s], np.nan))
                ntxt.set_text("Neural network: " + (f"{100 * errN[k]:.1f}%" if np.isfinite(errN[k]) else "blew up"))
            if k_cross is not None and k >= k_cross:
                mark.set_xdata([tt[k_cross], tt[k_cross]])
                mtxt.set_position((tt[k_cross], 0.03 * ae.get_ylim()[1]))
                mtxt.set_text(f" useful until here ({_g(tt[k_cross])})")
            stamp.set_text(f"Forecast: {_g(tt[k])} time units after the last snapshot the agents saw")
            blow(k)

        idx = np.unique(np.linspace(0, len(tt) - 1, min(len(tt), 160)).round().astype(int))
        fps = int(np.clip(len(idx) / 9, 6, 20))
        writer = FFMpegWriter(fps=fps, codec="h264", bitrate=3000, extra_args=["-movflags", "+faststart"])
        with writer.saving(fig, str(out_path), dpi=100):
            for k in idx:
                upd(int(k))
                writer.grab_frame()
            for _ in range(int(1.5 * fps)):
                writer.grab_frame()
    return str(out_path)


def _training_field2d_fig(z):
    """The first field of the first run: first snapshot, last snapshot the agents saw, last held-back snapshot."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    name = str(z["names"][0])
    n, t, U = int(z["n_train"]), z["t"], z["U"][0, ..., 0]
    picks = [(0, f"t = {_g(t[0])}: first"), (n - 1, f"t = {_g(t[n - 1])}: last seen"),
             (len(t) - 1, f"t = {_g(t[-1])}: held back")]
    fig = make_subplots(rows=1, cols=3, subplot_titles=[p[1] for p in picks], horizontal_spacing=0.03)
    lo, hi = float(np.nanmin(U)), float(np.nanmax(U))
    for c, (k, _) in enumerate(picks):
        fig.add_trace(go.Heatmap(z=U[k].T, colorscale="RdBu_r", zmin=lo, zmax=hi, showscale=False,
                                 hovertemplate=f"{name} %{{z:.3g}}<extra></extra>"), row=1, col=c + 1)
        fig.update_xaxes(showticklabels=False, row=1, col=c + 1)
        fig.update_yaxes(showticklabels=False, scaleanchor=f"x{c + 1 if c else ''}", row=1, col=c + 1)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    return fig


def _run_colors(n):
    import viz
    pal = [viz.AQUA, viz.BLUE, viz.ORANGE, "#6b7280"]          # violet is reserved for the neural network
    return [pal[j % len(pal)] for j in range(n)]


def training_fig(npz_path, max_runs=6):
    """What the agents saw (solid) and what was held back (dotted), per variable."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    z = np.load(npz_path)
    if z["U"].ndim == 5:
        return _training_field2d_fig(z)
    if z["U"].ndim == 4:
        return _training_field_fig(z)
    names = [str(s) for s in z["names"]][:3]
    n, t, U = int(z["n_train"]), z["t"], z["U"]
    runs = min(U.shape[0], max_runs)
    cols = _run_colors(runs)
    fig = make_subplots(rows=len(names), cols=1, shared_xaxes=True, vertical_spacing=0.06)
    for i, v in enumerate(names):
        for j in range(runs):
            first = i == 0
            fig.add_trace(go.Scatter(x=t[:n], y=U[j, :n, i], mode="lines", line=dict(color=cols[j], width=1.6),
                                     name="seen by the agents" if runs == 1 else f"run {j + 1}", legendgroup=str(j),
                                     showlegend=first), row=i + 1, col=1)
            fig.add_trace(go.Scatter(x=t[n - 1:], y=U[j, n - 1:, i], mode="lines", opacity=0.55,
                                     line=dict(color=cols[j], width=1.6, dash="dot"), name="held back",
                                     legendgroup=str(j), showlegend=first and runs == 1), row=i + 1, col=1)
        fig.add_vline(x=float(t[n - 1]), line=dict(color="rgba(120,120,120,0.85)", dash="dash", width=1),
                      row=i + 1, col=1)
        fig.update_yaxes(title_text=v, row=i + 1, col=1)
    fig.update_xaxes(title_text="time  (dotted: held back)", row=len(names), col=1)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      legend=dict(orientation="h", y=1.08, x=0))
    return fig


def _training_field_fig(z):
    """Space-time of the first run per field; the dashed line marks where the held-back part starts."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    names = [str(s) for s in z["names"]][:2]
    n, t, x, U = int(z["n_train"]), z["t"], z["x"], z["U"][0]
    st = max(1, len(t) // 300)
    fig = make_subplots(rows=1, cols=len(names), subplot_titles=names, horizontal_spacing=0.08)
    for i in range(len(names)):
        fig.add_trace(go.Heatmap(z=U[::st, :, i], x=x, y=t[::st], colorscale="RdBu_r", showscale=False,
                                 hovertemplate="x %{x:.3g}<br>t %{y:.3g}<br>value %{z:.3g}<extra></extra>"),
                      row=1, col=i + 1)
        fig.add_hline(y=float(t[n - 1]), line=dict(color="black", dash="dash", width=1.5), row=1, col=i + 1)
        fig.update_xaxes(title_text="x", row=1, col=i + 1)
    fig.update_yaxes(title_text="time  (above the line: held back)", row=1, col=1)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    return fig


def error_fig(npz_path, max_runs=6):
    """Distance from reality over the held-back window, per run."""
    import plotly.graph_objects as go
    z = np.load(npz_path)
    n, t, err = int(z["n_train"]), z["t"], z["err"]
    tt = t[n - 1:] - t[n - 1]
    runs = min(err.shape[0], max_runs)
    cols = _run_colors(runs)
    fig = go.Figure()
    for j in range(runs):
        e = np.where(np.isfinite(err[j]), err[j], np.nan) * 100
        fig.add_trace(go.Scatter(x=tt, y=e, mode="lines", line=dict(color=cols[j], width=2.2),
                                 name="Discovered law" if runs == 1 else f"run {j + 1}"))
    fig.add_hline(y=THRESHOLD * 100, line=dict(color="rgba(120,120,120,0.85)", dash="dash", width=1),
                  annotation_text="useful limit", annotation_position="top left")
    if "errN" in z.files:
        lab = f"Neural network ({str(z['nn_kind'])})" if "nn_kind" in z.files else "Neural network"
        for j in range(runs):
            e = np.where(np.isfinite(z["errN"][j]), z["errN"][j], np.nan) * 100
            fig.add_trace(go.Scatter(x=tt, y=e, mode="lines", line=dict(color=NN_COLOR, width=2, dash="dot"),
                                     name=lab, legendgroup="nn", showlegend=j == 0))
    blown = [(j, int(np.where(np.isinf(err[j]))[0][0])) for j in range(runs) if np.isinf(err[j]).any()]
    for j, k in blown[:1]:
        fig.add_vline(x=float(tt[k]), line=dict(color="#dc2626", width=2))
        fig.add_annotation(x=float(tt[k]), y=0.92, yref="paper", xanchor="left", showarrow=False,
                           font=dict(color="#dc2626"), text=f" law broke down at {_g(tt[k])}"
                           + (f" (run {j + 1})" if runs > 1 else ""))
    noise = float(z["noise_rel"])
    if np.isfinite(noise) and noise < THRESHOLD:
        fig.add_hline(y=noise * 100, line=dict(color="rgba(160,160,160,0.9)", dash="dot", width=1),
                      annotation_text="noise level", annotation_position="bottom right")
    allv = np.concatenate([err[:runs].ravel()] + ([z["errN"][:runs].ravel()] if "errN" in z.files else []))
    top = np.nanmax(allv[np.isfinite(allv)]) * 100 if np.isfinite(allv).any() else 100
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified", showlegend=runs > 1 or "errN" in z.files,
                      legend=dict(orientation="h", y=1.08, x=0),
                      xaxis_title="time since the last sample seen",
                      yaxis=dict(title="error (% of typical spread)", range=[0, max(60, min(300, 1.1 * top))]))
    return fig


# ----------------------------------------------------------------------------- what the equation means
def _names_in(exprs, extra=()):
    toks = set(extra)
    for e in exprs:
        toks |= set(re.findall(r"[A-Za-z_]\w*", str(e)))
    return sorted(toks - FUNCS)


def _deriv(sym, variables):
    """'u_xx' -> ('u', 'xx') for a field u; None otherwise."""
    m = re.fullmatch(r"(.+)_([xyz]+)", sym)
    return (m.group(1), m.group(2)) if m and m.group(1) in variables else None


def _describe(v, c, rest, variables):
    """Plain facts about one additive term c*rest in dv/dt."""
    import sympy as sp
    up = "raises" if c > 0 else "lowers"
    if rest == 1:
        return f"Constant rate: {v} changes by {_g(c)} per unit of time, whatever the state."
    syms = sorted(map(str, rest.free_symbols))
    ders = [(s, _deriv(s, variables)) for s in syms if _deriv(s, variables)]
    if ders:
        s, (f, ax) = ders[0]
        k = len(ax)
        if rest == sp.Symbol(s):
            if k == 1:
                return f"Transport: the pattern in {f} drifts along {ax} at speed {_g(-c)}."
            if k == 2:
                return (f"Diffusion: smooths out bumps in {f} (strength {_g(c)})." if c > 0 else
                        f"Anti-diffusion: makes long waves in {f} grow (strength {_g(-c)}); another term must stop them.")
            if k == 3:
                return f"Dispersion: waves of different lengths travel at different speeds (strength {_g(c)})."
            if k == 4:
                return (f"Hyper-diffusion: damps short ripples in {f} strongly (strength {_g(-c)})." if c < 0 else
                        f"Makes short ripples in {f} grow; unstable on its own (strength {_g(c)}).")
        if k == 1 and sp.Symbol(f) in rest.free_symbols and sp.degree(rest, sp.Symbol(s)) == 1:
            return f"Nonlinear transport: larger values of {f} move faster, which steepens fronts (strength {_g(c)})."
        return f"Spatial nonlinear term {rest}: {up} {v} (strength {_g(c)})."
    if rest == sp.Symbol("t"):
        return f"Steady drift in time: the rate of change of {v} shifts by {_g(c)} per unit of time."
    if rest == sp.Symbol(v):
        if c < 0:
            return f"Decay: {v} relaxes back toward zero at rate {_g(-c)} per unit time (time scale about {_g(-1 / c)})."
        return f"Self-growth: {v} grows in proportion to itself at rate {_g(c)} (doubling time about {_g(math.log(2) / c)})."
    if rest.is_Symbol and str(rest) in variables:
        w = str(rest)
        if abs(c - 1) < 0.02:
            return f"{v} changes at the rate {w}: {w} acts as the velocity of {v}."
        return f"Linear coupling: {w} {up} the rate of change of {v} (strength {_g(c)})."
    if rest.is_Pow and rest.base.is_Symbol and rest.exp.is_Integer and int(rest.exp) >= 2:
        b, p = str(rest.base), int(rest.exp)
        if b == v and p == 2:
            return (f"Self-limiting (crowding): slows {v} more and more as it grows ({_g(c)}·{v}²)." if c < 0 else
                    f"Self-reinforcing: grows like {v}², so large values run away faster.")
        if b == v and p % 2 == 1:
            return (f"Nonlinear restoring push: pulls {v} back toward zero, much harder at large values ({v}^{p})."
                    if c < 0 else f"Nonlinear push away from zero, stronger at large values ({v}^{p}).")
        return f"Nonlinear term {b}^{p}: {up} {v} (strength {_g(c)}), strongest where |{b}| is large."
    if rest.is_Mul and all(f.is_Symbol or (f.is_Pow and f.base.is_Symbol and f.exp.is_Integer) for f in rest.args) \
            and len(rest.free_symbols) >= 2:
        fs = " and ".join(sorted(str(s) for s in rest.free_symbols))
        return f"Interaction: proportional to {rest}, so it acts only when {fs} are both non-zero; {up} {v} (strength {_g(c)})."
    trig = [a for a in rest.atoms(sp.sin, sp.cos)]
    if trig:
        arg = trig[0].args[0]
        s = sorted(arg.free_symbols, key=str)
        if len(s) == 1:
            k = float(sp.Poly(arg, s[0]).coeffs()[0]) if arg.is_polynomial(s[0]) and sp.degree(arg, s[0]) == 1 else None
            per = f", repeating every {_g(2 * math.pi / abs(k))} in {s[0]}" if k else ""
            what = "forcing in time" if str(s[0]) == "t" else f"dependence on {s[0]}"
            small = (f"; for small {s[0]} it acts like {_g(c * k)}·{s[0]}" if k and rest == sp.sin(arg) and str(s[0]) != "t"
                     else "")
            return f"Periodic {what}{per}{small} (strength {_g(c)})."
        return f"Periodic term {rest}: {up} {v} (strength {_g(c)})."
    if rest.has(sp.exp):
        return f"Exponential dependence {rest}: {up} {v}; small changes in its argument have large effects."
    if any(p.exp.is_negative for p in rest.atoms(sp.Pow)):
        return f"Rational term {rest}: its effect levels off or changes sharply where the denominator is small ({up} {v})."
    return f"Nonlinear term {rest}: {up} {v} (strength {_g(c)})."


def term_notes(rhs):
    """{var: expr} -> one plain-facts note per additive term."""
    import sympy as sp
    from eqdisc.solvers import parse
    variables = list(rhs)
    names = _names_in(rhs.values(), variables + ["t"])
    out = []
    for v in variables:
        try:
            e = sp.expand(parse(rhs[v], names))
        except Exception:  # noqa: BLE001
            out.append({"var": v, "term": str(rhs[v]), "coef": None, "note": f"d{v}/dt = {rhs[v]}"})
            continue
        for term in sp.Add.make_args(e):
            c, rest = term.as_coeff_Mul()
            try:
                note = _describe(v, float(c), rest, variables)
            except Exception:  # noqa: BLE001
                note = f"Term {rest} (coefficient {_g(float(c))})."
            out.append({"var": v, "term": str(rest), "coef": float(c), "note": note})
    return out


def static_notes(expr, names, target):
    """y = f(x): one plain-facts note per additive term."""
    import sympy as sp
    from eqdisc.solvers import parse
    try:
        e = parse(expr, _names_in([expr], names))
    except Exception:  # noqa: BLE001
        return [{"var": target, "term": expr, "coef": None, "note": f"{target} = {expr}"}]
    out = []
    for term in sp.Add.make_args(e):
        c, rest = term.as_coeff_Mul()
        c = float(c)
        if rest == 1:
            note = f"Offset: adds {_g(c)} to {target}."
        elif rest.is_Symbol:
            note = f"Linear: {target} changes by {_g(c)} per unit of {rest}."
        elif rest.is_Pow and rest.base.is_Symbol and rest.exp.is_number:
            note = f"Power law: this part scales as {rest.base}^{_g(float(rest.exp))} (factor {_g(c)})."
        else:
            note = f"Depends on {', '.join(sorted(map(str, rest.free_symbols)))} through {rest} (factor {_g(c)})."
        out.append({"var": target, "term": str(rest), "coef": c, "note": note})
    return out


MEANING_PROMPT = """A data-driven discovery produced this law from measured data. Explain to the person who measured \
the data what the law means, in plain language.

Kind: {kind}
Variables (the person's column names): {names}
The law: {law}
Term-by-term facts computed from the law (exact; use these numbers): {notes}
How well the data pin down each coefficient: {terms}
What the person said about the data (may be empty): {prompt}
Forecast check on data the discovery never saw: {forecast}

Write one bullet per term, or per group of closely related terms: what it does, in words, with a time scale or size \
where the facts give one. If the law as a whole has the form of a well-known model, add one bullet that says "has the \
same form as ..." and names it; the discovery was not told what the data are, so present this as a reading of the \
form, not a fact about the person's system. Skip that bullet if nothing fits well. If the person described the data, \
you may relate terms to that description.

Rules: say nothing the law or the facts do not support; short sentences; use the column names as given; numbers to \
2-3 significant figures; no LaTeX, use plain text and unicode (², ·, √).
Answer with JSON only: {{"bullets": [{{"title": "short label", "text": "one or two sentences"}}], \
"summary": "one sentence: what kind of system this is, in plain words"}}"""


def _claude(prompt):
    from eqdisc.agent import Usage, make_client
    client = make_client()
    resp = client.beta.messages.create(model=EXPLAIN_MODEL, max_tokens=4000, output_config={"effort": "low"},
                                       betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                                       messages=[{"role": "user", "content": prompt}])
    usage = Usage(EXPLAIN_MODEL)
    usage.add(resp)
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    return json.loads(text[text.index("{"): text.rindex("}") + 1]), usage.cost()


def explain(kind, law, names, notes, assessment=None, prompt="", fc=None, use_llm=True):
    """Plain-language reading of the found law. Falls back to the rule-based notes without an API call."""
    def title(n):
        if n["coef"] is None:
            return n["term"]
        t = _g(n["coef"]) if n["term"] == "1" else f"{_g(n['coef'])}·{n['term']}"
        return f"{t} in d{n['var']}/dt" if kind == "dynamics" else t
    rules = {"bullets": [{"title": title(n), "text": n["note"]} for n in notes],
             "summary": None, "source": "rules", "cost_usd": 0.0}
    if not use_llm:
        return rules
    terms = [{k: t.get(k) for k in ("var", "term", "coef", "ci90", "rel_uncertainty", "significant")}
             for t in ((assessment or {}).get("terms") or [])]
    fcs = caption(fc) if fc and fc.get("runs") else "not run"
    try:
        out, cost = _claude(MEANING_PROMPT.format(kind=kind, names=", ".join(names), law=law,
                                                  notes=json.dumps([n["note"] for n in notes]),
                                                  terms=json.dumps(terms, default=str)[:3000], prompt=prompt or "",
                                                  forecast=fcs))
        bullets = [b for b in out.get("bullets") or [] if isinstance(b, dict) and b.get("text")]
        if not bullets:
            raise ValueError("no bullets")
        return {"bullets": bullets, "summary": out.get("summary"), "source": "claude", "cost_usd": round(cost, 4),
                "used_prompt": bool(prompt)}
    except Exception as e:  # noqa: BLE001
        return {**rules, "llm_error": f"{type(e).__name__}: {e}"[:300]}


def _full_names(model, train_ds, split):
    """The training file can name variables differently from the full file (e.g. an .npz field array 'U' vs 'uu');
    the order is the same, so rename the law's symbols (and their derivatives, U_xx -> uu_xx) by position."""
    from eqdisc.evaluate import load
    if not model or not train_ds:
        return model
    try:
        old, new = load(train_ds)[0]["variables"], load(split["full_ds"])[0]["variables"]
    except Exception:  # noqa: BLE001
        return model
    if old == new or len(old) != len(new) or set(old) != set(model):
        return model
    m = dict(zip(old, new))
    pat = re.compile(r"\b(" + "|".join(map(re.escape, sorted(old, key=len, reverse=True))) + r")(?=_[xyz]+\b|\b)")
    return {m[v]: pat.sub(lambda g: m[g.group(1)], e) for v, e in model.items()}


# ----------------------------------------------------------------------------- background jobs (wrap live.py backends)
def _stage(on_event, text):
    on_event({"type": "stage", "text": text})


def _recorder(on_event):
    events = []

    def push(ev):
        events.append(ev)
        on_event(ev)
    return push, events


def dynamics_job(backend, csv_path, run_dir, holdout, use_llm, on_event, **kw):
    """Forecast check (optional) -> discovery on the training part -> forecast video -> what the law means.
    The result is saved to <run_dir>/demo/result.json so the page can recall it later."""
    import time
    t0 = time.time()
    on_event, events = _recorder(on_event)
    res = _dynamics(backend, csv_path, run_dir, holdout, use_llm, on_event, **kw)
    save_result(run_dir, res, events, {"file": Path(csv_path).name, "prompt": kw.get("context") or "",
                                       "rehearsal": bool(res.get("rehearsal")), "holdout": holdout,
                                       "elapsed": time.time() - t0})
    return res


def _dynamics(backend, csv_path, run_dir, holdout, use_llm, on_event, **kw):
    run_dir = Path(run_dir)
    split = None
    if holdout:
        _stage(on_event, f"[+] forecast check: holding back the last {holdout:.0%} of every run; the agents never see it")
        try:
            split = split_holdout(csv_path, run_dir, holdout)
        except Exception as e:  # noqa: BLE001
            split = {"skipped": f"{type(e).__name__}: {e}"[:200]}
        if split.get("skipped"):
            _stage(on_event, f"forecast check skipped: {split['skipped']}")
    res = backend(csv_path=split["train_path"] if split and not split.get("skipped") else csv_path, run_dir=str(run_dir),
                  on_event=on_event, **kw)
    if split:
        fc = {"skipped": split["skipped"]} if split.get("skipped") else None
        if fc is None:
            _stage(on_event, "[+] forecast check: the found law predicts the held-back data")
            try:
                fc = forecast(_full_names(res.get("final_model"), res.get("dataset_path"), split), split,
                              run_dir / "demo")
                if fc.get("npz"):
                    _stage(on_event, outcome(fc))
                    _stage(on_event, f"[+] baseline: training a plain neural network on the same data (up to "
                                     f"{NN_MINUTES:g} min)")
                    try:
                        fc["nn"] = nn_baseline(fc)
                    except Exception as e:  # noqa: BLE001
                        fc["nn"] = {"skipped": f"neural-network baseline failed: {type(e).__name__}: {e}"[:200]}
                    if nn_sentence(fc):
                        _stage(on_event, nn_sentence(fc))
                    _stage(on_event, "[+] rendering the forecast video")
                    video = (render_field2d_video if fc.get("field2d") else render_field_video if fc.get("field")
                             else render_video)
                    fc["video"] = video(fc["npz"], run_dir / "demo" / "forecast.mp4", run=fc["video_run"])
            except Exception as e:  # noqa: BLE001
                fc = {**(fc or {}), "error": f"{type(e).__name__}: {e}"[:300]}
                _stage(on_event, f"forecast check failed: {fc['error']}")
        res["forecast"] = fc
    if res.get("final_model"):
        _stage(on_event, "[+] what the equation means" + ("" if use_llm else " (rule-based, no API call)"))
        law = "; ".join(f"d{v}/dt = {e}" for v, e in res["final_model"].items())
        res["meaning"] = explain("dynamics", law, list(res["final_model"]), term_notes(res["final_model"]),
                                 res.get("assessment"), kw.get("context") or "", res.get("forecast"), use_llm)
        res["cost_usd"] = (res.get("cost_usd") or 0) + res["meaning"]["cost_usd"]
    return res


def static_job(backend, use_llm, on_event, run_dir=None, **kw):
    import time
    t0 = time.time()
    on_event, events = _recorder(on_event)
    res = backend(on_event=on_event, **kw)
    if res.get("expr"):
        _stage(on_event, "[+] what the law means" + ("" if use_llm else " (rule-based, no API call)"))
        res["meaning"] = explain("static law y = f(x)", f"{res['target']} = {res['expr']}", res["names"],
                                 static_notes(res["expr"], res["names"], res["target"]), res.get("assessment"),
                                 kw.get("context") or "", None, use_llm)
        res["cost_usd"] = (res.get("cost_usd") or 0) + res["meaning"]["cost_usd"]
    if run_dir:
        save_result(run_dir, res, events, {"file": Path(kw.get("csv_path", "data")).name, "prompt": kw.get("context") or "",
                                           "rehearsal": bool(res.get("rehearsal")), "elapsed": time.time() - t0})
    return res


# ----------------------------------------------------------------------------- saved results
SAVED = "result.json"


def save_result(run_dir, res, events, meta):
    import time
    out = Path(run_dir) / "demo" / SAVED
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {"version": 1, "saved_at": time.strftime("%Y-%m-%d %H:%M"), "run_dir": str(Path(run_dir).resolve()),
           "meta": meta, "result": res, "events": events}
    out.write_text(json.dumps(doc, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    return str(out)


def list_saved(root):
    """Saved results under the live-runs folder, newest first: [(label, path)]."""
    items = []
    for p in sorted(Path(root).glob(f"*/demo/{SAVED}"), reverse=True):
        try:
            d = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        r, m = d.get("result") or {}, d.get("meta") or {}
        status = ((r.get("verdict") or {}).get("status") or "no verdict").replace("_", " ").lower()
        cost = r.get("cost_usd")
        label = (f"{d.get('saved_at', p.parent.parent.name)} · {m.get('file', '?')} · {status}"
                 + (f" · ${cost:.2f}" if cost else "") + (" · rehearsal" if m.get("rehearsal") else ""))
        items.append((label, str(p)))
    return items


def _relocate(path, old_root, new_root):
    """Paths saved inside a run folder still work if the folder was moved or copied."""
    if not path or Path(path).exists() or not old_root:
        return path
    try:
        cand = Path(new_root) / Path(path).relative_to(old_root)
    except ValueError:
        return path
    return str(cand) if cand.exists() else path


class SavedJob:
    """A finished run read back from disk; quacks like live.Job for the result page."""

    def __init__(self, path):
        d = json.loads(Path(path).read_text())
        run_dir = Path(path).parent.parent
        old = d.get("run_dir")
        r = d["result"]
        for k in ("report", "dataset_path"):
            r[k] = _relocate(r.get(k), old, run_dir)
        fc = r.get("forecast") or {}
        for k in ("video", "npz"):
            if fc.get(k):
                fc[k] = _relocate(fc[k], old, run_dir)
        self.result, self.events, self.error = r, d.get("events") or [], None
        self.prompt = (d.get("meta") or {}).get("prompt", "")
        self.run_dir, self.done, self.saved_at = str(run_dir), True, d.get("saved_at")
        self.elapsed = (d.get("meta") or {}).get("elapsed") or 0.0

    def drain(self):
        return self.events


# ----------------------------------------------------------------------------- page pieces
def _esc(s):
    """Plain text -> markdown that shows it literally (terms like 0.8*N*x must not turn into italics)."""
    return re.sub(r"([\\*_$`#<>\[\]])", r"\\\1", str(s).strip())


def meaning_md(m):
    lines = [f"- **{_esc(b['title'])}:** {_esc(b['text'])}" if b.get("title") else f"- {_esc(b['text'])}"
             for b in m.get("bullets") or []]
    if m.get("summary"):
        lines.insert(0, f"**{_esc(m['summary'])}**\n")
    return "\n".join(lines)


def meaning_note(m):
    if m.get("source") == "claude":
        src = "the law, its uncertainties" + (" and your prompt" if m.get("used_prompt") else "")
        return (f"Written by Claude after the discovery, from {src}. It is a reading of the equation, not checked "
                f"against the data (cost ${m.get('cost_usd', 0):.3f}).")
    why = " (Claude unavailable: " + m["llm_error"][:120] + ")" if m.get("llm_error") else ""
    return "Rule-based reading of each term, computed from the law itself; no API call" + why + "."
