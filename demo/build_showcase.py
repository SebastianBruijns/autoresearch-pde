"""Collect the demo's artifacts into demo/showcase/ so the app reads nothing outside demo/ and renders instantly.

    cd /Users/danield/eqdisc && PYTHONPATH=. python demo/build_showcase.py              # everything
    PYTHONPATH=. python demo/build_showcase.py gray_scott lageos                         # refresh some cases

Cases (honest out-of-sample protocol, produced by eqdisc/oos.py):
  lageos      runs/oos_lageos/              real LAGEOS-1 orbit, trained on 2017, forecast the unseen next month
  ks          runs/oos_ks_agent/            blinded Kuramoto-Sivashinsky, 2% noise, forecast the unseen future
  gray_scott  runs/oos_gs_spirals_n0.05/    The Well Gray-Scott, forecast a held-out trajectory (may be pending)
Plus `rehearsal`: data for the live tab's scripted no-API mode, and the example CSVs.
No Claude API calls are made here.
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

DEMO = Path(__file__).resolve().parent
REPO = DEMO.parent
sys.path.insert(0, str(REPO))

OUT = DEMO / "showcase"
AR = Path("/Users/danield/iterate-hackathon/autoresearch")
import os as _os
LAGEOS_CSV = Path(_os.environ.get("EQDISC_LAGEOS_CSV", "/Users/danield/iterate-hackathon/orbit_discover/data/lageos1.csv"))


def _f32(a):
    return np.asarray(a, dtype=np.float32)


def _json(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def _thumb(video, dst, at=4.0):
    """First-look thumbnail from an mp4 (ffmpeg); returns True on success."""
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", str(at), "-i", str(video), "-frames:v", "1",
                        "-vf", "scale=560:315:force_original_aspect_ratio=increase,crop=560:315", str(dst)], check=True, timeout=60)
        return Path(dst).exists()
    except Exception:  # noqa: BLE001
        return False


def _copy(src, dst):
    """Copy a file; mp4s are re-encoded (H.264, CRF 26, faststart) to keep the repo small, falling back to a copy."""
    src = Path(src)
    if not src.exists():
        return False
    if src.suffix == ".mp4":
        try:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-crf", "26",
                            "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)],
                           check=True, timeout=600)
            return True
        except Exception:  # noqa: BLE001
            pass
    shutil.copy(src, dst)
    return True


def _rationale(transcript):
    """The agent's final submission rationale (its own chain of reasoning), from transcript.json."""
    f = Path(transcript)
    if not f.exists():
        return None
    r = [e.get("input", {}).get("rationale") for e in json.loads(f.read_text()) if e.get("name") == "submit"]
    return r[-1] if r else None


def _tools(*transcripts):
    """Tool names actually called by the agent(s), in order, from transcript.json files."""
    out = []
    for t in transcripts:
        for f in sorted(Path(t).glob("**/transcript.json")) if Path(t).is_dir() else [Path(t)]:
            if f.exists():
                out += [e["name"] for e in json.loads(f.read_text()) if e.get("type") == "tool"]
    return out


def _uq_slim(a, verdict=None):
    """What the confidence panel needs from an eqdisc assessment."""
    if not a:
        return None
    from eqdisc import insights
    return {"verdict": verdict or insights.verdict(a), "terms": a.get("terms"), "missing": a.get("missing_term_evidence"),
            "validation": a.get("validation"), "experiments": ((a.get("experiments") or {}).get("ranked") or [])[:3],
            "data_advice": a.get("data_advice"), "confidence": a.get("confidence")}


def _fresh(case):
    d = OUT / case
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


# ----------------------------------------------------------------------------- A. LAGEOS-1
def build_lageos():
    from eqdisc.oos import RE_E, T_E
    src = REPO / "runs/oos_lageos"
    res = _json(src / "results.json")
    if res is None:
        raise FileNotFoundError(src / "results.json")
    d = _fresh("lageos")
    # agent arm: data-only runs (neutral variable names, no context, no domain guidance, sandboxed code).
    # Headline: the stricter run in random units (mu and the Earth radius are not 1); the other is shown in details.
    agent = _json(REPO / "runs/oos_lageos_dataonly_units/agent_results.json")
    agent_alt = _json(REPO / "runs/oos_lageos_dataonly/agent_results.json")
    _copy(src / "lageos_forecast.mp4", d / "video.mp4")
    _copy(src / "lageos_errors.png", d / "errors.png")
    _thumb(d / "video.mp4", d / "thumb.jpg", at=6.0)
    z = np.load(src / "forecasts.npz")
    arrays = {"days": _f32(z["tt"] * T_E / 86400)}
    models = [k[2:] for k in z.files if k.startswith("P_")]
    for i, m in enumerate(models):
        arrays[f"err{i}"] = _f32(np.linalg.norm(z[f"P_{m}"][:, :3] - z["truth"][:, :3], axis=1) * RE_E / 1e3)
    fa = REPO / "runs/oos_lageos_dataonly_units/agent_forecast.npz"
    if fa.exists():
        S = np.load(fa)["S"]
        arrays["err_agent"] = _f32(np.linalg.norm(S[:, :3] - z["truth"][:len(S), :3], axis=1) * RE_E / 1e3)
    arrays["years_long"] = _f32(z["days_long"] / 365.25)
    for k in ("data", "Kepler", "Kepler + J2"):
        v = z[f"node_{k}"]
        arrays[f"node_{k}"] = _f32(v - v[0])
    # training data: one densified orbit every ~2 weeks of 2017, to show the orbit plane turning over the year
    if LAGEOS_CSV.exists():
        from eqdisc.oos import _densify, load_lageos
        t_, X_, dates_ = load_lageos(LAGEOS_CSV)
        starts = np.arange(0, 24 * 365 - 6, 24 * 14)
        arrays["train_orbits"] = _f32(np.stack([_densify(X_[i:i + 5], t_[1] - t_[0], 24) for i in starts]))
        arrays["train_full"] = _f32(X_[:24 * 365, :3])
        train_dates = [str(dates_.iloc[i].date()) for i in starts]
        h = np.cross(X_[:24 * 365, :3], X_[:24 * 365, 3:])
        arrays["train_h"] = _f32(h / np.linalg.norm(h, axis=1, keepdims=True))
    else:
        train_dates = []
    uq = _json(REPO / "runs/assess_lageos.json")
    if uq:
        uq["data_advice"] = (uq.get("data_advice") or []) + [
            "The ± ranges are statistical only: the agent noted a slow drift its law does not explain (other forces), "
            "so the true bulge value sits ~0.03% away, just outside the 90% range."]
    info = {"results": res, "agent": agent, "agent_alt": agent_alt, "models": models, "RE_km": RE_E / 1e3,
            "T_s": float(T_E), "train_dates": train_dates, "uq": uq,
            "rationale": _rationale(REPO / "runs/oos_lageos_dataonly_units/agent/transcript.json")}
    (d / "case.json").write_text(json.dumps(info, indent=1, default=str))
    np.savez_compressed(d / "arrays.npz", **arrays)


# ----------------------------------------------------------------------------- A2. synthetic orbit, big Earth bulge
ORBIT_CSV = Path(_os.environ.get("EQDISC_ORBIT_CSV", "/Users/danield/iterate-hackathon/orbit_discover/data/Challenge1.csv"))


def _orbit_elements(S):
    """S (..., 6) -> RAAN, argument of perigee (rad), mu = 1."""
    r, v = S[..., :3], S[..., 3:]
    h = np.cross(r, v)
    raan = np.arctan2(h[..., 0], -h[..., 1])
    rn = np.linalg.norm(r, axis=-1, keepdims=True)
    e = np.cross(v, h) - r / rn
    n = np.stack([-h[..., 1], h[..., 0], np.zeros_like(h[..., 0])], -1)
    cosw = np.sum(n * e, -1) / (np.linalg.norm(n, axis=-1) * np.linalg.norm(e, axis=-1) + 1e-300)
    w = np.arccos(np.clip(cosw, -1, 1))
    return raan, np.where(e[..., 2] < 0, 2 * np.pi - w, w)


def build_orbit():
    import pandas as pd
    from scipy.integrate import solve_ivp
    from eqdisc.oos import RE_E, T_E, _std_rhs
    src = REPO / "runs/oos_orbit"
    res = _json(src / "results.json")
    if res is None:
        raise FileNotFoundError(src / "results.json")
    d = _fresh("orbit")
    _copy(src / "orbit_forecast.mp4", d / "video.mp4")
    _thumb(d / "video.mp4", d / "thumb.jpg", at=6.0)
    z = np.load(src / "forecasts.npz")
    n_tr = int(z["n_tr"])
    df = pd.read_csv(ORBIT_CSV)
    X = np.hstack([df[["rx", "ry", "rz"]].values / RE_E, df[["vx", "vy", "vz"]].values / (RE_E / T_E)])
    Xn = X + res["noise"] * X.std(0) * np.random.default_rng(0).standard_normal(X.shape)   # exactly what was used
    t = np.arange(len(X)) * 30.0 / T_E
    hrs_all = t * T_E / 3600
    tt = z["tt"]
    hrs = tt * T_E / 3600 + hrs_all[n_tr - 1]
    arrays = {"t_train": _f32(t[:n_tr:5]), "U_train": _f32(Xn[:n_tr:5, :3]), "hrs": _f32(hrs)}
    models = [k[2:] for k in z.files if k.startswith("P_")]
    for i, m in enumerate(models):
        arrays[f"err{i}"] = _f32(np.linalg.norm(z[f"P_{m}"][:, :3] - z["truth"][:, :3], axis=1) * RE_E / 1e3)
    # orbital elements: measured (one-orbit moving average, all 6 days) vs forecasts over the unseen half
    win = int(round(18.4 / (t[1] - t[0])))
    ker = np.ones(win) / win
    sm = lambda a_: np.convolve(a_, ker, mode="valid")
    ra, w = _orbit_elements(Xn)
    arrays["el_hrs"] = _f32(sm(hrs_all))
    arrays["raan_data"] = _f32(sm(np.degrees(np.unwrap(ra))))
    arrays["argp_data"] = _f32(sm(np.degrees(np.unwrap(w))))
    for key, m in (("disc", "data-only agent"), ("kep", "Kepler")):
        if f"P_{m}" in z.files:
            r_, w_ = _orbit_elements(z[f"P_{m}"])
            for nm, ang, ref in (("raan", r_, arrays["raan_data"]), ("argp", w_, arrays["argp_data"])):
                ang = np.degrees(np.unwrap(ang))
                at = float(np.interp(hrs[0], arrays["el_hrs"], ref))     # join the measured curve (same branch)
                arrays[f"{nm}_{key}"] = _f32(ang + 360.0 * np.round((at - ang[0]) / 360.0))
    # same start, two laws, ~6 orbits: the bulge turns the orbit plane, round-Earth gravity does not
    # (uses the TRUE generator law, labelled as such: the data-only agent did not recover it)
    ag = res.get("agent") or {}
    from eqdisc.oos import ORBIT_TRUTH
    if True:
        tl = {"x": "vx", "y": "vy", "z": "vz",
              **{k: v.replace("r**", "sqrt(x**2+y**2+z**2)**") for k, v in ORBIT_TRUTH.items()}}
        f_ag = _std_rhs(["x", "y", "z", "vx", "vy", "vz"], tl)
        f_k = lambda y: np.r_[y[3:], -y[:3] / np.linalg.norm(y[:3]) ** 3]
        x0 = z["P_Kepler + J2"][0]
        ts = np.linspace(0, 110, 3000)
        run = lambda f: solve_ivp(lambda s_, y: f(y), (0, ts[-1]), x0, t_eval=ts, rtol=1e-10, atol=1e-12).y.T
        arrays["kj_t"], arrays["kj_disc"], arrays["kj_kep"] = _f32(ts), _f32(run(f_ag)[:, :3]), _f32(run(f_k)[:, :3])
    info = {"results": res, "models": models, "agent": ag, "uq": _json(REPO / "runs/assess_orbit.json")}
    (d / "case.json").write_text(json.dumps(info, indent=1, default=str))
    np.savez_compressed(d / "arrays.npz", **arrays)


# ----------------------------------------------------------------------------- B. Kuramoto-Sivashinsky (blinded)
def build_ks():
    from eqdisc.oos import rel_err_t
    src = REPO / "runs/oos_ks_dataonly"
    res = _json(src / "results.json")
    if res is None:
        raise FileNotFoundError(src / "results.json")
    d = _fresh("ks")
    disc = _json(src / "discover/discovery.json") or {}
    _copy(src / "ks_spacetime.mp4", d / "video.mp4") or _copy(src / "ks_oos.mp4", d / "video.mp4")
    _copy(src / "ks_spacetime.png", d / "spacetime.png")
    _copy(src / "discover/report.html", d / "report.html")
    _thumb(d / "video.mp4", d / "thumb.jpg", at=9.0)
    z = np.load(src / "rollouts.npz")
    labels = [str(s) for s in z["labels"]]
    lam = float(res["lyapunov_exponent"])
    arrays = {"t_lyap": _f32((z["t"] - z["t"][0]) * lam)}
    for i in range(1, len(labels)):
        arrays[f"err{i}"] = _f32(np.minimum(rel_err_t(z[f"Y{i}"], z["Y0"]), 5.0))
    a = disc.get("assessment") or {}
    info = {"results": res, "labels": labels, "story": disc.get("story"), "verdict": disc.get("verdict"),
            "assessment": {k: a.get(k) for k in ("terms", "confidence", "noise_floor", "missing_term_evidence")},
            "uq": _uq_slim(a, disc.get("verdict")),
            "cost_usd": (res.get("agent") or {}).get("cost_usd") or disc.get("cost_usd"), "wall_s": disc.get("wall_s"),
            "tools": _tools(src / "discover")}
    (d / "case.json").write_text(json.dumps(info, indent=1, default=str))
    np.savez_compressed(d / "arrays.npz", **arrays)


# ----------------------------------------------------------------------------- C. Gray-Scott (The Well)
def build_gray_scott(regime="spirals", noise=0.05):
    from eqdisc import solvers
    from eqdisc.well_gs import REGIMES
    import os
    src = Path(os.environ.get("EQDISC_GS_OOS_DIR") or REPO / "runs/oos_gs_dataonly")   # data-only agent run
    d = _fresh("gray_scott")
    res = _json(src / "results.json")
    sweep = None   # the earlier noise sweep showed the agent the dataset name and a context line: not shown
    ds = REPO / f"datasets/well_gs_{regime}_n{noise:g}"
    truth = _json(ds / "hidden/truth.json")
    has_video = _copy(src / "gs_forecast.mp4", d / "video.mp4")
    arrays = {}
    if has_video:
        _thumb(d / "video.mp4", d / "thumb.jpg", at=2.0)
    else:
        # fallback hero: held-out trajectory vs the true PDE re-simulated from its first frame (B field, 31 frames)
        meta = _json(ds / "meta.json")
        te = np.load(ds / "hidden/test.npz")
        Ut, t = te["U"][0][:31], te["t"][:31]
        Y = solvers.integrate_pde_general(["A", "B"], truth["rhs"], solvers.pde_layout(meta), Ut[0], t, dt_sim=1.0)
        arrays.update(fb_t=_f32(t), fb_data=_f32(Ut[..., 1]), fb_model=_f32(Y[..., 1]))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        img = Ut[10, ..., 1].T
        plt.imsave(d / "thumb.jpg", img[(img.shape[0] - 72) // 2:(img.shape[0] + 72) // 2], cmap="magma", origin="lower")
    cache = REPO / "datasets/_well_cache"
    for r in REGIMES:
        f = cache / f"{r}_t50_60_tr0-1-2.npz"
        if f.exists():
            arrays[f"gal_{r}"] = _f32(np.load(f)["U"][0, 0, ::2, ::2, 1])
    info = {"results": res, "sweep": sweep, "regime": regime, "noise": noise, "truth": (truth or {}).get("rhs"),
            "params": (truth or {}).get("params"), "regimes": {k: list(v) for k, v in REGIMES.items()},
            "has_video": has_video, "tools": _tools(REPO / "runs/dataonly/gs_agent"),
            "uq": _json(REPO / "runs/assess_gray_scott.json"),
            "rationale": _rationale(REPO / "runs/dataonly/gs_agent/transcript.json"),
            "agent_cost": (_json(REPO / "runs/dataonly/gs_agent/result.json") or {}).get("cost_usd")}
    (d / "case.json").write_text(json.dumps(info, indent=1, default=str))
    np.savez_compressed(d / "arrays.npz", **arrays)


# ----------------------------------------------------------------------------- rehearsal data for the live tab
def build_rehearsal():
    """Precomputed results replayed by the live tab's scripted (no-API) mode, plus the example CSVs."""
    import pandas as pd
    d = _fresh("rehearsal")
    (DEMO / "examples").mkdir(exist_ok=True)
    # dynamics: the pendulum discovery (COLLECT MORE DATA) + its dataset + report
    ds = AR / "datasets/pendulum_n0.05red_dt1_s0"
    disc = json.loads((AR / "runs/live_pendulum/discovery.json").read_text())
    keep = {k: disc.get(k) for k in ("verdict", "final_model", "story", "insights", "assessment", "winner_branch",
                                     "cost_usd", "wall_s")}
    (d / "pendulum.json").write_text(json.dumps(keep, indent=1, default=str))
    (d / "pendulum_dataset").mkdir()
    shutil.copy(ds / "data.npz", d / "pendulum_dataset/data.npz")
    shutil.copy(ds / "meta.json", d / "pendulum_dataset/meta.json")
    _copy(AR / "runs/live_pendulum/report.html", d / "pendulum_report.html")
    z = np.load(ds / "data.npz")
    t, U = z["t"], z["U"]
    pd.concat([pd.DataFrame({"traj": j, "t": t, "theta": U[j, :, 0], "omega": U[j, :, 1]}) for j in range(U.shape[0])]
              ).to_csv(DEMO / "examples/pendulum.csv", index=False, float_format="%.6g")
    # static: the E. coli law (LLM-SR bactgrow); the example CSV lives in demo/examples (MIT, see ATTRIBUTION.txt)
    r = _json(REPO / "runs/sr_llmsr/llmsr_bactgrow.json") or {}
    (d / "ecoli.json").write_text(json.dumps({"expr": r.get("expr")}, indent=1))



# ----------------------------------------------------------------------------- hidden oscillator (secret_test_1)
HIDDEN_DATA = REPO / "datasets/secret_test_1_ingested"
HIDDEN_RUN = sorted(REPO.glob("runs/discover_secret_test_1_ingested_*"))
HIDDEN_BARE = REPO / "runs/secret1_bare_v2/secret_test_1_ingested/result.json"
HIDDEN_COLORS = {"Reality": "#9ca3af", "Discovered law": "#16a34a", "Claude alone": "#ea580c", "True law": "#111827"}


def hidden_truth_rhs(_, X):
    """The generating law (never shown to the agent): polar dynamics about a hidden centre, seen through a tilt+stretch."""
    c, th = np.array([-1.6947, 2.9225]), 1.1592
    A = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) @ np.diag([1.8091, 0.7989])
    p = np.linalg.solve(A, X[:2] - c)
    r, ang = np.hypot(*p), np.arctan2(p[1], p[0])
    rd, thd = r * (0.7029 - 0.8495 * r), 1.9022 + 0.5183 * r
    pd = np.array([rd * np.cos(ang) - r * thd * np.sin(ang), rd * np.sin(ang) + r * thd * np.cos(ang)])
    return np.concatenate([A @ pd, [-0.9497 * X[2] + 0.3493 * r]])


def _time_avg(t, e):
    """Average over time (frames are denser in the slow-motion part, so a plain mean would over-weight it)."""
    return float(np.trapezoid(e, t) / (t[-1] - t[0]))


def hidden_video(path, tf, t, data, ref, sims, train, fps=20, tail=1.6, slow_until=8.0):
    """Phase plane of the held-out run (comet-tail lines), u2 over time, and distance from (smoothed) reality.
    tf: frame times (dense early = slow motion); t, data: the noisy samples; ref: smoothed reality at tf."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.animation as anim
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mt
    import imageio_ffmpeg
    from eqdisc.oos import _video_style
    _video_style(plt)
    plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    fig = plt.figure(figsize=(16, 9), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], left=0.03, right=0.98, bottom=0.1, top=0.86, wspace=0.12)
    ax, ae = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    allxy = np.concatenate([train[..., :2].reshape(-1, 2), data[:, :2]])
    pad = 0.08 * np.ptp(allxy, axis=0)
    ax.set_xlim(allxy[:, 0].min() - pad[0], allxy[:, 0].max() + pad[0])
    ax.set_ylim(allxy[:, 1].min() - pad[1], allxy[:, 1].max() + pad[1])
    for tr in train:                                     # training runs: faint context
        ax.plot(tr[:, 0], tr[:, 1], color="#eceef1", lw=1.0, zorder=0)
    ax.set_title("A run it never saw", pad=10)
    ax.set_axis_off()                                    # unnamed variables: no axes, ticks or frame
    scale = data.std(0)
    err = {k: np.sqrt(np.mean(((S - ref) / scale) ** 2, axis=1)) for k, S in sims.items()}
    top = max(float(np.nanmax(e)) for e in err.values())
    ae.set_xlim(tf[0], tf[-1]); ae.set_yscale("log"); ae.set_ylim(2e-3, max(0.5, 1.6 * top))
    ae.yaxis.set_major_locator(mt.FixedLocator([0.003, 0.01, 0.03, 0.1, 0.3, 1.0]))
    ae.yaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{100 * v:g}%"))
    ae.yaxis.set_minor_formatter(mt.NullFormatter())
    ae.set_title("Distance from reality", pad=8); ae.set_xlabel("time")
    ae.set_ylabel("error (relative to each variable's spread)")
    ae.axvspan(tf[0], slow_until, color="#fef3c7", alpha=0.6, zorder=0, lw=0)
    rd, = ax.plot([], [], "o", ms=4.5, color=HIDDEN_COLORS["Reality"], alpha=0.8, zorder=1, label="Reality (noisy)")
    order = ["Discovered law", "Claude alone", "True law"]          # True law drawn last: thin dashed, on top
    label = {"Discovered law": "Discovered law (eqdisc)", "Claude alone": "Claude alone (no tools)", "True law": "True law (hidden)"}
    sty = {"True law": dict(ls=(0, (3, 2)), lw=2.2), "Discovered law": dict(lw=5.0), "Claude alone": dict(lw=5.0)}
    zo = {"Claude alone": 4, "Discovered law": 5, "True law": 7}
    faint, bold, head, eline = {}, {}, {}, {}
    for k in order:
        col = HIDDEN_COLORS[k]
        faint[k], = ax.plot([], [], color=col, lw=1.3, alpha=0.3 if k != "True law" else 0.0, zorder=zo[k])
        bold[k], = ax.plot([], [], color=col, solid_capstyle="round", zorder=zo[k] + 1, label=label[k], **sty[k])
        head[k], = ax.plot([], [], "o", ms=14 if k != "True law" else 7, color=col, mec="white", mew=2.0, zorder=zo[k] + 2)
        eline[k], = ae.plot([], [], color=col, lw=3.2 if k != "True law" else 2.0, ls=sty[k].get("ls", "-"), zorder=zo[k],
                            label=f"{label[k].split(' (')[0]}   avg {100 * _time_avg(tf, err[k]):.1f}%")
    ax.legend(loc="upper left", framealpha=0.95, handlelength=2.4)
    fig.text(0.06, 0.955, "Hidden Oscillator: forecasting a new run from one noisy observation", ha="left", va="center")
    clock = fig.text(0.98, 0.955, "", ha="right", va="center")
    ae.legend(loc="upper right", framealpha=0.95, handlelength=2.2, title="average over the run", title_fontsize=15,
              fontsize=15)
    dt_tail = tail

    def upd(i):
        now = tf[i]
        a = np.searchsorted(tf, now - dt_tail)
        nd = np.searchsorted(t, now + 1e-9)
        rd.set_data(data[:nd, 0], data[:nd, 1])
        for k, S in sims.items():
            faint[k].set_data(S[:i + 1, 0], S[:i + 1, 1]); bold[k].set_data(S[a:i + 1, 0], S[a:i + 1, 1])
            head[k].set_data([S[i, 0]], [S[i, 1]])
            eline[k].set_data(tf[:i + 1], np.maximum(err[k][:i + 1], 2.2e-3))
        clock.set_text(("slow motion   " if now < slow_until else "") + f"t = {now:5.2f}")
        return []
    a = anim.FuncAnimation(fig, upd, frames=len(tf), interval=1000 / fps)
    a.save(path, writer=anim.FFMpegWriter(fps=fps, bitrate=5000))
    upd(int(np.searchsorted(tf, 1.2)))
    fig.savefig(Path(path).with_name("still.png"), dpi=60)
    plt.close(fig)
    return err


def _hidden_thumb(d, t, data, sims, train):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for tr in train:
        ax.plot(tr[:, 0], tr[:, 1], color="#e5e7eb", lw=0.8)
    ax.plot(data[:, 0], data[:, 1], ".", ms=2, color=HIDDEN_COLORS["Reality"])
    for k in ("Claude alone", "Discovered law"):
        ax.plot(sims[k][:, 0], sims[k][:, 1], color=HIDDEN_COLORS[k], lw=2.4)
    ax.set_axis_off()
    fig.savefig(d / "thumb.jpg", dpi=110, facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    im = Image.open(d / "thumb.jpg").convert("RGB")
    w, h = im.size
    W, H = max(w, int(round(h * 16 / 9))), max(h, int(round(w * 9 / 16)))
    canvas = Image.new("RGB", (W, H), "white")
    canvas.paste(im, ((W - w) // 2, (H - h) // 2))
    canvas.save(d / "thumb.jpg", quality=90)


def build_hidden_oscillator():
    """secret_test_1: 3 variables, 4 noisy runs; trained on runs 0-2, forecast run 3 from its first (noisy) sample."""
    from scipy.integrate import solve_ivp
    from eqdisc.evaluate import load
    from eqdisc.solvers import integrate_ode
    if not HIDDEN_RUN:
        raise FileNotFoundError("no runs/discover_secret_test_1_ingested_* run")
    run = HIDDEN_RUN[-1]
    disc = _json(run / "discovery.json")
    bare = _json(HIDDEN_BARE)
    meta, data = load(HIDDEN_DATA)
    U, t = data["U"], data["t"]
    held, train = U[-1], U[:-1]
    from scipy.signal import savgol_filter
    x0 = held[0]                                         # one noisy observation: the honest starting point
    slow = 8.0                                           # the models differ while the run settles: show it slowly
    tf = np.unique(np.concatenate([np.arange(0, slow, 0.02), np.arange(slow, t[-1] + 1e-9, 0.1), [t[-1]]]))
    sims = {"True law": solve_ivp(hidden_truth_rhs, (tf[0], tf[-1]), x0, t_eval=tf, rtol=1e-9, atol=1e-9).y.T,
            "Discovered law": integrate_ode(meta["variables"], disc["final_model"], x0, tf, max_seconds=60),
            "Claude alone": integrate_ode(meta["variables"], bare["submitted"]["rhs"], x0, tf, max_seconds=60)}
    sm = savgol_filter(held, 21, 3, axis=0)              # reality without the measurement noise, for the distance
    ref = np.stack([np.interp(tf, t, sm[:, j]) for j in range(held.shape[1])], 1)
    d = _fresh("hidden_oscillator")
    err = hidden_video(d / "video.mp4", tf, t, held, ref, sims, train, slow_until=slow)
    _hidden_thumb(d, tf, held, sims, train)
    t = tf
    tools = {}
    for f in sorted(run.glob("*/transcript.json")):
        for e in json.loads(f.read_text()):
            if e.get("type") == "tool" and e.get("name") not in ("submit",):
                tools[e["name"]] = tools.get(e["name"], 0) + 1
    import re
    fm = disc.get("final_model") or {}
    polar = {}
    try:                                                 # read the polar-form constants off the submitted model
        polar["growth"] = float(re.search(r"([\d.]+)\*\(1 - sqrt", fm["u1"]).group(1))
        w = re.search(r"([\d.]+)\*sqrt\(.*?\) \+ ([\d.]+)\)", fm["u1"])
        polar["omega1"], polar["omega0"] = float(w.group(1)), float(w.group(2))
        z = re.search(r"-([\d.]+)\*u2 \+ ([\d.]+)\*sqrt", fm["u2"])
        polar["decay"], polar["drive"] = float(z.group(1)), float(z.group(2))
    except Exception:  # noqa: BLE001
        polar = {}
    a = disc.get("assessment") or {}
    sq = next((t_["term"] for t_ in a.get("terms") or [] if t_["term"].startswith("sqrt(")), None)
    info = {"verdict": disc.get("verdict"), "story": disc.get("story"), "final_model": disc.get("final_model"),
            "winner_branch": disc.get("winner_branch"), "cost_usd": disc.get("cost_usd"), "wall_s": disc.get("wall_s"),
            "uq": _uq_slim(a, disc.get("verdict")), "radius_term": sq, "tools": tools, "polar": polar,
            "bare": {"rhs": bare["submitted"]["rhs"], "rationale": bare["submitted"].get("rationale"),
                     "cost_usd": bare.get("cost_usd"), "n_tool_calls": bare.get("n_tool_calls")},
            "errors_mean": {k: _time_avg(t, e) for k, e in err.items()},
            "errors_at": {k: {str(tt): float(e[np.searchsorted(t, tt)]) for tt in (2, 5, 10, 30)} for k, e in err.items()}}
    (d / "case.json").write_text(json.dumps(info, indent=1, default=str))
    np.savez_compressed(d / "arrays.npz", t=_f32(t), held=_f32(held), train=_f32(train),
                        **{f"sim_{i}": _f32(S) for i, S in enumerate(sims.values())},
                        **{f"err_{i}": _f32(e) for i, e in enumerate(err.values())}, names=np.array(list(sims)))

CASES = {"lageos": build_lageos, "orbit": build_orbit, "ks": build_ks, "gray_scott": build_gray_scott, "rehearsal": build_rehearsal,
         "hidden_oscillator": build_hidden_oscillator}


# ----------------------------------------------------------------------------- clean card thumbnails (no text)
def clean_thumbs():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(fig, case):
        from PIL import Image
        path = OUT / case / "thumb.jpg"
        fig.savefig(path, dpi=110, facecolor="white", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        im = Image.open(path).convert("RGB")             # pad to a uniform 16:9 card image
        w, h = im.size
        W, H = max(w, int(round(h * 16 / 9))), max(h, int(round(w * 9 / 16)))
        canvas = Image.new("RGB", (W, H), "white")
        canvas.paste(im, ((W - w) // 2, (H - h) // 2))
        canvas.resize((560, 315)).save(path, quality=90)

    def earth(ax, zoom=1.9):
        u_, v_ = np.mgrid[0:2 * np.pi:60j, 0:np.pi:30j]
        ax.plot_surface(np.cos(u_) * np.sin(v_), np.sin(u_) * np.sin(v_), np.cos(v_), color="#2f6db5", alpha=.9,
                        linewidth=0, shade=True)
        ax.set_box_aspect((1, 1, 1), zoom=zoom)
        ax.set_axis_off()

    z = np.load(OUT / "lageos" / "arrays.npz")
    if "train_orbits" in z.files:
        fig = plt.figure(figsize=(5.6, 3.15))
        ax = fig.add_axes([0, 0, 1, 1], projection="3d")
        earth(ax, zoom=1.25)
        O = z["train_orbits"]
        for i, o in enumerate(O):
            ax.plot(o[:, 0], o[:, 1], o[:, 2], color=plt.cm.cool(i / len(O)), lw=0.8, alpha=.85)
        lim = 1.6
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim))
        ax.view_init(18, 35)
        save(fig, "lageos")
    z = np.load(OUT / "orbit" / "arrays.npz")
    fig = plt.figure(figsize=(5.6, 3.15))
    ax = fig.add_axes([0, 0, 1, 1], projection="3d")
    earth(ax)
    U = z["U_train"]
    ax.plot(U[:, 0], U[:, 1], U[:, 2], color="#eb6834", lw=0.5, alpha=.8)
    lim = 2.4
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim))
    ax.view_init(20, 40)
    save(fig, "orbit")
    f = np.load(REPO / "runs/oos_ks_dataonly/rollouts.npz", allow_pickle=True)
    fig, ax = plt.subplots(figsize=(5.6, 3.15))
    Y = f["Y0"]
    ax.imshow(Y.T, aspect="auto", origin="lower", cmap="RdBu_r", vmin=-np.abs(Y).max(), vmax=np.abs(Y).max())
    ax.set_axis_off()
    save(fig, "ks")
    f = np.load(REPO / "runs/oos_gs_dataonly/rollouts.npz", allow_pickle=True)
    fig, ax = plt.subplots(figsize=(5.6, 3.15))
    B = f["Y0"][-1, ..., 1]
    ax.imshow(np.tile(B.T, (1, 2))[:, : int(B.shape[0] * 1.78)], origin="lower", cmap="magma", aspect="auto")
    ax.set_axis_off()
    save(fig, "gray_scott")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name in (sys.argv[1:] or list(CASES)):
        t0 = time.time()
        try:
            CASES[name]()
            print(f"[{name}] ok ({time.time() - t0:.1f}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] FAILED: {type(e).__name__}: {e}", flush=True)
    try:
        clean_thumbs()
    except Exception as e:  # noqa: BLE001
        print(f"[thumbs] FAILED: {e}")
    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"showcase total: {total / 1e6:.1f} MB")
