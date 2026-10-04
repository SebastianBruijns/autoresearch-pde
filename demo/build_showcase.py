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


CASES = {"lageos": build_lageos, "orbit": build_orbit, "ks": build_ks, "gray_scott": build_gray_scott, "rehearsal": build_rehearsal}


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

    def earth(ax):
        u_, v_ = np.mgrid[0:2 * np.pi:60j, 0:np.pi:30j]
        ax.plot_surface(np.cos(u_) * np.sin(v_), np.sin(u_) * np.sin(v_), np.cos(v_), color="#2f6db5", alpha=.9,
                        linewidth=0, shade=True)
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()

    z = np.load(OUT / "lageos" / "arrays.npz")
    if "train_orbits" in z.files:
        fig = plt.figure(figsize=(5.6, 3.15))
        ax = fig.add_subplot(111, projection="3d")
        earth(ax)
        O = z["train_orbits"]
        for i, o in enumerate(O):
            ax.plot(o[:, 0], o[:, 1], o[:, 2], color=plt.cm.cool(i / len(O)), lw=0.8, alpha=.85)
        lim = 1.6
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim))
        ax.view_init(18, 35)
        save(fig, "lageos")
    z = np.load(OUT / "orbit" / "arrays.npz")
    fig = plt.figure(figsize=(5.6, 3.15))
    ax = fig.add_subplot(111, projection="3d")
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
