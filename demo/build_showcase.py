"""Collect everything the demo Showcase needs into demo/showcase/<case>/ (json + trimmed float32 arrays +
precomputed rollouts), so the app never reads paths outside the repo and renders instantly.

    PYTHONPATH=. python demo/build_showcase.py            # all cases
    PYTHONPATH=. python demo/build_showcase.py gray_scott                              # refresh one case

Re-run `gray_scott` once runs/well_gs/results.json exists: it then also rolls out the agent's discovered PDE.
No Claude API calls are made here.
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

DEMO = Path(__file__).resolve().parent
REPO = DEMO.parent
sys.path.insert(0, str(REPO))

OUT = DEMO / "showcase"
import os
# Maintainer-only: where the original run outputs live (the app itself only reads demo/showcase/).
AR = Path(os.environ.get("EQDISC_RUNS_ROOT", "../iterate-hackathon/autoresearch"))
SCRATCH_SRB = Path(os.environ.get("EQDISC_LLMSR_REPO", "external/LLM-SR")) / "data" / "bactgrow"   # clone of deep-symbolic-mathematics/LLM-SR


def _f32(a):
    return np.asarray(a, dtype=np.float32)


def _dedupe_insights(ins):
    seen, out = set(), []
    for i in ins or []:
        key = (i.get("step"), (i.get("finding") or "")[:80])
        if key in seen:
            continue
        seen.add(key)
        out.append(i)
    return out


def _slim_discovery(d):
    """Keep only what the app renders."""
    a = d.get("assessment") or {}
    keep_a = {k: a.get(k) for k in ("terms", "missing_term_evidence", "model_ambiguity", "noise_floor", "validation",
                                     "predictability", "experiments", "data_advice", "confidence", "questions_for_human",
                                     "statistics_basis")}
    return {"verdict": d.get("verdict"), "final_model": d.get("final_model"), "story": d.get("story"),
            "insights": _dedupe_insights(d.get("insights")), "assessment": keep_a,
            "branches": {k: {"model": b.get("model"), "rationale": b.get("rationale"), "cost_usd": b.get("cost_usd")}
                         for k, b in (d.get("branches") or {}).items()},
            "winner_branch": d.get("winner_branch"),
            "tournament": {k: (d.get("tournament") or {}).get(k) for k in ("winner", "verdict", "ranking")},
            "adversary": {k: (d.get("adversary") or {}).get(k) for k in ("rationale", "cost_usd")},
            "cost_usd": d.get("cost_usd"), "wall_s": d.get("wall_s")}


def _write(case, info, arrays=None, report=None):
    d = OUT / case
    d.mkdir(parents=True, exist_ok=True)
    if report and Path(report).exists():
        shutil.copy(report, d / "report.html")
    (d / "case.json").write_text(json.dumps(info, indent=1, default=str))
    if arrays:
        np.savez_compressed(d / "arrays.npz", **arrays)
    print(f"  wrote {d} ({sum(f.stat().st_size for f in d.rglob('*') if f.is_file()) / 1e6:.1f} MB)", flush=True)


# ----------------------------------------------------------------------------- orbit
def _orbit_elements(S):
    """S (..., 6) -> RAAN, argument of perigee, inclination (rad, unwrapped along axis 0 later)."""
    r, v = S[..., :3], S[..., 3:]
    h = np.cross(r, v)
    raan = np.arctan2(h[..., 0], -h[..., 1])
    inc = np.arccos(np.clip(h[..., 2] / np.linalg.norm(h, axis=-1), -1, 1))
    rn = np.linalg.norm(r, axis=-1, keepdims=True)
    e = np.cross(v, h) - r / rn                    # mu = 1
    n = np.stack([-h[..., 1], h[..., 0], np.zeros_like(h[..., 0])], -1)
    cosw = np.sum(n * e, -1) / (np.linalg.norm(n, axis=-1) * np.linalg.norm(e, axis=-1) + 1e-300)
    w = np.arccos(np.clip(cosw, -1, 1))
    w = np.where(e[..., 2] < 0, 2 * np.pi - w, w)
    return raan, w, inc, np.linalg.norm(e, axis=-1)


def build_orbit():
    from scipy.integrate import solve_ivp
    from scipy.optimize import least_squares
    from eqdisc.solvers import make_ode_rhs
    ds = AR / "datasets/orbit_challenge1_n0.01"
    disc = json.loads((AR / "runs/live_orbit/discovery.json").read_text())
    truth = json.loads((ds / "hidden/truth.json").read_text())
    meta = json.loads((ds / "meta.json").read_text())
    z = np.load(ds / "data.npz")
    t, U = z["t"], z["U"][0]
    names = meta["variables"]
    f_disc = make_ode_rhs(names, disc["final_model"])
    f_kep = make_ode_rhs(names, {"x": "vx", "y": "vy", "z": "vz", "vx": "-x/(x**2+y**2+z**2)**1.5",
                                 "vy": "-y/(x**2+y**2+z**2)**1.5", "vz": "-z/(x**2+y**2+z**2)**1.5"})

    def run(f, x0, tt):
        s = solve_ivp(lambda tau, y: np.asarray(f(y, tau), float), (tt[0], tt[-1]), x0, t_eval=tt, rtol=1e-10,
                      atol=1e-12, method="DOP853")
        return s.y.T

    # denoise the initial state by shooting the discovered model over the first ~2 orbits
    k = 250
    res = least_squares(lambda x0: (run(f_disc, x0, t[:k]) - U[:k]).ravel(), U[0], method="lm", max_nfev=200)
    x0 = res.x
    T_long = float(t[-1] - t[0])
    tt = np.linspace(t[0], t[0] + T_long, 3000)
    S_disc = run(f_disc, x0, tt)
    S_kep = run(f_kep, x0, tt)
    el = {}
    for nm, S in (("data", U), ("disc", S_disc), ("kep", S_kep)):
        raan, w, inc, ecc = _orbit_elements(S)
        el[nm] = (np.degrees(np.unwrap(raan)), np.degrees(np.unwrap(w)), np.degrees(inc), ecc)
    # data elements are noisy: one-orbit moving average (period ~18.4 time units)
    win = max(3, int(round(18.4 / float(t[1] - t[0]))))
    ker = np.ones(win) / win
    sm = lambda a: np.convolve(a, ker, mode="valid")
    td_s = sm(t)
    a = el
    info = {"title": "Satellite orbit with J2", "kind": "dynamics", "discovery": _slim_discovery(disc),
            "truth": truth["rhs"], "names": names, "units": meta.get("units"), "source": meta.get("source"),
            "latex": [r"\dot{\mathbf r} = \mathbf v",
                      r"\dot{v}_{q} = -\frac{q}{r^{3}}\left[1 + \frac{0.75}{r^{2}}\left(1 - \frac{5z^{2}}{r^{2}}\right)\right], \quad q \in \{x, y\}",
                      r"\dot{v}_{z} = -\frac{z}{r^{3}}\left[1 + \frac{0.75}{r^{2}}\left(3 - \frac{5z^{2}}{r^{2}}\right)\right]"],
            "truth_latex": [r"\dot{v}_{q} = -\frac{q}{r^{3}}\left[1 + \frac{\tfrac32 J_2}{r^{2}}\left(1 - \frac{5z^{2}}{r^{2}}\right)\right]",
                            r"\dot{v}_{z} = -\frac{z}{r^{3}}\left[1 + \frac{\tfrac32 J_2}{r^{2}}\left(3 - \frac{5z^{2}}{r^{2}}\right)\right]",
                            r"J_2 = 0.5 \;\Rightarrow\; \tfrac32 J_2 = 0.75"],
            "raan_rate_deg_per_unit": {nm: float(np.polyfit(tt if nm != "data" else t, a[nm][0], 1)[0]) for nm in a},
            "argp_rate_deg_per_unit": {nm: float(np.polyfit(tt if nm != "data" else t, a[nm][1], 1)[0]) for nm in a},
            "time_unit_s": 806.8}
    arrays = {"t": _f32(t), "U": _f32(U[:, :3]), "tt": _f32(tt), "S_disc": _f32(S_disc[:, :3]), "S_kep": _f32(S_kep[:, :3]),
              "td_s": _f32(td_s), "raan_data": _f32(sm(a["data"][0])), "argp_data": _f32(sm(a["data"][1])),
              "raan_disc": _f32(a["disc"][0]), "argp_disc": _f32(a["disc"][1]),
              "raan_kep": _f32(a["kep"][0]), "argp_kep": _f32(a["kep"][1]), "x0": x0}
    _write("orbit", info, arrays, report=AR / "runs/live_orbit/report.html")


# ----------------------------------------------------------------------------- real KS
def build_ks():
    from eqdisc import solvers
    ds = AR / "datasets/real_ks"
    disc = json.loads((AR / "runs/live_real_ks/discovery.json").read_text())
    meta = json.loads((ds / "meta.json").read_text())
    z = np.load(ds / "data.npz")
    t, x, U = z["t"], z["x"], z["U"][0, :, :, :]
    t0 = time.time()
    Y = solvers.integrate_pde_general(meta["variables"], disc["final_model"], solvers.pde_layout(meta), U[0], t)
    print(f"  KS rollout {time.time() - t0:.1f}s, finite={np.isfinite(Y).all()}", flush=True)
    err = np.sqrt(np.mean((Y[..., 0] - U[..., 0]) ** 2, axis=1)) / np.std(U[..., 0])
    info = {"title": "Real Kuramoto–Sivashinsky data", "kind": "dynamics", "discovery": _slim_discovery(disc),
            "truth": {"u": "-u*u_x - u_xx - u_xxxx"}, "truth_note": "textbook KS (PySINDy tutorial data)",
            "names": meta["variables"], "pde": True, "source": "KS_data.mat (PySINDy tutorial), 1024 x 251 grid"}
    s = 2
    _write("ks", info, {"t": _f32(t), "x": _f32(x[::s]), "U": _f32(U[:, ::s, 0]), "Y": _f32(Y[:, ::s, 0]),
                        "rel_err": _f32(err)}, report=AR / "runs/live_real_ks/report.html")


# ----------------------------------------------------------------------------- Gray-Scott (The Well)
def build_gray_scott(regime="spirals", noise=0.05):
    from eqdisc import solvers
    from eqdisc.well_gs import REGIMES
    ds = REPO / f"datasets/well_gs_{regime}_n{noise:g}"
    meta = json.loads((ds / "meta.json").read_text())
    truth = json.loads((ds / "hidden/truth.json").read_text())
    te = np.load(ds / "hidden/test.npz")
    Ut, t = te["U"][0], te["t"]
    nf = 31
    import os
    res_path = Path(os.environ.get("EQDISC_GS_RESULTS") or REPO / "runs/well_gs/results.json")
    results = json.loads(res_path.read_text()) if res_path.exists() else None
    agent_rhs = None
    if results:
        agent_rhs = ((results.get(f"{regime}|{noise}") or results.get(f"{regime}|{noise:g}") or {}).get("agent") or {}).get("rhs")
    model_rhs = agent_rhs or truth["rhs"]
    t0 = time.time()
    Y = solvers.integrate_pde_general(["A", "B"], model_rhs, solvers.pde_layout(meta), Ut[0], t[:nf], dt_sim=1.0)
    print(f"  Gray-Scott rollout ({'agent' if agent_rhs else 'truth'} model) {time.time() - t0:.1f}s, "
          f"finite={np.isfinite(Y).all()}", flush=True)

    def vrmse(a, b):
        return float(np.mean([np.sqrt(np.mean((a[..., f] - b[..., f]) ** 2) / (np.var(b[..., f]) + 1e-12)) for f in range(2)]))
    vr = [vrmse(Y[i], Ut[i]) if np.all(np.isfinite(Y[i])) else None for i in range(nf)]
    gallery = {}
    cache = REPO / "datasets/_well_cache"
    for r in REGIMES:
        f = cache / f"{r}_t50_60_tr0-1-2.npz"
        if f.exists():
            gallery[r] = _f32(np.load(f)["U"][0, 0, :, :, 1])
    info = {"title": "Gray–Scott reaction–diffusion (The Well)", "kind": "dynamics", "regime": regime, "noise": noise,
            "truth": truth["rhs"], "params": truth["params"], "regimes": {k: list(v) for k, v in REGIMES.items()},
            "model_source": "agent" if agent_rhs else "truth", "model_rhs": model_rhs, "results": results,
            "vrmse_per_frame": vr, "dt": float(t[1] - t[0]), "names": ["A", "B"], "pde": True}
    arrays = {"t": _f32(t[:nf]), "data": _f32(Ut[:nf]), "model": _f32(Y), **{f"gal_{k}": v for k, v in gallery.items()}}
    _write("gray_scott", info, arrays)


# ----------------------------------------------------------------------------- pendulum
def build_pendulum():
    from scipy.integrate import solve_ivp
    ds = AR / "datasets/pendulum_n0.05red_dt1_s0"
    disc = json.loads((AR / "runs/live_pendulum/discovery.json").read_text())
    truth = json.loads((ds / "hidden/truth.json").read_text())
    z = np.load(ds / "data.npz")
    t, U = z["t"], z["U"]
    terms = disc["assessment"]["terms"]
    ex = disc["assessment"]["experiments"]["ranked"]
    rng = np.random.default_rng(0)
    # coefficient samples from the bootstrap 90% intervals (normal approx: halfwidth / 1.645)
    mu = np.array([tm["coef"] for tm in terms])
    sd = np.array([(tm["ci90"][1] - tm["ci90"][0]) / 2 / 1.645 for tm in terms])
    tt = np.linspace(0, 10, 501)

    def rhs_from(c):
        def f(_, y):
            th, om = y
            d = {"theta": 0.0, "omega": 0.0}
            for tm, ci in zip(terms, c):
                val = {"omega": om, "sin(theta)": np.sin(th), "theta": th}[tm["term"]]
                d[tm["var"]] += ci * val
            return [d["theta"], d["omega"]]
        return f
    fans = []
    for k in range(min(3, len(ex))):
        ic = ex[k]["initial_condition"]
        traj = []
        for j in range(40):
            c = mu + sd * rng.standard_normal(len(mu))
            s = solve_ivp(rhs_from(c), (0, tt[-1]), ic, t_eval=tt, rtol=1e-9, atol=1e-9)
            traj.append(s.y.T)
        fans.append(np.stack(traj))
    branch_models = {k: b["model"] for k, b in disc["branches"].items()}
    info = {"title": "Noisy pendulum → collect more data here", "kind": "dynamics", "discovery": _slim_discovery(disc),
            "truth": truth["rhs"], "names": ["theta", "omega"], "noise_note": "5% red (time-correlated) noise",
            "branch_models": branch_models, "fan_terms": [{"var": tm["var"], "term": tm["term"], "coef": tm["coef"],
                                                           "sd": float(s_)} for tm, s_ in zip(terms, sd)]}
    _write("pendulum", info, {"t": _f32(t), "U": _f32(U), "tt": _f32(tt), "fans": _f32(np.stack(fans)),
                              "ics": _f32([e["initial_condition"] for e in ex])},
           report=AR / "runs/live_pendulum/report.html")
    # a copy of the dataset (tiny) so rehearsal mode can draw eqdisc.plots figures without outside paths
    dd = OUT / "pendulum" / "dataset"
    dd.mkdir(exist_ok=True)
    shutil.copy(ds / "data.npz", dd / "data.npz")
    shutil.copy(ds / "meta.json", dd / "meta.json")
    # bundled example CSV for the live tab (time, trajectory id, theta, omega)
    import pandas as pd
    rows = [pd.DataFrame({"traj": j, "t": t, "theta": U[j, :, 0], "omega": U[j, :, 1]}) for j in range(U.shape[0])]
    pd.concat(rows).to_csv(DEMO / "examples/pendulum.csv", index=False, float_format="%.6g")


# ----------------------------------------------------------------------------- fresh oscillators
def build_osc():
    """Batch B (runs/sr_variants_b): held-out problems, the main card. Batch A (runs/sr_variants): development
    problems, kept only as a summary for honesty."""
    import pandas as pd
    from eqdisc.sr import evaluate_expr
    variants, dev, arrays = {}, {}, {}
    for v in ("osc_v0", "osc_v1", "osc_v2", "osc_v3"):
        fa = REPO / f"runs/sr_variants/{v}.json"
        if fa.exists():
            r = json.loads(fa.read_text())
            dev[v] = {"truth": r["truth"], **{m: {"expr": r[m]["expr"], "ID": r[m]["ID"], "OOD": r[m]["OOD"]}
                                              for m in ("agent", "pysr", "sparse")}}
        f = REPO / f"runs/sr_variants_b/{v}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        dd = REPO / f"datasets/osc_variants_b/{v}"
        desc = (dd / "description.txt").read_text().strip() if (dd / "description.txt").exists() else ""
        variants[v] = {"truth": r["truth"], "description": desc,
                       **{m: {"expr": r[m]["expr"], "ID": r[m]["ID"], "OOD": r[m]["OOD"], "cost_usd": r[m].get("cost_usd")}
                          for m in ("agent", "pysr", "sparse")}}
        variants[v]["agent"]["verdict"] = r["agent"].get("verdict")
        variants[v]["agent"]["assessment"] = r["agent"].get("assessment")
        names = [c for c in pd.read_csv(dd / "train.csv", nrows=1).columns if c != "a"]
        variants[v]["names"] = names
        rng = np.random.default_rng(0)
        for split in ("train", "test_id", "test_ood"):
            df = pd.read_csv(dd / f"{split}.csv")
            idx = rng.choice(len(df), size=min(1500, len(df)), replace=False)
            X = df[names].values[idx]
            arrays[f"{v}_{split}_X"] = _f32(df[["t", "x", "v"]].values[idx] if "t" in df else
                                            np.column_stack([np.zeros(len(idx)), X]))
            arrays[f"{v}_{split}_y"] = _f32(df["a"].values[idx])
            if split != "train":
                for m in ("agent", "pysr", "sparse"):
                    arrays[f"{v}_{split}_{m}"] = _f32(evaluate_expr(r[m]["expr"], names, X))
    info = {"title": "Fresh (unpublished) oscillators — static symbolic regression", "kind": "static",
            "variants": variants, "dev_batch": dev, "featured": "osc_v3" if "osc_v3" in variants else next(iter(variants)),
            "names": ["t", "x", "v"], "target": "a"}
    _write("osc", info, arrays)


# ----------------------------------------------------------------------------- E. coli growth (LLM-SR)
def build_ecoli():
    import pandas as pd
    from eqdisc.sr import evaluate_expr
    r = json.loads((REPO / "runs/sr_llmsr/llmsr_bactgrow.json").read_text())
    d = OUT / "ecoli"
    d.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test_id", "test_ood"):
        src = SCRATCH_SRB / f"{split}.csv"
        if src.exists():
            shutil.copy(src, d / f"{split}.csv")
        elif not (d / f"{split}.csv").exists():
            raise FileNotFoundError(f"{src} missing and no copy in {d}")
    (d / "ATTRIBUTION.txt").write_text(
        "train.csv / test_id.csv / test_ood.csv: E. coli growth (bactgrow) problem from the LLM-SR repository,\n"
        "https://github.com/deep-symbolic-mathematics/LLM-SR (MIT License).\n"
        "Shojaee, Meidani, Gupta, Barati Farimani, Reddy. LLM-SR: Scientific Equation Discovery via Programming with\n"
        "Large Language Models. ICLR 2025 (arXiv:2404.18400).\n")
    names = ["b", "s", "temp", "pH"]
    df = pd.read_csv(d / "train.csv")
    pd.concat([pd.read_csv(d / "train.csv"), pd.read_csv(d / "test_id.csv")]).to_csv(
        DEMO / "examples/ecoli_growth.csv", index=False, float_format="%.6g")
    info = {"title": "E. coli growth (LLM-SR benchmark)", "kind": "static", "names": names, "target": "db",
            "expr": r["expr"], "candidates": r.get("candidates"), "cost_usd": r.get("cost_usd"), "wall_s": r.get("wall_s"),
            "test_id": r.get("test_id"), "test_ood": r.get("test_ood"),
            "latex": r"\frac{dB}{dt} = 0.5\,\frac{B\,S}{1+S}\cdot\frac{1}{1+\left(\frac{T-35}{3.76}\right)^{4}}"
                     r"\cdot e^{-|\mathrm{pH}-7|}\,\sin^{2}\!\left(\frac{\pi(\mathrm{pH}-2)}{10}\right)",
            "comparison": [{"method": "eqdisc agent (ours)", "ID": r.get("test_id"), "OOD": r.get("test_ood")},
                           {"method": "LLM-SR (Mixtral)", "ID": 0.0026, "OOD": 0.0037},
                           {"method": "LLM-SR (GPT-3.5)", "ID": 0.0214, "OOD": 0.0264},
                           {"method": "PySR", "ID": 0.0376, "OOD": 1.0141},
                           {"method": "uDSR", "ID": 0.3322, "OOD": 5.4584}]}
    preds = {}
    for split in ("test_id", "test_ood"):
        dfs = pd.read_csv(d / f"{split}.csv")
        preds[f"{split}_pred"] = _f32(evaluate_expr(r["expr"], names, dfs[names].values))
    (d / "case.json").write_text(json.dumps(info, indent=1, default=str))
    np.savez_compressed(d / "arrays.npz", **preds)
    print(f"  wrote {d}", flush=True)


CASES = {"orbit": build_orbit, "ks": build_ks, "gray_scott": build_gray_scott, "pendulum": build_pendulum,
         "osc": build_osc, "ecoli": build_ecoli}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    (DEMO / "examples").mkdir(exist_ok=True)
    for name in (sys.argv[1:] or list(CASES)):
        t0 = time.time()
        print(f"[{name}]", flush=True)
        CASES[name]()
        print(f"  done in {time.time() - t0:.1f}s", flush=True)
    total = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    print(f"showcase total: {total / 1e6:.1f} MB")
