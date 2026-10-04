"""Precompute the artifacts of the demo's "evidence layer" page into demo/showcase/evidence/ (no API calls).

    export PYTHONPATH=$PWD
    .venv/bin/python demo/build_evidence.py               # everything (orbit takes minutes: it runs the full assessment)
    .venv/bin/python demo/build_evidence.py scoreboard    # refresh only the benchmark scoreboard (seconds)
    .venv/bin/python demo/build_evidence.py grid orbit    # any subset of: orbit, grid, scoreboard, thumb
    .venv/bin/python demo/build_evidence.py orbit!        # recompute the orbit models (else cached per model)

Parts
  orbit       Challenge1 orbit (first 3 days, 2-min samples of the 30 s data, 1% noise), as in
              eqdisc/tests/test_audit_slices.py. Models: the true law (Kepler + J2), the data-only agent's submitted
              polynomial (refit constants, from demo/showcase/orbit/case.json; blinded units), and a degree-3 SINDy
              polynomial (no-LLM proxy). For each: the slice audit, coefficients refitted on distance-from-centre
              terciles (same weak-form helpers as eqdisc/audit/slices.py), and the verdict
              (assess.assess -> insights.verdict).                              -> orbit.json
  grid        One representative corruption case per type (report split, burgers, seed 10): data audit + repair,
              then audit_model with the TRUE base equation (hidden/truth.json). Thumbnails as PNG. Plus the detection
              rates over all calibration splits from runs/calib/{dev,report,blind}.json.   -> grid.json, thumb_*.png
  scoreboard  runs/evidence_bench/outcomes.jsonl (written by the benchmark agent; read only).  -> scoreboard.json
"""
import json
import sys
import time
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

DEMO = Path(__file__).resolve().parent
REPO = DEMO.parent
sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore")

OUT = DEMO / "showcase" / "evidence"
ORBIT_CSV = REPO / "examples" / "data" / "Challenge1.csv"
INDEX = REPO / "datasets" / "corrupt" / "index.json"
CALIB = REPO / "runs" / "calib"
OUTCOMES = REPO / "runs" / "evidence_bench" / "outcomes.jsonl"

CORRUPTIONS = ["clean", "outliers", "gaps_random", "gaps_state", "forcing_time", "source_space", "traj_coeffs", "amp_term"]
EXPECTED = {"outliers": ["outliers"], "gaps_random": ["gaps"], "gaps_state": ["gaps_state_dependent"],
            "forcing_time": ["residual_time_only", "slice_time"], "source_space": ["residual_space_only", "slice_space"],
            "traj_coeffs": ["slice_trajectory"], "amp_term": ["slice_amplitude", "residual_amplitude"]}


def _write(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, default=float))


def _slim(f):
    """A Finding without its bulky details (keep the few numbers the page shows)."""
    d = f.get("details") or {}
    keep = {k: d[k] for k in ("nan_fraction", "variable", "amplitude_variable", "inconsistent_coefficients", "z",
                              "mean_edge_percentile", "n_flagged", "n_tested", "excess_ratio") if k in d}
    return {k: f.get(k) for k in ("id", "stage", "statistic", "threshold", "fired", "severity", "response", "scope",
                                  "message", "resolved", "repair")} | {"details": keep}


# ============================================================================= orbit
def orbit_data():
    """Exactly the construction of eqdisc/tests/test_audit_slices.py::test_orbit_polynomial_fails_amplitude...:
    first 3 days, every 4th 30 s sample, 1% noise (seed 0). Standard units (mu = Re = 1) and the agent's blinded
    units (lengths x0.53, times x1.7; oos.orbit_case's default), on the same noisy samples."""
    import pandas as pd
    from eqdisc.oos import RE_E, T_E
    df = pd.read_csv(ORBIT_CSV)
    X = np.hstack([df[["rx", "ry", "rz"]].values / RE_E, df[["vx", "vy", "vz"]].values / (RE_E / T_E)])
    t = np.arange(len(X)) * 30.0 / T_E
    Xn = X + 0.01 * X.std(0) * np.random.default_rng(0).standard_normal(X.shape)
    n = len(X) // 2
    std = ["x", "y", "z", "vx", "vy", "vz"]
    meta_s = {"kind": "ode", "variables": std, "dt": float(t[4] - t[0]), "allowed_symbols": std + ["t"]}
    data_s = {"U": Xn[None, :n:4], "t": t[:n:4]}
    Ls, Ts = 0.53, 1.7
    sc = np.array([Ls] * 3 + [Ls / Ts] * 3)
    bl = [f"u{i}" for i in range(1, 7)]
    meta_b = {"kind": "ode", "variables": bl, "dt": float(t[4] - t[0]) * Ts, "allowed_symbols": bl + ["t"]}
    data_b = {"U": (Xn[:n:4] * sc)[None], "t": t[:n:4] * Ts}
    return (meta_s, data_s), (meta_b, data_b), {"days": float(t[n] * T_E / 86400), "n_samples": int(data_s["U"].shape[1]),
                                                 "cadence_s": 120.0, "noise": 0.01}


def orbit_models():
    from eqdisc import toolbox as tb
    from eqdisc.oos import ORBIT_TRUTH
    (ms, ds), _, _ = orbit_data()
    truth = {"x": "vx", "y": "vy", "z": "vz",
             **{k: v.replace("r**", "sqrt(x**2+y**2+z**2)**") for k, v in ORBIT_TRUTH.items()}}
    case = json.loads((DEMO / "showcase" / "orbit" / "case.json").read_text())
    agent = (case.get("agent") or {}).get("refit_rhs")
    models = {"truth": {"label": "True law: gravity + bulge", "units": "std", "rhs": truth}}
    if agent:
        models["agent"] = {"label": "What the AI agent submitted: a polynomial", "units": "blind", "rhs": agent}
    models["sindy"] = {"label": "Degree-3 polynomial (SINDy, no LLM)", "units": "std",
                       "rhs": tb.run_sindy(ms, ds, poly_degree=3)["rhs"]}
    return models


BLIND = {"u1": "x", "u2": "y", "u3": "z", "u4": "vx", "u5": "vy", "u6": "vz"}


def term_label(term, var, units):
    """Readable term name: r for sqrt(x^2+y^2+z^2), blinded names mapped back to x..vz."""
    import sympy as sp
    from eqdisc.solvers import parse
    names = list(BLIND) if units == "blind" else ["x", "y", "z", "vx", "vy", "vz"]
    e = parse(term, names + ["t"])
    if units == "blind":
        e = e.xreplace({sp.Symbol(k): sp.Symbol(v) for k, v in BLIND.items()})
        var = BLIND.get(var, var)
    x, y, z, r = sp.symbols("x y z r", positive=True)
    e = sp.factor(e.xreplace({sp.Symbol("x"): x, sp.Symbol("y"): y, sp.Symbol("z"): z}))
    e = sp.simplify(e.subs(x ** 2 + y ** 2 + z ** 2, r ** 2).subs(sp.sqrt(r ** 2), r))
    s = str(e).replace("**", "^").replace("*", "·")
    for a, b in (("vx", "v_x"), ("vy", "v_y"), ("vz", "v_z")):
        s = s.replace(a, b)
    acc = {"vx": "a_x", "vy": "a_y", "vz": "a_z"}.get(var, f"d{var}/dt")
    return s, acc


def slice_by_distance(meta, data, rhs, length=1.0, n_bins=3):
    """Refit the fixed structure of rhs on terciles of the distance from the centre (|position| at each weak-form
    test function's centre), with the helpers of eqdisc/audit/slices.py (weak form, cluster-robust + jackknife se)."""
    from eqdisc.audit import slices as sl
    from eqdisc.uq import _structure
    struct = _structure(meta, rhs)
    sysw = sl.weak_system(meta, data, struct)
    full = sl._slice_stats(sysw, struct, np.zeros(len(sysw["lhs"]), int), ["all"])
    U = np.asarray(data["U"], float)
    r = np.linalg.norm(U[sysw["traj"], sysw["tc"], :3], axis=-1)
    qs = np.quantile(r, np.linspace(0, 1, n_bins + 1))
    lab = np.clip(np.searchsorted(qs[1:-1], r, side="right"), 0, n_bins - 1)
    st = sl._slice_stats(sysw, struct, lab, [str(b) for b in range(n_bins)])
    out = {}
    for key, s in st.items():
        h = sl._hetero(s["b"], s["se"])
        fb, fse = float(full[key]["b"][0]), float(full[key]["se"][0])
        out[key] = {"per_slice": [None if not np.isfinite(b) else float(b) for b in s["b"]],
                    "per_slice_se": [None if not np.isfinite(v) else float(v) for v in s["se"]],
                    "full": fb, "full_se": fse,
                    "I2": None if h is None else round(h["I2"], 4), "p": None if h is None else h["p"],
                    "rel_range": None if h is None else h["rel_range"]}
    return {"edges_planet_radii": [float(q / length) for q in qs], "coefficients": out}


def _orbit_one(args):
    name, spec = args
    from eqdisc import assess, insights
    from eqdisc.audit import slices
    (ms, ds), (mb, db), _ = orbit_data()
    meta, data = (mb, db) if spec["units"] == "blind" else (ms, ds)
    rhs = spec["rhs"]
    t0 = time.time()
    audit = slices.audit(meta, data, rhs)
    dist = slice_by_distance(meta, data, rhs, length=0.53 if spec["units"] == "blind" else 1.0)
    a = assess.assess(meta, data, rhs)
    v = insights.verdict(a)
    findings = [_slim(f) for f in a.get("findings") or []]
    return name, {**{k: spec[k] for k in ("label", "units", "rhs")},
                  "slice_audit": [_slim(f) for f in audit],
                  "slice_amplitude_detail": next(({"variable": f["details"].get("amplitude_variable"),
                                                   "slices": f["details"].get("slices"), "i2": f["details"].get("i2")}
                                                  for f in audit if f["id"] == "slice_amplitude"), None),
                  "by_distance": dist, "verdict": v, "grade": (a.get("confidence") or {}),
                  "fired": [f for f in findings if f["fired"]], "seconds": round(time.time() - t0, 1)}


def pick_coefficients(m, k=3):
    """Coefficients to plot: the true law's three acceleration terms of a_x; for a polynomial the k most
    inconsistent (highest I^2) non-kinematic terms with an estimate on every slice."""
    co = m["by_distance"]["coefficients"]
    ok = {key: c for key, c in co.items() if all(b is not None for b in c["per_slice"])
          and key.split(":")[0] not in ("x", "y", "z", "u1", "u2", "u3") and abs(c["full"]) > 0}
    if m["units"] == "std" and m["label"].startswith("True"):
        keys = [key for key in ok if key.startswith("vx:")][:k]
    else:
        keys = sorted(ok, key=lambda key: -(ok[key]["I2"] or 0))[:k]
    out = []
    for key in keys:
        var, term = key.split(":", 1)
        lab, acc = term_label(term, var, m["units"])
        c = ok[key]
        out.append({"key": key, "term": lab, "equation": acc, **c})
    return out


def build_orbit(force=False):
    """Per-model results are cached in orbit_<model>.json (the assessment of a polynomial takes many minutes);
    a model is recomputed when its rhs changed, or with `orbit!` (force)."""
    from concurrent.futures import as_completed
    models = orbit_models()
    res, todo = {}, {}
    old = json.loads((OUT / "orbit.json").read_text())["models"] if (OUT / "orbit.json").exists() else {}
    for name, spec in models.items():
        p = OUT / f"orbit_{name}.json"
        cached = json.loads(p.read_text()) if p.exists() else old.get(name)
        cached = None if force else cached
        if cached and cached.get("rhs") == spec["rhs"]:
            res[name] = cached
        else:
            todo[name] = spec
    print(f"  cached: {sorted(res)}; computing: {sorted(todo)}", flush=True)
    if todo:
        with ProcessPoolExecutor(len(todo)) as ex:
            for fut in as_completed([ex.submit(_orbit_one, it) for it in todo.items()]):
                name, r = fut.result()
                _write(f"orbit_{name}.json", r)
                res[name] = r
                print(f"  {name} done ({r['seconds']} s)", flush=True)
    res = {k: res[k] for k in models}
    for name, m in res.items():
        m["shown"] = pick_coefficients(m)
        print(f"  {name}: {m['verdict']['status']} ({m['seconds']} s); fired: "
              f"{[(f['id'], f['severity']) for f in m['fired']]}", flush=True)
        for c in m["shown"]:
            print(f"     {c['equation']} {c['term']}: {[None if b is None else round(b, 4) for b in c['per_slice']]}"
                  f" full {c['full']:.4g} I2 {c['I2']}", flush=True)
    _, _, setup = orbit_data()
    _write("orbit.json", {"setup": setup, "models": res, "built": time.strftime("%Y-%m-%d %H:%M")})
    home_thumb()


def home_thumb():
    """Home-card thumbnail (560x315): per-slice coefficients of the true law vs the polynomial, as in section 1."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = json.loads((OUT / "orbit.json").read_text())["models"]
    wrong = "agent" if "agent" in d else "sindy"
    fig, axs = plt.subplots(1, 2, figsize=(5.6, 3.15), dpi=100, sharey=True)
    for ax, key, title in zip(axs, ("truth", wrong), ("true law", "AI's polynomial")):
        ax.axhspan(0.95, 1.05, color="#16a34a", alpha=0.12, lw=0)
        ax.axhline(1, color="#888888", lw=1, ls="--")
        for i, (c, col) in enumerate(zip(d[key]["shown"], ("#2a78d6", "#eb6834", "#1baf7a"))):
            x = np.arange(3) + (i - 1) * 0.12
            y = [np.nan if b is None else b / c["full"] for b in c["per_slice"]]
            e = [0 if s_ is None else 1.645 * s_ / abs(c["full"]) for s_ in c["per_slice_se"]]
            ax.errorbar(x, y, yerr=e, color=col, marker="o", ms=6, lw=2, capsize=3, mec="white")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(range(3), ["near", "mid", "far"], fontsize=10)
        ax.tick_params(axis="y", labelsize=9)
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
    axs[0].set_ylabel("coefficient / overall", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "thumb.jpg", facecolor="white")
    plt.close(fig)


# ============================================================================= corruption grid
def _grid_one(e):
    from eqdisc.audit import audit_model
    from eqdisc.audit.repair import audit_and_repair
    from eqdisc.evaluate import load
    m, D = load(e["path"])
    truth = json.loads((Path(e["path"]) / "hidden" / "truth.json").read_text())
    m2, D2, df, rep = audit_and_repair(m, D)
    mf = audit_model(m2, D2, truth["rhs"])
    thumb(np.asarray(D["U"], float), np.asarray(D["t"]), OUT / f"thumb_{e['corruption']}.png",
          spikes=e["corruption"] == "outliers")
    return e["corruption"], {"path": e["path"], "system": e["system"], "seed": e["seed"], "truth": truth["rhs"],
                             "findings": [_slim(f) for f in df + mf if f["fired"]],
                             "repairs": [{"tool": a.get("tool"), "ok": bool(a.get("ok")), "note": a.get("note")}
                                         for a in rep]}


def thumb(U, t, path, spikes=False):
    """Space-time heatmaps of every run side by side; missing samples light grey; isolated spikes as black dots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import median_filter
    U = U[..., 0]
    n = U.shape[0]
    vmax = float(np.nanpercentile(np.abs(U), 99.5))
    cmap = matplotlib.colormaps["RdBu_r"].copy()
    cmap.set_bad("#bdbdbd")
    fig, axs = plt.subplots(1, n, figsize=(3.6, 1.55), dpi=160, sharey=True)
    for j, ax in enumerate(np.atleast_1d(axs)):
        ax.imshow(U[j], aspect="auto", origin="lower", cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="nearest")
        if spikes:
            V = np.nan_to_num(U[j])
            d = V - median_filter(V, size=(3, 5), mode="nearest")
            s = 1.4826 * np.median(np.abs(d - np.median(d)))
            ti, xi = np.where(np.abs(d) > 6 * s)
            ax.scatter(xi, ti, s=1.6, c="black", linewidths=0, marker="s")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp_ in ax.spines.values():
            sp_.set_visible(False)
        ax.set_title(f"run {j + 1}", fontsize=6, pad=2, color="#555555")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.86, bottom=0.02, wspace=0.06)
    fig.savefig(path, transparent=True)
    plt.close(fig)


def detection_rates():
    rows = []
    for split in ("dev", "report", "blind"):
        p = CALIB / f"{split}.json"
        if p.exists():
            rows += [dict(e, split=split) for e in json.loads(p.read_text())]
    out = {"splits": sorted({r["split"] for r in rows}), "per_corruption": {}}
    for c in CORRUPTIONS:
        rs = [r for r in rows if r["corruption"] == c and "error" not in r]
        fired = [{f[0] for f in r.get("fired") or []} for r in rs]
        if c == "clean":
            hit = sum(bool(f) for f in fired)
        else:
            hit = sum(bool(f & set(EXPECTED[c])) for f in fired)
        out["per_corruption"][c] = {"n": len(rs), "hit": hit, "errors": sum("error" in r for r in rows
                                                                            if r["corruption"] == c)}
    return out


def build_grid():
    idx = json.loads(INDEX.read_text())
    pick = {}
    for e in idx:
        if e["split"] == "report" and e["system"] == "burgers" and e["seed"] == 10 and e["corruption"] in CORRUPTIONS:
            pick[e["corruption"]] = e
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(4) as ex:
        res = dict(ex.map(_grid_one, [pick[c] for c in CORRUPTIONS if c in pick]))
    for c, r in res.items():
        print(f"  {c}: {[(f['id'], f['severity'], f['response'], f.get('resolved')) for f in r['findings']]}", flush=True)
    _write("grid.json", {"cases": res, "rates": detection_rates(), "built": time.strftime("%Y-%m-%d %H:%M")})


# ============================================================================= scoreboard
CATS = ["right+confident", "right+cautious", "wrong+flagged", "wrong+confident", "crashed"]


def build_scoreboard():
    expected = Counter(e["split"] for e in json.loads(INDEX.read_text())) if INDEX.exists() else Counter()
    if not OUTCOMES.exists():
        _write("scoreboard.json", {"available": False, "built": time.strftime("%Y-%m-%d %H:%M")})
        print("  outcomes.jsonl not found: placeholder written", flush=True)
        return
    last = {}
    for ln in OUTCOMES.read_text().splitlines():
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        last[(r.get("arm"), r.get("case") or r.get("path"))] = r          # re-scored rows replace older ones
    counts = defaultdict(lambda: defaultdict(Counter))
    for (arm, _), r in last.items():
        cat = "crashed" if r.get("crashed") else r.get("category") or "crashed"
        for sp in (r.get("split") or "?", "all"):
            counts[sp][arm][cat] += 1
    _write("scoreboard.json", {"available": True, "categories": CATS,
                               "expected": dict(expected) | {"all": sum(expected.values())},
                               "counts": {sp: {a: dict(c) for a, c in d.items()} for sp, d in counts.items()},
                               "n_rows": len(last), "built": time.strftime("%Y-%m-%d %H:%M"),
                               "source_mtime": time.strftime("%Y-%m-%d %H:%M",
                                                             time.localtime(OUTCOMES.stat().st_mtime))})
    for sp, d in counts.items():
        print(f"  {sp}: " + "; ".join(f"{a}: {dict(c)}" for a, c in sorted(d.items())), flush=True)


PARTS = {"orbit": build_orbit, "orbit!": lambda: build_orbit(force=True), "grid": build_grid, "scoreboard": build_scoreboard, "thumb": home_thumb}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name in (sys.argv[1:] or ["scoreboard", "grid", "orbit"]):
        t0 = time.time()
        print(f"[{name}] ...", flush=True)
        PARTS[name]()
        print(f"[{name}] ok ({time.time() - t0:.1f}s)", flush=True)
