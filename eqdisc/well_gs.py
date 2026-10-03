"""Gray-Scott reaction-diffusion from The Well (Ohana et al., NeurIPS 2024), streamed, as an equation-discovery
benchmark with controlled noise.

Only short windows of a few trajectories are read (HTTP range requests on the uncompressed HDF5 files); nothing
else is downloaded. Ground truth (The Well, App. C):  A_t = dA*lap(A) - A*B^2 + F*(1 - A),
                                                      B_t = dB*lap(B) + A*B^2 - (F + k)*B,   dA = 2e-5, dB = 1e-5.

    python -m eqdisc.well_gs fetch            # all 6 regimes -> datasets/well_gs_<regime>_n<noise>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = "datasets/polymathic-ai/gray_scott_reaction_diffusion/data/test/"
REGIMES = {"bubbles": (0.098, 0.057), "gliders": (0.014, 0.054), "maze": (0.029, 0.057),
           "spirals": (0.018, 0.051), "spots": (0.03, 0.062), "worms": (0.058, 0.065)}
DA, DB = 2e-5, 1e-5


def fetch_window(regime, trajs=(0, 1, 2), t0=50, nt=60, stride_xy=1, cache="datasets/_well_cache"):
    """Return (t, x, y, U) with U shape (n_traj, nt, nx, ny, 2) for fields (A, B). Cached locally as .npz."""
    import h5py
    from huggingface_hub import HfFileSystem
    F, k = REGIMES[regime]
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    cf = cache / f"{regime}_t{t0}_{nt}_tr{'-'.join(map(str, trajs))}.npz"
    if cf.exists():
        z = np.load(cf)
        return z["t"], z["x"], z["y"], z["U"]
    path = f"{REPO}gray_scott_reaction_diffusion_{regime}_F_{F}_k_{k}.hdf5"
    fs = HfFileSystem()
    with fs.open(path, "rb", block_size=8 * 2 ** 20) as f, h5py.File(f, "r") as h:
        t = h["dimensions/time"][t0:t0 + nt]
        x, y = h["dimensions/x"][:], h["dimensions/y"][:]
        U = np.stack([np.stack([h["t0_fields/A"][j, t0:t0 + nt], h["t0_fields/B"][j, t0:t0 + nt]], -1) for j in trajs])
    np.savez_compressed(cf, t=t, x=x, y=y, U=U)
    return t, x, y, U


def build(regime, noise=0.0, out_root="datasets", t0=50, nt=60, stride_xy=1, seed=0):
    """Train = trajectories 0, 1 (noisy); hidden test = trajectory 2 (clean). Grid optionally subsampled by
    stride_xy (periodic, so spectral derivatives remain valid) to keep regression matrices small."""
    from .datagen import add_noise
    from .solvers import derivative_symbols
    F, k = REGIMES[regime]
    t, x, y, U = fetch_window(regime, (0, 1, 2), t0, nt)
    U = U[:, :, ::stride_xy, ::stride_xy].astype(float)
    n = U.shape[2]
    L = float(len(x) * (x[1] - x[0]))                # periodic period: 128 points, spacing 2/127, no duplicate edge
    xs = -1 + np.arange(n) * L / n
    tt = np.round(t - t[0], 6).astype(float)
    name = f"well_gs_{regime}_n{noise:g}"
    d = Path(out_root) / name
    (d / "hidden").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    train = add_noise(U[:2], noise, rng) if noise > 0 else U[:2]
    np.savez_compressed(d / "data.npz", t=tt, U=train, x=xs, y=xs)
    np.savez_compressed(d / "hidden" / "test.npz", t=tt, U=U[2:3], x=xs, y=xs)
    grid = {"x": {"n": n, "L": L, "dx": L / n, "x0": -1.0}, "y": {"n": n, "L": L, "dx": L / n, "x0": -1.0}}
    meta = {"name": name, "kind": "pde", "variables": ["A", "B"], "dt": float(tt[1] - tt[0]), "n_traj": 2,
            "shape": list(train.shape), "shape_doc": "(n_traj, nt, nx, ny, n_fields)", "spatial_dims": ["x", "y"],
            "boundary": "periodic", "grid": grid, "L": L, "nx": n,
            "allowed_symbols": list(dict.fromkeys(derivative_symbols(["A", "B"], ["x", "y"], 4) + ["x", "y"])),
            "source": f"The Well / gray_scott_reaction_diffusion, regime {regime}, snapshots {t0}..{t0 + nt - 1}, "
                      f"grid subsampled x{stride_xy}", "system": None}
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    rhs = {"A": f"{DA}*(A_xx + A_yy) - A*B**2 + {F}*(1 - A)", "B": f"{DB}*(B_xx + B_yy) + A*B**2 - {F + k}*B"}
    truth = {"system": f"gray_scott_{regime}", "kind": "pde", "variables": ["A", "B"], "rhs": rhs, "noise": noise,
             "noise_type": "gaussian", "seed": seed, "eval_horizon": float(tt[min(30, len(tt) - 1)]),
             "spatial_dims": ["x", "y"], "boundary": "periodic", "grid": grid, "L": L, "nx": n, "dt_sim": 0.5,
             "params": {"F": F, "k": k, "dA": DA, "dB": DB}}
    (d / "hidden" / "truth.json").write_text(json.dumps(truth, indent=2))
    return d


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "fetch":
    for r in REGIMES:
        t0 = time.time()
        fetch_window(r)
        print(r, f"{time.time() - t0:.0f}s", flush=True)


# ----------------------------------------------------------------------------- benchmark
def vrmse_windows(meta, rhs, d, windows=((6, 12), (13, 30)), dt_sim=1.0):
    """The Well's VRMSE (per field sqrt(<|u-v|^2> / <|u-mean u|^2>), averaged over fields), averaged over rollout
    windows of snapshot steps, starting from the first snapshot of the held-out trajectory."""
    from . import solvers
    te = np.load(Path(d) / "hidden" / "test.npz")
    U, t = te["U"][0], te["t"]
    nmax = max(w[1] for w in windows)
    Y = solvers.integrate_pde_general(meta["variables"], rhs, solvers.pde_layout(meta), U[0], t[:nmax + 1], dt_sim=dt_sim)
    out = {}
    for a, b in windows:
        vals = []
        for i in range(a, b + 1):
            if not np.all(np.isfinite(Y[i])):
                vals.append(np.inf)
                continue
            per = [np.sqrt(np.mean((Y[i, ..., f] - U[i, ..., f]) ** 2) / (np.mean((U[i, ..., f] - U[i, ..., f].mean()) ** 2) + 1e-12))
                   for f in range(U.shape[-1])]
            vals.append(float(np.mean(per)))
        out[f"vrmse_{a}-{b}"] = float(np.mean(vals))
    return out


def coefficient_errors(rhs, truth_params):
    """Relative errors of the recovered physical constants F, k, dA, dB (None if the term is missing)."""
    import sympy as sp
    from .solvers import parse
    names = ["A", "B", "A_xx", "A_yy", "B_xx", "B_yy"]
    tA = {str(m): float(c) for c, m in (t.as_coeff_Mul() for t in sp.Add.make_args(sp.expand(parse(rhs.get("A", "0"), names))))}
    tB = {str(m): float(c) for c, m in (t.as_coeff_Mul() for t in sp.Add.make_args(sp.expand(parse(rhs.get("B", "0"), names))))}
    F, k, dA, dB = (truth_params[x] for x in ("F", "k", "dA", "dB"))
    est = {"F": tA.get("1"), "dA": np.mean([tA[x] for x in ("A_xx", "A_yy") if x in tA]) if any(x in tA for x in ("A_xx", "A_yy")) else None,
           "dB": np.mean([tB[x] for x in ("B_xx", "B_yy") if x in tB]) if any(x in tB for x in ("B_xx", "B_yy")) else None,
           "F+k": -tB["B"] if "B" in tB else None}
    err = {}
    for key, true in (("F", F), ("dA", dA), ("dB", dB), ("F+k", F + k)):
        err[key] = None if est[key] is None else abs(est[key] - true) / abs(true)
    return err


CONTEXT = "Measured concentrations of two interacting chemical species A and B on a periodic 2-D domain."


def run_benchmark(noises=(0.0, 0.01, 0.05, 0.1), regimes=None, out="runs/well_gs", agent=True, workers=2,
                  max_tools=18):
    """Arms: plain SINDy, weak SINDy (both fixed defaults, no LLM), and one agent session (no skills/memory; a
    one-line domain context). Metrics: term F1 vs truth, relative errors of F, F+k, dA, dB, and The Well's VRMSE
    over rollout windows 6-12 and 13-30 from the held-out trajectory."""
    from concurrent.futures import ThreadPoolExecutor
    from . import toolbox as tb
    from .agent import make_client, run_agent
    from .evaluate import evaluate, load
    from .weakform import weak_sindy
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    regimes = regimes or list(REGIMES)
    jobs = [(r, n) for r in regimes for n in noises]
    rows = {}

    def score(d, m, rhs, truth):
        ev = evaluate(d, {"rhs": rhs}, reveal=True)
        return {"f1": ev.get("f1"), **vrmse_windows(m, rhs, d), **{f"err_{k}": v for k, v in
                                                                   coefficient_errors(rhs, truth["params"]).items()},
                "rhs": rhs}
    for r, n in jobs:
        d = build(r, noise=n)
        m, D = load(d)
        truth = json.loads((d / "hidden" / "truth.json").read_text())
        row = {"regime": r, "noise": n, "truth": score(d, m, truth["rhs"], truth)}
        for name, fit in (("sindy", lambda: tb.run_sindy(m, D, poly_degree=3, max_deriv=2)),
                          ("weak_sindy", lambda: weak_sindy(m, D, poly_degree=3, max_deriv=2))):
            try:
                row[name] = score(d, m, fit()["rhs"], truth)
            except Exception as e:  # noqa: BLE001
                row[name] = {"error": str(e)[:200]}
        rows[(r, n)] = row
        print(json.dumps({"regime": r, "noise": n, **{a: {k: row[a].get(k) for k in ("f1", "vrmse_6-12", "vrmse_13-30")}
                                                      for a in ("truth", "sindy", "weak_sindy")}}, default=str), flush=True)
    if agent:
        client = make_client()

        def one(job):
            r, n = job
            d = Path("datasets") / f"well_gs_{r}_n{n:g}"
            m, _ = load(d)
            truth = json.loads((d / "hidden" / "truth.json").read_text())
            try:
                res = run_agent(d, client=client, verbose=False, max_tools=max_tools, use_skills=False,
                                use_memory=False, context=CONTEXT, out_dir=out / f"agent_{r}_n{n:g}",
                                final_assessment=False, report=True)
                rhs = (res.get("submitted") or {}).get("rhs")
                return job, ({**score(d, m, rhs, truth), "cost_usd": res["cost_usd"]} if rhs else
                             {"error": "no submission", "cost_usd": res["cost_usd"]})
            except Exception as e:  # noqa: BLE001
                return job, {"error": str(e)[:200]}
        with ThreadPoolExecutor(workers) as ex:
            for job, s in ex.map(one, jobs):
                rows[job]["agent"] = s
                print(json.dumps({"regime": job[0], "noise": job[1], "agent": {k: s.get(k) for k in
                                  ("f1", "vrmse_6-12", "vrmse_13-30", "cost_usd", "error")}}, default=str), flush=True)
    (out / "results.json").write_text(json.dumps({f"{r}|{n}": v for (r, n), v in rows.items()}, indent=1, default=str))
    return rows


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "bench":
    run_benchmark(agent="--no-agent" not in sys.argv)
