"""Compare toolbox.run_sindy vs weakform.weak_sindy on noisy ODE/PDE datasets.

    python -m eqdisc.tests.test_weakform [dataset_name ...]      # comparison table
    python -m eqdisc.tests.test_weakform --checks                # assertions only (or use pytest)

Uses evaluate(..., reveal=True) (hidden test data) only to MEASURE quality.
"""
import sys
import time
from pathlib import Path

import numpy as np

from eqdisc import toolbox
from eqdisc.evaluate import evaluate, load, terms
from eqdisc.solvers import parse
from eqdisc.weakform import weak_sindy

ROOT = Path(__file__).resolve().parents[2] / "datasets"
DATASETS = ["kdv_n0.01_dt1_s0", "kdv_n0.05_dt2_s0", "kdv_n0.05_dt1_s0", "kuramoto_sivashinsky_n0.05_dt1_s0",
            "burgers_n0.05_dt1_s0", "allen_cahn_n0.1_dt1_s0", "lorenz_n0.05_dt1_s0", "vanderpol_n0.1_dt1_s0"]


def _fmt(x):
    return "  -  " if x is None else f"{x:.3f}"


def run(names, verbose=True):
    rows = []
    for name in names:
        d = ROOT / name
        meta, data = load(d)
        for label, fn in (("sindy", toolbox.run_sindy), ("weak", weak_sindy)):
            t0 = time.time()
            out = fn(meta, data)
            secs = time.time() - t0
            r = evaluate(d, out, reveal=True)
            rows.append((name, label, r.get("score"), r.get("f1"), r.get("coef_rel_err"), secs, out["rhs"]))
            if verbose:
                print(f"{name:36s} {label:5s} score={r.get('score'):7.3f} f1={_fmt(r.get('f1'))} "
                      f"coef_err={_fmt(r.get('coef_rel_err'))} t={secs:5.1f}s  rhs={out['rhs']}", flush=True)
        if verbose:
            print(f"{'':36s} truth {r['truth']}")
    if verbose:
        print("\n| dataset | method | score | f1 | coef_rel_err | sec |")
        print("|---|---|---|---|---|---|")
        for name, label, s, f1, ce, secs, _ in rows:
            print(f"| {name} | {label} | {s:.3f} | {_fmt(f1)} | {_fmt(ce)} | {secs:.1f} |")
    return rows


# ----------------------------------------------------------------------------- assertions
def test_weak_beats_sindy_on_noisy_pdes():
    for name in ["kdv_n0.05_dt1_s0", "kuramoto_sivashinsky_n0.05_dt1_s0", "burgers_n0.05_dt1_s0"]:
        (_, _, s0, f0, *_), (_, _, s1, f1, _, secs, _) = run([name], verbose=False)
        assert f1 == 1.0 and s1 > s0 + 1.0 and secs < 30, name


def test_single_trajectory_and_nonperiodic_paths():
    meta, data = load(ROOT / "kdv_n0.05_dt1_s0")
    one = weak_sindy(meta, {**data, "U": data["U"][:1]})
    nonper = weak_sindy({**meta, "boundary": "dirichlet"}, data)
    for out in (one, nonper):
        r = evaluate(ROOT / "kdv_n0.05_dt1_s0", out, reveal=True)
        assert r["f1"] == 1.0 and r["coef_rel_err"] < 0.02


def test_2d_heat_equation():
    """Exact 2-D diffusion solution (u_t = 0.1 (u_xx + u_yy)) with 2% noise, on an (x, y) grid."""
    L, n, D = 2 * np.pi, 48, 0.1
    x = np.arange(n) * L / n
    X, Y = np.meshgrid(x, x, indexing="ij")
    t = np.arange(60) * 0.05
    rng = np.random.default_rng(1)
    U = np.zeros((3, len(t), n, n))
    for j in range(3):
        for _ in range(8):
            kx, ky = rng.integers(-4, 5, 2)
            U[j] += rng.normal() * np.cos(kx * X + ky * Y + rng.uniform(0, 2 * np.pi))[None] * \
                np.exp(-D * (kx ** 2 + ky ** 2) * t)[:, None, None]
    U = U[..., None] + 0.02 * U.std() * rng.normal(size=U.shape + (1,))
    meta = {"kind": "pde", "variables": ["u"], "dt": 0.05, "L": L, "nx": n, "spatial_dims": ["x", "y"],
            "boundary": "periodic", "allowed_symbols": ["u", "u_x", "u_y", "u_xx", "u_xy", "u_yy", "x", "y"]}
    real_validate = toolbox.validate
    toolbox.validate = lambda *a, **k: {"skipped": "1-D validator"}
    try:
        out = weak_sindy(meta, {"t": t, "U": U, "x": x, "y": x}, poly_degree=2, max_deriv=2)
    finally:
        toolbox.validate = real_validate
    found = {str(k): v for k, v in terms(parse(out["rhs"]["u"], meta["allowed_symbols"])).items()}
    assert set(found) == {"u_xx", "u_yy"} and all(abs(c - D) < 0.005 for c in found.values()), found


def run_checks():
    for fn in (test_weak_beats_sindy_on_noisy_pdes, test_single_trajectory_and_nonperiodic_paths,
               test_2d_heat_equation):
        fn()
        print(f"PASS {fn.__name__}", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--checks"]
    if "--checks" in sys.argv:
        run_checks()
    else:
        run(args or DATASETS)
