"""Weak-form SINDy on a cross-field product term (u_t = -v*u_x, variable-coefficient advection by a frozen spatial
field v), in 1-D, 2-D and 3-D. Product terms like v*u_x are not exact under integration by parts, so weak_sindy
differentiates the fields numerically; this checks that it does so along the SPATIAL axes (regression test for the
axis-offset bug where it differentiated along time: the coefficient came out +7.9 instead of -1 in 1-D).

Known open issue, deliberately not tested here: with BOTH v*u_x and a diffusion term in the library and a varying v,
weak_sindy under-weights or drops the diffusion term (each term alone is recovered to <0.5%).

    python -m eqdisc.tests.test_weak_crossfield
"""
import json
import tempfile
from pathlib import Path

import numpy as np

from eqdisc.evaluate import load, terms
from eqdisc.solvers import derivative_symbols, integrate_pde_general, parse
from eqdisc.weakform import weak_sindy



def _dataset(nd, n, nt=40, dt=0.01, seed=0):
    dims = ["x", "y", "z"][:nd]
    L = 2 * np.pi
    meta = {"kind": "pde", "variables": ["u", "v"], "spatial_dims": dims, "boundary": "periodic",
            "grid": {d: {"n": n, "L": L, "x0": 0.0} for d in dims}}
    g = np.arange(n) * L / n
    G = np.meshgrid(*([g] * nd), indexing="ij")
    rhs = {"u": "-v*u_x", "v": "0"}
    rng = np.random.default_rng(seed)
    t = np.arange(nt) * dt
    trajs = []
    for _ in range(3):
        ph = rng.uniform(0, 2 * np.pi, 4)
        u = np.sin(G[0] + ph[0]) + 0.5 * np.cos(2 * G[0] + ph[1]) + (0.4 * np.sin(G[-1] + ph[2]) if nd > 1 else 0)
        v = 1.0 + 0.5 * np.cos(G[0] + ph[3]) + (0.3 * np.sin(G[1]) if nd > 1 else 0)
        trajs.append(integrate_pde_general(["u", "v"], rhs, meta, np.stack([u, v], -1), t, dt / 20))
    U = np.stack(trajs)
    out = Path(tempfile.mkdtemp()) / f"cross{nd}d"
    out.mkdir()
    np.savez(out / "data.npz", t=t, U=U, **{d: g for d in dims})
    meta.update({"name": out.name, "dt": dt, "n_traj": 3, "shape": list(U.shape), "L": L, "nx": n,
                 "allowed_symbols": derivative_symbols(["u", "v"], dims, 2)})
    (out / "meta.json").write_text(json.dumps(meta))
    return out, dims


def _check(nd, n):
    ds, dims = _dataset(nd, n)
    meta, data = load(ds)
    r = weak_sindy(meta, data, poly_degree=1, max_deriv=0, custom_terms=["v*u_x"], exclude_terms=["1", "u", "v"],
                   targets=["u"], thresholds=[0.0])
    got = {str(k): c for k, c in terms(parse(r["rhs"]["u"], derivative_symbols(["u", "v"], dims, 2))).items()}
    print(f"  {nd}-D: u_t = {r['rhs']['u']}")
    assert abs(got.get("u_x*v", 0) + 1) < 0.03, (nd, got)


def test_crossfield_1d():
    _check(1, 64)


def test_crossfield_2d():
    _check(2, 32)


def test_crossfield_3d():
    _check(3, 16)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name, flush=True)
    print("WEAK CROSS-FIELD TESTS PASSED")
