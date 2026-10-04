"""Offline tests of 3-D PDE support: tools on an exact 3-D solution, and the MHD_64 loader on a synthetic file in
The Well's schema (no network, no API calls).

    python -m eqdisc.tests.test_3d

1. Advection-diffusion u_t = -a u_x - b u_y - c u_z + D lap(u) on a 16^3 periodic grid: every agent tool must run,
   weak_sindy must recover the coefficients, repair must find a missing z-term.
2. MHD: constant velocity v, density and a divergence-free magnetic field advected rigidly with it. Then
   rho_t = -div(rho v) and B_t = curl(v x B) hold exactly, so the loader's fitted unit constant must be ~1 and the
   true equations must score as a match.
"""
import json
import tempfile
from pathlib import Path

import h5py
import numpy as np

from eqdisc import agent, hf_well, repair
from eqdisc.solvers import derivative_symbols


def _advdiff_dataset(out, n=16, nt=30, dt=0.05, a=0.5, b=-0.3, c=0.2, D=0.05, seed=0):
    rng = np.random.default_rng(seed)
    L = 2 * np.pi
    x = np.arange(n) * L / n
    t = np.arange(1, nt + 1) * dt
    T, X, Y, Z = np.meshgrid(t, x, x, x, indexing="ij")

    def traj():
        u = np.zeros_like(X)
        for _ in range(4):
            k = rng.integers(-2, 3, 3)
            k[0] = k[0] or 1
            arg = k[0] * (X - a * T) + k[1] * (Y - b * T) + k[2] * (Z - c * T) + rng.uniform(0, 2 * np.pi)
            u += rng.normal() * np.exp(-D * (k @ k) * T) * np.cos(arg)
        return u[..., None]
    U = np.stack([traj() for _ in range(3)])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dims = ["x", "y", "z"]
    np.savez(out / "data.npz", t=t, U=U, x=x, y=x, z=x)
    meta = {"name": "synthetic3d", "kind": "pde", "variables": ["u"], "dt": dt, "n_traj": 3, "shape": list(U.shape),
            "system": None, "L": L, "nx": n, "boundary": "periodic", "spatial_dims": dims,
            "grid": {d: {"n": n, "L": L, "x0": 0.0} for d in dims}, "allowed_symbols": derivative_symbols(["u"], dims, 2)}
    (out / "meta.json").write_text(json.dumps(meta))
    return out, (a, b, c, D)


def test_tools_on_3d():
    ds, (a, b, c, D) = _advdiff_dataset(Path(tempfile.mkdtemp()) / "ds")
    s = agent.Session(ds, workdir=Path(tempfile.mkdtemp()))
    w = s.call("weak_sindy", {"poly_degree": 1, "max_deriv": 2})
    want = {"u_x": -a, "u_y": -b, "u_z": -c, "u_xx": D, "u_yy": D, "u_zz": D}
    from eqdisc.evaluate import terms
    from eqdisc.solvers import parse
    got = {str(k): v for k, v in terms(parse(w["rhs"]["u"], derivative_symbols(["u"], ["x", "y", "z"], 2))).items()}
    assert all(abs(got.get(k, 0) - v) < 0.05 * max(abs(v), 0.05) for k, v in want.items()), got
    r = s.call("run_sindy", {"poly_degree": 1, "max_deriv": 4})
    assert "capped" in r.get("note", ""), r.get("note")                    # 3-D: derivative order capped at 2
    rep = s.call("repair", {"rhs": {"u": f"{-a}*u_x + {-b}*u_y + {-c}*u_z + {D}*(u_xx + u_yy)"}})
    assert rep["top_edits"][0]["term"] == "u_zz", rep["top_edits"][:2]    # needs z-terms in the pool
    for name, args in [("diagnose", {}), ("intuit", {}), ("plot_data", {}), ("plot_model", {"rhs": w["rhs"]}),
                       ("compare_models", {"candidates": {"a": w["rhs"]}}), ("ensemble_sindy", {"n_models": 5, "poly_degree": 1})]:
        out = s.call(name, args)
        assert not (isinstance(out, dict) and out.get("error")), (name, out.get("error"))
    assert "error" in s.call("find_invariants", {})                         # declines cleanly in 3-D
    assert "u_yy" in repair.default_pool(s.meta) and "u_z" in repair.default_pool(s.meta)


def _write_mhd_file(path, n_traj, n=16, nt=12, dt=0.01, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(nt) * dt
    g = np.arange(n) / n                                   # physical period 1; the file stores linspace(0, 1, n)
    T, X, Y, Z = np.meshgrid(t, g, g, g, indexing="ij")
    rho, vel, mag = [], [], []
    for _ in range(n_traj):
        v = rng.uniform(-1, 1, 3)
        Xs, Ys, Zs = X - v[0] * T, Y - v[1] * T, Z - v[2] * T
        k = rng.integers(1, 3, 4)
        rho.append(1 + 0.3 * np.cos(2 * np.pi * (k[0] * Xs + Ys) + rng.uniform(0, 6)))
        # each component independent of its own coordinate -> div B = 0
        mag.append(np.stack([np.sin(2 * np.pi * k[1] * Ys), np.sin(2 * np.pi * k[2] * Zs), np.sin(2 * np.pi * k[3] * Xs)], -1))
        vel.append(np.broadcast_to(v, X.shape + (3,)).copy())
    with h5py.File(path, "w") as h:
        h.attrs.update({"Ma": 2.0, "Ms": 0.5, "dataset_name": "MHD_64", "n_spatial_dims": 3, "n_trajectories": n_traj})
        d = h.create_group("dimensions")
        d.attrs["spatial_dims"] = ["x", "y", "z"]
        d["time"] = t
        for ax in ("x", "y", "z"):
            d[ax] = np.linspace(0, 1, n)
            b = h.create_group(f"boundary_conditions/{ax}_periodic")
            b.attrs["bc_type"], b.attrs["associated_dims"] = "PERIODIC", [ax]
        h["scalars/Ma"], h["scalars/Ms"] = np.float32(2.0), np.float32(0.5)
        h["t0_fields/density"] = np.asarray(rho)
        h["t1_fields/velocity"] = np.asarray(vel)
        h["t1_fields/magnetic_field"] = np.asarray(mag)
        h.create_group("t2_fields")


def test_mhd_loader_calibration_and_score():
    tmp = Path(tempfile.mkdtemp())
    files = {s: tmp / f"{s}.hdf5" for s in ("train", "test")}
    # 31 frames: the weak-form calibration needs long enough time windows (with 11 frames its quadrature bias is ~3%;
    # with 31, ~0.1%; the real MHD_64 runs use ~29 frames)
    _write_mhd_file(files["train"], 3, nt=31, seed=0)
    _write_mhd_file(files["test"], 1, nt=31, seed=1)
    hf_well.open_remote = lambda ref, split, fname, block_size=None: h5py.File(files[split], "r")
    ds = hf_well.build_dataset("MHD_64", "MHD_Ma_2_Ms_0.5.hdf5", out_root=tmp / "out", n_train=2, n_test=1, coarsen=1)
    meta = json.loads((ds / "meta.json").read_text())
    assert meta["variables"] == ["rho", "vx", "vy", "vz", "bx", "by", "bz"] and meta["max_deriv_cap"] == 2
    assert "rho_xx" in meta["allowed_symbols"] and "rho_xxx" not in meta["allowed_symbols"]
    truth = json.loads((ds / "hidden" / "well_truth.json").read_text())
    assert abs(truth["calibration"]["C"] - 1) < 0.01, truth["calibration"]
    sc = hf_well.score_well(ds, truth["rhs"])
    assert sc["all_equivalent"] and sc["n_scored"] == 4, sc
    assert set(sc["per_var"]) == {"rho", "bx", "by", "bz"}, sc["per_var"].keys()     # velocity is not scored
    assert all(p["truth_residual"] < 0.05 for p in sc["per_var"].values())
    wrong = dict(truth["rhs"], rho="0")
    assert hf_well.score_well(ds, wrong)["n_equivalent"] == 3


def test_mhd_disguise():
    """Disguised MHD: neutral names, shuffled order, rescaled grid/clock; truth still exact with C = C_factor."""
    tmp = Path(tempfile.mkdtemp())
    files = {s: tmp / f"{s}.hdf5" for s in ("train", "test")}
    _write_mhd_file(files["train"], 3, nt=31, seed=0)
    _write_mhd_file(files["test"], 1, nt=31, seed=1)
    hf_well.open_remote = lambda ref, split, fname, block_size=None: h5py.File(files[split], "r")
    ds = hf_well.build_dataset("MHD_64", "MHD_Ma_2_Ms_0.5.hdf5", out_root=tmp / "out", n_train=2, n_test=1, coarsen=1,
                               disguise_seed=3)
    meta = json.loads((ds / "meta.json").read_text())
    assert meta["variables"] == [f"q{k}" for k in range(1, 8)], meta["variables"]
    assert not any(n in json.dumps(meta) for n in ("rho", "vx", "bx", "MHD"))
    truth = json.loads((ds / "hidden" / "well_truth.json").read_text())
    d = truth["disguise"]
    assert abs(meta["grid"]["x"]["L"] - d["space_scale"]) < 1e-6 and abs(meta["dt"] - 0.01 * d["time_scale"]) < 1e-6
    assert abs(truth["calibration"]["C"] / d["C_factor"] - 1) < 0.01, (truth["calibration"], d)
    sc = hf_well.score_well(ds, truth["rhs"])
    assert sc["all_equivalent"] and sc["n_scored"] == 4, sc
    assert set(sc["per_var"]) == {d["rename"][v] for v in ("rho", "bx", "by", "bz")}
    assert all(p["truth_residual"] < 0.05 for p in sc["per_var"].values())


def test_mhd_solve_dry_run():
    tmp = Path(tempfile.mkdtemp())
    files = {s: tmp / f"{s}.hdf5" for s in ("train", "test")}
    _write_mhd_file(files["train"], 3, seed=0)
    _write_mhd_file(files["test"], 1, seed=1)
    hf_well.open_remote = lambda ref, split, fname, block_size=None: h5py.File(files[split], "r")
    spec = {"ref": "MHD_64", "param_file": "MHD_Ma_2_Ms_0.5.hdf5", "n_train": 2, "n_test": 1, "t_end": None, "coarsen": 1}
    r = hf_well.solve_well(spec, {"dry_run": True, "max_tools": 5, "workdir": str(tmp / "w"), "critic": False, "verbose": False})
    assert "error" not in r, r.get("trace")
    assert r["eval"]["well"]["scored"] and r["eval"]["well"]["n_scored"] == 4


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name, flush=True)
    print("3-D TESTS PASSED")
