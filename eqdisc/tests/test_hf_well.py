"""Offline tests of eqdisc.hf_well: a tiny synthetic file in The Well's HDF5 schema stands in for Hugging Face.

    python -m eqdisc.tests.test_hf_well

The synthetic flow is an exact Navier-Stokes + passive-tracer solution on the shear_flow domain (x in [0,1),
y in [-1,1), periodic): a decaying shear u = A exp(-nu ky^2 t) sin(ky y), v = 0, and a tracer
s = B exp(-D ky2^2 t) cos(ky2 y). Advection vanishes (nothing depends on x), so omega_t = nu lap(omega) and
s_t = D lap(s) hold exactly; the true shear_flow equations must reach the time-differencing floor.
"""
import tempfile
from pathlib import Path

import h5py
import numpy as np

from eqdisc import hf_well


def _write_well_file(path, n_traj, re=50.0, sc=0.5, nx=32, ny=64, nt=31, dt=0.1, seed=0):
    rng = np.random.default_rng(seed)
    nu, D = 1 / re, 1 / re / sc
    t = np.arange(nt) * dt
    x = np.arange(nx) / nx                                  # physical grid; the file stores coords rescaled to [0, 1]
    y = -1 + 2 * np.arange(ny) / ny
    T, X, Y = np.meshgrid(t, x, y, indexing="ij")
    tracer, vel = [], []
    for _ in range(n_traj):
        A, B = rng.uniform(0.5, 1.5, 2)
        ky, ky2 = np.pi * rng.integers(1, 4), np.pi * rng.integers(1, 4)
        u = A * np.exp(-nu * ky ** 2 * T) * np.sin(ky * Y)
        tracer.append(B * np.exp(-D * ky2 ** 2 * T) * np.cos(ky2 * Y))
        vel.append(np.stack([u, np.zeros_like(u)], -1))
    with h5py.File(path, "w") as h:
        h.attrs.update({"Reynolds": re, "Schmidt": sc, "dataset_name": "shear_flow", "n_spatial_dims": 2,
                        "n_trajectories": n_traj})
        g = h.create_group("dimensions")
        g.attrs["spatial_dims"] = ["x", "y"]
        g["time"], g["x"], g["y"] = t, np.linspace(0, 1, nx).astype("f4"), np.linspace(0, 1, ny).astype("f4")
        for d in ("x", "y"):
            b = h.create_group(f"boundary_conditions/{d}_periodic")
            b.attrs["bc_type"], b.attrs["associated_dims"] = "PERIODIC", [d]
        h["scalars/Reynolds"], h["scalars/Schmidt"] = np.float32(re), np.float32(sc)
        h["t0_fields/tracer"] = np.asarray(tracer, "f8")
        h["t0_fields/pressure"] = np.zeros_like(np.asarray(tracer))
        h["t1_fields/velocity"] = np.asarray(vel, "f8")
        h.create_group("t2_fields")


def _patched(tmp):
    files = {s: tmp / f"{s}.hdf5" for s in ("train", "test")}
    _write_well_file(files["train"], 3, seed=0)
    _write_well_file(files["test"], 2, seed=1)
    hf_well.open_remote = lambda ref, split, fname, block_size=None: h5py.File(files[split], "r")


def test_parse_ref():
    for r in ("https://huggingface.co/datasets/polymathic-ai/shear_flow", "polymathic-ai/shear_flow", "well:shear_flow",
              "shear_flow", "https://huggingface.co/datasets/polymathic-ai/shear_flow/tree/main"):
        assert hf_well.parse_ref(r) == ("polymathic-ai/shear_flow", "shear_flow"), r


def test_spectral_coarsen_exact_for_band_limited():
    x = np.arange(64) / 64
    f = np.sin(2 * np.pi * 3 * x)[None, :] * np.cos(2 * np.pi * 2 * x)[:, None]
    g = hf_well.spectral_coarsen(f, 2, (0, 1))
    xc = np.arange(32) / 32
    want = np.sin(2 * np.pi * 3 * xc)[None, :] * np.cos(2 * np.pi * 2 * xc)[:, None]
    assert g.shape == (32, 32) and np.max(np.abs(g - want)) < 1e-12


def test_vorticity():
    x, y = np.arange(32) / 32, -1 + 2 * np.arange(64) / 64
    X, Y = np.meshgrid(x, y, indexing="ij")
    u, v = np.sin(np.pi * Y), np.cos(2 * np.pi * X)
    w = hf_well.vorticity(u, v, 1.0, 2.0)
    assert np.max(np.abs(w - (-2 * np.pi * np.sin(2 * np.pi * X) - np.pi * np.cos(np.pi * Y)))) < 1e-10


def test_build_and_score():
    tmp = Path(tempfile.mkdtemp())
    _patched(tmp)
    ds = hf_well.build_dataset("shear_flow", "x.hdf5", out_root=tmp / "out", n_train=2, n_test=2, t_end=3.0, coarsen=2)
    import json
    meta = json.loads((ds / "meta.json").read_text())
    assert meta["variables"] == ["u", "v", "omega", "s"] and meta["grid"]["y"]["L"] == 2.0 and meta["grid"]["x"]["n"] == 16
    assert not (ds / "hidden" / "truth.json").exists()              # keeps eqdisc's rollout evaluator off
    truth = json.loads((ds / "hidden" / "well_truth.json").read_text())["rhs"]
    good = hf_well.score_well(ds, truth)
    assert good["all_equivalent"] and good["mean_truth_residual"] < 5e-3, good          # weak-form floor
    assert all(p["truth_strong_residual"] < 1e-3 for p in good["per_var"].values()), good
    bad = hf_well.score_well(ds, {"omega": "-u*omega_x - v*omega_y", "s": "-u*s_x - v*s_y + 0.1*(s_xx + s_yy)"})
    assert bad["n_equivalent"] == 0 and bad["mean_test_residual"] > 0.5, bad


def test_weak_residual_tracks_coefficient_error():
    """Diffusion 1.5x too strong -> weak residual ~0.5 (the extra diffusion term is half the true rate of change)."""
    tmp = Path(tempfile.mkdtemp())
    _patched(tmp)
    ds = hf_well.build_dataset("shear_flow", "x.hdf5", out_root=tmp / "out", n_train=2, n_test=2, t_end=3.0, coarsen=2)
    import json
    truth = json.loads((ds / "hidden" / "well_truth.json").read_text())["rhs"]
    nu, d = 1 / 50, 1 / 50 / 0.5
    off = {"omega": f"-u*omega_x - v*omega_y + {1.5 * nu:g}*(omega_xx + omega_yy)",
           "s": f"-u*s_x - v*s_y + {1.5 * d:g}*(s_xx + s_yy)"}
    sc = hf_well.score_well(ds, off)
    assert all(0.4 < p["test_residual"] < 0.6 for p in sc["per_var"].values()), sc


def test_solve_well_dry_run():
    tmp = Path(tempfile.mkdtemp())
    _patched(tmp)
    spec = {"ref": "shear_flow", "param_file": "x.hdf5", "n_train": 2, "n_test": 1, "t_end": 3.0, "coarsen": 2}
    r = hf_well.solve_well(spec, {"dry_run": True, "max_tools": 5, "workdir": str(tmp / "w"), "critic": True})
    assert "error" not in r, r.get("trace")
    assert r["submitted"] and r["eval"]["well"]["scored"] and r["eval"]["symbolic_match"] is False   # no diffusion
    assert r["cost_usd"] > 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("HF WELL TESTS PASSED")
