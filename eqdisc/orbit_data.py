"""Build benchmark datasets from the orbit_discover Challenge1 orbit (J2-perturbed Kepler problem).

Public data = the real Challenge1 trajectory, nondimensionalised (lengths / Re, time / sqrt(Re^3/mu)), subsampled;
hidden test = new orbits simulated with the ground truth (two-body + zonal J2 with J2 = 0.5, as in the
challenge's data-generation script), so the evaluator can score discovered models on unseen orbits.

    python -m eqdisc.orbit_data            # writes datasets/orbit_challenge1 and datasets/orbit_challenge1_n0.01
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .datagen import add_noise, simulate
from .systems import SYSTEMS

MU, RE = 3.986004414498200e14, 6.378136460000000e6
T = np.sqrt(RE ** 3 / MU)
SRC = Path(__file__).resolve().parents[1] / "examples" / "data" / "Challenge1.csv"   # from SymbolicModel/orbit_discover (MIT)


def build(noise=0.0, every=4, out_root="datasets", seed=0):
    df = pd.read_csv(SRC)
    t = (pd.to_datetime(df.Time) - pd.to_datetime(df.Time)[0]).dt.total_seconds().values / T
    X = np.hstack([df[["rx", "ry", "rz"]].values / RE, df[["vx", "vy", "vz"]].values / (RE / T)])
    t, X = t[::every], X[::every]
    U = X[None]
    rng = np.random.default_rng(seed)
    U = add_noise(U, noise, rng)
    name = "orbit_challenge1" + (f"_n{noise:g}" if noise else "")
    d = Path(out_root) / name
    (d / "hidden").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "data.npz", t=np.round(t, 10), U=U)
    sys = SYSTEMS["kepler_j2"]
    sim_sys = type(sys)(**{**sys.__dict__, "dt": float(t[1] - t[0]), "t_end": 120.0})
    tt, Ut = simulate(sim_sys, 4, np.random.default_rng(seed + 1))
    np.savez_compressed(d / "hidden" / "test.npz", t=tt, U=Ut)
    meta = {"name": name, "kind": "ode", "variables": sys.variables, "dt": float(t[1] - t[0]), "n_traj": 1,
            "shape": list(U.shape), "shape_doc": "(n_traj, nt, n_vars)", "system": None,
            "allowed_symbols": sys.variables + ["t"],
            "units": "lengths in Earth radii Re, time in sqrt(Re^3/mu) (= 806.8 s); velocities in Re/that time",
            "source": "orbit_discover/data/Challenge1.csv (satellite orbit, 6 days, 30 s cadence), subsampled x%d" % every}
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    truth = {"system": "kepler_j2", "kind": "ode", "variables": sys.variables, "rhs": sys.rhs, "noise": noise,
             "noise_type": "gaussian", "dt_mult": every, "seed": seed, "eval_horizon": sys.eval_horizon,
             "tags": list(sys.tags)}
    (d / "hidden" / "truth.json").write_text(json.dumps(truth, indent=2))
    return d


if __name__ == "__main__":
    for n in (0.0, 0.01):
        print("wrote", build(noise=n))
