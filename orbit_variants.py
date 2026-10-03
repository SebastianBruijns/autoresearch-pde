"""Orbit datasets in units that do not make the answer look clean.

eqdisc.orbit_data nondimensionalises by Re and sqrt(Re^3/mu), so gravity's coefficient is exactly 1 and J2's is 0.75.
This rescales those datasets (data, hidden test set and truth) to:
    orbit_si      SI units (m, s, m/s): gravity coefficient mu = 3.986e14 (recognisable as Earth's)
    orbit_scaled  arbitrary length and time units: coefficients are unremarkable numbers

With lengths scaled by a and times by b, r' = a r and t' = b t, so dv'/dt' = (a / b^2) f(r' / a).

    python -m eqdisc.orbit_data && python orbit_variants.py
"""
import json
import shutil
from pathlib import Path

import numpy as np

from eqdisc.orbit_data import RE, T

SRC = ["datasets/orbit_challenge1", "datasets/orbit_challenge1_n0.01"]
POS, VEL = ["x", "y", "z"], ["vx", "vy", "vz"]


def rescale(src, tag, a, b, note):
    src = Path(src)
    dst = src.with_name(src.name.replace("orbit_challenge1", f"orbit_{tag}"))
    if dst.exists():
        shutil.rmtree(dst)
    (dst / "hidden").mkdir(parents=True)
    scale = np.array([a] * 3 + [a / b] * 3)
    for f in ("data.npz", "hidden/test.npz"):
        z = dict(np.load(src / f))
        np.savez_compressed(dst / f, t=z["t"] * b, U=z["U"] * scale)
    truth = json.loads((src / "hidden" / "truth.json").read_text())
    # nondimensional truth: -r/|r|^3 - 0.75 * J2 term; rescaled, gravity gets mu = a^3/b^2 and J2 gets 0.75 a^5/b^2
    assert truth["system"] == "kepler_j2"
    mu, k = a ** 3 / b ** 2, 0.75 * a ** 5 / b ** 2
    r = "sqrt(x**2 + y**2 + z**2)"
    for v, p, c in zip(VEL, POS, (1, 1, 3)):
        truth["rhs"][v] = f"-{mu:.10g}*{p}/{r}**3 - {k:.10g}*{p}/{r}**5*({c} - 5*z**2/{r}**2)"
    truth["eval_horizon"] *= b
    truth["tags"] = [t for t in truth["tags"] if not t.startswith("nondimensional")] + [note]
    (dst / "hidden" / "truth.json").write_text(json.dumps(truth, indent=2))
    meta = json.loads((src / "meta.json").read_text())
    for k in ("units", "source"):           # these would tell the agent it is an orbit
        meta.pop(k, None)
    meta.update(name=dst.name, dt=meta["dt"] * b)
    (dst / "meta.json").write_text(json.dumps(meta, indent=2))
    return dst


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    a, b = 10 ** rng.uniform(-1, 1), 10 ** rng.uniform(-1, 1)
    for s in SRC:
        print(rescale(s, "si", RE, T, "SI units (m, s)"))
        print(rescale(s, "scaled", a, b, f"arbitrary units: length x{a:.4g}, time x{b:.4g} of (Re, sqrt(Re^3/mu))"))
