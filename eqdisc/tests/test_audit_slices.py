"""Slice-consistency audit (eqdisc/audit/slices.py): fires on heterogeneous data, quiet on clean data."""
import json

import numpy as np
import pytest

from eqdisc.audit import slices
from eqdisc.blind import _perturb
from eqdisc.datagen import add_noise
from eqdisc.solvers import derivative_symbols, grid_coords, integrate_ode, integrate_pde_general, parse
from eqdisc.systems import SYSTEMS


def _ode(system, rhs_per_traj, ics, noise=0.02, seed=0):
    sys = SYSTEMS[system]
    t = np.round(np.arange(0, sys.t_end + 1e-9, sys.dt), 10)
    U = np.stack([integrate_ode(sys.variables, r, ic, t) for r, ic in zip(rhs_per_traj, ics)])
    U = add_noise(U, noise, np.random.default_rng(seed))
    meta = {"kind": "ode", "variables": list(sys.variables), "dt": sys.dt, "allowed_symbols": list(sys.variables) + ["t"]}
    return meta, {"U": U, "t": t}


def _pde(system, rhs_per_traj, seed=0, noise=0.02, t_end=None, amp=None):
    sys = SYSTEMS[system]
    lay = sys.layout()
    rng = np.random.default_rng(seed)
    t = np.round(np.arange(0, (t_end or sys.t_end) + 1e-9, sys.dt), 10)
    U = []
    for j, r in enumerate(rhs_per_traj):
        ic = sys.ic(rng, *grid_coords(lay)) * (amp[j] if amp else 1.0)
        U.append(integrate_pde_general(sys.fields, r, lay, ic, t, sys.dt_sim))
    U = add_noise(np.stack(U), noise, rng)
    meta = {"kind": "pde", "variables": list(sys.fields), "dt": sys.dt, "L": sys.L, "nx": sys.nx,
            "boundary": lay["boundary"], "spatial_dims": lay["spatial_dims"], "grid": lay["grid"],
            "allowed_symbols": derivative_symbols(sys.fields, lay["spatial_dims"], 4)}
    return meta, {"U": U, "t": t, "x": grid_coords(lay)[0]}


def _by_id(fs):
    return {f["id"]: f for f in fs}


def _check_contract(fs):
    for f in fs:
        json.dumps(f)
        assert f["stage"] == "model"
        assert "re_intervals" in f["details"] and "i2" in f["details"]
        for k, (lo, hi) in f["details"]["re_intervals"].items():
            assert ":" in k and lo <= hi


def _lv_ics(n, rng):
    return [np.array([rng.uniform(2, 15), rng.uniform(2, 8)]) for _ in range(n)]


def test_clean_ode_quiet():
    sys = SYSTEMS["lotka_volterra"]
    rng = np.random.default_rng(1)
    meta, data = _ode("lotka_volterra", [sys.rhs] * 4, _lv_ics(4, rng))
    fs = slices.audit(meta, data, sys.rhs)
    _check_contract(fs)
    ids = _by_id(fs)
    assert {"slice_trajectory", "slice_time", "slice_amplitude"} <= set(ids)
    assert not any(f["fired"] for f in fs)
    assert ids["slice_amplitude"]["scope"]["variable"]
    assert set(ids["slice_trajectory"]["details"]["re_intervals"]) == {"x:x", "x:x*y", "y:y", "y:x*y"}


def test_trajectory_coefficients_fire_with_repair():
    sys = SYSTEMS["lotka_volterra"]
    rng = np.random.default_rng(2)
    names = sys.variables + ["t"]
    rhs = [{v: str(_perturb(parse(e, names), rng, 0.2)) for v, e in sys.rhs.items()} for _ in range(4)]
    meta, data = _ode("lotka_volterra", rhs, _lv_ics(4, rng))
    f = _by_id(slices.audit(meta, data, sys.rhs))["slice_trajectory"]
    assert f["fired"] and f["response"] == "repair"
    assert f["fix"] == {"tool": "per_trajectory", "args": {}}
    assert len(next(iter(f["details"]["per_trajectory"].values()))) == 4
    assert "hidden parameter" in f["message"]


def test_drift_fires_time_slice():
    sys = SYSTEMS["lotka_volterra"]
    rng = np.random.default_rng(3)
    drift = dict(sys.rhs, x="1.1*(0.8 + 0.4*t/40)*x - 0.4*x*y")
    meta, data = _ode("lotka_volterra", [drift] * 4, _lv_ics(4, rng))
    f = _by_id(slices.audit(meta, data, sys.rhs))["slice_time"]
    assert f["fired"] and f["response"] == "widen"
    assert "drift" in f["message"]
    # widened: the drifting coefficient's interval is wider than on clean data
    meta0, data0 = _ode("lotka_volterra", [sys.rhs] * 4, _lv_ics(4, np.random.default_rng(3)))
    f0 = _by_id(slices.audit(meta0, data0, sys.rhs))["slice_time"]
    w = lambda iv: iv[1] - iv[0]
    assert w(f["details"]["re_intervals"]["x:x"]) > 2 * w(f0["details"]["re_intervals"]["x:x"])


def test_missing_cubic_fires_amplitude_with_scope():
    sys = SYSTEMS["vanderpol"]
    rng = np.random.default_rng(4)
    true = dict(sys.rhs, y=sys.rhs["y"] + " - 0.3*x**3")
    ics = [sys.ic(rng) * a for a in (0.5, 1.0, 1.5, 2.0)]
    meta, data = _ode("vanderpol", [true] * 4, ics)
    f = _by_id(slices.audit(meta, data, sys.rhs))["slice_amplitude"]
    assert f["fired"] and f["response"] == "widen"
    lo, hi = f["scope"]["range"]
    assert hi < f["details"]["observed_range"][1]
    # the true model is consistent
    assert not _by_id(slices.audit(meta, data, true))["slice_amplitude"]["fired"]


def test_nan_rows_dropped():
    sys = SYSTEMS["lotka_volterra"]
    meta, data = _ode("lotka_volterra", [sys.rhs] * 3, _lv_ics(3, np.random.default_rng(5)))
    data["U"][1, 100:130] = np.nan
    data["U"][0, 400, 1] = np.nan
    fs = slices.audit(meta, data, sys.rhs)
    assert fs and not any(f["fired"] for f in fs)
    assert fs[0]["details"]["n_rows_dropped_nan"] > 0


def test_pde_clean_and_per_trajectory():
    sys = SYSTEMS["advection_diffusion"]
    meta, data = _pde("advection_diffusion", [sys.rhs] * 3, seed=0, t_end=2.0)
    fs = slices.audit(meta, data, sys.rhs)
    _check_contract(fs)
    ids = _by_id(fs)
    assert set(ids) == {"slice_trajectory", "slice_time", "slice_space", "slice_amplitude"}
    assert not any(f["fired"] for f in fs)
    rhs = [sys.rhs, {"u": "-0.8*u_x + 0.05*u_xx"}, {"u": "-1.2*u_x + 0.06*u_xx"}]
    meta, data = _pde("advection_diffusion", rhs, seed=1, t_end=2.0)
    f = _by_id(slices.audit(meta, data, sys.rhs))["slice_trajectory"]
    assert f["fired"] and "u:u_x" in f["details"]["inconsistent_coefficients"]


def test_orbit_polynomial_fails_amplitude_true_law_passes():
    """A polynomial fitted in place of inverse-square gravity + J2 is locally fine but inconsistent across radii."""
    import pandas as pd
    from pathlib import Path
    from eqdisc import toolbox as tb
    from eqdisc.oos import ORBIT_TRUTH, RE_E, T_E
    path = Path(__file__).resolve().parents[2] / "examples" / "data" / "Challenge1.csv"
    if not path.exists():
        pytest.skip("Challenge1.csv not available")
    df = pd.read_csv(path)
    X = np.hstack([df[["rx", "ry", "rz"]].values / RE_E, df[["vx", "vy", "vz"]].values / (RE_E / T_E)])
    t = np.arange(len(X)) * 30.0 / T_E
    Xn = X + 0.01 * X.std(0) * np.random.default_rng(0).standard_normal(X.shape)
    n = len(X) // 2
    names = ["x", "y", "z", "vx", "vy", "vz"]
    meta = {"kind": "ode", "variables": names, "dt": float(t[4] - t[0]), "allowed_symbols": names + ["t"]}
    data = {"U": Xn[None, :n:4], "t": t[:n:4]}
    truth = {"x": "vx", "y": "vy", "z": "vz",
             **{k: v.replace("r**", "sqrt(x**2+y**2+z**2)**") for k, v in ORBIT_TRUTH.items()}}
    assert not any(f["fired"] for f in slices.audit(meta, data, truth))
    poly = tb.run_sindy(meta, data, poly_degree=2)["rhs"]
    f = _by_id(slices.audit(meta, data, poly))["slice_amplitude"]
    assert f["fired"] and f["severity"] == "critical"


def test_limit_cycle_true_model_quiet(tmp_path):
    """Hopf on its limit cycle: x and x*(x^2+y^2) are nearly collinear, the radial coefficients are identified only by
    short transients. Delete-one-cluster jackknife se and the identifiability guards must keep the true model quiet."""
    from eqdisc import datagen
    from eqdisc.evaluate import load
    for seed in (1, 5):
        d = datagen.generate("hopf", out_root=str(tmp_path), noise=0.02, seed=seed, plot=False)
        meta, data = load(d)
        assert not any(f["fired"] for f in slices.audit(meta, data, SYSTEMS["hopf"].rhs))
