"""Offline tests of the isothermal MHD forward simulator (eqdisc.mhd_sim). No network, ~1 minute.

    python -m eqdisc.tests.test_mhd_sim
"""
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from eqdisc import mhd_sim as ms


def _ic(n=32, seed=1, **kw):
    g = ms.Grid(n)
    a = dict(v_rms=0.3, b_rms=0.3, rho_amp=0.1, kmax=3)
    a.update(kw)
    return g, ms.random_initial_condition(g, np.random.default_rng(seed), **a)


def test_conservation():
    g, U0 = _ic()
    p = {"cs2": 1.0}
    for form in ("conservative", "primitive"):
        _, U = ms.simulate(U0, [0.0, 0.1], dict(p, form=form), g, scheme="rk4")
        assert abs(ms.mass(U[-1]) - ms.mass(U0)) < 1e-12, form
        dE = abs(ms.energy(U[-1], p) - ms.energy(U0, p)) / ms.energy(U0, p)
        assert dE < 2e-6, (form, dE)
        assert ms.div_b(U[-1], g) < 1e-12
        if form == "conservative":
            assert np.abs(ms.momentum(U[-1]) - ms.momentum(U0)).max() < 1e-12
    # SSP-RK3 drift converges with the step (3rd order); the card form (no magnetic pressure) does not conserve E
    _, U = ms.simulate(U0, [0.0, 0.1], p, g, cfl=0.2)
    assert abs(ms.energy(U[-1], p) / ms.energy(U0, p) - 1) < 2e-6
    _, U = ms.simulate(U0, [0.0, 0.1], dict(p, mag_pressure=0.0), g, scheme="rk4")
    assert abs(ms.energy(U[-1], p) / ms.energy(U0, p) - 1) > 1e-3


def test_divb_without_projection_and_dissipation():
    g, U0 = _ic()
    _, U = ms.simulate(U0, [0.0, 0.1], {"cs2": 1.0, "nu": 1e-3, "eta": 1e-3}, g, project_B=False,
                       forcing={"amp": 1.0, "seed": 3})
    assert np.all(np.isfinite(U)) and ms.div_b(U[-1], g) < 1e-12
    p = {"cs2": 1.0, "nu": 5e-3, "eta": 5e-3}
    _, U = ms.simulate(U0, [0.0, 0.1], p, g)
    assert ms.energy(U[-1], p) < ms.energy(U0, p)                     # dissipation removes energy


def _phase_speed(U0, field, axis, T, params, n):
    """propagation speed of the |n| = 1 mode of `field` along `axis` from its Fourier phase after time T."""
    _, U = ms.simulate(U0, [0.0, T], params, ms.Grid(n), scheme="rk4", cfl=0.3)
    i = ms.FIELDS.index(field)
    idx = [0, 0, 0]
    idx[axis] = 1
    c0, c1 = (np.fft.fftn(u[..., i])[tuple(idx)] for u in (U0, U[-1]))
    dphi = np.angle(c1 / c0)
    return -dphi / (2 * np.pi * T)


def _wave_ic(n, axis, comps, a=1e-4, b0=0.7):
    x = np.arange(n) / n
    s = [1, 1, 1]
    s[axis] = n
    c = np.cos(2 * np.pi * x).reshape(s) * np.ones((n, n, n))
    U = np.zeros((n, n, n, 7))
    U[..., 0] = 1.0
    U[..., 4] = b0
    for f, amp in comps.items():
        U[..., ms.FIELDS.index(f)] += a * amp * c
    return U


def test_linear_wave_speeds():
    n, T, b0, cs2 = 16, 0.2, 0.7, 1.3
    cs = np.sqrt(cs2)
    p = {"cs2": cs2}
    # Alfven wave along B0 = b0 x: v_A = b0 / sqrt(rho0)
    c = _phase_speed(_wave_ic(n, 0, {"vy": 1.0, "by": -1.0}, b0=b0), "vy", 0, T, p, n)
    assert abs(c - b0) < 1e-3 * b0, (c, b0)
    # sound wave along B0 (unaffected by B)
    c = _phase_speed(_wave_ic(n, 0, {"rho": 1.0, "vx": cs}, b0=b0), "rho", 0, T, p, n)
    assert abs(c - cs) < 1e-3 * cs, (c, cs)
    # fast wave across B0: sqrt(cs2 + m2 * vA^2); the card form (m2 = 0) propagates at cs
    for m2 in (1.0, 0.0):
        cf = np.sqrt(cs2 + m2 * b0 ** 2)
        c = _phase_speed(_wave_ic(n, 1, {"rho": 1.0, "vy": cf, "bx": b0}, b0=b0), "rho", 1, T,
                         dict(p, mag_pressure=m2), n)
        assert abs(c - cf) < 1e-3 * cf, (m2, c, cf)


def test_rhs_matches_symbolic_truth():
    """rhs() equals eqdisc's symbolic templates (hf_well._mhd_truth_template at C = 1 and the momentum features
    with a = 1, K = cs2, m1 = m2 = 1, or m2 = 0) evaluated by make_pde_rhs_general, on random smooth fields."""
    from eqdisc.solvers import make_pde_rhs_general
    n = 24
    g, U = _ic(n, seed=4, b0=(0.4, -0.2, 0.1))
    meta = {"spatial_dims": ["x", "y", "z"], "boundary": "periodic",
            "grid": {d: {"n": n, "L": 1.0, "x0": 0.0} for d in "xyz"}}
    for m2 in (1.0, 0.0):
        p = {"cs2": 1.7, "mag_pressure": m2, "dealias": False, "C": 1.3}
        truth = ms.truth_expressions(p)
        F = make_pde_rhs_general(ms.FIELDS, truth, meta)(U)
        for form in ("primitive", "conservative"):
            R = ms.rhs(U, dict(p, form=form), g)
            for i, f in enumerate(ms.FIELDS):
                err = np.abs(R[..., i] - F[..., i]).max() / np.abs(F[..., i]).max()
                assert err < (1e-10 if form == "primitive" else 1e-5), (m2, form, f, err)
    # dissipation terms too
    p = {"cs2": 1.0, "nu": 0.01, "eta": 0.02, "zeta": 0.005, "dealias": False, "form": "primitive"}
    F = make_pde_rhs_general(ms.FIELDS, ms.truth_expressions(p), meta)(U)
    assert np.abs(ms.rhs(U, p, g) - F).max() < 1e-9 * np.abs(F).max()


def test_compare_frames_recovers_constants():
    """compare_frames on simulated frames recovers C, cs2 and the magnetic-pressure coefficient."""
    g, U0 = _ic(32, seed=5, b0=(0.6, 0, 0))
    for m2 in (1.0, 0.0):
        p = {"cs2": 0.8, "mag_pressure": m2, "C": 1.0}
        _, fr = ms.simulate(U0, np.arange(5) * 0.002, p, g, scheme="rk4", cfl=0.2)
        rep = ms.compare_frames(fr, 0.002, verbose=False)
        assert abs(rep["rho"]["C"] - 1) < 1e-3 and abs(rep["B"]["C"] - 1) < 1e-3, rep["rho"]
        free = rep["v"]["standard(free: a, K, m1, m2)"]["no_forcing"]
        assert free["r2"] > 0.999 and abs(free["coefs"]["K"] - 0.8) < 0.01 and abs(free["coefs"]["m2"] - m2) < 0.02, free


def test_generate_dataset_roundtrip():
    from eqdisc.solvers import make_pde_rhs_general
    tmp = Path(tempfile.mkdtemp(prefix="mhd_ds_", dir="/tmp"))
    try:
        d = ms.generate_dataset(tmp / "ds", n=16, n_traj=1, n_test=1, t_end=0.1, n_frames=6,
                                params={"cs2": 1.0, "nu": 2e-3, "eta": 2e-3}, seed=0)
        meta = json.loads((d / "meta.json").read_text())
        truth = json.loads((d / "hidden" / "truth.json").read_text())
        data = np.load(d / "data.npz")
        assert data["U"].shape == (1, 6, 16, 16, 16, 7) and meta["variables"] == ms.FIELDS
        assert set(truth["closed_variables"]) == set(ms.FIELDS)
        # the stored truth expressions reproduce the simulated tendencies (dealiasing aside)
        g = ms.Grid(16)
        U = np.moveaxis(g.ifft(g.fft(np.moveaxis(data["U"][0, 3], -1, 0)) * g.mask), 0, -1)   # band-limited
        F = make_pde_rhs_general(ms.FIELDS, truth["rhs"], meta)(U)
        R = ms.rhs(U, dict(truth["params"], dealias=False, form="primitive"), g)
        assert np.abs(F - R).max() < 1e-9 * np.abs(R).max()
        # interventions: forcing makes velocity unclosed; a clamp holds the field
        d2 = ms.generate_dataset(tmp / "ds2", n=16, n_traj=1, n_test=1, t_end=0.05, n_frames=3,
                                 forcing={"amp": 1.0}, clamp={"vz": 0.0}, seed=1)
        t2 = json.loads((d2 / "hidden" / "truth.json").read_text())
        assert t2["closed_variables"] == ["rho", "bx", "by", "bz"]
        assert np.abs(np.load(d2 / "data.npz")["U"][..., 3]).max() == 0.0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    import time
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            t0 = time.time()
            fn()
            print(f"ok {name} ({time.time() - t0:.1f}s)", flush=True)
    print("MHD SIM TESTS PASSED")
