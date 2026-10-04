"""Residual decomposition (eqdisc/audit/residual.py): fast checks on small generated cases.

Data are simulated here (own small integrators) so the tests do not depend on datasets/ or on solver changes:
  clean           the model given to audit is the true rhs
  forcing_time    data with + A*sin(w*t) in one equation, model = base rhs
  source_space    (PDE) data with + A*cos(2*pi*x/L), model = base rhs
  missing_term    data from the full rhs, model drops one u-dependent term (the confound for residual_time_only)
  amp_term        data with - c*v**3 (matters only at the largest amplitudes), model = base rhs
"""
import time

import numpy as np
import pytest
import sympy as sp

from eqdisc.audit import SEVERITIES, residual
from eqdisc.solvers import (MAX_DERIV, _etdrk4_coeffs, integrate_ode, make_ode_rhs, parse, pde_symbols,
                            spectral_derivs, split_linear, wavenumbers)


# ----------------------------------------------------------------------------- tiny generators
def _ode_case(corruption, seed=0, noise=0.02, amp=0.3):
    """Van der Pol, 4 trajectories."""
    rng = np.random.default_rng(seed)
    names = ["x", "y"]
    base = {"x": "y", "y": "2*(1 - x**2)*y - x"}
    t = np.round(np.arange(0, 20 + 1e-9, 0.02), 10)
    ics = [rng.uniform(-3, 3, 2) for _ in range(4)]
    pilot = np.stack([integrate_ode(names, base, x0, t) for x0 in ics])
    scale = float(np.sqrt(np.mean(make_ode_rhs(names, base)(pilot)[..., 1] ** 2)))
    true = dict(base)
    if corruption == "forcing_time":
        true["y"] = f"{base['y']} + {amp * scale:.6g}*sin({2 * np.pi * 3 / 20:.6g}*t + {rng.uniform(0, 6):.4g})"
    elif corruption == "amp_term":
        true["y"] = f"{base['y']} - {amp * scale / np.abs(pilot[..., 1]).max() ** 3:.6g}*y**3"
    U = pilot if true == base else np.stack([integrate_ode(names, true, x0, t) for x0 in ics])
    U = U + noise * U.reshape(-1, 2).std(0) * rng.standard_normal(U.shape)
    meta = {"name": "vdp", "kind": "ode", "variables": names, "dt": 0.02, "n_traj": 4, "shape": list(U.shape),
            "allowed_symbols": names + ["t"]}
    model = dict(base)
    if corruption == "missing_term":
        model["y"] = "2*(1 - x**2)*y"
    return meta, {"t": t, "U": U}, model


def _etdrk4(rhs, L, u0, t, dt_sim, forcing=None):
    nx = u0.size
    x = np.arange(nx) * L / nx
    k = wavenumbers(nx, L)
    lin, nonlin = split_linear(["u"], rhs)
    Lop = sum(lin["u"][n] * (1j * k) ** n for n in range(MAX_DERIV + 1)).astype(complex)
    dealias = np.arange(k.size) < nx / 3
    names = pde_symbols(["u"])
    fn = sp.lambdify([sp.Symbol(n) for n in names], parse(str(nonlin["u"]), names), "numpy")

    def N(vh, tt):
        D = spectral_derivs(np.fft.irfft(vh, n=nx)[:, None], L)
        out = np.broadcast_to(fn(*[D[j][:, 0] for j in range(MAX_DERIV + 1)], x), (nx,)).astype(float)
        if forcing is not None:
            out = out + forcing(tt, x)
        return dealias * np.fft.rfft(out)

    ns = int(round((t[1] - t[0]) / dt_sim))
    h = (t[1] - t[0]) / ns
    E, E2, Q, f1, f2, f3 = _etdrk4_coeffs(Lop, h)
    v = np.fft.rfft(u0)
    out = [u0]
    for i in range(len(t) - 1):
        for j in range(ns):
            tn = t[i] + j * h
            Nv = N(v, tn)
            a = E2 * v + Q * Nv
            Na = N(a, tn + h / 2)
            b = E2 * v + Q * Na
            Nb = N(b, tn + h / 2)
            c = E2 * a + Q * (2 * Nb - Nv)
            Nc = N(c, tn + h)
            v = E * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        out.append(np.fft.irfft(v, n=nx))
    return np.array(out)


def _pde_case(corruption, seed=0, noise=0.02, amp=0.3):
    """Advection-diffusion u_t = -u_x + 0.05 u_xx on [0, 2pi), 3 trajectories, 128 points, t in [0, 4]."""
    rng = np.random.default_rng(seed)
    L, nx = 2 * np.pi, 128
    x = np.arange(nx) * L / nx
    t = np.round(np.arange(0, 4 + 1e-9, 0.02), 10)
    base = {"u": "-1.0*u_x + 0.05*u_xx"}
    ics = []
    for j in range(3):
        u = sum(rng.normal() / k * np.cos(k * x + 2 * np.pi * rng.random()) for k in range(1, 5))
        u = u / np.abs(u).max()
        ics.append(u * (0.4 + 0.3 * j if corruption == "amp_term" else 1.0))
    A = amp * 0.6                                   # ~ amp x RMS of the rhs
    forcing, true = None, dict(base)
    if corruption == "forcing_time":
        w, ph = 2 * np.pi * 3 / 4, rng.uniform(0, 6)
        forcing = lambda tt, xx: A * np.sin(w * tt + ph) + 0 * xx
    elif corruption == "source_space":
        ph = rng.uniform(0, 6)
        forcing = lambda tt, xx: A * np.cos(xx + ph)
    elif corruption == "amp_term":
        true = {"u": f"{base['u']} - {2 * A:.6g}*u**3"}
    U = np.stack([_etdrk4(true, L, u0, t, 4e-3, forcing) for u0 in ics])[..., None]
    U = U + noise * U.std() * rng.standard_normal(U.shape)
    meta = {"name": "ad", "kind": "pde", "variables": ["u"], "dt": 0.02, "n_traj": 3, "shape": list(U.shape),
            "L": L, "nx": nx, "boundary": "periodic", "spatial_dims": ["x"],
            "grid": {"x": {"n": nx, "L": L, "x0": 0.0}}}
    model = dict(base)
    if corruption == "missing_term":
        model = {"u": "0.05*u_xx"}
    return meta, {"t": t, "U": U, "x": x}, model


def _by_id(findings):
    return {f["id"]: f for f in findings}


# ----------------------------------------------------------------------------- tests
def test_contract_and_clean_ode():
    meta, data, model = _ode_case("clean", seed=1)
    F = residual.audit(meta, data, model)
    ids = [f["id"] for f in F]
    assert ids == ["residual_time_only", "residual_amplitude", "residual_white"]
    for f in F:
        assert f["stage"] == "model" and f["severity"] in SEVERITIES
        assert isinstance(f["statistic"], float) and isinstance(f["threshold"], float)
        assert not f["fired"], f
        assert f["response"] is None and f["fix"] is None and f["scope"] is None


def test_ode_time_forcing_fires_with_profile():
    meta, data, model = _ode_case("forcing_time", seed=2)
    f = _by_id(residual.audit(meta, data, model))["residual_time_only"]
    assert f["fired"] and f["response"] == "widen"
    assert f["message"].startswith("residual follows a function of time only")
    assert f["fix"]["tool"] == "add_forcing" and f["fix"]["args"]["basis"] == "time"
    assert f["fix"]["args"]["variable"] == "y"
    prof = f["details"]["profile"]
    assert 10 <= len(prof) <= 50


def test_ode_missing_term_is_not_forcing():
    meta, data, model = _ode_case("missing_term", seed=3)
    F = _by_id(residual.audit(meta, data, model))
    assert not F["residual_time_only"]["fired"]
    assert F["residual_white"]["fired"]               # the misfit itself is still reported


def test_ode_amplitude_term_scoped():
    meta, data, model = _ode_case("amp_term", seed=4, amp=0.5)
    F = _by_id(residual.audit(meta, data, model))
    f = F["residual_amplitude"]
    assert f["fired"] and f["response"] == "scope"
    assert f["scope"]["variable"] == "y"
    lo, hi = f["scope"]["range"]
    assert lo < 0 < hi < np.abs(data["U"][..., 1]).max()
    assert "excess_kurtosis" in f["details"]["per_variable"]["y"]
    assert not F["residual_time_only"]["fired"]


def test_pde_clean_and_runtime():
    meta, data, model = _pde_case("clean", seed=0)
    t0 = time.time()
    F = residual.audit(meta, data, model)
    assert time.time() - t0 < 15
    assert [f["id"] for f in F] == ["residual_time_only", "residual_space_only", "residual_amplitude",
                                    "residual_white"]
    assert not any(f["fired"] for f in F), [(f["id"], f["statistic"]) for f in F if f["fired"]]


@pytest.mark.parametrize("corruption,fires,quiet", [("forcing_time", "residual_time_only", "residual_space_only"),
                                                    ("source_space", "residual_space_only", "residual_time_only")])
def test_pde_forcing_and_source(corruption, fires, quiet):
    meta, data, model = _pde_case(corruption, seed=1)
    F = _by_id(residual.audit(meta, data, model))
    assert F[fires]["fired"] and F[fires]["severity"] == "critical"
    assert F[fires]["fix"]["args"]["basis"] == ("time" if "time" in fires else "space")
    assert len(F[fires]["details"]["profile"]) <= 50
    assert not F[quiet]["fired"]


def test_pde_missing_term_confound():
    meta, data, model = _pde_case("missing_term", seed=2)
    F = _by_id(residual.audit(meta, data, model))
    assert not F["residual_time_only"]["fired"] and not F["residual_space_only"]["fired"]
    assert F["residual_white"]["fired"]


def test_pde_amplitude_term():
    meta, data, model = _pde_case("amp_term", seed=3, amp=0.6)
    F = _by_id(residual.audit(meta, data, model))
    assert F["residual_amplitude"]["fired"]
    assert F["residual_amplitude"]["scope"]["variable"] == "u"
    assert not F["residual_time_only"]["fired"] and not F["residual_space_only"]["fired"]


def test_refitted_coefficients_still_detect_forcing():
    """The tournament fits the base structure ON the forced data: biased coefficients must not hide the forcing."""
    from eqdisc import uq
    meta, data, model = _pde_case("forcing_time", seed=4)
    prep = uq._prepare(meta, data)
    struct = uq._structure(meta, model)
    cols = uq._structure_columns(prep, struct)
    c = uq._lstsq(cols["u"], prep["Y"][:, 0])
    refit = uq._rhs_from(struct, {"u": c})
    F = _by_id(residual.audit(meta, data, refit))
    assert F["residual_time_only"]["fired"]


def test_nan_rows_dropped():
    meta, data, model = _ode_case("forcing_time", seed=5)
    U = data["U"].copy()
    U[0, 100:140] = np.nan
    U[2, 600:603, 1] = np.nan
    F = residual.audit(meta, {"t": data["t"], "U": U}, model)
    assert all(not f["id"].endswith("_error") for f in F)
    f = _by_id(F)["residual_time_only"]
    assert f["fired"] and f["details"]["n_rows_used"] < f["details"]["n_rows_total"]
