"""Corruption benchmark generator (eqdisc.corrupt) and the time-dependent PDE rhs in solvers. Fast: short KS records."""
import json

import numpy as np
import pytest

from eqdisc import corrupt, evaluate
from eqdisc.solvers import integrate_pde, make_pde_rhs
from eqdisc.systems import SYSTEMS

KS = "kuramoto_sivashinsky"
T_SHORT = 20.0


@pytest.fixture(scope="module")
def ks_cases(tmp_path_factory):
    root = tmp_path_factory.mktemp("corrupt")
    return {c: corrupt.make_case(KS, c, seed=0, out_root=root, t_end=T_SHORT) for c in corrupt.CORRUPTIONS}


def _hidden(d, name):
    return json.loads((d / "hidden" / name).read_text())


def test_time_dependent_rhs_backward_compatible():
    s = SYSTEMS["advection_diffusion"]
    x = np.arange(s.nx) * s.L / s.nx
    U0 = s.ic(np.random.default_rng(1), x)
    t = np.arange(0, 2.0001, s.dt)
    f = make_pde_rhs(s.fields, s.rhs, s.L)
    assert np.array_equal(f(U0, x), f(U0, x, 3.7))                     # t-free rhs ignores t
    U = integrate_pde(s.fields, {"u": s.rhs["u"] + " + 0.5*sin(2.0*t)"}, s.L, U0, t, s.dt_sim)
    mean = U[:, :, 0].mean(axis=1) - U0.mean()
    assert np.allclose(mean, 0.25 * (1 - np.cos(2 * t)), atol=1e-8)   # spatial mean integrates the forcing exactly


def test_cases_are_valid_datasets(ks_cases):
    base = SYSTEMS[KS].rhs
    for c, d in ks_cases.items():
        meta, data = evaluate.load(d)
        U = data["U"]
        assert U.shape[0] >= 3 and U.shape[-1] == 1
        assert meta["system"] is None and meta["name"].startswith("case_")
        public = (d / "meta.json").read_text()
        assert KS not in public and c not in public.replace("allowed_symbols", "")
        truth, corr = _hidden(d, "truth.json"), _hidden(d, "corruption.json")
        assert truth["rhs"] == base
        assert corr["id"] == c and corr["expected_detector"] == corrupt.EXPECTED[c]
        test = dict(np.load(d / "hidden" / "test.npz"))
        assert np.all(np.isfinite(test["U"]))
        if c.startswith("gaps"):
            frac = np.isnan(U).mean()
            assert 0.05 < frac < 0.3 and np.all(np.isfinite(U[:, 0]))
        else:
            assert np.all(np.isfinite(U))


def test_measurement_corruptions(ks_cases):
    _, clean = evaluate.load(ks_cases["clean"])
    _, out = evaluate.load(ks_cases["outliers"])

    def spikes(U):                       # fraction of samples > 5 noise sigmas from a spectrally smoothed field
        Uh = np.fft.rfft(U, axis=2)
        Uh[:, :, U.shape[2] // 8:] = 0
        dev = np.abs(U - np.fft.irfft(Uh, n=U.shape[2], axis=2))
        return (dev > 5 * corrupt.NOISE * U.std()).mean()
    assert spikes(clean["U"]) < 1e-3 and spikes(out["U"]) > 4e-3
    _, gs = evaluate.load(ks_cases["gaps_state"])
    test_U = dict(np.load(ks_cases["clean"] / "hidden" / "test.npz"))["U"]
    obs = gs["U"][np.isfinite(gs["U"])]
    assert np.abs(obs).max() < np.abs(clean["U"]).max()      # tails censored
    assert test_U.shape[0] == 2


def test_forcing_and_source_visible_in_residual(ks_cases):
    for c, basis in (("forcing_time", "t"), ("source_space", "x")):
        d = ks_cases[c]
        meta, data = evaluate.load(d)
        p = _hidden(d, "corruption.json")["params"]
        U, t, x = data["U"], data["t"], data["x"]
        Uh = np.fft.rfft(U, axis=2)
        Uh[:, :, U.shape[2] // 8:] = 0
        Us = np.fft.irfft(Uh, n=U.shape[2], axis=2)
        r = (np.gradient(Us, t, axis=1) - make_pde_rhs(["u"], SYSTEMS[KS].rhs, meta["L"])(Us, x))[:, 2:-2, :, 0]
        if basis == "t":
            est = 2 * np.mean(r.mean(axis=2) * np.sin(p["w"] * t[2:-2]))
        else:
            est = 2 * np.mean(r @ np.cos(2 * np.pi * x / meta["L"])) / len(x)
        assert est == pytest.approx(p["A"], rel=0.3)


def test_traj_coeffs_differ(ks_cases):
    corr = _hidden(ks_cases["traj_coeffs"], "corruption.json")
    assert len({json.dumps(r) for r in corr["rhs_train"]}) == 3
    assert corr["rhs_test"] == SYSTEMS[KS].rhs


def test_clean_truth_score(tmp_path):
    d = corrupt.make_case("advection_diffusion", "clean", seed=0, out_root=tmp_path)
    truth = _hidden(d, "truth.json")
    assert evaluate.evaluate(d, {"rhs": truth["rhs"]}, reveal=True)["score"] > 6.0


def test_blind_variant_and_index(tmp_path):
    paths = corrupt.make_suite([0], out_root=tmp_path, blind=True, systems=[KS], corruptions=["forcing_time"],
                               workers=1)
    assert len(paths) == 1
    d = paths[0]
    meta, _ = evaluate.load(d)
    assert meta["variables"] == ["q"]
    truth = _hidden(d, "truth.json")
    assert "q_xxxx" in truth["rhs"]["q"] and truth["rhs"] != SYSTEMS[KS].rhs
    index = json.loads((tmp_path / "index.json").read_text())
    assert index[0]["split"] == "blind" and index[0]["corruption"] == "forcing_time"
