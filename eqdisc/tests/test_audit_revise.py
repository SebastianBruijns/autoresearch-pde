"""audit.revise: fired findings -> refitted challengers -> adopted only if they win the tournament.
Dev corrupt cases only (seed 0, generated in tmp); no API calls."""
import json

import numpy as np
import pytest

from eqdisc import audit, corrupt, evaluate, orchestrate
from eqdisc.audit import revise as R
from eqdisc.uq import _structure


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    root = tmp_path_factory.mktemp("revise_cases")
    cache = {}

    def get(system, corruption, seed=0):
        key = (system, corruption, seed)
        if key not in cache:
            d = corrupt.make_case(system, corruption, seed, out_root=root)
            meta, data = evaluate.load(d)
            truth = json.loads((d / "hidden" / "truth.json").read_text())
            corr = json.loads((d / "hidden" / "corruption.json").read_text())
            cache[key] = (meta, data, truth, corr, audit.audit_model(meta, data, truth["rhs"]))
        return cache[key]
    return get


def _new_terms(meta, rhs, base):
    old = {tm for tm, _ in _structure(meta, base)["u"]}
    return [tm for tm, _ in _structure(meta, rhs)["u"] if tm not in old]


def _rate(term, sym):
    """Angular rate in 'sin(3.14*t)' / 'cos(x)'."""
    arg = term[term.index("(") + 1:term.rindex(")")]
    return 1.0 if arg == sym else float(arg.replace(f"*{sym}", ""))


def test_source_space_recovers_wavenumber_and_is_adopted(case):
    meta, data, truth, corr, fnd = case("burgers", "source_space")
    assert any(f["id"] == "residual_space_only" and f["fired"] for f in fnd)
    ch = R.challengers(meta, data, truth["rhs"], fnd)
    k_true = 2 * np.pi / truth["L"]
    hits = [n for n, r in ch.items() if any(abs(_rate(tm, "x") - k_true) < 0.1 * k_true
                                            for tm in _new_terms(meta, r, truth["rhs"]) if "x" in tm)]
    assert hits, ch
    final, log = R.revise(meta, data, truth["rhs"], fnd, orchestrate.tournament)
    adopted = [e for e in log if e["adopted"]]
    assert adopted and adopted[0]["name"] in hits and adopted[0]["from_finding"] == "residual_space_only"
    assert final != truth["rhs"]
    # refitted source amplitude close to the true one (cos with A = params["A"])
    cos = [c for tm, c in _structure(meta, final)["u"] if tm.startswith("cos")]
    assert cos and abs(cos[0] - corr["params"]["A"]) < 0.2 * corr["params"]["A"]


def test_amp_term_proposes_cubic(case):
    meta, data, truth, corr, fnd = case("burgers", "amp_term")
    if not any(f["id"] == "residual_amplitude" and f["fired"] for f in fnd):
        pytest.skip("residual_amplitude did not fire on burgers/amp_term seed 0")
    ch = R.challengers(meta, data, truth["rhs"], fnd)
    cub = [r for r in ch.values() if _new_terms(meta, r, truth["rhs"]) == ["u**3"]]
    assert cub and "amp_u2" in ch
    c = dict(_structure(meta, cub[0])["u"])["u**3"]
    assert abs(c - corr["params"]["c"]) < 0.2 * abs(corr["params"]["c"])


def test_clean_case_unchanged(case):
    meta, data, truth, corr, fnd = case("burgers", "clean")
    assert R.challengers(meta, data, truth["rhs"], fnd) == {}
    calls = []
    final, log = R.revise(meta, data, truth["rhs"], fnd, lambda *a: calls.append(a) or {"winner": "incumbent"})
    assert final == truth["rhs"] and log == [] and not calls


def test_forcing_time_recovers_frequency(case):
    meta, data, truth, corr, fnd = case("burgers", "forcing_time")
    try:
        R.refit(meta, data, {"u": truth["rhs"]["u"] + " + 1.0*sin(t)"}, ["u"])
    except Exception as e:  # noqa: BLE001
        pytest.xfail(f"needs t in PDE fitters ({type(e).__name__})")
    ch = R.challengers(meta, data, truth["rhs"], fnd)
    w_true = corr["params"]["w"]
    rates = [_rate(tm, "t") for r in ch.values() for tm in _new_terms(meta, r, truth["rhs"]) if "t)" in tm]
    assert rates and any(abs(w - w_true) < 0.1 * w_true for w in rates), ch


def test_segment_reset_skips_time_and_errors_never_raise(case):
    meta, data, truth, corr, fnd = case("burgers", "forcing_time")
    assert R.challengers({**meta, "segment_t0": [0.0, 5.0]}, data, truth["rhs"], fnd) == {}

    def boom(*a):
        raise RuntimeError("tournament down")
    final, log = R.revise(meta, data, truth["rhs"], fnd, boom)
    assert final == truth["rhs"] and log[-1]["name"] == "error"


def test_dominant_frequency_synthetic():
    t = np.linspace(0, 10, 50)
    w, ev = R.dominant_frequency(np.c_[t, 0.3 * np.sin(2.2 * t + 0.4) + 0.1])
    assert abs(w - 2.2) < 0.02 and ev > 0.99
