"""WS8: glitches vs real events. Glitches are removed (exactly), kicks split, bursts kept, model check never crashes."""
import json

import numpy as np
import pytest

from eqdisc import corrupt
from eqdisc.audit import audit_model
from eqdisc.audit import data as D
from eqdisc.audit import events as E
from eqdisc.audit.repair import audit_and_repair
from eqdisc.datagen import generate
from eqdisc.evaluate import load
from eqdisc.solvers import integrate_ode
from eqdisc.systems import SYSTEMS


@pytest.fixture(scope="module")
def lorenz(tmp_path_factory):
    meta, data = load(generate("lorenz", out_root=tmp_path_factory.mktemp("ws8"), noise=0.0, seed=3, plot=False))
    U = data["U"].astype(float)
    sd = 0.02 * U.reshape(-1, 3).std(0)
    Un = U + sd * np.random.default_rng(0).standard_normal(U.shape)
    return meta, data, U, Un, sd


def by_id(fs):
    return {f["id"]: f for f in fs}


def test_glitch_removed_exactly(lorenz):
    meta, data, U, Un, sd = lorenz
    A = Un.copy()
    pts = [(0, 300, 0), (1, 520, 1), (2, 700, 2), (3, 150, 0)]
    for j, i, v in pts:
        A[j, i, v] += 12 * sd[v]
    d = dict(data, U=A)
    f = by_id(D.audit(meta, d))
    assert f["outliers"]["fired"] and f["outliers"]["fix"]["tool"] == "despike"
    assert not f["external_shock"]["fired"]
    clean, n = D.despike(meta, d, **f["outliers"]["fix"]["args"])
    edited = set(map(tuple, np.argwhere(clean["U"] != A)))
    assert edited == set(pts)


def test_huge_glitch_edits_one_sample(lorenz):
    meta, data, U, Un, sd = lorenz
    A = Un.copy()
    A[0, 500, 0] += 10 * U[..., 0].std()
    clean, n = D.despike(meta, dict(data, U=A))
    assert set(map(tuple, np.argwhere(clean["U"] != A))) == {(0, 500, 0)}


def test_kick_is_shock_not_glitch(lorenz):
    meta, data, U, Un, sd = lorenz
    s = SYSTEMS["lorenz"]
    k = 450
    tail = integrate_ode(s.variables, s.rhs, U[1, k] + np.array([30 * sd[0], 0, 0]), data["t"][k:])
    A = Un.copy()
    A[1, k + 1:] = tail[1:] + (Un[1, k + 1:] - U[1, k + 1:])
    d = dict(data, U=A)
    f = by_id(D.audit(meta, d))
    assert f["external_shock"]["fired"] and f["external_shock"]["fix"]["tool"] == "split_at_events"
    assert "kept, not removed" in f["external_shock"]["message"]
    clean, _ = D.despike(meta, d)
    assert not (clean["U"] != A)[1, k - 8:k + 9].any()
    m2, d2 = D.split_at_events(meta, d, **f["external_shock"]["fix"]["args"])
    assert d2["U"].shape[0] > A.shape[0] and len(m2["segment_t0"]) == d2["U"].shape[0]
    assert not by_id(D.audit(m2, d2))["external_shock"]["fired"]
    # the repair loop routes it
    m3, d3, fs, applied = audit_and_repair(meta, d)
    assert any(a["tool"] == "split_at_events" and a["ok"] for a in applied)
    # model check: the true equation cannot produce a kick (info by default), runs fast and never crashes
    fm = by_id(E.audit(meta, d, s.rhs))
    assert "event_unexplained" in fm and fm["event_unexplained"]["severity"] in ("info", "warn")


def test_smooth_burst_kept(lorenz):
    meta, data, U, Un, sd = lorenz
    A = Un.copy()
    k = np.arange(-4, 5)
    A[2, 600 + k, 1] += 6 * U[..., 1].std() * np.exp(-0.5 * (k / 1.5) ** 2)
    d = dict(data, U=A)
    f = by_id(D.audit(meta, d))
    assert not f["outliers"]["fired"]
    assert f["extreme_event"]["fired"] and f["extreme_event"]["severity"] == "info"
    clean, n = D.despike(meta, d)
    assert n == 0


def test_clean_quiet_and_model_check_safe(lorenz):
    meta, data, U, Un, sd = lorenz
    d = dict(data, U=Un)
    f = by_id(D.audit(meta, d))
    assert not any(f[k]["fired"] for k in ("outliers", "external_shock", "extreme_event"))
    fs = E.audit(meta, d, SYSTEMS["lorenz"].rhs)
    assert len(fs) == 1 and not fs[0]["fired"]
    bad = E.audit(meta, dict(d, U=np.where(np.arange(U.shape[1])[None, :, None] == 5, np.nan, Un)), {"x": "1/0"})
    json.dumps(bad)
    assert any(x["id"].startswith("event") for x in audit_model(meta, d, SYSTEMS["lorenz"].rhs))


@pytest.mark.parametrize("corruption", corrupt.EVENT_CORRUPTIONS)
def test_event_corruptions(tmp_path, corruption):
    dd = corrupt.make_case("kuramoto_sivashinsky", corruption, seed=0, out_root=tmp_path, t_end=40.0)
    meta, data = load(dd)
    corr = json.loads((dd / "hidden" / "corruption.json").read_text())
    assert corr["expected_handling"] == corrupt.HANDLING[corruption] and corr["events"]
    f = by_id(D.audit(meta, data))
    clean, n = D.despike(meta, data)
    edited = set(map(tuple, np.argwhere(clean["U"] != data["U"])))
    if corruption == "glitch":
        truth = {(j, i, x, v) for j, i, x, v, _ in corr["events"]}
        assert f["outliers"]["fired"] and edited and edited <= truth
    else:
        assert not f["outliers"]["fired"] and not edited
