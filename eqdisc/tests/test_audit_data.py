"""WS1 data audit: detectors fire on their corruption, stay quiet on clean data; repair tools work."""
import json

import numpy as np
import pytest
from scipy.ndimage import uniform_filter

from eqdisc import toolbox as tb
from eqdisc import weakform as wf
from eqdisc.audit import audit_data
from eqdisc.audit import data as D
from eqdisc.datagen import add_noise, generate
from eqdisc.evaluate import load


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    root = tmp_path_factory.mktemp("ws1")
    out = {}
    for name, system, noise in (("ode", "lorenz", 0.02), ("pde", "kuramoto_sivashinsky", 0.02),
                                ("pde0", "kuramoto_sivashinsky", 0.0)):
        out[name] = load(generate(system, out_root=root, noise=noise, seed=7, plot=False))
    return out


def by_id(findings):
    return {f["id"]: f for f in findings}


def with_U(data, U):
    d = dict(data)
    d["U"] = U
    return d


def gaps_random(U, rng, frac=0.10):
    U = U.astype(float).copy()
    nt = U.shape[1]
    for j in range(U.shape[0]):
        rows = np.zeros(nt, bool)
        while rows.mean() < frac:
            L = max(2, int(rng.uniform(0.02, 0.05) * nt))
            s = rng.integers(0, nt - L)
            rows[s:s + L] = True
        U[j, rows] = np.nan
    return U


def gaps_state(U, q=0.8, w=5):
    """NaN where the local |u| is in the top 20% (censored tails)."""
    U = U.astype(float).copy()
    a = uniform_filter(np.abs(U), size=(1, w) + (w,) * (U.ndim - 3) + (1,), mode="nearest")
    U[a > np.quantile(a.reshape(-1, U.shape[-1]), q, axis=0)] = np.nan
    return U


def spikes(U, rng, level=0.02):
    """The spike part of datagen.add_noise(kind="outliers") added to already-noisy data."""
    sc = U.reshape(-1, U.shape[-1]).std(0)
    m = rng.random(U.shape) < 0.01
    return np.where(m, U + 10 * level * sc * rng.choice([-1, 1], U.shape), U)


@pytest.mark.parametrize("name", ["ode", "pde"])
def test_clean_nothing_fires(ds, name):
    meta, data = ds[name]
    fs = audit_data(meta, data)
    json.dumps(fs)
    assert not any(f["id"].endswith("_error") for f in fs)
    assert not [f for f in fs if f["fired"]], [f["message"] for f in fs if f["fired"]]
    assert D.despike(meta, data)[1] == 0


@pytest.mark.parametrize("name", ["ode", "pde"])
def test_outliers_fire_and_despike(ds, name):
    meta, data = ds[name]
    U = data["U"]
    Us = spikes(U, np.random.default_rng(0))
    f = by_id(audit_data(meta, with_U(data, Us)))["outliers"]
    assert f["fired"] and f["response"] == "repair" and f["fix"]["tool"] == "despike"
    clean, n = D.despike(meta, with_U(data, Us), **f["fix"]["args"])
    assert n > 0.5 * (Us != U).sum()
    err = lambda A: np.sqrt(np.mean((A - U) ** 2))  # noqa: E731
    assert err(clean["U"]) < 0.4 * err(Us)
    assert not by_id(audit_data(meta, clean))["outliers"]["fired"]


def test_outliers_datagen_kind(ds):
    meta, data = ds["ode"]
    rng = np.random.default_rng(1)
    U = add_noise(data["U"], 0.02, rng, "outliers")
    assert by_id(audit_data(meta, with_U(data, U)))["outliers"]["fired"]


@pytest.mark.parametrize("name", ["ode", "pde"])
def test_random_gaps(ds, name):
    meta, data = ds[name]
    for seed in range(4):
        U = gaps_random(data["U"], np.random.default_rng(seed))
        f = by_id(audit_data(meta, with_U(data, U)))
        assert f["gaps"]["fired"] and f["gaps"]["fix"]["tool"] == "split_at_gaps"
        assert not f["gaps_state_dependent"]["fired"], f["gaps_state_dependent"]["message"]
        assert not f["outliers"]["fired"]


@pytest.mark.parametrize("name", ["ode", "pde"])
def test_state_dependent_gaps(ds, name):
    meta, data = ds[name]
    f = by_id(audit_data(meta, with_U(data, gaps_state(data["U"]))))
    g = f["gaps_state_dependent"]
    assert g["fired"] and g["severity"] == "critical" and g["response"] == "scope"
    lo, hi = g["scope"]["range"]
    assert lo < hi
    assert "under-sampled" in g["message"]


@pytest.mark.parametrize("name", ["ode", "pde"])
def test_split_at_gaps_feeds_fitters(ds, name):
    meta, data = ds[name]
    U = gaps_random(data["U"], np.random.default_rng(3))
    meta2, data2 = D.split_at_gaps(meta, with_U(data, U), min_len=D._default_min_len(meta))
    U2 = data2["U"]
    assert np.isfinite(U2).all()
    assert meta2["n_traj"] == U2.shape[0] and meta2["shape"] == list(U2.shape)
    assert len(data2["t"]) == U2.shape[1] and len(meta2["segment_t0"]) == U2.shape[0]
    assert U2.shape[2:] == data["U"].shape[2:]
    assert U2.shape[1] >= 20
    for fit in (tb.run_sindy, wf.weak_sindy):
        res = fit(meta2, data2)
        assert set(res["rhs"]) == set(meta["variables"])
    assert not by_id(audit_data(meta2, data2))["gaps"]["fired"]


def test_split_prefers_long_pieces(ds):
    meta, data = ds["ode"]
    U = data["U"].copy()
    U[:, 100:140] = np.nan
    meta2, data2 = D.split_at_gaps(meta, with_U(data, U))
    assert data2["U"].shape[0] == U.shape[0] and data2["U"].shape[1] >= 0.8 * (U.shape[1] - 140)
    assert all(t0 == pytest.approx(data["t"][140]) for t0 in meta2["segment_t0"])


def test_split_no_nan_is_identity(ds):
    meta, data = ds["ode"]
    meta2, data2 = D.split_at_gaps(meta, data)
    assert data2["U"].shape == data["U"].shape


def test_holes_in_every_snapshot(ds):
    """PDE with scattered NaNs in every snapshot: split cannot help; no crash, no fix offered."""
    meta, data = ds["pde"]
    U = data["U"].astype(float).copy()
    U[:, :, ::17] = np.nan
    f = by_id(audit_data(meta, with_U(data, U)))
    assert f["gaps"]["fired"] and f["gaps"]["fix"] is None
    assert not any(k.endswith("_error") for k in f)
    with pytest.raises(ValueError):
        D.split_at_gaps(meta, with_U(data, U))


def test_existing_signals(ds):
    meta, data = ds["ode"]
    f = by_id(audit_data(meta, with_U(data, data["U"][:1])))
    assert f["single_trajectory"]["fired"]
    m = dict(meta, dt=meta["dt"] * 25)
    d = dict(data, U=data["U"][:, ::25], t=data["t"][::25])
    assert by_id(audit_data(m, d))["coarse_sampling"]["fired"]
    meta0, data0 = ds["pde0"]    # noise-free KS: spectrum reaches the grid scale; no false spikes
    f0 = by_id(audit_data(meta0, data0))
    assert f0["grid_scale_signal"]["fired"]
    assert not f0["outliers"]["fired"]
