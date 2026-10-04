"""Regression: the data repair must never throw away real dynamics.

PR #5 labelled chaotic Kuramoto-Sivashinsky evolution as "external shocks" and cut whole time rows around them,
keeping 66-83% of the record. Without NaN, repair may only change values (clip isolated glitches); it never removes
rows, and on clean data it changes nothing.
"""
import numpy as np
import pytest

from eqdisc.audit.repair import audit_and_repair
from eqdisc.corrupt import make_case
from eqdisc.datagen import generate
from eqdisc.evaluate import load


@pytest.fixture(scope="module")
def ks(tmp_path_factory):
    root = tmp_path_factory.mktemp("ks")
    return {"clean0": load(generate("kuramoto_sivashinsky", out_root=root, noise=0.0, seed=7, plot=False)),
            "clean2": load(generate("kuramoto_sivashinsky", out_root=root, noise=0.02, seed=7, plot=False)),
            "outliers": load(make_case("kuramoto_sivashinsky", "outliers", 2, out_root=root))}   # dev seed


@pytest.mark.parametrize("name", ["clean0", "clean2"])
def test_clean_ks_untouched(ks, name):
    meta, data = ks[name]
    U0 = np.asarray(data["U"], float)
    _, d2, _, _ = audit_and_repair(meta, data)
    assert np.array_equal(np.asarray(d2["U"]), U0)


def test_ks_outliers_keep_every_row(ks):
    meta, data = ks["outliers"]
    U0 = np.asarray(data["U"], float)
    _, d2, _, _ = audit_and_repair(meta, data)
    U2 = np.asarray(d2["U"])
    assert U2.shape == U0.shape
    assert (U2 != U0).mean() < 0.02          # 1% of samples are spikes; only (some of) those may change
