"""The time `t` is usable in PDE right-hand sides (explicitly forced models such as + A*sin(w*t)): strong form
(feature_arrays, validate, uq.compare_models incl. rollouts), weak form (weak_sindy) and the model checks
(audit_model). Default libraries must not change. Data: a short forced Burgers case from eqdisc.corrupt."""
import hashlib
import json

import numpy as np
import pytest

from eqdisc import corrupt, toolbox, uq, weakform
from eqdisc.audit import audit_model
from eqdisc.evaluate import load
from eqdisc.solvers import integrate_pde


@pytest.fixture(scope="module")
def forced(tmp_path_factory):
    d = corrupt.make_case("burgers", "forcing_time", 0, out_root=tmp_path_factory.mktemp("corrupt"), t_end=2.0)
    meta, data = load(d)
    corr = json.loads((d / "hidden" / "corruption.json").read_text())
    base = json.loads((d / "hidden" / "truth.json").read_text())["rhs"]
    return meta, data, corr["rhs_test"], base, corr["params"]["w"]


def test_t_is_a_symbol_and_feature(forced):
    meta, data, full, base, _ = forced
    assert "t" not in meta["allowed_symbols"] and "t" in toolbox.symbols(meta)
    U, t = data["U"], data["t"]
    f = toolbox.feature_arrays(meta, U, t)
    assert f["t"].shape == U.shape[:-1] and np.allclose(f["t"][1, :, 7], t)
    v = toolbox.validate(meta, data, full)
    assert "error" not in v and v["rollout_nrmse_full"] < toolbox.validate(meta, data, base)["rollout_nrmse_full"]
    # 2-D / non-periodic path: t is a feature too
    m2 = {"kind": "pde", "variables": ["u"], "spatial_dims": ["x"], "boundary": "dirichlet",
          "grid": {"x": {"n": 32, "L": 1.0}}, "dt": 0.1}
    f2 = toolbox.feature_arrays(m2, np.random.default_rng(0).normal(size=(2, 5, 32, 1)), np.arange(5) * 0.1)
    assert np.allclose(f2["t"][0, :, 3], np.arange(5) * 0.1)


def test_compare_models_prefers_forced_rhs(forced):
    meta, data, full, base, _ = forced
    r = uq.compare_models(meta, data, {"full": full, "base": base})
    assert r["ranking"][0] == "full" and r["preferred"] == "full"
    assert r["candidates"]["full"]["cv_deriv_nrmse"] < r["candidates"]["base"]["cv_deriv_nrmse"]


def test_rollout_uses_absolute_time(forced):
    meta, data, full, _, _ = forced
    prep = uq._prepare(meta, data)
    te, Us = prep["te"], prep["Us"]
    a, n = len(te) // 2, 40
    h = toolbox._pde_step(meta, prep["U"])
    sub = max(1, int(round(meta["dt"] / h)))
    err = {}
    for name, tt in (("abs", te[a:a + n]), ("shifted", te[a:a + n] - te[a])):
        roll = integrate_pde(meta["variables"], full, meta["L"], Us[0, a], tt, meta["dt"] / sub)
        err[name] = float(np.sqrt(np.mean((roll - Us[0, a:a + n]) ** 2)))
    assert err["abs"] < 0.5 * err["shifted"]
    assert uq._rollout(prep, full, (0, a, a + n))["valid_frac"] == 1.0


def test_weak_sindy_recovers_forcing(forced):
    meta, data, _, _, w = forced
    res = weakform.weak_sindy(meta, data, custom_terms=[f"sin({w}*t)", f"cos({w}*t)"])
    rhs = res["rhs"]["u"]
    coef = float(rhs.split(f"*sin({w}*t)")[0].split()[-1])
    assert 0.5 * 0.08 < coef < 1.5 * 0.08, rhs


def test_audit_model_runs_on_forced_rhs(forced):
    meta, data, full, base, _ = forced
    f = audit_model(meta, data, full)
    assert f and not [x for x in f if x["id"].endswith("_error")]
    assert not [x for x in f if x["id"] == "residual_time_only" and x["fired"]]
    fb = audit_model(meta, data, base)
    assert [x for x in fb if x["id"] == "residual_time_only" and x["fired"]]


def test_default_libraries_unchanged(forced):
    """Snapshot of the 1-D PDE default term lists (run_sindy / weak_sindy) taken before `t` became a symbol."""
    meta, data, _, _, _ = forced
    U, t = data["U"][:, :20], data["t"][:20]
    strong, _ = toolbox.build_library(meta, toolbox.feature_arrays(meta, U, t))
    weak = weakform._finalise_terms(weakform._pde_library_terms(meta, ["u"], ["x"], 3, 4), (), (),
                                    toolbox.symbols(meta))
    for terms in (strong, weak):
        assert len(terms) == 16 and not any("t" in tm.replace("u_", "").replace("x", "") for tm in terms)
        assert hashlib.sha1("|".join(terms).encode()).hexdigest() == "7c7060190b614dba5b46ec04f852a39203370a8e"
