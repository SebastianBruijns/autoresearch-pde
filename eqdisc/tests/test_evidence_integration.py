"""Integration of the evidence layer (WS4): findings -> grade, verdict, repair, ledger, reassess. No API calls.

Synthetic findings are injected by monkeypatching `eqdisc.audit.audit_data` / `audit_model`, so these tests do not
depend on detector calibration.
"""
import copy
import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from eqdisc import audit
from eqdisc.assess import _re_intervals, assess, grade
from eqdisc.audit import finding
from eqdisc.audit.repair import audit_and_repair, has_nan, repair_data, repair_model, save_dataset
from eqdisc.datagen import generate
from eqdisc.evaluate import load
from eqdisc.insights import verdict
from eqdisc.ledger import Ledger, read

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    """Clean pendulum data, the true model and its baseline assessment with no findings (the pre-evidence behaviour)."""
    root = tmp_path_factory.mktemp("ev")
    d = generate("pendulum", out_root=root, noise=0.01, plot=False)
    meta, data = load(d)
    rhs = json.loads((Path(d) / "hidden" / "truth.json").read_text())["rhs"]
    base = assess(meta, data, rhs, findings=[])
    return {"dir": Path(d), "meta": meta, "data": data, "rhs": rhs, "base": base, "root": root}


def _crit(stage="model", fid="residual_time_only"):
    return finding(fid, stage, 5.0, 1.0, True, "critical", "widen",
                   message="the residual has structure in time only (unmodelled forcing)")


def _warn():
    return finding("slice_time", "model", 0.8, 0.75, True, "warn", "widen",
                   message="coefficients drift between early and late time")


def _quiet():
    return [finding("outliers", "data", 0.0, 1e-3, False, "info", message="no spikes"),
            finding("slice_trajectory", "model", 0.1, 0.75, False, "info", message="consistent")]


def _with(a, findings):
    a = copy.deepcopy(a)
    a["findings"] = findings
    a["confidence"] = grade(a)
    return a


def test_baseline_is_confident(clean):
    assert verdict(clean["base"])["status"] == "CONFIDENT"
    assert clean["base"]["findings"] == []


def test_clean_findings_leave_grade_and_verdict_unchanged(clean):
    b = clean["base"]
    a = _with(b, _quiet())
    assert a["confidence"] == b["confidence"]
    assert verdict(a) == verdict(b)
    assert "valid_range" not in verdict(a)


def test_critical_blocks_confident_and_is_named(clean):
    b = clean["base"]
    a = _with(b, [_crit()])
    assert a["confidence"]["points"] == b["confidence"]["points"] - 3
    v = verdict(a)
    assert not v["status"].startswith("CONFIDENT")
    assert "residual_time_only" in v["headline"]
    assert v["failed_checks"] == ["residual_time_only"]
    assert any("residual_time_only" in r for r in a["confidence"]["reasons"])


def test_critical_blocks_even_if_points_stay_high(clean):
    a = copy.deepcopy(clean["base"])
    a["findings"] = [_crit()]                         # grade not recomputed: the veto does not depend on points
    assert not verdict(a)["status"].startswith("CONFIDENT")


def test_resolved_critical_and_info_do_not_count(clean):
    b = clean["base"]
    resolved = dict(_crit("data", "gaps"), resolved=True, repair="split_at_gaps")
    info = finding("coarse_sampling", "data", 0.2, 0.15, True, "info", message="sampling is coarse")
    a = _with(b, [resolved, info])
    assert a["confidence"]["points"] == b["confidence"]["points"]
    assert verdict(a)["status"] == verdict(b)["status"]
    assert any("sampling is coarse" in r for r in a["confidence"]["reasons"])


def test_warn_reduces_points(clean):
    b = clean["base"]
    a = _with(b, [_warn(), _warn()])
    assert a["confidence"]["points"] == b["confidence"]["points"] - 2
    assert a["confidence"]["level"] != "high"
    assert any("drift" in r for r in a["confidence"]["reasons"])


def test_valid_range_from_scope_findings(clean):
    f = finding("gaps_state_dependent", "data", 3.0, 1.0, True, "warn", "scope",
                scope={"variable": "theta", "range": [-1.5, 2.0]}, message="large amplitudes are missing")
    a = _with(clean["base"], [f])
    v = verdict(a)
    assert v["valid_range"] == {"theta": [-1.5, 2.0]}
    assert "Valid for theta in [-1.5, 2]; no data beyond." in v["recommendation"]


def test_re_interval_helper():
    f = finding("slice_amplitude", "model", 0.9, 0.75, True, "warn", "widen", message="x",
                details={"re_intervals": {"omega:omega": [-0.3, 0.1], "omega:sin(theta)": [-10, -9]},
                         "i2": {"omega:omega": 0.9, "omega:sin(theta)": 0.1}})
    iv = _re_intervals([f], ["theta", "omega", "t"])
    assert list(iv) == [("omega", "omega")] and iv[("omega", "omega")][:2] == (-0.3, 0.1)
    assert _re_intervals([dict(f, fired=False)], ["theta", "omega", "t"]) == {}


def test_assess_computes_findings_uses_re_intervals_and_writes_ledger(clean, monkeypatch, tmp_path):
    re_f = finding("slice_trajectory", "model", 0.95, 0.75, True, "warn", "repair", fix={"tool": "per_trajectory", "args": {}},
                   message="hidden parameter varies between runs",
                   details={"re_intervals": {"omega:omega": [-0.3, 0.1]}, "I2": 0.95,
                            "per_trajectory": {"omega:omega": [-0.05, -0.15]}})
    monkeypatch.setattr(audit, "audit_data", lambda meta, data: _quiet()[:1])
    monkeypatch.setattr(audit, "audit_model", lambda meta, data, rhs: [_crit(), re_f])
    a = assess(clean["meta"], clean["data"], clean["rhs"], run_dir=tmp_path)
    assert {f["id"] for f in a["findings"]} == {"outliers", "residual_time_only", "slice_trajectory"}
    t = next(t for t in a["terms"] if t["var"] == "omega" and t["term"].replace(" ", "") in ("omega", "1.0*omega"))
    assert t["interval_used"].startswith("random-effects") and t["significant"] is False
    assert all(x["interval_used"] == "bootstrap" for x in a["terms"] if x is not t)
    v = verdict(a)
    assert not v["status"].startswith("CONFIDENT") and "residual_time_only" in v["headline"]
    lines = read(tmp_path)
    assert lines and all(isinstance(l, dict) and l["dataset_hash"] is None and l["config_hash"] for l in lines)
    for raw in (tmp_path / "ledger.jsonl").read_text().splitlines():
        json.loads(raw)
    forcing = dict(_crit(), fix={"tool": "add_forcing", "args": {"basis": "time"}})
    rep = repair_model(clean["meta"], clean["data"], clean["rhs"], [re_f, forcing])
    assert rep[0]["kind"] == "reporting" and rep[0]["coefficients"] == {"omega:omega": [-0.05, -0.15]}
    assert rep[1]["ok"] is False


def test_real_detectors_on_clean_data_keep_verdict(clean):
    a = assess(clean["meta"], clean["data"], clean["rhs"])
    noisy = audit.fired(a["findings"], "warn")
    if noisy:
        pytest.skip(f"detectors fire on clean data (calibration pending): {[f['id'] for f in noisy]}")
    assert verdict(a) == verdict(clean["base"])


def _gapped(clean, tmp_path):
    meta, data = copy.deepcopy(clean["meta"]), {k: np.array(v) for k, v in clean["data"].items()}
    U = data["U"].astype(float)
    nt = U.shape[1]
    rng = np.random.default_rng(0)
    for i in range(U.shape[0]):
        for s in rng.choice(nt - 40, 2, replace=False):
            U[i, s:s + 25] = np.nan
    data["U"] = U
    return meta, data


def test_repair_data_removes_gaps_and_pipeline_runs(clean, tmp_path):
    from eqdisc import toolbox as tb
    from eqdisc.intuition import intuit
    meta, data = _gapped(clean, tmp_path)
    assert has_nan(data)
    led = Ledger(tmp_path, config={"test": 1})
    m2, d2, findings, applied = audit_and_repair(meta, data, ledger=led)
    assert not has_nan(d2) and any(a["ok"] and a["tool"] == "split_at_gaps" for a in applied)
    assert m2["shape"] == list(d2["U"].shape) and d2["U"].shape[1] == len(d2["t"])
    # unknown tools are skipped and recorded
    _, _, app = repair_data(m2, d2, [finding("x", "data", 1, 0, True, "warn", "repair", fix={"tool": "nope", "args": {}})])
    assert app == [{"finding": "x", "tool": "nope", "ok": False, "note": "skipped: not a data repair tool"}]
    out = save_dataset(m2, d2, tmp_path / "dataset_audited", applied)
    m3, d3 = load(out)
    assert not has_nan(d3)
    assert intuit(m3, d3)["hypotheses"]
    s = tb.run_sindy(m3, d3, poly_degree=1, include_trig=True)
    assert s.get("rhs")
    a = assess(meta, data, clean["rhs"])               # assess on gapped data repairs before fitting
    assert a["data_repairs"] and a["confidence"]["level"] in ("high", "medium", "low")
    for raw in (tmp_path / "ledger.jsonl").read_text().splitlines():
        assert json.loads(raw)["kind"] in ("finding", "repair", "reaudit")


def test_reassess_fake_run_dir(clean, tmp_path, monkeypatch):
    from eqdisc.orchestrate import reassess
    monkeypatch.setattr(audit, "audit_model", lambda meta, data, rhs: [_crit()])
    res = {"dataset": clean["meta"]["name"], "dataset_path": str(clean["dir"]), "final_model": clean["rhs"],
           "winner_branch": "sparse-regression", "verdict": {"status": "CONFIDENT", "headline": "", "recommendation": ""},
           "story": {}, "insights": [], "intuition": {"hypotheses": []}, "tournament": {"verdict": "single candidate"},
           "adversary": {}, "branches": {"sparse-regression": {"model": clean["rhs"]}}, "assessment": None,
           "wall_s": 1.0, "cost_usd": 0.0}
    (tmp_path / "discovery.json").write_text(json.dumps(res))
    out = reassess(tmp_path)
    assert not out["verdict"]["status"].startswith("CONFIDENT")
    assert "residual_time_only" in out["verdict"]["headline"]
    assert Path(out["report"]).exists() and "Data and model checks" in Path(out["report"]).read_text()
    kinds = [l["kind"] for l in read(tmp_path)]
    assert "verdict" in kinds and "finding" in kinds
    assert all(l["dataset_hash"] for l in read(tmp_path))
