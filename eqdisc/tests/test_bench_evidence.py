"""Fast tests for the evidence benchmark scorer (no API, no runs)."""
import json

from eqdisc import bench_evidence as be

TRUTH = {"kind": "pde", "variables": ["u"], "spatial_dims": ["x"], "rhs": {"u": "-u*u_x + 0.05*u_xx"}}


def rec(right, status, findings=(), crashed=False, headline="x", failed=None):
    v = {"status": status, "headline": headline}
    if failed:
        v["failed_checks"] = failed
    r = {"crashed": crashed, "right": right, "confident": be.is_confident(v), "flagged": be.is_flagged(v, list(findings))}
    r["category"] = be.classify(r)
    return r


def test_classify():
    crit = [{"id": "gaps", "fired": True, "severity": "critical"}]
    assert rec(True, "CONFIDENT")["category"] == "right+confident"
    assert rec(True, "CONFIDENT IN PREDICTIONS")["category"] == "right+confident"
    assert rec(True, "COLLECT MORE DATA")["category"] == "right+cautious"
    assert rec(False, "CONFIDENT", crit)["category"] == "wrong+confident"       # a finding does not excuse confidence
    assert rec(False, "INCONCLUSIVE", failed=["gaps"])["category"] == "wrong+flagged"
    assert rec(False, "INCONCLUSIVE", headline="No assessment available.")["category"] == "wrong+unflagged"
    assert rec(False, "INCONCLUSIVE", crit, headline="No assessment available.")["category"] == "wrong+flagged"
    assert rec(False, "", crashed=True)["category"] == "crashed"


def test_structural_right():
    names = ["u", "u_x", "u_xx", "u_xxx", "u_xxxx", "x"]
    t = TRUTH["rhs"]
    ok, _ = be.structural_right(t, {"u": "-1.02*u*u_x + 0.051*u_xx + 0.1*sin(1.3*t)"}, names, "forcing")
    assert ok
    ok, d = be.structural_right(t, {"u": "-u*u_x + 0.05*u_xx + 0.3*u"}, names, "forcing")
    assert not ok and d["bad_extra"]
    ok, d = be.structural_right(t, {"u": "-u*u_x"}, names, "forcing")
    assert not ok and d["missing"]
    ok, d = be.structural_right(t, {"u": "-1.5*u*u_x + 0.05*u_xx"}, names, "forcing")
    assert not ok and d["bad_coef"]
    assert be.structural_right(t, {"u": "-u*u_x + 0.05*u_xx - 0.4*u**3"}, names, "amp", fields=["u"])[0]
    assert be.structural_right(t, {"u": "-u*u_x + 0.05*u_xx"}, names, "amp", fields=["u"])[0]
    assert not be.structural_right(t, {"u": "-u*u_x + 0.05*u_xx + 0.1*u**2"}, names, "amp", fields=["u"])[0]


def test_correctness_judge_sympy(tmp_path, monkeypatch):
    monkeypatch.setattr(be, "ROOT", tmp_path)
    assert be.correctness(TRUTH, {"u": "-1.01*u*u_x + 0.0502*u_xx"}, "clean", use_llm=False)["right"]
    assert not be.correctness(TRUTH, None, "clean", use_llm=False)["right"]
    assert be.correctness(TRUTH, {"u": "-u*u_x + 0.05*u_xx + 0.2*cos(x)"}, "source_space")["right"]


def test_table_and_outcomes(tmp_path):
    p = tmp_path / "o.jsonl"
    base = {"split": "report", "corruption": "clean", "system": "kdv", "f1": 1.0, "cost_usd": 0.5, "wall_s": 100,
            "exact_structure": True, "right": True, "confident": True, "category": "right+confident"}
    be.append_outcome(dict(base, arm="A", case="c1", category="wrong+confident", right=False), p)
    be.append_outcome(dict(base, arm="A", case="c1"), p)                     # idempotent: latest wins
    be.append_outcome(dict(base, arm="A", case="c2", corruption="gaps_state", category="wrong+confident", right=False), p)
    be.append_outcome(dict(base, arm="B", case="c2", corruption="gaps_state", category="wrong+flagged", right=False,
                           confident=False), p)
    recs = be.load_outcomes(p)
    assert len(recs) == 3
    t = be.table(recs)
    assert "| A | report | all | 2 | 1 | 0 | 0 | 1 | 0 | 0 | 0 | 1/2 (50%)" in t
    assert "| A | report | gaps | 1 | 0 | 0 | 0 | 1 |" in t
    assert "| B | report | all | 1 |" in t and "gaps_state" in t


def test_select_and_names():
    idx = [{"split": "dev", "seed": 0, "system": "kdv", "corruption": "clean", "path": "p"},
           {"split": "report", "seed": 10, "system": "kdv", "corruption": "amp_term", "path": "q"}]
    s = be.select(idx, "report", [10])
    assert len(s) == 1 and be.case_name(s[0]) == "report_kdv_amp_term_s10"
    json.dumps(be.provenance())


def test_dynamic_full_vs_base():
    full = {"u": "-u*u_x + 0.05*u_xx + 0.06*sin(1.2566*t)"}
    base_only = {"u": "-1.01*u*u_x + 0.05*u_xx"}
    with_forcing = {"u": "-1.01*u*u_x + 0.05*u_xx + 0.05*sin(1.26*t) + 0.01*cos(1.26*t)"}
    wrong_freq = {"u": "-u*u_x + 0.05*u_xx + 0.05*sin(2.0*t)"}
    assert be.full_right(TRUTH, full, with_forcing, "forcing_time")[0]
    assert not be.full_right(TRUTH, full, base_only, "forcing_time")[0]
    assert not be.full_right(TRUTH, full, wrong_freq, "forcing_time")[0]
    c = be.correctness(TRUTH, base_only, "forcing_time", full_rhs=full)
    assert c["right_base"] and not c["right"]
    src = {"u": "-u*u_x + 0.05*u_xx + 0.05*cos(1.0*x)"}
    assert be.full_right(TRUTH, src, {"u": "-u*u_x + 0.05*u_xx + 0.04*cos(x)"}, "source_space")[0]
    amp = {"u": "-u*u_x + 0.05*u_xx - 0.4*u**3"}
    assert be.full_right(TRUTH, amp, {"u": "-u*u_x + 0.05*u_xx - 0.3*u**3"}, "amp_term")[0]
    assert not be.full_right(TRUTH, amp, {"u": "-u*u_x + 0.05*u_xx + 0.3*u**3"}, "amp_term")[0]
    # incomplete law: base-only model
    r = {"right": False, "right_base": True, "confident": True, "flagged": False}
    assert be.classify(r) == "wrong+confident" and be.classify(r, base_view=True) == "right+confident"
    r = {"right": False, "right_base": True, "confident": False, "flagged": True}
    assert be.classify(r) == "wrong+flagged" and be.classify(r, base_view=True) == "right+cautious"
    r = {"right": False, "right_base": True, "confident": False, "flagged": False}
    assert be.classify(r) == "incomplete+cautious"
