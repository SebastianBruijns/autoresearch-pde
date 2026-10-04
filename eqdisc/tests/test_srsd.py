"""Offline tests of the static (SRSD) mode: no network, no API calls, no PySR.

    python -m eqdisc.tests.test_srsd
"""
import numpy as np

from eqdisc import srsd


def _problem(truth, ranges, n=3000, seed=0, dummy=0):
    """Synthetic SRSD-style problem: inputs sampled independently (log-uniform for positive ranges)."""
    rng = np.random.default_rng(seed)
    k = len(ranges) + dummy
    names = [f"x{i}" for i in range(k)]

    def sample(m):
        cols = [10 ** rng.uniform(np.log10(lo), np.log10(hi), m) if lo > 0 else rng.uniform(lo, hi, m) for lo, hi in ranges]
        cols += [rng.uniform(1, 10, m) for _ in range(dummy)]
        X = np.column_stack(cols)
        return X, srsd.predict(truth, X, names)
    p = {"name": "synthetic", "truth": truth, "n_vars": k, "variables": names, "descriptions": {}}
    for split, m in (("train", n), ("val", n // 8), ("test", n // 8)):
        p[f"X_{split}"], p[f"y_{split}"] = sample(m)
    return p


def test_power_law_and_skeleton():
    p = _problem("6.674e-11*x0*x1/x2**2", [(1, 100), (1, 100), (0.1, 10)])
    m, d = srsd.public_view(p)
    pl = srsd.diagnose_static(m, d)["power_law_fit"]
    assert pl["r2_in_log_space"] > 0.999 and abs(pl["exponents"]["x2"] + 2) < 1e-3
    sk = srsd.skeleton_static(m, d, "p0*x0*x1/x2**2")
    ev = srsd.evaluate_static(p, sk["expr"])
    assert ev["symbolic_match"] and ev["numeric_exact"], ev


def test_tiny_constant_near_domain_wall():
    # relativistic factor: the residual has a needle minimum at 1/c^2 ~ 1.1e-17 next to a sqrt domain wall
    p = _problem("(x0 - x1*x2)/sqrt(1 - 1.11265e-17*x1**2)", [(-50, 50), (-1e8, 1e8), (1e-6, 1e-5)])
    m, d = srsd.public_view(p)
    sk = srsd.skeleton_static(m, d, "(x0 - p0*x1*x2)/sqrt(1 - p1*x1**2)")
    assert sk["validation"]["rel_err_median"] < 1e-6, sk
    assert srsd.evaluate_static(p, sk["expr"])["symbolic_match"]


def test_equivalence_negatives():
    p = _problem("sqrt(2)*exp(-x0**2/(2*x1**2))/(2*sqrt(pi)*x1)", [(-3, 3), (0.5, 3)])
    assert srsd.evaluate_static(p, "0.398942*exp(-0.5*x0**2/x1**2)/x1")["symbolic_match"]
    assert not srsd.evaluate_static(p, "0.4*exp(-0.5*x0**2/x1**2)/x1")["symbolic_match"]
    assert not srsd.evaluate_static(p, "0.398942*exp(-0.5*x0**2/x1**2)/x1**2")["symbolic_match"]


def test_dependence_flags_dummy():
    p = _problem("x0**2*x1/x2", [(1, 10), (1, 10), (1, 10)], dummy=1)
    m, d = srsd.public_view(p)
    v = {r["variable"]: r["verdict"] for r in srsd.dependence_static(m, d)["inputs"]}
    assert v["x3"] == "irrelevant?" and all(v[f"x{i}"] != "irrelevant?" for i in range(3)), v


def test_agent_loop_dry_run():
    p = _problem("3.0*x0**2*x1", [(1, 10), (1, 10)])
    r = srsd.solve_problem(p, {"max_tools": 5, "max_cost_usd": 5.0}, client=srsd.FakeClient())
    assert r["submitted"] and r["eval"]["symbolic_match"], r.get("eval")
    assert abs(r["cost_usd"] - 0.33) < 1e-6          # 3 scripted calls x (20k in, 1.5k out) on Opus 5.5 prices


def test_cost_cap_stops_session():
    p = _problem("3.0*x0**2*x1", [(1, 10), (1, 10)])
    r = srsd.solve_problem(p, {"max_tools": 5, "max_cost_usd": 0.15}, client=srsd.FakeClient())
    assert r["stop"] == "cost cap reached" and r["usage"]["llm_calls"] == 2, (r["stop"], r["usage"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("SRSD TESTS PASSED")
