"""Smoke test: the whole pipeline without any API calls (Claude replaced by a scripted stand-in).

    python -m eqdisc.tests.smoke
"""
import json
import tempfile
import warnings
from pathlib import Path
from types import SimpleNamespace as NS

warnings.filterwarnings("ignore")


def main():
    from eqdisc import coordinates as co, toolbox as tb
    from eqdisc.agent import run_agent
    from eqdisc.assess import assess
    from eqdisc.datagen import generate
    from eqdisc.evaluate import evaluate, load
    from eqdisc.intuition import intuit
    from eqdisc.weakform import weak_sindy

    tmp = Path(tempfile.mkdtemp())
    d = generate("pendulum", out_root=tmp, noise=0.01, plot=False)
    m, D = load(d)
    assert intuit(m, D)["hypotheses"], "intuition produced no hypotheses"
    s = tb.run_sindy(m, D, poly_degree=1, include_trig=True)
    w = weak_sindy(m, D, poly_degree=1, include_trig=True)
    sk = tb.fit_skeleton(m, D, {"theta": "omega", "omega": "-p0*sin(theta) - p1*omega"})
    print("SINDy", round(evaluate(d, s)["score"], 2), "| weak", round(evaluate(d, w)["score"], 2),
          "| skeleton", round(evaluate(d, sk)["score"], 2), sk["params"])
    a = assess(m, D, sk["rhs"])
    print("assessment:", a["confidence"]["level"], "| experiments:", len(a["experiments"]["ranked"]))

    T = lambda i, n, x: NS(type="tool_use", id=i, name=n, input=x)
    script = [[T("a", "intuit", {})], [T("b", "fit_skeleton", {"rhs_with_params": {"theta": "omega", "omega": "-p0*sin(theta) - p1*omega"}})],
              [T("c", "submit", {"rhs": sk["rhs"], "rationale": "smoke"})]]

    class M:
        i = 0

        def create(self, **kw):
            if "tools" not in kw:
                return NS(content=[NS(type="text", text='{"verdict": "accept", "issues": []}')], stop_reason="end_turn", usage=None)
            c = script[self.i]
            self.i += 1
            return NS(content=c, stop_reason="tool_use", usage=None)

    r = run_agent(d, client=NS(beta=NS(messages=M())), out_dir=tmp / "run", judge_llm=False, verbose=False)
    assert r["submitted"] and Path(r["report"]).exists()
    print("agent (scripted):", round(r["hidden_eval"]["score"], 2), "| report:", r["report"])
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
