"""Symbolic-equivalence judge (LLM-SRBench / KeplerAgent 'symbolic accuracy').

judge(truth_rhs, cand_rhs) -> {"equivalent": bool, "per_var": {...}, "method": "sympy"|"llm"}
1. sympy: exact simplification of (truth - candidate) after rounding constants (fast, no LLM);
2. structural: same set of terms after cancel/together, coefficients within rel tol;
3. otherwise ask Claude to judge equivalence up to small coefficient differences.
"""
import json

import sympy as sp

from .solvers import parse


def _names(rhs_a, rhs_b, extra=()):
    import re
    s = " ".join(list(rhs_a.values()) + list(rhs_b.values()))
    return sorted(set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", s)) - {"sin", "cos", "exp", "log", "sqrt", "tan",
                                                                    "atan2", "tanh", "cot", "Abs"} | set(extra))


def _skeleton(e):
    """Replace every float constant with a placeholder to compare structure."""
    return e.xreplace({a: sp.Symbol("C") for a in e.atoms(sp.Float)})


def _coef_close(a, b, tol):
    fa = sorted(float(x) for x in a.atoms(sp.Float))
    fb = sorted(float(x) for x in b.atoms(sp.Float))
    return len(fa) == len(fb) and all(abs(x - y) <= tol * max(abs(x), abs(y), 1e-12) for x, y in zip(fa, fb))


def sympy_check(t_str, c_str, names, tol=0.05):
    t = sp.nsimplify(parse(t_str, names), rational=False)
    c = parse(c_str, names)
    try:
        if sp.simplify(sp.expand(t - c)) == 0:
            return True, "exact"
    except Exception:  # noqa: BLE001
        pass
    try:
        tc, cc = sp.cancel(sp.together(sp.expand(t))), sp.cancel(sp.together(sp.expand(c)))
        tn, td = sp.fraction(tc)
        cn, cd = sp.fraction(cc)
        # normalise so that denominators are monic in their largest coefficient
        def norm(n, d):
            cs = [abs(float(x)) for x in sp.Poly(d, *sorted(d.free_symbols, key=str)).coeffs()] if d.free_symbols else [abs(float(d))]
            k = max(cs) if cs else 1.0
            return sp.expand(n / k), sp.expand(d / k)
        tn, td = norm(tn, td)
        cn, cd = norm(cn, cd)
        if all(len(sp.Add.make_args(a)) == len(sp.Add.make_args(b)) for a, b in [(tn, cn), (td, cd)]):
            ts = {_skeleton(x.as_coeff_Mul()[1]) for x in sp.Add.make_args(tn) + sp.Add.make_args(td)}
            cs = {_skeleton(x.as_coeff_Mul()[1]) for x in sp.Add.make_args(cn) + sp.Add.make_args(cd)}
            if ts == cs and _coef_close(sp.Add(tn, td), sp.Add(cn, cd), tol):
                return True, "structure+coefficients"
    except Exception:  # noqa: BLE001
        pass
    return None, "undecided"


JUDGE_PROMPT = """Decide whether a discovered equation is symbolically equivalent to the ground truth, allowing
small numerical differences in coefficients (about 5% relative) and algebraic rearrangement, but NOT
missing/extra terms with non-negligible effect.
Ground truth:   d{v}/dt = {t}
Discovered:     d{v}/dt = {c}
Answer with JSON only: {{"equivalent": true|false, "reason": "<one sentence>"}}"""


def llm_check(v, t, c, client=None, model="claude-opus-5-5"):
    import anthropic
    client = client or anthropic.Anthropic()
    msg = client.messages.create(model=model, max_tokens=2000, output_config={"effort": "low"},
                                 messages=[{"role": "user", "content": JUDGE_PROMPT.format(v=v, t=t, c=c)}])
    text = "".join(b.text for b in msg.content if b.type == "text")
    try:
        j = json.loads(text[text.index("{"): text.rindex("}") + 1])
        return bool(j["equivalent"]), j.get("reason", "")
    except Exception:  # noqa: BLE001
        return False, f"unparseable judge output: {text[:200]}"


def judge(truth_rhs, cand_rhs, use_llm=True, client=None):
    names = _names(truth_rhs, cand_rhs)
    per = {}
    for v, t in truth_rhs.items():
        c = cand_rhs.get(v, "0")
        ok, how = sympy_check(t, c, names)
        if ok is None and use_llm:
            try:
                ok, how = llm_check(v, t, c, client)
                how = "llm: " + how
            except Exception as e:  # noqa: BLE001
                ok, how = False, f"llm unavailable ({type(e).__name__})"
        per[v] = {"equivalent": bool(ok), "how": how}
    return {"equivalent": all(p["equivalent"] for p in per.values()), "per_var": per}
