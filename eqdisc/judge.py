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


def _consts(e):
    """Numeric constants that a fit may change: floats, and non-integer rationals (nsimplify turns 0.5 into 1/2)."""
    return [a for a in e.atoms(sp.Float, sp.Rational) if not a.is_Integer]


def _skeleton(e):
    """Replace every numeric constant (float or non-integer rational) with a placeholder to compare structure."""
    return e.xreplace({a: sp.Symbol("C") for a in _consts(e)})


def _term_map(e):
    """{structure of a term (floats -> C): [(numeric coefficient, sorted inner floats), ...]} of an expanded sum."""
    out = {}
    for x in sp.Add.make_args(sp.expand(e)):
        c, m = x.as_coeff_Mul()
        out.setdefault(_skeleton(m), []).append((float(c), sorted(float(f) for f in _consts(m))))
    return out


def _coef_close(a, b, tol):
    """Same terms with coefficients (and constants inside functions) within relative tol, matched term by term.
    Integer coefficients count (a coefficient of exactly 1 is a coefficient), unlike comparing sorted Float atoms."""
    close = lambda x, y: abs(x - y) <= tol * max(abs(x), abs(y), 1e-12)
    A, B = _term_map(a), _term_map(b)
    if A.keys() != B.keys():
        return False
    for k in A:
        la, lb = sorted(A[k]), sorted(B[k])
        if len(la) != len(lb):
            return False
        for (ca, fa), (cb, fb) in zip(la, lb):
            if not close(ca, cb) or len(fa) != len(fb) or not all(close(x, y) for x, y in zip(fa, fb)):
                return False
    return True


def _terms(e):
    """{monomial: coefficient} of an expanded expression, or None if a coefficient is not a number."""
    out = {}
    for t in sp.Add.make_args(sp.expand(e)):
        k, m = t.as_coeff_Mul()
        if not k.is_number:
            return None
        out[m] = out.get(m, 0.0) + float(k)
    return out


def sympy_check(t_str, c_str, names, tol=0.05):
    t = sp.nsimplify(parse(t_str, names), rational=False)
    c = parse(c_str, names)
    try:
        if sp.simplify(sp.expand(t - c)) == 0:
            return True, "exact"
    except Exception:  # noqa: BLE001
        pass
    try:     # sums of monomials (the usual case): same terms, each coefficient within tol
        tt, ct = _terms(t), _terms(c)
        if tt is not None and ct is not None and tt.keys() == ct.keys() and all(
                abs(tt[m] - ct[m]) <= tol * max(abs(tt[m]), abs(ct[m]), 1e-12) for m in tt):
            return True, "terms+coefficients"
    except Exception:  # noqa: BLE001
        pass
    try:
        tc, cc = sp.cancel(sp.together(sp.expand(t))), sp.cancel(sp.together(sp.expand(c)))
        tn, td = sp.fraction(tc)
        cn, cd = sp.fraction(cc)
        # normalise so that denominators are monic in their largest coefficient
        def norm(n, d):
            try:
                cs = [abs(float(x)) for x in sp.Poly(d, *sorted(d.free_symbols, key=str)).coeffs()] if d.free_symbols \
                    else [abs(float(d))]
            except Exception:  # noqa: BLE001  (non-polynomial denominator, e.g. 1/exp(u/2): leave unnormalised)
                cs = []
            k = max(cs) if cs else 1.0
            return sp.expand(n / k), sp.expand(d / k)
        tn, td = norm(tn, td)
        cn, cd = norm(cn, cd)
        if all(len(sp.Add.make_args(a)) == len(sp.Add.make_args(b)) for a, b in [(tn, cn), (td, cd)]):
            ts = {_skeleton(x.as_coeff_Mul()[1]) for x in sp.Add.make_args(tn) + sp.Add.make_args(td)}
            cs = {_skeleton(x.as_coeff_Mul()[1]) for x in sp.Add.make_args(cn) + sp.Add.make_args(cd)}
            if ts == cs and _coef_close(tn, cn, tol) and _coef_close(td, cd, tol):
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
