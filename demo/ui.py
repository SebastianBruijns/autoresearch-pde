"""Shared rendering for the demo: verdict banner, equations, key-steps timeline, confidence, experiments.
Used by both the precomputed Showcase and the live "Run on your data" tab."""
import html
import re

import streamlit as st

VERDICT_STYLE = {
    "CONFIDENT": ("#16a34a", "✅", "Confident"),
    "CONFIDENT IN PREDICTIONS": ("#0891b2", "🔷", "Confident in predictions"),
    "COLLECT MORE DATA": ("#d97706", "🧪", "Collect more data — here"),
    "INCONCLUSIVE": ("#dc2626", "❔", "Inconclusive"),
    "BENCHMARK": ("#7c3aed", "📊", "Benchmark result"),
    "RESULT": ("#475569", "🧮", "Result"),
}

CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1400px;}
.vbanner {border-radius: 14px; padding: 18px 22px; margin: 4px 0 14px 0; color: #fff;
          box-shadow: 0 4px 18px rgba(0,0,0,.18);}
.vbanner .status {font-size: 1.65rem; font-weight: 800; letter-spacing: .02em;}
.vbanner .headline {font-size: 1.05rem; margin-top: 4px; opacity: .97;}
.vbanner .rec {font-size: .95rem; margin-top: 8px; opacity: .92; border-top: 1px solid rgba(255,255,255,.35);
               padding-top: 8px;}
.step {border-left: 4px solid #6366f1; padding: 8px 14px; margin: 0 0 12px 6px; border-radius: 0 10px 10px 0;
       background: rgba(99,102,241,.08);}
.step .n {font-weight: 800; color: #6366f1; margin-right: 6px;}
.step .o {font-size: .93rem;}
.step .d {font-size: .93rem; margin-top: 3px;}
.step .r {font-size: .93rem; margin-top: 3px; font-weight: 600;}
.pill {display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .78rem; font-weight: 700;
       margin-right: 6px; background: rgba(127,127,127,.18);}
.casecard {border: 1px solid rgba(127,127,127,.25); border-radius: 12px; padding: 10px 12px; height: 100%;}
.casecard .t {font-weight: 800; font-size: 1.0rem;}
.casecard .s {font-size: .84rem; opacity: .8;}
.footnote {font-size: .82rem; opacity: .8; border-top: 1px dashed rgba(127,127,127,.4); padding-top: 6px;}
.hero {font-size: 1.15rem; opacity: .9; margin-bottom: .3rem;}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def verdict_banner(verdict):
    verdict = verdict or {}
    status = (verdict.get("status") or "RESULT").upper()
    color, icon, label = VERDICT_STYLE.get(status, VERDICT_STYLE["RESULT"])
    head = html.escape(verdict.get("headline") or "")
    rec = html.escape(verdict.get("recommendation") or "")
    st.markdown(
        f'<div class="vbanner" style="background: linear-gradient(135deg, {color}, {color}cc);">'
        f'<div class="status">{icon} {status}</div>'
        f'<div class="headline">{head}</div>'
        + (f'<div class="rec"><b>Recommendation.</b> {rec}</div>' if rec else "")
        + "</div>", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- equations
def _sym_names(rhs, extra=()):
    names = set(extra)
    for e in rhs.values():
        names.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(e)))
    funcs = {"sin", "cos", "tan", "exp", "log", "sqrt", "tanh", "Abs", "abs", "pi", "E", "sinh", "cosh", "atan"}
    return sorted(n for n in names if n not in funcs)


def expr_latex(expr, names, digits=4):
    import sympy as sp
    from eqdisc.solvers import parse
    try:
        e = parse(str(expr).replace("abs(", "Abs("), list(names))
        e = e.xreplace({a: sp.Float(float(a), digits) for a in e.atoms(sp.Float)})
        return sp.latex(e)
    except Exception:  # noqa: BLE001
        return r"\texttt{" + str(expr).replace("_", r"\_").replace("**", "^") + "}"


def lhs_latex(var, pde=False):
    import sympy as sp
    v = sp.latex(sp.Symbol(var))
    return rf"\partial_t {v}" if pde else rf"\dot{{{v}}}"


def show_equations(rhs, pde=False, title=None):
    if not rhs:
        st.info("No model.")
        return
    if title:
        st.markdown(f"**{title}**")
    names = _sym_names(rhs, rhs.keys())
    for v, e in rhs.items():
        st.latex(f"{lhs_latex(v, pde)} = {expr_latex(e, names)}")


# ----------------------------------------------------------------------------- story
def key_steps(story, insights=None, max_steps=6):
    steps = (story or {}).get("key_steps") or []
    if (story or {}).get("headline"):
        st.markdown(f"<div class='hero'>{html.escape(story['headline'])}</div>", unsafe_allow_html=True)
    for i, s in enumerate(steps[:max_steps], 1):
        st.markdown(
            f"<div class='step'><span class='n'>{i}</span>"
            f"<div class='o'>🔍 <b>Saw:</b> {html.escape(s.get('observation', ''))}</div>"
            f"<div class='d'>🧭 <b>Decided:</b> {html.escape(s.get('decision', ''))}</div>"
            f"<div class='r'>✅ {html.escape(s.get('outcome', ''))}</div></div>", unsafe_allow_html=True)
    if insights:
        with st.expander("Automatic insights from the research trail"):
            for ins in insights:
                why = f" — *{ins['why_it_matters']}*" if ins.get("why_it_matters") else ""
                st.markdown(f"- **{ins.get('step', '')}:** {ins.get('finding', '')}{why}")
    if (story or {}).get("physical_interpretation"):
        st.markdown(f"**Physical interpretation.** {story['physical_interpretation']}")
    if (story or {}).get("caveats"):
        with st.expander("Caveats the agents flagged"):
            for c in story["caveats"]:
                st.markdown(f"- {c}")


# ----------------------------------------------------------------------------- confidence
def _fmt(x, d=3):
    if x is None:
        return "—"
    try:
        x = float(x)
    except Exception:  # noqa: BLE001
        return str(x)
    if x == 0:
        return "0"
    return f"{x:.{d}g}"


def confidence(assessment, cost=None, wall=None, static=False):
    import pandas as pd
    a = assessment or {}
    if not a or "confidence" not in a:
        st.caption("No assessment available for this result." + (f" ({a['error']})" if a.get("error") else ""))
        return
    conf = a.get("confidence") or {}
    nf = a.get("noise_floor") or {}
    pr = a.get("predictability") or {}
    cols = st.columns(4)
    cols[0].metric("Confidence", (conf.get("level") or "—").upper())
    r = nf.get("error_to_floor_ratio")
    cols[1].metric("Error / noise floor", f"{_fmt(r)}×" if r is not None else "n/a",
                   help="1× means the remaining error is pure measurement noise")
    cols[2].metric("Predictability horizon", _fmt(pr.get("horizon")) + (" (full span)" if pr.get("horizon_is_full_data_span") else ""),
                   help=pr.get("meaning"))
    if cost is not None:
        cols[3].metric("Cost / wall time", f"${cost:.2f}" + (f" · {wall / 60:.0f} min" if wall else ""))
    if conf.get("reasons"):
        st.markdown("".join(f"<span class='pill'>{html.escape(r)}</span> " for r in conf["reasons"]),
                    unsafe_allow_html=True)
    terms = a.get("terms") or []
    if terms:
        df = pd.DataFrame([{"equation": t.get("var") if static else f"d{t.get('var')}/dt", "term": t.get("term"), "coef": t.get("coef"),
                            "90% CI": f"[{_fmt((t.get('ci90') or [None, None])[0], 4)}, {_fmt((t.get('ci90') or [None, None])[1], 4)}]",
                            "rel. unc.": t.get("rel_uncertainty"), "significant": bool(t.get("significant")),
                            "ΔBIC if removed": t.get("dBIC_if_removed")} for t in terms])
        st.dataframe(df, hide_index=True, width="stretch",
                     column_config={"coef": st.column_config.NumberColumn(format="%.4g"),
                                    "rel. unc.": st.column_config.NumberColumn(format="%.2g"),
                                    "ΔBIC if removed": st.column_config.NumberColumn(format="%.0f")})
    amb = a.get("model_ambiguity") or {}
    miss = a.get("missing_term_evidence") or []
    c1, c2 = st.columns(2)
    with c1:
        if amb.get("verdict"):
            st.markdown("**Competing models**")
            st.caption(amb["verdict"])
    with c2:
        if miss:
            st.markdown("**Candidate missing terms tested**")
            st.caption("; ".join(f"{m['term']} in {m['var'] if static else 'd' + m['var'] + '/dt'} (ΔBIC {_fmt(m.get('dBIC_if_added'))})"
                                 for m in miss[:4]))


def experiments(assessment, names=None, top=4):
    ex = ((assessment or {}).get("experiments") or {}).get("ranked") or []
    if not ex:
        st.caption("No experiment design available.")
        return
    for i, e in enumerate(ex[:top], 1):
        where = e.get("description") or ("start at (" + ", ".join(
            (f"{n}={x:.3g}" if names else f"{x:.3g}") for n, x in zip(names or [""] * 99, e.get("initial_condition") or [])) + ")")
        pins = ", ".join(f"{c['coefficient']} ({_fmt(c.get('info_gain_vs_existing'))}×)"
                         for c in (e.get("informs_coefficients") or [])[:3])
        gain = e.get("gain_vs_existing_data")
        st.markdown(f"**#{i}** · {where}  \n"
                    f"<span class='pill'>score {_fmt(e.get('score'))}</span>"
                    + (f"<span class='pill'>{_fmt(gain)}× more informative than repeating</span>" if gain else "")
                    + (f"<br><small>pins down: {html.escape(pins)}</small>" if pins else ""), unsafe_allow_html=True)
    qs = (assessment or {}).get("questions_for_human") or []
    if qs:
        with st.expander("Questions the system would ask you"):
            for q in qs:
                st.markdown(f"- {q}")


def result_layout(verdict, rhs, story=None, insights=None, assessment=None, truth=None, pde=False, names=None,
                  cost=None, wall=None, latex_lines=None, truth_latex=None, truth_title="Ground truth (hidden from the agents)"):
    """The standard result card: verdict, equation (+ truth), key steps, confidence, experiments."""
    verdict_banner(verdict)
    c1, c2 = st.columns([1.15, 1]) if (truth or truth_latex) else (st.container(), None)
    with c1:
        st.markdown("#### Discovered equation")
        if latex_lines:
            for line in latex_lines:
                st.latex(line)
        else:
            show_equations(rhs, pde)
    if c2 is not None:
        with c2:
            st.markdown(f"#### {truth_title}")
            if truth_latex:
                for line in truth_latex:
                    st.latex(line)
            else:
                show_equations(truth, pde)
    t1, t2, t3 = st.tabs(["🧭 How we got there", "📏 Confidence", "🧪 Next experiments"])
    with t1:
        key_steps(story, insights)
    with t2:
        confidence(assessment, cost, wall)
    with t3:
        experiments(assessment, names)
