"""Compact presentation components: one screen per case (question, hero, equation + verdict chip + tiles + chips,
one chart, one collapsed details expander)."""
import html
import numpy as np
import re

import streamlit as st

VERDICT = {  # status -> (colour, label)
    "CONFIDENT": ("#16a34a", "✓ CONFIDENT"),
    "CONFIDENT IN PREDICTIONS": ("#0891b2", "✓ CONFIDENT IN PREDICTIONS"),
    "COLLECT MORE DATA": ("#d97706", "◐ COLLECT MORE DATA"),
    "INCONCLUSIVE": ("#dc2626", "? INCONCLUSIVE"),
    "VALIDATED": ("#16a34a", "✓ VALIDATED OUT OF SAMPLE"),
    "PENDING": ("#64748b", "… RESULTS PENDING"),
    "NOT RECOVERED": ("#dc2626", "✗ NOT RECOVERED"),
    "RESULT": ("#475569", "RESULT"),
}

CSS = """
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1380px;}
h1, h2, h3 {letter-spacing: -0.01em;}
.q {font-size: 1.45rem; font-weight: 750; line-height: 1.25; margin: 0 0 .7rem 0;}
.vchip {display:inline-block; padding: 5px 14px; border-radius: 999px; color: #fff; font-weight: 800;
        font-size: .86rem; letter-spacing: .03em; margin-bottom: .25rem;}
.chips {display:flex; flex-wrap: wrap; gap: 6px; margin-top: .35rem;}
.chip {padding: 3px 10px; border-radius: 999px; font-size: .76rem; border: 1px solid rgba(99,102,241,.35);
       background: rgba(99,102,241,.08); white-space: nowrap;}
.chip b {color: #6366f1;}
.protocol {border-radius: 12px; padding: 10px 16px; margin: .4rem 0 1rem 0; font-size: .95rem;
           border: 1px solid rgba(22,163,74,.35); background: rgba(22,163,74,.08);}
.card-num {font-size: 1.55rem; font-weight: 800; line-height: 1.2;}
.card-sub {font-size: .85rem; opacity: .75;}
.card-name {font-size: 1.05rem; font-weight: 750; margin-top: .3rem;}
.small {font-size: .8rem; opacity: .75;}
div[data-testid="stMetricValue"] {font-size: 1.6rem;}
</style>
"""


def inject_css():
    st.html(CSS)


def question(text):
    st.markdown(f"<div class='q'>{html.escape(text)}</div>", unsafe_allow_html=True)


def verdict_chip(status, note=None):
    status = (status or "RESULT").upper()
    color, label = VERDICT.get(status, VERDICT["RESULT"])
    st.markdown(f"<span class='vchip' style='background:{color}'>{label}</span>"
                + (f" <span class='small'>{html.escape(note)}</span>" if note else ""), unsafe_allow_html=True)


def chips(items, title="how it was found"):
    """items: list of (tool, finding) or plain strings; ≤6 words each."""
    parts = []
    for it in items[:6]:
        if isinstance(it, (tuple, list)):
            parts.append(f"<span class='chip'><b>{html.escape(it[0])}</b> → {html.escape(it[1])}</span>")
        else:
            parts.append(f"<span class='chip'>{html.escape(it)}</span>")
    st.markdown(f"<div class='small'>{html.escape(title)}</div><div class='chips'>{''.join(parts)}</div>",
                unsafe_allow_html=True)


def tiles(items, cols=2):
    """items: list of dict(label, value, delta=None, help=None)."""
    rows = [items[i:i + cols] for i in range(0, len(items), cols)]
    for row in rows:
        cs = st.columns(cols)
        for c, it in zip(cs, row):
            c.metric(it["label"], it["value"], it.get("delta"), delta_color="off", help=it.get("help"), border=True)


# ----------------------------------------------------------------------------- equations
def _sym_names(exprs, extra=()):
    names = set(extra)
    for e in exprs:
        names.update(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(e)))
    funcs = {"sin", "cos", "tan", "exp", "log", "sqrt", "tanh", "Abs", "abs", "pi", "E", "sinh", "cosh", "atan"}
    return sorted(n for n in names if n not in funcs)


def expr_latex(expr, names=None, digits=4):
    import sympy as sp
    from eqdisc.solvers import parse
    names = names or _sym_names([expr])
    try:
        e = parse(str(expr).replace("abs(", "Abs("), list(names))
        e = e.xreplace({a: sp.Float(float(a), digits) for a in e.atoms(sp.Float)})
        return sp.latex(e)
    except Exception:  # noqa: BLE001
        return r"\texttt{" + str(expr).replace("_", r"\_").replace("**", "^") + "}"


def _laplacians(e, fields):
    """Fold c*f_xx + c*f_yy (equal coefficients within 2%) into c*nabla^2 f for compact 2-D PDEs."""
    import sympy as sp
    for f in fields:
        fx, fy = sp.Symbol(f"{f}_xx"), sp.Symbol(f"{f}_yy")
        c1, c2 = e.coeff(fx), e.coeff(fy)
        try:
            ok = c1 != 0 and c2 != 0 and abs(float(c1) - float(c2)) <= 0.02 * abs(float(c1))
        except TypeError:
            ok = False
        if ok:
            e = sp.expand(e - c1 * fx - c2 * fy) + sp.Float((float(c1) + float(c2)) / 2) * sp.Symbol(rf"\nabla^{{2}} {f}")
    return e


def rhs_latex(rhs, pde=False, digits=4):
    import sympy as sp
    from eqdisc.solvers import parse
    names = _sym_names(rhs.values(), rhs.keys())
    out = []
    for v, e in rhs.items():
        lhs = rf"\partial_t {sp.latex(sp.Symbol(v))}" if pde else rf"\dot{{{sp.latex(sp.Symbol(v))}}}"
        body = expr_latex(e, names, digits)
        if pde:
            try:
                ex = _laplacians(sp.expand(parse(str(e), names)), list(rhs))
                ex = ex.xreplace({a: sp.Float(float(a), digits) for a in ex.atoms(sp.Float)})
                body = sp.latex(ex)
            except Exception:  # noqa: BLE001
                pass
        out.append(f"{lhs} = {body}")
    return out


def equations(lines, small=False):
    for ln in lines:
        st.latex((r"\small " if small else "") + ln)


# ----------------------------------------------------------------------------- details helpers
def _fmt(x, d=3):
    if x is None:
        return "—"
    try:
        x = float(x)
    except Exception:  # noqa: BLE001
        return str(x)
    return "0" if x == 0 else f"{x:.{d}g}"


def terms_table(assessment, static=False):
    import pandas as pd
    terms = (assessment or {}).get("terms") or []
    if not terms:
        return
    st.dataframe(pd.DataFrame([{
        "equation": t.get("var") if static else f"d{t.get('var')}/dt", "term": t.get("term"), "coef": t.get("coef"),
        "90% CI": f"[{_fmt((t.get('ci90') or [None, None])[0], 4)}, {_fmt((t.get('ci90') or [None, None])[1], 4)}]",
        "significant": bool(t.get("significant")), "ΔBIC if removed": t.get("dBIC_if_removed")} for t in terms]),
        hide_index=True, column_config={"coef": st.column_config.NumberColumn(format="%.5g"),
                                        "ΔBIC if removed": st.column_config.NumberColumn(format="%.0f")})


def experiments(assessment, top=3):
    ex = ((assessment or {}).get("experiments") or {}).get("ranked") or []
    for i, e in enumerate(ex[:top], 1):
        where = e.get("description") or ("start at (" + ", ".join(f"{x:.3g}" for x in e.get("initial_condition") or []) + ")")
        gain = e.get("gain_vs_existing_data")
        st.markdown(f"**#{i}** {where}" + (f" · {_fmt(gain)}× more informative than repeating" if gain else ""))


# ----------------------------------------------------------------------------- confidence + next data
PLAIN_VERDICT = {
    "CONFIDENT": ("#16a34a", "✓ Very likely the law", "Every check passed."),
    "CONFIDENT IN PREDICTIONS": ("#0891b2", "✓ Predictions trustworthy", "Rival forms fit equally well, but they all "
                                 "predict the same behaviour over the data range."),
    "COLLECT MORE DATA": ("#d97706", "◐ Not sure yet: collect more data", "The best model so far is not confirmed."),
    "INCONCLUSIVE": ("#dc2626", "✗ Not established", "The data do not pin down a law."),
}


def _checks(a):
    terms = a.get("terms") or []
    weak = [t for t in terms if not t.get("significant")]
    strong_add = [m for m in (a.get("missing") or []) if (m.get("dBIC_if_added") or 0) < -10
                  and (m.get("error_reduction") is None or m["error_reduction"] >= 0.02)]
    val = a.get("validation") or {}
    ok_roll = (not val.get("rollout_blew_up")) and val.get("rollout_valid_time") is not None and \
        val.get("rollout_horizon") and val["rollout_valid_time"] >= 0.7 * val["rollout_horizon"]
    out = [("Every term earns its place", not weak and bool(terms),
            f"{len(terms)} terms, all 90% ranges exclude zero" if not weak and terms
            else (f"{len(weak)} term(s) could be zero: " + ", ".join(t['term'] for t in weak[:3]) if weak else "no term statistics"))]
    out.append(("Nothing obvious is missing", not strong_add,
                "no extra term improves the fit enough to justify itself" if not strong_add
                else "data favour adding " + ", ".join(m["term"] for m in strong_add[:2])))
    if val:
        frac = (val.get("rollout_valid_time") or 0) / (val.get("rollout_horizon") or 1)
        crit = f" ({val['criterion']})" if val.get("criterion") else ""
        out.append(("Forecasts data it was not fitted to", bool(ok_roll),
                    ("the forecast blew up" if val.get("rollout_blew_up") else
                     f"refit without the last part of the training data, it forecasts that part: valid for {frac:.0%} of it{crit}")))
    return out


def _decades(vals):
    v = [x for x in vals if x and x > 0] or [1.0]
    lo, hi = int(np.floor(np.log10(min(v)))), int(np.ceil(np.log10(max(v))))
    step = max(1, (hi - lo) // 4)
    return [10.0 ** e for e in range(lo, hi + 1, step)]


def confidence_panel(a, key, names=None):
    """a: slim assessment {verdict, terms, missing, validation, experiments}. Verdict, three checks, coefficient
    precision chart, and the single most useful next measurement."""
    import plotly.graph_objects as go
    names = names or {}
    v = (a.get("verdict") or {})
    color, label, sub = PLAIN_VERDICT.get(v.get("status"), PLAIN_VERDICT["INCONCLUSIVE"])
    st.markdown("#### How sure are we, and what next?")
    c1, c2, c3 = st.columns([1.05, 1, 1], gap="large")
    with c1:
        st.markdown(f"<div style='background:{color};color:white;border-radius:10px;padding:12px 14px;"
                    f"font-weight:700;font-size:1.15rem'>{html.escape(label)}</div>"
                    f"<div class='small' style='margin:6px 0 10px'>{html.escape(sub)}</div>", unsafe_allow_html=True)
        for name, ok, detail in _checks(a):
            st.markdown(f"<div style='margin:4px 0'>{'✅' if ok else '⚠️'} <b>{html.escape(name)}</b><br>"
                        f"<span class='small'>{html.escape(detail)}</span></div>", unsafe_allow_html=True)
    with c2:
        terms = a.get("terms") or []
        if terms:
            lab = [f"{names.get(t['term'], t['term'])}  (d{t['var']}/dt)" for t in terms]
            pct = [100 * float(t.get("rel_uncertainty") or 0) for t in terms]
            col = ["#16a34a" if (t.get("significant") and p < 10) else "#d97706" if t.get("significant") else "#dc2626"
                   for t, p in zip(terms, pct)]
            fig = go.Figure(go.Bar(x=pct, y=lab, orientation="h", marker_color=col,
                                   text=[f"±{p:.2g}%" for p in pct], textposition="outside",
                                   hovertemplate="%{y}: ±%{x:.3g}%<extra></extra>"))
            fig.update_layout(height=60 + 34 * len(terms), margin=dict(l=10, r=40, t=30, b=10),
                              title=dict(text="How precisely each coefficient is known (90%)", font=dict(size=13)),
                              xaxis=dict(type="log", title=None, tickvals=_decades(pct), ticktext=[f"{v:g}%" for v in _decades(pct)],
                                         range=[np.log10(_decades(pct)[0]), np.log10(_decades(pct)[-1]) + 0.3]),
                              yaxis=dict(autorange="reversed"),
                              showlegend=False)
            st.plotly_chart(fig, key=f"conf_{key}", config={"displayModeBar": False})
    with c3:
        ex = (a.get("experiments") or [])
        st.markdown("<div class='small'><b>Most useful next measurement</b></div>", unsafe_allow_html=True)
        if ex:
            e = ex[0]
            what = e.get("description") or ("start at (" + ", ".join(f"{x:.3g}" for x in e.get("initial_condition", [])) + ")")
            gains = e.get("informs_coefficients") or []
            pins = "".join(f"<li>{html.escape(names.get(g['coefficient'].split(' in ')[0], g['coefficient']))}: "
                           f"{g['info_gain_vs_existing']:.3g}× more information than repeating existing runs</li>"
                           for g in gains[:2] if g.get("info_gain_vs_existing"))
            st.markdown(f"<div style='border:1px solid rgba(128,128,128,.35);border-radius:10px;padding:10px 12px'>"
                        f"📍 <b>{html.escape(str(what))}</b>"
                        + (f"<ul style='margin:6px 0 0 0;padding-left:18px' class='small'>{pins}</ul>" if pins else "")
                        + "</div>", unsafe_allow_html=True)
        for adv in (a.get("data_advice") or [])[:1]:
            st.markdown(f"<div class='small' style='margin-top:8px'>💡 {html.escape(adv)}</div>", unsafe_allow_html=True)
        if v.get("recommendation") and not ex:
            st.markdown(f"<div class='small'>{html.escape(v['recommendation'])}</div>", unsafe_allow_html=True)
