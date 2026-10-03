"""Compact presentation components: one screen per case (question, hero, equation + verdict chip + tiles + chips,
one chart, one collapsed details expander)."""
import html
import re

import streamlit as st

VERDICT = {  # status -> (colour, label)
    "CONFIDENT": ("#16a34a", "✓ CONFIDENT"),
    "CONFIDENT IN PREDICTIONS": ("#0891b2", "✓ CONFIDENT IN PREDICTIONS"),
    "COLLECT MORE DATA": ("#d97706", "◐ COLLECT MORE DATA"),
    "INCONCLUSIVE": ("#dc2626", "? INCONCLUSIVE"),
    "VALIDATED": ("#16a34a", "✓ VALIDATED OUT OF SAMPLE"),
    "PENDING": ("#64748b", "… RESULTS PENDING"),
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
    for it in items[:4]:
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
