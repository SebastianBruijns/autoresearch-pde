"""Demo page: the evidence layer ("knowing when not to trust the answer").

Reads only demo/showcase/evidence/ (built by demo/build_evidence.py); never computes at view time.
"""
import html
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import ui
from viz import AQUA, BLUE, ORANGE
from eqdisc.corrupt import EXPECTED

DEMO = Path(__file__).resolve().parent
EV = DEMO / "showcase" / "evidence"
GOOD, CAUTION, WARN, BAD, MUTED = "#16a34a", "#0891b2", "#d97706", "#dc2626", "#94a3b8"
PRINCIPLE = ("A correct equation has the same coefficients on every slice of the data, "
             "and leaves only noise behind.")

CSS = """
<style>
.ev-card-t {font-size: 1.02rem; font-weight: 750; margin: 0 0 .1rem 0;}
.ev-card-s {font-size: .78rem; opacity: .7; margin-bottom: .35rem; min-height: 2.1em;}
.ev-said {font-size: .86rem; line-height: 1.3; margin: .35rem 0 .3rem 0; min-height: 3.4em;}
.ev-chip {display:inline-block; padding: 2px 9px; border-radius: 999px; color: #fff; font-weight: 750;
          font-size: .7rem; letter-spacing: .03em; margin: 0 4px 3px 0;}
.ev-quiet {font-size: .72rem; opacity: .6; font-style: italic;}
.ev-principle {font-size: 1.25rem; font-weight: 650; border-left: 4px solid #6366f1; padding: .3rem 0 .3rem .9rem;
               margin: .2rem 0 1rem 0;}
.ev-model {font-size: 1.1rem; font-weight: 750; margin-bottom: .2rem;}
.ev-take {font-size: 1.12rem; font-weight: 650; text-align: center; margin: .6rem 0 .2rem 0;}
.ev-pending {border: 1px dashed rgba(128,128,128,.6); border-radius: 12px; padding: 18px; text-align: center;
             opacity: .8;}
</style>
"""


@st.cache_data(show_spinner=False)
def _load(name):
    p = EV / name
    return json.loads(p.read_text()) if p.exists() else None


def _chip(text, color, tip=None):
    t = f" title='{html.escape(tip, quote=True)}'" if tip else ""
    return f"<span class='ev-chip' style='background:{color}'{t}>{html.escape(text)}</span>"


def _show(fig, key):
    st.plotly_chart(fig, key=key, config={"displaylogo": False, "displayModeBar": False})


# ============================================================================= 1. orbit
SLICE_NAMES = ["closest third", "middle third", "farthest third"]
COEF_COLORS = [BLUE, ORANGE, AQUA]


def coef_chart(m, edges, ylim):
    """Per-slice estimate of each shown coefficient, relative to its whole-data value (1 = same everywhere)."""
    fig = go.Figure()
    fig.add_hrect(y0=0.95, y1=1.05, fillcolor="rgba(22,163,74,.10)", line_width=0)
    fig.add_hline(y=1, line=dict(color="rgba(120,120,120,.7)", width=1, dash="dash"))
    xs = np.arange(3)
    shown = m["shown"]
    for i, c in enumerate(shown):
        off = (i - (len(shown) - 1) / 2) * 0.16
        full = c["full"]
        y = [None if b is None else b / full for b in c["per_slice"]]
        e = [None if s is None else 1.645 * s / abs(full) for s in c["per_slice_se"]]
        hover = [f"{c['equation']}: {c['term']}<br>{SLICE_NAMES[j]} of the orbit "
                 f"({edges[j]:.2f}–{edges[j + 1]:.2f} planet radii)<br>coefficient {b:.4g} ± {1.645 * s:.2g} "
                 f"(90%)<br>whole data: {full:.4g}" if b is not None else "not estimable here"
                 for j, (b, s) in enumerate(zip(c["per_slice"], c["per_slice_se"]))]
        fig.add_trace(go.Scatter(
            x=xs + off, y=y, mode="markers+lines", name=f"{c['term']}  (in {c['equation']})",
            line=dict(color=COEF_COLORS[i], width=2), marker=dict(size=10, color=COEF_COLORS[i],
                                                                  line=dict(color="white", width=2)),
            error_y=dict(type="data", array=e, color=COEF_COLORS[i], thickness=2, width=5),
            hovertext=hover, hoverinfo="text"))
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=10, b=10), hovermode="closest",
                      legend=dict(orientation="h", y=-0.22, x=0, yanchor="top", font=dict(size=11)),
                      xaxis=dict(tickvals=xs, ticktext=[f"{n}<br>{edges[j]:.1f}–{edges[j + 1]:.1f} radii"
                                                       for j, n in enumerate(SLICE_NAMES)],
                                 range=[-0.5, 2.5], showgrid=False),
                      yaxis=dict(title="coefficient ÷ its whole-data value", range=ylim, zeroline=False,
                                 gridcolor="rgba(128,128,128,.15)"))
    return fig


def _km(x):
    return f"{x:,.0f} km" if x >= 10 else f"{x:.2g} km"


def section_orbit():
    data = _load("orbit.json")
    ui.section(1, "Same data, two laws", "a real failure, made visible")
    if not data:
        st.markdown("<div class='ev-pending'>Orbit case not built yet: "
                    "<code>python demo/build_evidence.py orbit</code></div>", unsafe_allow_html=True)
        return
    models = data["models"]
    wrong_key = "agent" if "agent" in models else "sindy"
    pair = [("truth", models["truth"]), (wrong_key, models[wrong_key])]
    orbit_case = _load_orbit_case()
    pe = (orbit_case or {}).get("results", {}).get("position_error_km", {})
    err3 = {"truth": (pe.get("Kepler + J2") or {}).get("72h"), "agent": (pe.get("data-only agent") or {}).get("72h")}
    s = data["setup"]
    st.markdown(f"A satellite around a planet with a large equatorial bulge: {s['days']:.0f} days of positions and "
                f"velocities, {s['noise']:.0%} noise. Each law's coefficients are refitted on the closest, middle and "
                "farthest third of the orbit.")
    vals = [b / c["full"] for _, m in pair for c in m["shown"] for b in c["per_slice"] if b is not None]
    lo, hi = min(vals + [0.8]), max(vals + [1.2])
    pad = 0.12 * (hi - lo)
    ylim = [lo - pad, hi + pad]
    cols = st.columns(2, gap="large")
    for col, (key, m) in zip(cols, pair):
        with col, st.container(border=True):
            title = "The true law: gravity + bulge" if key == "truth" else (
                "What an AI agent submitted: a polynomial" if key == "agent" else "A polynomial fit (SINDy)")
            st.markdown(f"<div class='ev-model'>{html.escape(title)}</div>", unsafe_allow_html=True)
            v = m["verdict"]
            note = None
            if v.get("failed_checks"):
                note = "check failed: coefficients change with altitude" if any(
                    f.startswith("slice_amplitude") for f in v["failed_checks"]) else "a critical check failed"
            ui.verdict_chip(v["status"], note)
            _show(coef_chart(m, m["by_distance"]["edges_planet_radii"], ylim), f"ev_orbit_{key}")
            e3 = err3.get(key)
            same = all((c["I2"] or 0) < 0.5 for c in m["shown"])
            msg = ("Same coefficients on every slice." if same else
                   "Its coefficients drift with altitude: it is a local approximation, not the law.")
            if e3:
                msg += f" Forecast error after 3 unseen days: <b>{_km(e3)}</b>."
            st.markdown(f"<div class='small' style='opacity:.85'>{msg}</div>", unsafe_allow_html=True)
    st.markdown("<div class='ev-take'>Same data. The wrong law's coefficients change with altitude, "
                "so the check refuses to call it confident.</div>", unsafe_allow_html=True)
    with st.expander("Orbit case: exact numbers"):
        for key, m in models.items():
            st.markdown(f"**{m['label']}**: verdict `{m['verdict']['status']}`. {m['verdict'].get('headline', '')}")
            rows = [{"coefficient": f"{c['equation']}: {c['term']}",
                     **{SLICE_NAMES[j]: (None if b is None else f"{b:.4g} ± {1.645 * se:.2g}")
                        for j, (b, se) in enumerate(zip(c["per_slice"], c["per_slice_se"]))},
                     "whole data": f"{c['full']:.4g}", "I² (disagreement, 0–1)": c["I2"]} for c in m["shown"]]
            st.dataframe(rows, hide_index=True)
        st.caption("Error bars: 90% intervals (cluster-robust / jackknife standard errors, weak form). I² is the share "
                   "of the spread between slices not explained by noise. The audit itself slices by several amplitude "
                   "measures (|x|, |y|, |z|, |v|, state norm) and fires when I² > 0.75 with a significant Cochran's Q; "
                   "the chart slices by distance from the centre for readability. The agent's law is shown with its "
                   "constants refitted on the training data (blinded units); a no-LLM degree-3 SINDy polynomial gives "
                   f"the verdict `{models.get('sindy', {}).get('verdict', {}).get('status', 'n/a')}`.")


@st.cache_data(show_spinner=False)
def _load_orbit_case():
    p = DEMO / "showcase" / "orbit" / "case.json"
    return json.loads(p.read_text()) if p.exists() else None


# ============================================================================= 2. corruption grid
CARDS = {
    "clean": ("Clean data", "2% noise, nothing else"),
    "outliers": ("Spikes", "a few wild readings (black dots)"),
    "gaps_random": ("Random gaps", "10% of time windows missing (grey)"),
    "gaps_state": ("Missing extremes", "data vanish exactly where values are largest"),
    "forcing_time": ("Hidden push in time", "an outside force that varies in time"),
    "source_space": ("Hidden source in space", "a fixed source at some positions"),
    "traj_coeffs": ("Runs that differ", "each run has ±20% different coefficients"),
    "amp_term": ("Extreme-only term", "a small u³ term that matters only at large values"),
}
INVISIBLE = {"forcing_time", "source_space", "traj_coeffs", "amp_term"}
CHECK_NAMES = {
    "glitches": "isolated spikes", "gaps": "missing data", "gaps_state_dependent": "missing data at the extremes",
    "slice_trajectory": "runs disagree", "slice_time": "early vs late disagree", "slice_space": "left vs right disagree",
    "slice_amplitude": "small vs large values disagree", "residual_time_only": "leftover follows the clock",
    "residual_space_only": "leftover follows position", "residual_amplitude": "leftover grows at extremes",
    "residual_white": "leftover is not just noise",
}


def plain(f):
    d = f.get("details") or {}
    i = f["id"]
    if i == "glitches":
        frac = d.get("fraction")
        pct = f"{100 * frac:.1g}% of points" if frac else "some points"
        return (f"{pct} are isolated spikes. Clipped: the fit is the same with or without them."
                if f.get("resolved") else f"{pct} are isolated spikes, and the fit changes without them: kept, "
                "result not trusted.")
    if i == "gaps":
        nf = d.get("nan_fraction")
        return (f"{nf:.0%} of the data are missing." if nf else "Data are missing.") + " Fit only on the complete pieces."
    if i == "gaps_state_dependent":
        sc = f.get("scope") or {}
        rng = sc.get("range")
        tail = f" Trust it only for {sc['variable']} in [{rng[0]:.2g}, {rng[1]:.2g}]." if rng else ""
        return "The missing data are the largest values: the extremes were never seen." + tail
    if i == "residual_time_only":
        return "What the equation leaves behind follows the clock: an outside force is acting."
    if i == "residual_space_only":
        return "What the equation leaves behind depends on position: a hidden source."
    if i == "slice_trajectory":
        return "Each run gives different coefficients: something differs between runs."
    if i == "slice_space":
        return "Left and right halves give different coefficients."
    if i == "slice_time":
        return "Early and late data give different coefficients."
    if i == "slice_amplitude":
        return "Small and large values give different coefficients."
    if i == "residual_amplitude":
        sc = f.get("scope") or {}
        rng = sc.get("range")
        tail = f" Trust it only up to |{sc['variable']}| = {rng[1]:.2g}." if rng else ""
        return "The error grows at the largest values: something is missing there." + tail
    if i == "residual_white":
        return "What is left behind is more than noise."
    return f.get("message") or i


def response_chips(fs):
    out = []
    if any(f.get("resolved") for f in fs):
        out.append(_chip("REPAIRED", GOOD, "a known fix was applied and re-checked"))
    if any(f.get("response") == "widen" and not f.get("resolved") for f in fs):
        out.append(_chip("WIDENED", WARN, "uncertainty widened: the coefficients are not the same on every slice"))
    if any(f.get("response") == "scope" for f in fs):
        out.append(_chip("SCOPED", CAUTION, "the claim is restricted to the range the data support"))
    if any(f["severity"] == "critical" and not f.get("resolved") for f in fs):
        out.append(_chip("NO CONFIDENT VERDICT", BAD, "a critical check failed: the verdict cannot be CONFIDENT"))
    return "".join(out)


def card(cor, case):
    title, sub = CARDS[cor]
    fs = case["findings"]
    with st.container(border=True):
        st.markdown(f"<div class='ev-card-t'>{html.escape(title)}</div><div class='ev-card-s'>{html.escape(sub)}</div>",
                    unsafe_allow_html=True)
        p = EV / f"thumb_{cor}.png"
        if p.exists():
            st.image(str(p), width="stretch")
        if cor in INVISIBLE:
            st.markdown("<div class='ev-quiet'>looks normal: the damage is in the dynamics</div>",
                        unsafe_allow_html=True)
        if not fs:
            st.markdown(f"<div class='ev-said'>No check fired.</div>{_chip('NO ALARMS', GOOD)}",
                        unsafe_allow_html=True)
            return
        exp = EXPECTED.get(cor, [])
        main = next((f for f in fs if f["id"] in exp), None) or next(
            (f for f in fs if f["severity"] == "critical"), fs[0])
        others = sorted({CHECK_NAMES.get(f["id"], f["id"]) for f in fs if f is not main})
        tip = " | ".join(f"{f['id']} ({f['severity']}): {f['message']}" for f in fs)
        st.markdown(f"<div class='ev-said' title='{html.escape(tip, quote=True)}'>{html.escape(plain(main))}</div>"
                    + response_chips(fs)
                    + (f"<div class='small'>also: {html.escape(', '.join(others))}</div>" if others else ""),
                    unsafe_allow_html=True)


def rates_line(rates):
    pc = rates.get("per_corruption") or {}
    clean = pc.get("clean")
    if not clean or not clean["n"]:
        return None
    dmg = {c: r for c, r in pc.items() if c != "clean" and r["n"]}
    hits = {(r["hit"], r["n"]) for r in dmg.values()}
    if len(hits) == 1:
        h, n = next(iter(hits))
        caught = f"each damage type caught <b>{h}/{n}</b>"
    else:
        caught = "caught: " + ", ".join(f"{CARDS[c][0].lower()} <b>{r['hit']}/{r['n']}</b>" for c, r in dmg.items())
    return (f"Across all test sets (4 equations × 3 seeds × {len(rates['splits'])} splits): clean data "
            f"<b>{clean['hit']}/{clean['n']}</b> false alarms; {caught}.")


def section_grid():
    data = _load("grid.json")
    ui.section(2, "Seven ways data goes wrong", "and what the checks say")
    if not data:
        st.markdown("<div class='ev-pending'>Not built yet: <code>python demo/build_evidence.py grid</code></div>",
                    unsafe_allow_html=True)
        return
    line = rates_line(data.get("rates") or {})
    if line:
        st.markdown(f"<div class='protocol'>{line}</div>", unsafe_allow_html=True)
    cases = data["cases"]
    order = [c for c in CARDS if c in cases]
    for row in (order[:4], order[4:]):
        cols = st.columns(4, gap="small")
        for col, cor in zip(cols, row):
            with col:
                card(cor, cases[cor])
    st.caption(f"Each card: Burgers' equation, 3 runs (colour = u over space → and time ↑), held-out reporting seed. "
               "This panel checks the TRUE base equation against each damaged data set, to show what each check sees "
               "on its own. Hover a check's sentence for its exact statistic.")


# ============================================================================= 3. scoreboard
ARMS = {"A": "A · plain eqdisc (AI agent)", "B": "B · checks only, no AI",
        "C": "C · A's answers re-judged by the checks", "D": "D · AI agent + checks"}
CAT_STYLE = {"right+confident": (GOOD, "right, confident"), "right+cautious": ("#86efac", "right, cautious"),
             "wrong+flagged": (WARN, "wrong, but flagged"), "wrong+confident": (BAD, "wrong and confident"),
             "crashed": (MUTED, "crashed")}


def score_chart(counts, arms):
    fig = go.Figure()
    ylab = [ARMS[a] for a in arms]
    for cat, (color, name) in CAT_STYLE.items():
        n = [counts.get(a, {}).get(cat, 0) for a in arms]
        tot = [max(1, sum(counts.get(a, {}).values())) for a in arms]
        pct = [100 * k / t for k, t in zip(n, tot)]
        fig.add_trace(go.Bar(y=ylab, x=pct, orientation="h", name=name, marker=dict(color=color,
                                                                                     line=dict(color="white", width=2)),
                             text=[f"{k}" if k else "" for k in n], textposition="inside",
                             insidetextanchor="middle", textfont=dict(color="white" if cat != "right+cautious" else "#14532d"),
                             customdata=np.stack([n, tot], 1),
                             hovertemplate="%{y}<br>" + name + ": %{customdata[0]} of %{customdata[1]} runs"
                                           " (%{x:.0f}%)<extra></extra>"))
    fig.update_layout(barmode="stack", height=70 + 52 * len(arms), margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=-0.25, x=0, yanchor="top", traceorder="normal"),
                      xaxis=dict(title="% of runs", range=[0, 100], gridcolor="rgba(128,128,128,.15)"),
                      yaxis=dict(autorange="reversed"), hovermode="closest")
    return fig


def section_scoreboard():
    sb = _load("scoreboard.json")
    ui.section(3, "Confidently wrong: before vs after", "the corruption benchmark")
    if not sb or not sb.get("available"):
        st.markdown("<div class='ev-pending'><b>Results pending.</b> The benchmark is running. Refresh with "
                    "<code>python demo/build_evidence.py scoreboard</code>.</div>", unsafe_allow_html=True)
        return
    counts = sb["counts"]
    splits = [s for s in ("report", "blind", "dev", "all") if s in counts]
    names = {"report": "reporting seeds", "blind": "blinded", "dev": "development seeds", "all": "all runs"}
    sp = st.segmented_control("Data set", splits, default=splits[0], format_func=lambda s: names.get(s, s),
                              key="ev_split", label_visibility="collapsed") or splits[0]
    c = counts.get(sp, {})
    arms = [a for a in ARMS if a in c]
    missing = [a for a in ARMS if a not in c]
    exp = sb.get("expected", {}).get(sp)
    left, right = st.columns([2.2, 1], gap="large")
    with left:
        if arms:
            _show(score_chart(c, arms), f"ev_score_{sp}")
    with right:
        for a in arms:
            n = sum(c[a].values())
            wc = c[a].get("wrong+confident", 0)
            partial = f" · {n}/{exp} runs scored" if exp and n < exp else f" · {n} runs"
            st.metric(f"Confidently wrong, arm {a}", f"{wc}/{n}", partial, delta_color="off", border=True,
                      help=ARMS[a])
    notes = []
    if missing:
        notes.append("pending: " + ", ".join(ARMS[a] for a in missing))
    if exp and any(sum(c[a].values()) < exp for a in arms):
        notes.append("partial results: the benchmark is still running")
    notes.append(f"outcomes as of {sb.get('source_mtime', '?')}")
    st.caption(" · ".join(notes) + ". Right = equivalent to the true base equation; confident = verdict starts with "
               "CONFIDENT; flagged = the verdict names a failed check.")


# ============================================================================= 4. how it works
def section_how():
    with st.expander("How it works"):
        st.markdown(
            "**Every check tests the principle**: the same coefficients on every slice, and only noise left behind. "
            "Checks are plain statistics (numpy/scipy); no AI is involved in the checks, the grade or the verdict.\n\n"
            "**Three responses**, in order:\n"
            "1. **Repair**: the only data changes: fit on gap-free pieces, and clip isolated spikes when the fit is "
            "the same with or without them. Rows are never deleted otherwise.\n"
            "2. **Widen**: when coefficients differ between slices, widen their uncertainty by the between-slice "
            "spread (random effects), so the grade drops by itself.\n"
            "3. **Scope**: restrict the claim to where the data are (\"valid for |u| ≤ X\") and ask for data there.\n\n"
            "A critical failed check makes CONFIDENT impossible; the verdict names it.\n\n"
            "**The checks**\n"
            "- *Before fitting*: isolated spikes (a sample that disagrees with its spatial neighbours at one time step); missing data and uneven "
            "sampling; missing data concentrated at the extremes (censored tails).\n"
            "- *Same coefficients on every slice*: refit on each run, early vs late, left vs right half, small vs "
            "large values; fire when the disagreement exceeds noise (Cochran's Q, I² > 0.75).\n"
            "- *Only noise left behind*: leftover explained by time alone (forcing), by position alone (source), "
            "growing at large values (missing term at the extremes), or simply larger than the noise floor.\n\n"
            "**Honesty.** Thresholds were frozen on development seeds of generated data only "
            "(`eqdisc/audit/thresholds.json`), never on the reporting seeds or the blinded set.\n\n"
            "Docs: `docs/evidence.md`, `docs/honest_oos.md` (section 1b). "
            "Code: `eqdisc/audit/`.")


def page():
    st.html(CSS)
    ui.header("Knowing when not to trust the answer",
              "Checks that stop an equation-discovery system from being confidently wrong")
    st.markdown(f"<div class='ev-principle'>{html.escape(PRINCIPLE)}</div>", unsafe_allow_html=True)
    section_orbit()
    section_grid()
    section_scoreboard()
    section_how()


def home_card():
    """(number, subtitle) for the Home page card, or None if nothing is built yet."""
    g = _load("grid.json")
    if not g:
        return None
    pc = (g.get("rates") or {}).get("per_corruption") or {}
    dmg = [r for c, r in pc.items() if c != "clean" and r["n"]]
    clean = pc.get("clean") or {}
    if not dmg or not clean.get("n"):
        return None
    caught = sum(r["hit"] for r in dmg)
    total = sum(r["n"] for r in dmg)
    return (f"{caught}/{total} caught",
            f"damaged data sets flagged by the checks; clean: {clean['hit']}/{clean['n']} false alarms")
