"""eqdisc demo.   cd /Users/danield/eqdisc && PYTHONPATH=. streamlit run demo/app.py

Three out-of-sample cases (one screen each) + a live "Run on your data" page. Reads only demo/showcase/ and
demo/examples/ (build with demo/build_showcase.py). Rehearsal mode (EQDISC_DEMO_FAKE=1 or the sidebar toggle)
replays a scripted live run without API calls.
"""
import html
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

DEMO = Path(__file__).resolve().parent
REPO = DEMO.parent
for p in (str(REPO), str(DEMO)):
    if p not in sys.path:
        sys.path.insert(0, p)

BRAND = DEMO / "brand"  # Lorenz logo: an abstract vortex (demo/brand/make_logo.py)
st.set_page_config(page_title="Equation Discovery AutoScientist", page_icon=str(BRAND / "icon-64.png"), layout="wide",
                   initial_sidebar_state="collapsed")

import evidence  # noqa: E402
import live  # noqa: E402
import ui  # noqa: E402
import viz  # noqa: E402
import yourdata  # noqa: E402

SHOW = DEMO / "showcase"
ui.inject_css()

PROTOCOL = ("Data only: the agent gets numbers with neutral names, no description of the system, no domain guidance, "
            "and its code cannot read anything else. Honest test: noisy training data → forecast an unseen future or "
            "held-out trajectory from a noisy observed state; coefficients refitted; a neural net trained on the same data.")
TOOL_LABEL = {"diagnose": "diagnose", "intuit": "intuition", "run_sindy": "SINDy", "weak_sindy": "weak SINDy",
              "run_pysr": "PySR", "fit_skeleton": "skeleton fit", "fit_flow": "flow-map fit", "fit_trajectories": "trajectory fit",
              "find_invariants": "invariants", "transform": "coordinates", "set_coordinates": "coordinates",
              "detect_symmetries": "symmetries", "equivariant_sindy": "equivariant SINDy",
              "ensemble_sindy": "ensemble SINDy", "compare_models": "model comparison",
              "coefficient_uncertainty": "coefficient UQ", "assess_model": "assessment", "run_python": "own Python",
              "plot_data": "plots", "plot_model": "plots", "repair": "repair", "validate": "validation"}


def tool_chips(tools, title="what the agent did (from its log)"):
    """Chips from the tool calls the agent actually made: first-use order, with counts."""
    counts = {}
    for t in tools or []:
        lab = TOOL_LABEL.get(t)
        if lab:
            counts[lab] = counts.get(lab, 0) + 1
    if counts:
        ui.chips([f"{k} ×{n}" if n > 1 else k for k, n in counts.items()], title=f"{title} · {len(tools)} tool calls")
J2_ACCEPTED = 1.08263e-3
PAGES = ["Home", "🛰️ LAGEOS-1 satellite", "🔥 Chaos (KS)", "🌀 Gray–Scott patterns", "⚡ Run on your data",
         "⚙️ How it works", "🌍 Orbit with a big bulge", "🔎 When not to trust it"]


@st.cache_data(show_spinner=False)
def load_case(name):
    d = SHOW / name
    if not (d / "case.json").exists():
        return None, {}
    info = json.loads((d / "case.json").read_text())
    arr = {}
    if (d / "arrays.npz").exists():
        with np.load(d / "arrays.npz") as z:
            arr = {k: z[k] for k in z.files}
    return info, arr


def show(fig, key):
    st.plotly_chart(fig, key=key, config={"displaylogo": False, "displayModeBar": False})


def hero_video(case):
    p = SHOW / case / "video.mp4"
    if p.exists():
        st.video(str(p), loop=True, autoplay=True, muted=True)
        return True
    return False


def km(x):
    return f"{x:,.0f} km" if x >= 10 else f"{x:.2g} km"


def go_to(page):
    st.session_state["nav"] = page


def missing(case):
    st.warning(f"No data for this case yet. Build it: `PYTHONPATH=. python demo/build_showcase.py {case}`")


# ============================================================================= LAGEOS-1
def lageos_numbers(info):
    r = info["results"]
    pe = r["position_error_km"]
    return {"J": pe["Kepler + J2"]["30d"], "K": pe["Kepler"]["30d"], "N": pe["neural step model (MLP)"]["30d"],
            "J2": r["Kepler + J2"]["params"]["p1"], "J2s": r["Kepler + J2"]["param_sigma"]["p1"],
            "one_step_gain": r["Kepler"]["one_step_rel_err_heldout"] / r["Kepler + J2"]["one_step_rel_err_heldout"],
            "node": r["node_rate_deg_per_day"]}


def lageos_agent_rhs(ag):
    """The agent's law in blinded names u1..u6 (u1-u3 position, u4-u6 velocity; the agent was not told)."""
    sub = ag.get("refit_rhs") or ag.get("submitted") or {}
    return {k: v for k, v in sub.items() if str(v).strip() not in sub}      # drop trivial u1' = u4 style lines


def lageos_latex(rhs):
    """Agent's law with r = |(u1, u2, u3)| substituted, for display."""
    import sympy as sp
    u = sp.symbols("u1:7")
    r = sp.Symbol("r", positive=True)
    out = []
    for k, e in rhs.items():
        try:
            ex = sp.sympify(e, locals={f"u{i + 1}": u[i] for i in range(6)})
            ex = sp.expand(ex.subs(u[0] ** 2 + u[1] ** 2 + u[2] ** 2, r ** 2))
            ex = ex.xreplace({a: sp.Float(float(a), 4) for a in ex.atoms(sp.Float) if abs(float(a)) < 0.9 or abs(float(a)) > 1.1})
            ex = ex.xreplace({p_: sp.Pow(p_.base, sp.nsimplify(p_.exp)) for p_ in ex.atoms(sp.Pow) if p_.exp.is_Float})
            out.append(rf"\dot{{{sp.latex(sp.Symbol(k))}}} = {sp.latex(ex)}")
        except Exception:  # noqa: BLE001
            out += ui.rhs_latex({k: e})
    return out


LEGEND_ORBIT = """
<div class='small'><b>What the lines mean.</b>
<b>Real satellite</b>: measured positions.
<b>eqdisc</b>: the law of motion the agent worked out from the numbers alone: Newton's gravity plus a correction for
Earth's equatorial bulge (the "J₂" term), which makes the orbit plane slowly turn.
<b>Neural network</b>: learns step-to-step motion directly from the same data, with no equations.
<b>Newton's gravity, perfectly round Earth</b>: the textbook ellipse (Kepler); it misses the bulge, so its orbit plane
never turns and it falls further behind every day.
<b>Textbook model incl. bulge</b>: what an expert would write down, fitted by us; shown for reference only.
</div>"""


def page_lageos():
    info, a = load_case("lageos")
    if not info:
        return missing("lageos")
    n = lageos_numbers(info)
    ag = info.get("agent") or {}
    pe_ag = ag.get("position_error_km") or {}
    ui.header("🛰️ A real satellite", "LAGEOS-1: one year of hourly positions in → its law of motion out")
    ui.section(1, "What it found", "from the data alone")
    c1, c2 = st.columns([1, 1.1], gap="large")
    with c1:
        ui.label("The data it was given: 2017, hourly")
        if "train_orbits" in a:
            show(viz.lageos_training(a["train_orbits"], info["train_dates"]), "lageos_train")
            st.caption("One orbit every two weeks. Press ▶: the orbit plane slowly turns over the year. "
                       "The agent saw only six unnamed columns of numbers.")
    with c2:
        ui.label("The law it found")
        if ag.get("submitted"):
            ui.equations(lageos_latex(lageos_agent_rhs(ag)), small=True)
            st.markdown("<div style='font-size:1.1rem;margin:.4rem 0 .8rem'>In words: <b>Newton's gravity plus a "
                        "correction for Earth's equatorial bulge</b>, inferred from unnamed numbers in random units "
                        "(r² = u1² + u2² + u3²).</div>", unsafe_allow_html=True)
            tool_chips(ag.get("tools"))
            st.caption(f"{ag.get('n_tool_calls', '?')} tool calls, ${ag.get('cost_usd', 0):.2f}")
    if info.get("uq"):
        ui.section(2, "How sure are we, and what next?", "computed from the training data only")
        ui.confidence_panel(info["uq"], "lageos")
    ui.benchmark_section(3)
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        hero_video("lageos")
        st.caption("The first 3 unseen days. Right: each forecast seen from the real satellite.")
    with right:
        if pe_ag:
            ok = pe_ag["30d"] < min(n["N"], n["K"])
            ui.verdict_chip("VALIDATED" if ok else "NOT RECOVERED")
            ui.tiles([
                {"label": "Discovered law: 30-day error", "value": km(pe_ag["30d"]), "delta": f"1 day: {km(pe_ag['1d'])}"},
                {"label": "Neural net, same data", "value": km(n["N"]), "delta": "30 days"},
                {"label": "Round-Earth gravity", "value": km(n["K"]), "delta": "30 days"},
                {"label": "Hand-built textbook law", "value": km(n["J"]), "delta": "30 days (reference)"},
            ])
        errs = {m: a[f"err{i}"] for i, m in enumerate(info["models"])}
        if "err_agent" in a:
            errs["agent"] = a["err_agent"]
        ui.label("Forecast error over the unseen month")
        show(viz.lageos_errors(a["days"], errs), "lageos_err")
    st.markdown(LEGEND_ORBIT, unsafe_allow_html=True)
    lageos_details(info, n, ag, pe_ag)


def lageos_details(info, n, ag, pe_ag):
    with st.expander("Details & caveats"):
        pe = info["results"]["position_error_km"]
        rows = {("physics reference (Kepler + J2)" if k == "Kepler + J2" else k): v for k, v in pe.items()}
        if pe_ag:
            rows["data-only agent"] = pe_ag
        st.markdown("**Forecast position error (km), autoregressive from the last training state**")
        st.dataframe(pd.DataFrame(rows).T.map(lambda v: f"{v:,.3g}"), width="content")
        if ag.get("submitted"):
            st.markdown("**The agent's submitted law** (blinded names and units) and the protocol refit of its constants")
            st.code(json.dumps({"submitted": ag["submitted"], "refit": ag.get("refit")}, indent=1, default=str),
                    language="json")
        alt = info.get("agent_alt") or {}
        if alt.get("position_error_km"):
            st.markdown(
                f"- **Second data-only run** (units where gravity constant = Earth radius = 1): the agent found the same "
                f"Kepler + J₂ structure; 30-day error {km(alt['position_error_km']['30d'])} after refit "
                f"(\\${alt.get('cost_usd', 0):.2f}, {alt.get('n_tool_calls', '?')} tool calls).")
        st.markdown(
            f"- **Protocol.** Train on 2017 only (hourly, real). Forecast January 2018 from the last observed state. "
            "The neural step model is trained on the same year.\n"
            "- **Data only.** The agent sees six unnamed columns (u1…u6) sampled hourly, in random units. No dataset "
            "name, no description, no domain guidance; its Python sandbox cannot read other files. It inferred an "
            "orbit itself (r nearly constant, r oscillating once per period, plane precessing about one axis).\n"
            "- **Refit.** Constants the agent typed (e.g. 4 significant figures) are re-estimated by flow-map shooting "
            "on 2017; only the refitted law is forecast. Unrefitted, rounding alone gave 5,133 km at 30 days.\n"
            "- **Neural net.** The MLP step model looks close at orbit scale, but its orbit plane is tilted: it swings "
            "±1,000 km out of plane every orbit (right panel of the video). It is worse than plain Kepler for the "
            "first day, better after a week (Kepler misses the precession).\n"
            f"- **Physics reference.** Kepler + J₂ fitted by flow-map shooting: J₂ = {n['J2'] * 1e3:.5f}×10⁻³ "
            f"(accepted {J2_ACCEPTED * 1e3:.5f}), node drift {n['node']['Kepler + J2']:.4f} vs measured "
            f"{n['node']['data']:.4f} °/day. This is a human-chosen model, shown only for comparison.\n"
            "- **Retracted.** An earlier agent result (13.6 km) was produced with the system named in the prompt and a "
            "domain skill available; it is not shown.\n"
            "- **Data.** orbit_discover workshop repository (MIT).")
        if (SHOW / "lageos" / "errors.png").exists():
            st.image(str(SHOW / "lageos" / "errors.png"), caption="Error vs time, and node drift over 6 years.")


# ============================================================================= synthetic orbit, big bulge
def page_orbit():
    info, a = load_case("orbit")
    if not info:
        return missing("orbit")
    r = info["results"]
    pe = r["position_error_km"]
    ag = info.get("agent") or {}
    pa = pe.get("data-only agent")
    ui.header("🌍 A planet with a huge bulge", "Synthetic satellite: 3 noisy days in → a law out. Here eqdisc gets it wrong, and says so.")
    ui.section(1, "What it found", "from the data alone")
    c1, c2 = st.columns([1, 1.1], gap="large")
    with c1:
        ui.label("The data it was given: 3 noisy days")
        show(viz.orbit_animation(a["t_train"], a["U_train"]), "orbit_anim")
        st.caption("Press ▶ to fly the first orbits. The faint tangle is all 3 days: the orbit never closes on itself.")
    with c2:
        ui.label("The law it found")
        n_c = len((ag.get("refit") or {}).get("refit_constants") or [])
        st.markdown(f"<div style='font-size:1.1rem;margin:.4rem 0 .8rem'><b>acceleration = position × (a polynomial in "
                    f"distances and speeds)</b>, with {n_c} fitted constants.</div>", unsafe_allow_html=True)
        tool_chips(ag.get("tools"))
        st.caption(f"{ag.get('n_tool_calls', '?')} tool calls, ${ag.get('cost_usd', 0):.2f}. Full law in the details.")
    if info.get("uq") and not info["uq"].get("error"):
        ui.section(2, "How sure are we, and what next?", "computed from the training data only")
        ui.confidence_panel(info["uq"], "orbit")
        st.caption("The ‘missing term’ check tries the same short menu of generic extra terms for any position-and-velocity "
                   "data (an inverse-square pull, drag, an extra radial power); it was not chosen knowing the answer.")
    ui.benchmark_section(3)
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        hero_video("orbit")
        st.caption("The first unseen day. Right: each forecast seen from the real satellite.")
    with right:
        if pa:
            ok = pa["24h"] < min(pe["neural step model (MLP)"]["24h"], pe["Kepler"]["24h"])
            ui.verdict_chip("VALIDATED" if ok else "NOT RECOVERED")
            ui.tiles([
                {"label": "Discovered law: 1-day error", "value": km(pa["24h"]), "delta": f"after 3 days: {km(pa['72h'])}"},
                {"label": "Neural network", "value": km(pe["neural step model (MLP)"]["24h"]), "delta": "after 1 day"},
                {"label": "Round-Earth gravity", "value": km(pe["Kepler"]["24h"]), "delta": "after 1 day"},
                {"label": "True law (best possible)", "value": km(pe["Kepler + J2"]["24h"]), "delta": "after 1 day"},
            ])
            if not ok:
                st.markdown("<div class='small'>🔒 <b>What it missed</b> (known only because we generated the data): "
                            "Newton's 1/r² pull and the bulge term. This matches what the checks above flagged.</div>",
                            unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        if "kj_t" in a:
            ui.label("🔒 The true law: why the bulge matters")
            show(viz.orbit_kepler_vs_j2(a["kj_t"], a["kj_disc"], a["kj_kep"]), "orbit_kj")
            st.caption("Same start, two laws: round-Earth gravity stays in one plane; the true law swings the plane around.")
    with c2:
        errs = {m: a[f"err{i}"] for i, m in enumerate(info["models"])}
        errs = {("agent" if m == "data-only agent" else m): v for m, v in errs.items()}
        ui.label("Forecast error over the unseen 3 days")
        show(viz.lageos_errors((a["hrs"] - a["hrs"][0]) / 24, errs, xlabel="days into the unseen second half"), "orbit_err")
    if "raan_disc" in a:
        ui.label("The tell-tale drift: measured vs forecast")
        show(viz.orbit_elements(a["el_hrs"], a["raan_data"], a["argp_data"], a["hrs"], a["raan_disc"], a["argp_disc"],
                                a["raan_kep"], a["argp_kep"]), "orbit_el")
    st.markdown(LEGEND_ORBIT.replace("Earth's equatorial bulge", "the planet's equatorial bulge").replace(
        "fitted by us; shown for reference only", "here the exact law that generated the data"), unsafe_allow_html=True)
    orbit_details(info, pe, ag)


def orbit_details(info, pe, ag):
    with st.expander("Details & caveats"):
        st.dataframe(pd.DataFrame({PLAIN_ROW.get(k, k): v for k, v in pe.items()}).T.map(lambda v: f"{v:,.3g}"),
                     width="content")
        if ag.get("refit_rhs"):
            st.markdown("**The agent's law (refitted constants, blinded units)**")
            ui.equations(lageos_latex(lageos_agent_rhs(ag)), small=True)
        if ag.get("rationale"):
            st.markdown("**The agent's own reasoning (from its submission)**")
            st.markdown("> " + ag["rationale"][:1500].replace("\n", " "))
        st.markdown(
            "- **Protocol.** 30 s samples over 6 days; 1% noise on every column. Train on the first 3 days only, "
            "forecast the last 3. Each law estimates its starting state by fitting its own trajectory to the last two "
            "training orbits; the neural network gets the true law's estimate (the most generous start).\n"
            "- **Data only.** Six unnamed columns in random units, no description, no domain guidance.\n"
            "- **Retracted.** The earlier version of this case gave the agent the context 'satellite orbiting Earth' "
            "and scored in-sample; it is replaced by this run.")


PLAIN_ROW = {"data-only agent": "eqdisc (data-only agent, refit)", "Kepler": "round-Earth gravity",
             "Kepler + J2": "true law (generator)", "neural step model (MLP)": "neural network"}


# ============================================================================= Kuramoto-Sivashinsky
def _coef_dict(expr):
    import sympy as sp
    from eqdisc.solvers import parse
    names = ui._sym_names([expr])
    e = sp.expand(parse(expr, names))
    return {str(m): float(c) for m, c in e.as_coefficients_dict().items()}


def _canon(term):
    from eqdisc.solvers import parse
    return str(parse(term, ui._sym_names([term])))


def page_ks():
    info, a = load_case("ks")
    if not info:
        return missing("ks")
    r = info["results"]
    vt = r["valid_time_lyapunov"]
    ag = r["agent"]
    truth = _coef_dict(r["truth"]["u"])
    rows = []
    for c in ag["coefficients"]:
        tv = truth.get(_canon(c["term"]))
        rows.append({"term": c["term"], "truth": tv, "refit": c["refit"], "90% CI": c["ci90"],
                     "inside": tv is not None and c["ci90"][0] <= tv <= c["ci90"][1]})
    n_in = sum(x["inside"] for x in rows)
    ui.header("🔥 Chaos", "A chaotic field with 2% noise in → its equation out. How long can it forecast?")
    ui.section(1, "What it found", "from the data alone")
    c1, c2 = st.columns([1, 1.1], gap="large")
    with c1:
        ui.label("The data it was given")
        st.markdown("<div style='font-size:1.1rem'>One unnamed field u(x, t): 1,024 points × 400 snapshots, "
                    "2% noise, rescaled so no textbook coefficient applies.</div>", unsafe_allow_html=True)
    with c2:
        ui.label("The equation it found")
        ui.equations(ui.rhs_latex(ag["refit"] if isinstance(ag["refit"], dict) else {"u": ag["refit"]}, pde=True, digits=5))
        tool_chips(info.get("tools"))
    if info.get("uq"):
        ui.section(2, "How sure are we, and what next?", "computed from the training data only")
        ui.confidence_panel(info["uq"], "ks", names={"u_xx": "u_xx (anti-diffusion)", "u_xxxx": "u_xxxx (hyper-diffusion)",
                                                       "u*u_x": "u·u_x (steepening)"})
    ui.benchmark_section(3)
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        hero_video("ks")
        st.caption("The unseen future unfolding left to right: truth, eqdisc's equation, a neural operator; the dark rows "
                   "show where each is wrong. Chaos defeats every forecast eventually.")
    with right:
        ui.tiles([
            {"label": "eqdisc forecast", "value": f"{vt['eqdisc agent (refit)']:.2f} λ",
             "delta": f"true equation: {vt['true PDE from noisy state']:.2f}",
             "help": "Lyapunov times until the relative error exceeds 0.5."},
            {"label": "Neural operator", "value": f"{vt['FNO (same noisy data)']:.2f} λ", "delta": "same noisy data"},
            {"label": "True numbers inside its ± ranges", "value": f"{n_in}/{len(rows)}", "delta": "from section 2"},
            {"label": "Agent cost", "value": f"${info.get('cost_usd') or 0:.2f}",
             "delta": f"{(info.get('wall_s') or 0) / 60:.0f} min" if info.get("wall_s") else None},
        ])
        errs = {lab: a[f"err{i}"] for i, lab in enumerate(info["labels"]) if i > 0}
        ui.label("Forecast error, in Lyapunov times")
        show(viz.ks_errors(a["t_lyap"], errs), "ks_err")
    ks_details(info, r, rows)


def ks_details(info, r, rows):
    with st.expander("Details & caveats"):
        st.dataframe(pd.DataFrame([{**x, "90% CI": f"[{x['90% CI'][0]:.4g}, {x['90% CI'][1]:.4g}]"} for x in rows]),
                     hide_index=True)
        st.markdown(
            f"- **Protocol.** Real KS data, blinded by rescaling x, t and u, so the coefficients are not textbook values. "
            f"Add 2% noise, train on t ∈ [0, {r['train_window'][1]:.1f}], forecast t ∈ [{r['test_window'][0]:.1f}, "
            f"{r['test_window'][1]:.1f}] from the noisy last state. Lyapunov exponent {r['lyapunov_exponent']:.3f}.\n"
            "- **Data only.** One unnamed field u(x, t); no description, no domain guidance.\n"
            "- **Caveat.** Here weak SINDy alone (no LLM) does as well as the agent. The agent adds the verdict, the "
            "per-term evidence and the refit with confidence intervals.\n"
            f"- FNO trained for {r.get('fno', {}).get('train_minutes', '?')} min on the same noisy training window.")
        if (SHOW / "ks" / "spacetime.png").exists():
            st.image(str(SHOW / "ks" / "spacetime.png"), caption="Space–time forecasts (top) and errors (bottom).")
        rp = SHOW / "ks" / "report.html"
        if rp.exists():
            st.download_button("Full agent report (HTML)", rp.read_bytes(), "ks_report.html", "text/html")


# ============================================================================= Gray-Scott
GS_PENDING = "true PDE (reference; forecasts pending)"
SHORT = {"eqdisc agent (refit)": "Agent", "weak SINDy (no LLM)": "SINDy", "FNO (same noisy data)": "FNO",
         "true PDE from noisy frame": "True PDE"}


def page_gs():
    info, a = load_case("gray_scott")
    if not info:
        return missing("gray_scott")
    res = info.get("results")
    vr = (res or {}).get("vrmse") or {}
    agent = (res or {}).get("agent")
    ui.header("🌀 Patterns", "Two noisy movies of a reacting, diffusing pair of chemicals in → their equations out")
    ui.section(1, "What it found", "from the data alone")
    c1, c2 = st.columns([1, 1.1], gap="large")
    with c1:
        ui.label("The data it was given")
        st.markdown("<div style='font-size:1.1rem'>Two unnamed fields A, B on a 128 × 128 grid: two runs × 60 frames, "
                    "5% noise. (Data: The Well, Gray–Scott.)</div>", unsafe_allow_html=True)
    with c2:
        ui.label("The equations it found")
        if agent:
            ui.equations(ui.rhs_latex(agent["refit"], pde=True, digits=3), small=True)
            tool_chips(info.get("tools"))
    if info.get("uq") and not info["uq"].get("error"):
        ui.section(2, "How sure are we, and what next?", "computed from the training data only")
        ui.confidence_panel(info["uq"], "gs", names={"A_xx": "diffusion of A (x)", "A_yy": "diffusion of A (y)",
                                                      "B_xx": "diffusion of B (x)", "B_yy": "diffusion of B (y)",
                                                      "A*B**2": "reaction A·B²", "A": "decay of A", "B": "decay of B",
                                                      "1": "feed (constant)"})
    ui.benchmark_section(3)
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        if info.get("has_video") and hero_video("gray_scott"):
            st.caption("A third run it never saw, forecast from its noisy first frame. Bottom row: where each forecast is wrong.")
    with right:
        if agent:
            ui.verdict_chip("VALIDATED")
        if vr:
            best = "eqdisc agent (refit)" if "eqdisc agent (refit)" in vr else "weak SINDy (no LLM)"
            t = []
            for lab in (best, "FNO (same noisy data)", "true PDE from noisy frame"):
                if lab in vr:
                    t.append({"label": f"{SHORT.get(lab, lab)}: error", "value": f"{vr[lab]['6-12']:.3g}",
                              "delta": f"later steps: {vr[lab]['13-30']:.3g}", "help": "VRMSE, rollout steps 6–12"})
            t.append({"label": "Best published neural net", "value": "0.29", "delta": "trained on 100s of runs"})
            ui.tiles(t[:4])
            ui.label("Forecast error (lower is better)")
            show(viz.gs_vrmse(vr), "gs_vrmse")
        if info.get("uq"):
            st.markdown("<div class='small'>🔒 Its equations are in fact the true ones, and forecast as well as the true "
                        "equations do. Section 2 could not know that: with two noisy runs, close rival versions fit "
                        "equally well, so ‘collect more data’ was the right call.</div>", unsafe_allow_html=True)
    gs_details(info, agent, vr)


def gs_details(info, agent, vr):
    with st.expander("Details & caveats"):
        if agent and agent.get("coefficients"):
            st.dataframe(pd.DataFrame(agent["coefficients"]), hide_index=True)
        if vr:
            st.dataframe(pd.DataFrame(vr).T, width="content")
        sw = info.get("sweep")
        if sw:
            rows = []
            for k, v in sw.items():
                r_, n_ = k.split("|")
                for arm in ("sindy", "weak_sindy", "agent"):
                    s = v.get(arm) or {}
                    rows.append({"regime": r_, "noise": n_, "arm": arm, "term F1": s.get("f1"),
                                 "VRMSE 6–12": s.get("vrmse_6-12"), "VRMSE 13–30": s.get("vrmse_13-30")})
            st.markdown("**Noise sweep, all regimes**")
            st.dataframe(pd.DataFrame(rows), hide_index=True)
        p = info.get("params") or {}
        st.markdown(
            f"- **Data.** The Well (Ohana et al., NeurIPS 2024), gray_scott_reaction_diffusion, regime '{info['regime']}' "
            f"(F = {p.get('F')}, k = {p.get('k')}). Train: 2 trajectories × 60 frames with {info['noise']:.0%} noise. "
            "Test: a third trajectory, forecast from its noisy first frame.\n"
            "- **Data only.** Two unnamed fields A, B on a grid; no dataset name, no description, no domain guidance. "
            "Coefficients are refitted from the data, so remembered parameter values earn nothing.\n"
            "- **Retracted.** An earlier noise sweep showed the agent the dataset name and a one-line description; "
            "it typed The Well's published parameters exactly, so it is not shown.\n"
            "- **Reference lines.** The Well paper's neural surrogates, trained on hundreds of trajectories, rollout VRMSE "
            "windows 6–12 / 13–30: FNO 0.89 / >10, U-net 0.57 / >10, CNextU-net 0.29 / 7.62. Their windows start at "
            "the trajectory's beginning; ours at snapshot 50, so the comparison is indicative.")


# ============================================================================= home
def page_home():
    st.markdown("# eqdisc: equations that forecast")
    st.markdown("Give it measurements; Claude agents return the governing equation, how sure they are, and what to "
                "measure next.")
    st.markdown(f"<div class='protocol'>✅ {PROTOCOL}</div>", unsafe_allow_html=True)
    st.caption("🔒 Card numbers are benchmark scores against the hidden future, which eqdisc never sees.")
    cards = []
    info, _ = load_case("lageos")
    if info:
        n = lageos_numbers(info)
        pe_ag = (info.get("agent") or {}).get("position_error_km")
        if pe_ag:
            cards.append(("🛰️ LAGEOS-1 satellite (real)", f"{km(pe_ag['30d'])} vs {km(n['N'])}",
                          "30-day forecast error: data-only agent vs neural net", "lageos", PAGES[1]))
        else:
            cards.append(("🛰️ LAGEOS-1 satellite (real)", "agent pending",
                          f"30-day error: Kepler {km(n['K'])}, neural net {km(n['N'])}", "lageos", PAGES[1]))
    info, _ = load_case("orbit")
    if info and (info["results"]["position_error_km"].get("data-only agent")):
        pe = info["results"]["position_error_km"]
        ag_, nn_ = pe['data-only agent']['24h'], pe['neural step model (MLP)']['24h']
        if ag_ < nn_:
            cards.append(("🌍 Orbit with a big bulge (synthetic)", f"{km(ag_)} vs {km(nn_)}",
                          "1-day forecast error: data-only agent vs neural net", "orbit", PAGES[6]))
        else:
            cards.append(("🌍 Orbit with a big bulge (synthetic)", "✗ flags its own law",
                          f"its checks say ‘not established’; 🔒 rightly: after 1 day it is {km(ag_)} off",
                          "orbit", PAGES[6]))
    info, _ = load_case("ks")
    if info:
        vt = info["results"]["valid_time_lyapunov"]
        cards.append(("🔥 Chaos, blinded (KS)",
                      f"{vt['eqdisc agent (refit)']:.1f} vs {vt['FNO (same noisy data)']:.1f}",
                      "Lyapunov times forecast: agent vs FNO", "ks", PAGES[2]))
    info, _ = load_case("gray_scott")
    if info:
        vr = (info.get("results") or {}).get("vrmse") or {}
        lab = "eqdisc agent (refit)"
        if lab in vr:
            num, sub = f"{vr[lab]['6-12']:.2g}", "rollout VRMSE (steps 6–12); The Well paper's best: 0.29"
        else:
            num = "agent pending"
            fno = vr.get("FNO (same noisy data)")
            sub = (f"VRMSE 6–12: FNO on same data {fno['6-12']:.2g}; The Well paper's best 0.29" if fno
                   else "forecast a held-out trajectory")
        cards.append(("🌀 Gray–Scott (The Well)", num, sub, "gray_scott", PAGES[3]))
    ev = evidence.home_card()
    if ev:
        cards.append(("🔎 When not to trust it", ev[0], ev[1], "evidence", PAGES[7]))
    cols = st.columns(len(cards) or 1, gap="medium")
    for c, (name, num, sub, case, page) in zip(cols, cards):
        with c, st.container(border=True):
            th = SHOW / case / "thumb.jpg"
            if th.exists():
                st.image(str(th), width="stretch")
            st.markdown(f"<div class='card-name'>{name}</div><div class='card-num'>{num}</div>"
                        f"<div class='card-sub'>{sub}</div>", unsafe_allow_html=True)
            st.button("Open →", key=f"open_{case}", on_click=go_to, args=(page,), width="stretch")


# ============================================================================= live
EXAMPLES = {
    "Example: pendulum (time series)": (DEMO / "examples" / "pendulum.csv", ""),
    "Example: E. coli growth (static law)": (DEMO / "examples" / "ecoli_growth.csv", ""),
}


def live_chips(events):
    out = []
    for e in events:
        if e.get("type") != "tool":
            continue
        res = e.get("rhs") or e.get("expr")
        if isinstance(res, dict):
            nterms = sum(str(v).count("+") + str(v).lstrip("-").count("-") + 1 for v in res.values())
            out.append((e["name"], f"{nterms} terms"))
        elif res:
            out.append((e["name"], f"val NMSE {e['val_nmse']:.2g}" if e.get("val_nmse") is not None else "candidate"))
        elif e["name"] in ("intuit", "describe", "find_invariants", "detect_symmetries"):
            out.append((e["name"], "data probed"))
    seen, uniq = set(), []
    for c in reversed(out):
        if c[0] not in seen:
            seen.add(c[0])
            uniq.append(c)
    return list(reversed(uniq))[-4:]


@st.cache_data(show_spinner=False)
def _model_figure(dataset_path, rhs_json, out_png):
    from eqdisc.evaluate import load
    from eqdisc.plots import plot_model
    meta, data = load(dataset_path)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    try:
        return meta["kind"], plot_model(meta, data, json.loads(rhs_json), out_png)
    except Exception:  # noqa: BLE001
        return meta["kind"], None


def render_live_result(job):
    res = job.result
    if res["kind"] == "dynamics":
        kind, png = "ode", None
        if res.get("dataset_path") and res.get("final_model"):
            kind, png = _model_figure(res["dataset_path"], json.dumps(res["final_model"]),
                                      str(Path(job.run_dir) / "demo_figs" / "model.png"))
        v = res.get("verdict") or {}
        left, right = st.columns([1, 1.15], gap="large")
        with left:
            ui.verdict_chip(v.get("status"), "rehearsal" if res.get("rehearsal") else None)
            ui.equations(ui.rhs_latex(res.get("final_model") or {}, pde=kind == "pde"))
            if v.get("recommendation"):
                st.caption(v["recommendation"][:220])
            ui.chips(live_chips(job.events))
        with right:
            if png:
                st.image(png, caption="Model rollout vs a held-out trajectory, and residuals.")
        with st.expander("Details & caveats"):
            for s in ((res.get("story") or {}).get("key_steps") or []):
                st.markdown(f"- **{s.get('decision', '')}** {s.get('outcome', '')}")
            ui.terms_table(res.get("assessment"))
            ui.experiments(res.get("assessment"))
            if res.get("rehearsal_note"):
                st.caption(res["rehearsal_note"])
            c1, c2 = st.columns(2)
            rp = Path(res["report"]) if res.get("report") else None
            if rp and rp.exists():
                c1.download_button("Full report (HTML)", rp.read_bytes(), rp.name, "text/html")
            c2.download_button("Result JSON", json.dumps(res, default=str, indent=1), "discovery.json")
    else:
        from eqdisc.sr import evaluate_expr
        names, target, expr = res["names"], res["target"], res.get("expr")
        if not expr:
            st.error("No expression was found.")
            return
        v = res.get("verdict") or {"status": "RESULT"}
        df = pd.read_csv(res["csv"])
        _, _, X, y, _, _ = live.static_task_arrays(df, target)
        yh = evaluate_expr(expr, names, X)
        left, right = st.columns([1, 1.15], gap="large")
        with left:
            ui.verdict_chip(v.get("status"), "rehearsal" if res.get("rehearsal") else None)
            ui.equations([f"{ui.expr_latex(target, [target])} = {ui.expr_latex(expr, names)}"])
            ui.tiles([{"label": "Validation NMSE", "value": f"{res.get('val_nmse', float('nan')):.2g}"},
                      {"label": "Cost · time", "value": f"${res.get('cost_usd') or 0:.2f}", "delta": f"{job.elapsed:.0f} s"}])
            ui.chips(live_chips(job.events))
        with right:
            show(viz.pred_vs_true(y, yh), "live_pvt")
        with st.expander("Details & caveats"):
            if v.get("recommendation"):
                st.markdown(v["recommendation"])
            ui.terms_table(res.get("assessment"), static=True)
            ui.experiments(res.get("assessment"))
            st.download_button("Result JSON", json.dumps(res, default=str, indent=1), "sr_result.json")


def page_live(fake):
    ui.question("Run on your data")
    job = st.session_state.get("job")
    running = job is not None and not job.done
    c1, c2 = st.columns([1.3, 1], gap="large")
    with c1:
        src = st.radio("Data", ["Upload a CSV"] + list(EXAMPLES), horizontal=True, disabled=running, key="src",
                       label_visibility="collapsed")
        df, ctx_default, fname = None, "", None
        if src == "Upload a CSV":
            up = st.file_uploader("CSV: a time column + one column per variable (optional trajectory id), "
                                  "or one row per measurement", type=["csv", "tsv", "txt"], disabled=running)
            if up is not None:
                df, fname = pd.read_csv(up, sep=None, engine="python"), Path(up.name).stem
        else:
            path, ctx_default = EXAMPLES[src]
            if path.exists():
                df, fname = pd.read_csv(path), path.stem
        context = st.text_input("Context (optional; leave empty for a data-only run)", value=ctx_default,
                                disabled=running, key=f"ctx_{src}")
    with c2:
        mode = st.segmented_control("Mode", ["Auto", "Dynamics", "Static y = f(x)"], default="Auto",
                                    disabled=running, key="mode") or "Auto"
        resolved = mode
        if df is not None and mode == "Auto":
            resolved = "Dynamics" if live.detect_mode(df) == "dynamics" else "Static y = f(x)"
        target = None
        if df is not None and resolved.startswith("Static"):
            num = list(df.select_dtypes("number").columns)
            target = st.selectbox("Target", num, index=len(num) - 1, disabled=running)
        budget = st.segmented_control("Budget", ["Quick", "Full"], default="Quick", disabled=running, key="budget") or "Quick"
        go_btn = st.button("🚀 Discover", type="primary", disabled=running or df is None)
    st.caption((f"{df.shape[0]} rows × {df.shape[1]} columns · {resolved}" if df is not None else "")
               + (" · 🎭 rehearsal mode (no API calls)" if fake else ""))

    if go_btn and df is not None:
        run_dir = live.RUNS / time.strftime("%Y%m%d-%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        csv_path = run_dir / f"{live.ident(fname or 'data')}.csv"
        df.to_csv(csv_path, index=False)
        quick = budget == "Quick"
        if resolved.startswith("Dyn"):
            job = live.Job(live.fake_dynamics if fake else live.run_dynamics, csv_path=str(csv_path),
                           run_dir=str(run_dir), n_branches=2 if quick else 3, adversary=not quick, context=context)
        else:
            job = live.Job(live.fake_static if fake else live.run_static, csv_path=str(csv_path), target=target,
                           context=context, n_sessions=2 if quick else 3)
        job.run_dir = str(run_dir)
        st.session_state.job = job.start()
        running = True

    job = st.session_state.get("job")
    if job is None:
        return
    with st.status("Agents at work…" if not job.done else "Done", expanded=not job.done,
                   state="running" if not job.done else ("error" if job.error else "complete")) as status:
        timer = st.empty()
        logph = st.container(height=300).empty()

        def paint():
            lines = [live.fmt_event(e).replace("$", "\\$") for e in job.events]
            logph.markdown("\n\n".join(lines[-40:]) or "_starting…_", unsafe_allow_html=True)
            timer.caption(f"⏱️ {job.elapsed:.0f} s · {sum(e.get('type') == 'tool' for e in job.events)} tool calls")
        while not job.done:
            job.drain()
            paint()
            time.sleep(0.4)
        job.drain()
        paint()
        status.update(label="Failed" if job.error else f"Done in {job.elapsed:.0f} s",
                      state="error" if job.error else "complete", expanded=False)
    if running:
        st.rerun()
    if job.error:
        st.error(job.error)
        return
    render_live_result(job)


# ============================================================================= how it works
def page_how():
    ui.question("How it works")
    st.graphviz_chart("""
digraph G { rankdir=LR; bgcolor="transparent"; node [shape=box, style="rounded,filled", fillcolor="#eef2ff",
  color="#6366f1", fontname="Helvetica", fontsize=11]; edge [color="#888888"];
  D [label="data"]; I [label="ingest +\\ndata card"]; P [label="intuition"];
  B [label="parallel Claude agents\\n(weak SINDy, PySR, skeleton fits, flow-map fits,\\ninvariants, sandboxed Python, plots)"];
  T [label="tournament"]; A [label="red team", fillcolor="#fee2e2", color="#dc2626"];
  Q [label="assessment\\nCIs · ΔBIC · noise floor"]; V [label="verdict +\\nnext experiment", fillcolor="#dcfce7", color="#16a34a"];
  D->I->P->B->T->A->Q->V; }""")
    st.markdown(f"<div class='protocol'>✅ {PROTOCOL}</div>", unsafe_allow_html=True)
    st.markdown("| verdict | meaning |\n|---|---|\n"
                "| ✓ CONFIDENT | every term supported, error at the noise floor, predicts held-out data |\n"
                "| ◐ COLLECT MORE DATA | competing models remain; it names the experiment that separates them |\n"
                "| ? INCONCLUSIVE | the data cannot determine the model |")
    with st.expander("Retracted / earlier experiments (not shown as results)"):
        st.markdown(
            "- **Domain leakage (retracted).** Earlier agent runs could see the dataset name, a one-line description "
            "of the system, and domain 'skills' (orbital mechanics, oscillators, PDEs, kinetics). All are removed; "
            "every agent result shown now comes from a data-only rerun.")
        st.markdown(
            "- Synthetic orbit with exaggerated J₂ = 0.5: recovered, but noise-free generator data.\n"
            "- Noisy pendulum: verdict COLLECT MORE DATA with a ranked next experiment.\n"
            "- LLM-SR E. coli growth: low error, but the equation structure is published (possible recall).\n"
            "- Unpublished oscillators, 3/4 exact vs PySR 0/4: retracted (3-digit coefficients the agent typed exactly; rerun with full precision, noise and blinded names).\n"
            "- Real KS, unblinded: textbook coefficients, so recall cannot be excluded.")


# ============================================================================= main
# ============================================================================= presentation pages (v3)
CHART_FONT = "Inter, sans-serif"


def _bigfont(fig, h=None):
    """One font (the app's Inter) and one size for every chart text: ticks, axis titles, legend."""
    fig.update_layout(font=dict(size=15, family=CHART_FONT), legend=dict(font=dict(size=15, family=CHART_FONT)))
    fig.update_xaxes(title_font=dict(size=15), tickfont=dict(size=15))
    fig.update_yaxes(title_font=dict(size=15), tickfont=dict(size=15))
    if h:
        fig.update_layout(height=h)
    return fig


def _hero(case_dir, law_fn, uq, names=None, video_title="Forecast vs reality"):
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        ui.fig_title(video_title, locked=True)
        hero_video(case_dir)
    with right:
        ui.fig_title("The verdict")
        if uq:
            ui.verdict_box(uq)
        ui.fig_title("The law it found")
        law_fn()
        if uq:
            ui.next_box(uq, names)


def _reasoning(text, tools=None):
    if tools:
        tool_chips(tools, title="tools it used")
    if text:
        st.markdown(text.replace("\n", "\n\n"))


def v3_satellite():
    info, a = load_case("lageos")
    if not info:
        return missing("lageos")
    ag = info.get("agent") or {}
    uq = info.get("uq") or {}
    ui.page_title("Satellite", "Real satellite LAGEOS-1 · one year of hourly positions")

    def law():
        ui.equations(lageos_latex(lageos_agent_rhs(ag)), small=True)
        st.markdown("<div class='law'>Newton's gravity + Earth's equatorial bulge</div>", unsafe_allow_html=True)
    _hero("lageos", law, uq)
    errs = {m: a[f"err{i}"] for i, m in enumerate(info["models"])}
    if "err_agent" in a:
        errs["agent"] = a["err_agent"]
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        ui.fig_title("Training data")
        if "train_orbits" in a:
            show(_bigfont(viz.lageos_training(a["train_orbits"], info["train_dates"]), 430), "sat_train")
    with c2:
        ui.fig_title("Forecast error (km)", locked=True)
        show(_bigfont(viz.lageos_errors(a["days"], errs, xlabel="days into the unseen month"), 430), "sat_err")
    with c3:
        ui.fig_title("How precisely known")
        f = ui.precision_fig(uq, {"central pull (1/r²)": "gravity", "equatorial bulge (J₂)": "bulge (J₂)"})
        if f:
            show(_bigfont(f), "sat_prec")
    with st.expander("How it got there", icon=":material/psychology:"):
        _reasoning(info.get("rationale"), ag.get("tools"))
    with st.expander("What the equation means", icon=":material/function:"):
        st.markdown(
            "- **1/r² term:** Newton's gravity, the pull toward Earth's centre.\n"
            "- **Bulge term (J₂):** Earth is fatter at the equator; this extra pull makes the orbit's plane slowly turn "
            f"(measured ≈ {lageos_numbers(info)['node']['data']:.3f}°/day).\n"
            "- The agent inferred both from unnamed numbers in random units: it was never told this is a satellite.")
    with st.expander("The checks behind the verdict", icon=":material/fact_check:"):
        st.markdown(ui.checks_md(uq))
        for adv in uq.get("data_advice") or []:
            st.caption(adv)
    with st.expander("Benchmark details", icon=":material/lock:"):
        st.markdown(LEGEND_ORBIT, unsafe_allow_html=True)
        n = lageos_numbers(info)
        lageos_details(info, n, ag, ag.get("position_error_km") or {})


def v3_bulge():
    info, a = load_case("orbit")
    if not info:
        return missing("orbit")
    ag = info.get("agent") or {}
    uq = info.get("uq") or {}
    ui.page_title("Big Bulge Orbit", "Synthetic satellite · 3 noisy days · here it gets it wrong, and says so")

    def law():
        n_c = len((ag.get("refit") or {}).get("refit_constants") or [])
        st.markdown(f"<div class='law'>acceleration = position × polynomial<br><span style='opacity:.6;font-weight:500'>"
                    f"{n_c} fitted constants, no 1/r² gravity</span></div>", unsafe_allow_html=True)
    _hero("orbit", law, uq, names={"which law is right (current, or with an added 1/r² pull)": "which law is right"})
    errs = {m: a[f"err{i}"] for i, m in enumerate(info["models"])}
    errs = {("agent" if m == "data-only agent" else m): v for m, v in errs.items()}
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        ui.fig_title("Training data")
        show(_bigfont(viz.orbit_animation(a["t_train"], a["U_train"]), 470), "bb_train")
    with c2:
        ui.fig_title("Forecast error (km)", locked=True)
        show(_bigfont(viz.lageos_errors((a["hrs"] - a["hrs"][0]) / 24, errs, xlabel="days"), 470), "bb_err")
    with c3:
        if "kj_t" in a:
            ui.fig_title("What it missed", locked=True)
            show(_bigfont(viz.orbit_kepler_vs_j2(a["kj_t"], a["kj_disc"], a["kj_kep"]), 470), "bb_kj")
    with st.expander("How it got there", icon=":material/psychology:"):
        _reasoning(ag.get("rationale"), ag.get("tools"))
    with st.expander("What went wrong", icon=":material/function:"):
        st.markdown(
            "- It found that velocity is the rate of change of position, and that the motion is symmetric about one axis.\n"
            "- It then fitted a smooth polynomial instead of Newton's 1/r² gravity plus a bulge term.\n"
            "- Its own checks catch this from the training data alone: an added 1/r² pull is strongly favoured, and a "
            "hold-out forecast fails at once. So it says *don't trust this law* and where to measure next.\n"
            "- The ‘missing term’ check tries a short generic menu (a 1/r² pull, drag, an extra radial power) for any "
            "position-and-velocity data; it was not chosen knowing the answer.")
    with st.expander("The checks behind the verdict", icon=":material/fact_check:"):
        st.markdown(ui.checks_md(uq))
        f = ui.precision_fig(uq)
        if f:
            show(_bigfont(f), "bb_prec")
    with st.expander("Benchmark details", icon=":material/lock:"):
        if "raan_disc" in a:
            st.markdown("**The tell-tale drift: measured vs forecast**")
            show(viz.orbit_elements(a["el_hrs"], a["raan_data"], a["argp_data"], a["hrs"], a["raan_disc"], a["argp_disc"],
                                    a["raan_kep"], a["argp_kep"]), "bb_el")
        orbit_details(info, info["results"]["position_error_km"], ag)


def v3_chaos():
    info, a = load_case("ks")
    if not info:
        return missing("ks")
    r = info["results"]
    ag = r["agent"]
    uq = info.get("uq") or {}
    names = {"u_xx": "anti-diffusion", "u_xxxx": "hyper-diffusion", "u*u_x": "steepening"}
    ui.page_title("Blind Chaos (KS)", "A chaotic field · rescaled so no textbook numbers apply · 2% noise")
    import re as _re
    for e in uq.get("experiments") or []:            # plain words for the audience
        m_ = _re.match(r"single Fourier mode k=(\d+)", e.get("description") or "")
        if m_:
            e["description"] = f"Start from one clean ripple ({m_.group(1)} waves across)"

    def law():
        ui.equations(ui.rhs_latex(ag["refit"] if isinstance(ag["refit"], dict) else {"u": ag["refit"]}, pde=True, digits=4))
    _hero("ks", law, uq, names=names)
    truth = _coef_dict(r["truth"]["u"])
    rows = []
    for c in ag["coefficients"]:
        tv = truth.get(_canon(c["term"]))
        rows.append({"term": c["term"], "truth": tv, "refit": c["refit"], "90% CI": c["ci90"],
                     "inside": tv is not None and c["ci90"][0] <= tv <= c["ci90"][1]})
    c1, c2 = st.columns(2, gap="large")
    with c1:
        ui.fig_title("Forecast error (Lyapunov times)", locked=True)
        errs = {lab: a[f"err{i}"] for i, lab in enumerate(info["labels"]) if i > 0}
        show(_bigfont(viz.ks_errors(a["t_lyap"], errs), 400), "ks_err3")
    with c2:
        ui.fig_title("How precisely known")
        f = ui.precision_fig(uq, names)
        if f:
            show(_bigfont(f), "ks_prec3")
    with st.expander("How it got there", icon=":material/psychology:"):
        story = info.get("story") or {}
        tool_chips(info.get("tools"), title="tools it used")
        for s_ in story.get("key_steps") or []:
            st.markdown(f"- **{s_.get('observation', '')}** → {s_.get('decision', '')}")
    with st.expander("What the equation means", icon=":material/function:"):
        st.markdown(
            "- **u_xx with a minus sign (anti-diffusion):** pumps energy into long waves (the instability).\n"
            "- **u_xxxx (hyper-diffusion):** kills short waves, so cells of a preferred size form.\n"
            "- **u·u_x (steepening):** moves energy between scales; together these make cellular chaos.\n"
            "- This is the Kuramoto–Sivashinsky equation, but rescaled: its numbers appear in no textbook.")
    with st.expander("The checks behind the verdict", icon=":material/fact_check:"):
        st.markdown(ui.checks_md(uq))
    with st.expander("Benchmark details", icon=":material/lock:"):
        ks_details(info, r, rows)


def v3_reaction():
    info, a = load_case("gray_scott")
    if not info:
        return missing("gray_scott")
    res = info.get("results") or {}
    vr = res.get("vrmse") or {}
    agent = res.get("agent")
    uq = info.get("uq") or {}
    names = {"A_xx": "diffusion of A", "A_yy": "diffusion of A (y)", "B_xx": "diffusion of B", "B_yy": "diffusion of B (y)",
             "A*B**2": "reaction A·B²", "A": "decay of A", "B": "decay of B", "1": "feed"}
    ui.page_title("Reaction-Diffusion (Chemistry)", "Two chemicals reacting and spreading · two noisy movies")

    def law():
        if agent:
            ui.equations(ui.rhs_latex(agent["refit"], pde=True, digits=3), small=True)
    _hero("gray_scott", law, uq, names=names)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        ui.fig_title("Forecast error", locked=True)
        if vr:
            show(_bigfont(viz.gs_simple(vr), 400), "gs_err3")
    with c2:
        ui.fig_title("How precisely known")
        f = ui.precision_fig(uq, names)
        if f:
            show(_bigfont(f), "gs_prec3")
    with st.expander("How it got there", icon=":material/psychology:"):
        _reasoning(info.get("rationale"), info.get("tools"))
    with st.expander("What the equations mean", icon=":material/function:"):
        st.markdown(
            "- **Feed and decay:** chemical A is supplied, B is removed.\n"
            "- **A·B² reaction:** B converts A into more B (autocatalysis), the engine of the patterns.\n"
            "- **Diffusion:** A spreads faster than B; that mismatch is what makes spots and spirals (a Turing mechanism).")
    with st.expander("The checks behind the verdict", icon=":material/fact_check:"):
        st.markdown(ui.checks_md(uq))
        st.caption("Hidden truth: its equations turn out to be the true ones. The checks could not know that: with two noisy runs, "
                   "close rival versions fit equally well, so ‘collect more data’ is the right call.")
    with st.expander("Benchmark details", icon=":material/lock:"):
        gs_details(info, agent, vr)


def v3_hidden():
    info, a = load_case("hidden_oscillator")
    if not info:
        return missing("hidden_oscillator")
    uq = info.get("uq") or {}
    pol = info.get("polar") or {}
    ui.page_title("Hidden Oscillator", "Synthetic · 3 unnamed variables, 4 noisy runs · the law hides behind a tilted, "
                                       "stretched ellipse and a square root")

    def law():
        if pol:
            ui.equations([rf"\dot r = {pol['growth']:.3f}\, r\,(1 - r)",
                          rf"\dot\varphi = {pol['omega0']:.3f} + {pol['omega1']:.3f}\, r",
                          rf"\dot u_2 = -{pol['decay']:.3f}\, u_2 + {pol['drive']:.3f}\, r"], small=True)
        st.markdown("<div class='law'>a self-sustained oscillation, in coordinates it found itself<br>"
                    "<span style='opacity:.6;font-weight:500'>r, φ: distance and angle around a hidden centre, "
                    "after undoing a tilt and a stretch</span></div>", unsafe_allow_html=True)
    _hero("hidden_oscillator", law, uq, video_title="Forecast of a run it never saw")
    names = [str(n) for n in a.get("names", [])]
    errs = {n: a[f"err_{i}"] for i, n in enumerate(names)}
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        ui.fig_title("Training data (3 noisy runs)")
        if "train" in a:
            show(_bigfont(viz.hidden_training(a["train"]), 430), "ho_train")
    with c2:
        ui.fig_title("Forecast error over time", locked=True)
        if errs:
            show(_bigfont(viz.hidden_errors(a["t"], errs), 430), "ho_err")
    with c3:
        ui.fig_title("How precisely known")
        sq = info.get("radius_term")
        nm = {t["term"]: t["term"].replace(sq, "r") if sq else t["term"] for t in uq.get("terms") or []}
        f = ui.precision_fig(uq, nm)
        if f:
            show(_bigfont(f), "ho_prec")
    with st.expander("How it got there", icon=":material/psychology:"):
        steps = (info.get("story") or {}).get("key_steps") or []
        tool_chips(info.get("tools") or {}, title="tools it used (all branches)")
        for k, s_ in enumerate(steps, 1):
            st.markdown(f"**{k}. {s_.get('observation', '')}**  \n{s_.get('decision', '')}  \n*{s_.get('outcome', '')}*")
    with st.expander("What the equation means", icon=":material/function:"):
        st.markdown(
            "- **r(1 − r):** a limit cycle. Small swings grow, large ones shrink, so every run settles onto the same "
            "loop (r = 1).\n"
            "- **φ̇ = a + b·r:** it circles faster the further out it is (the frequency depends on amplitude).\n"
            "- **u₂:** a third variable that simply follows the size of the swing, with a lag.\n"
            "- In the measured variables none of this is visible: r is a square root of a tilted quadratic form, so no "
            "polynomial can write it. The agent found the centre, undid the tilt and stretch, and fitted in (r, φ).")
    with st.expander("The checks behind the verdict", icon=":material/fact_check:"):
        st.markdown(ui.checks_md(uq))
        for adv in uq.get("data_advice") or []:
            st.caption(adv)
    with st.expander("Benchmark details", icon=":material/lock:"):
        tr = {"growth": 0.7029, "omega0": 1.9022, "omega1": 0.5183 * 0.8274, "decay": 0.9497, "drive": 0.3493 * 0.8274}
        lab = {"growth": "growth rate (r term)", "omega0": "base frequency", "omega1": "frequency per unit r",
               "decay": "u₂ decay rate", "drive": "u₂ drive by r"}
        if pol:
            st.markdown("**Discovered vs. true law** (in the same coordinates; the limit cycle at r = 1)\n\n| | true | "
                        "discovered | off by |\n|---|---|---|---|\n" + "\n".join(
                            f"| {lab[k]} | {tr[k]:.4f} | {pol[k]:.4f} | {abs(pol[k] - tr[k]) / tr[k]:.1%} |" for k in tr))
        em = info.get("errors_mean") or {}
        b = info.get("bare") or {}
        st.markdown(
            f"**Forecast of the held-out run** (mean distance from reality over 30 time units): discovered law "
            f"{em.get('Discovered law', 0):.1%}, true law {em.get('True law', 0):.1%}, Claude alone "
            f"{em.get('Claude alone', 0):.1%}.\n\n"
            f"**Claude alone** is the same model with only a Python sandbox and the prompt *Gimme PDE!*, on the same "
            f"training runs (${b.get('cost_usd', 0):.2f}, {b.get('n_tool_calls', 0)} tool calls). It fitted a cubic "
            f"polynomial oscillator: close on the loop, but wrong while a run settles. eqdisc: "
            f"${info.get('cost_usd', 0):.2f}, {info.get('wall_s', 0) / 60:.0f} min; the winning law came from its "
            f"{info.get('winner_branch', '')} step.")

LORENZ_WHAT = {
    "clean": "Four noisy runs (2% noise), nothing else wrong.",
    "spikes": "The same runs with sensor glitches: 0.5% of the samples replaced by large spikes.",
    "forcing": "The same runs, but the system was kicked from outside a few times (short pushes nobody recorded).",
    "sensor": "The third variable was never measured: only two of the three columns exist.",
}


def v3_lorenz():
    info, a = load_case("lorenz")
    if not info:
        return missing("lorenz")
    conds = info["conditions"]
    ui.page_title("Lorenz: In and Out of Sample",
                  "A chaotic system · 4 short noisy runs · rescaled units · law vs a neural operator (FNO)")
    labels = {c["label"]: k for k, c in conds.items()}
    pick = st.segmented_control("What was wrong with the training data", list(labels), default="Clean",
                                key="lorenz_cond") or "Clean"
    cond = labels[pick]
    c = conds[cond]
    st.markdown(f"<div class='law'>{html.escape(LORENZ_WHAT[cond])}</div>", unsafe_allow_html=True)
    uq = c.get("uq") or {}
    tl = a["t_lyap"]
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        ui.fig_title("Forecast from a new starting point", locked=True)
        p = SHOW / "lorenz" / f"video_{cond}.mp4"
        if p.exists():
            st.video(str(p), loop=True, autoplay=True, muted=True)
    with right:
        if uq:
            ui.fig_title("The verdict")
            ui.verdict_box(uq)
        ui.fig_title("The law it found")
        ui.equations(ui.rhs_latex(c["laws"]["Discovered law"], digits=3), small=True)
        if cond == "sensor":
            st.caption("Only the first equation is checkable: the second is its stand-in for the missing variable.")
        if uq:
            ui.next_box(uq)
    keys = {"Discovered law": "law", "FNO": "fno", "SINDy": "sindy"}
    c1, c2 = st.columns(2, gap="large")
    with c1:
        ui.fig_title("In sample: a training run")
        show(_bigfont(viz.ode_errors(tl, {k: a[f"{cond}_err_in_{v}"] for k, v in keys.items()}), 400), f"lz_in_{cond}")
    with c2:
        ui.fig_title("Out of sample: new starting points", locked=True)
        show(_bigfont(viz.ode_errors(tl, {k: a[f"{cond}_err_oos_{v}"] for k, v in keys.items()}), 400), f"lz_oos_{cond}")
    ui.fig_title("How long each stays on track", locked=True)
    show(_bigfont(viz.ode_valid_bars(c["valid_in"], c["valid_oos"], float(tl[-1])), 400), f"lz_bars_{cond}")
    with st.expander("The trajectories (in sample and out of sample)", icon=":material/show_chart:"):
        names = c["names"]
        for tag, lab in (("in", "In sample: training run 1 (as given, with its noise)"),
                         ("oos", "Out of sample: new starting point 1 (clean)")):
            st.markdown(f"**{lab}**")
            show(_bigfont(viz.ode_series(tl, a[f"{cond}_X_{tag}"], {k: a[f"{cond}_Y_{tag}_{v}"] for k, v in keys.items()},
                                         names)), f"lz_ser_{tag}_{cond}")
    with st.expander("How it got there", icon=":material/psychology:"):
        _reasoning(c.get("rationale"), c.get("tools"))
    with st.expander("What this shows", icon=":material/function:"):
        st.markdown(
            "- **Clean data:** the law and the FNO both track a training run well; on new starting points the law "
            "lasts longer (about 2.3 vs 1.8 Lyapunov times). Plain SINDy also finds the law here.\n"
            "- **Spikes:** SINDy fits the glitches and gets a wrong law; the FNO learns the glitches too. The agent "
            "recovers the right law, which tracks new runs more than twice as long as either.\n"
            "- **Outside kicks:** all three degrade; the law is still the most accurate.\n"
            "- **Lost sensor:** the law's first equation is right, but its stand-in for the missing variable cannot "
            "forecast; the FNO does better on the two measured columns. The agent's verdict (inconclusive) says so.\n"
            "- Lorenz is chaotic: tiny errors grow about 2.7× every Lyapunov time, so every forecast is lost "
            "eventually. What matters is how long it lasts.")
    if uq:
        with st.expander("The checks behind the verdict", icon=":material/fact_check:"):
            st.markdown(ui.checks_md(uq))
    with st.expander("Benchmark details", icon=":material/lock:"):
        fi = c.get("fno_info") or {}
        rows = [{"method": k, "in sample": round(c["valid_in"][k], 2),
                 "out of sample (run 1, run 2)": ", ".join(f"{x:.2f}" for x in c["valid_oos_runs"][k]),
                 "out of sample (mean)": round(c["valid_oos"][k], 2)} for k in c["valid_in"]]
        st.markdown("**Lyapunov times before the error passes 0.5** (end of record: "
                    f"{float(tl[-1]):.1f})")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.markdown(
            f"- **Same data for everyone:** the FNO, SINDy and the agent all trained on the same 4 runs, as given "
            f"(noise, spikes, kicks, missing column).\n"
            f"- **FNO:** Fourier layers over a window of {info['fno_window']} time steps; it maps the last "
            f"{info['fno_window']} states to the next {info['fno_window']} and is chained to forecast. Window chosen on "
            f"held-out training windows; best checkpoint by held-out loss ({int(fi.get('epochs', 0))} epochs, "
            f"{fi.get('minutes', 0):.0f} min on a laptop CPU).\n"
            f"- **Head start:** the FNO is given the first {info['fno_window']} true states of each run; the laws only "
            f"the first one.\n"
            "- **Error:** RMS over the variables of |forecast − data| / spread of that variable. Out-of-sample runs are "
            "clean; in sample is scored against the training run with isolated spikes median-filtered out.\n"
            f"- **Agent cost:** ${c.get('cost_usd') or 0:.2f}.")


def v3_home():
    st.markdown("<div class='home-e'>Lorenz</div><div class='home-t'>Equation Discovery AutoScientist</div>"
                "<div class='home-s'>Data in. Out come the equation, how sure it is, and where to measure next.</div>",
                unsafe_allow_html=True)
    cards = [("lageos", "Satellite", V3_PAGES["sat"]), ("orbit", "Big Bulge Orbit", V3_PAGES["bulge"]),
             ("ks", "Blind Chaos (KS)", V3_PAGES["chaos"]), ("gray_scott", "Reaction-Diffusion (Chemistry)", V3_PAGES["rd"]),
             ("hidden_oscillator", "Hidden Oscillator", V3_PAGES["hidden"]),
             ("lorenz", "Lorenz (In vs Out of Sample)", V3_PAGES["lorenz"])]
    cols = [c for _ in range(0, len(cards), 3) for c in st.columns(3, gap="large")]
    for c, (case, name, page) in zip(cols, cards):
        with c, st.container(border=True):
            th = SHOW / case / "thumb.jpg"
            if th.exists():
                st.image(str(th), width="stretch")
            st.markdown(f"<div class='cardn'>{html.escape(name)}</div>", unsafe_allow_html=True)
            st.page_link(page, label="Open case", icon=":material/arrow_forward:")
    st.write("")
    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        if st.button("Try it on your own data", type="primary", icon=":material/upload:", width="stretch"):
            st.switch_page(V3_PAGES["yours"])
    with st.expander("How we keep it honest"):
        st.markdown(f"{PROTOCOL}\n\n‘Hidden truth’ marks anything that uses the hidden future or the true law: "
                    "eqdisc never sees it.")


def _slim(a, verdict=None):
    """Assessment -> what the verdict/next/check widgets need."""
    a = a or {}
    from eqdisc import insights
    ex = a.get("experiments") or {}
    return {"verdict": verdict or (insights.verdict(a) if a.get("confidence") else {}), "terms": a.get("terms"),
            "missing": a.get("missing_term_evidence") or a.get("missing"), "validation": a.get("validation"),
            "experiments": (ex.get("ranked") if isinstance(ex, dict) else ex) or [],
            "data_advice": a.get("data_advice"), "model_ambiguity": a.get("model_ambiguity")}


def _meaning_expander(res):
    m = res.get("meaning")
    if m and m.get("bullets"):
        with st.expander("What the equation means", icon=":material/function:"):
            st.markdown(yourdata.meaning_md(m))
            st.caption(yourdata.meaning_note(m))


def _yourdata_result(job):
    res = job.result
    if getattr(job, "saved_at", None):
        st.caption(f"Saved result from {job.saved_at} · {job.run_dir}")
    if getattr(job, "prompt", ""):
        st.caption(f"Prompt used: “{job.prompt}”")
    else:
        st.caption("Blind run: no prompt, data only.")
    uq = _slim(res.get("assessment"), res.get("verdict"))
    fc = res.get("forecast") or {}
    video = fc.get("video") if fc.get("video") and Path(fc["video"]).exists() else None
    if res["kind"] == "dynamics":
        kind, png = "ode", None
        if res.get("dataset_path") and res.get("final_model"):
            kind, png = _model_figure(res["dataset_path"], json.dumps(res["final_model"]),
                                      str(Path(job.run_dir) / "demo_figs" / "model.png"))
        left, right = st.columns([1.35, 1], gap="large")
        with left:
            if video:
                ui.fig_title("Forecast vs reality", locked=True)
                if yourdata.blew_up_early(fc):
                    st.error("The law it found breaks down right at the start of the forecast: its values become "
                             "infinite or invalid, so there is no forecast to show, only reality. Treat this law as "
                             "unusable for prediction.")
                st.video(video, loop=True, autoplay=True, muted=True)
            else:
                ui.fig_title("Model vs your data")
                if png:
                    st.image(png, width="stretch")
                if fc.get("skipped") or fc.get("error"):
                    st.caption(f"No forecast video: {fc.get('skipped') or fc.get('error')}")
        with right:
            ui.fig_title("The verdict")
            ui.verdict_box(uq)
            ui.fig_title("The law it found")
            ui.equations(ui.rhs_latex(res.get("final_model") or {}, pde=kind == "pde"), small=True)
            ui.next_box(uq)
        if fc.get("npz"):
            st.caption(yourdata.caption(fc))
            c1, c2, c3 = st.columns(3, gap="large")
            with c1:
                ui.fig_title("What the agents saw")
                show(_bigfont(yourdata.training_fig(fc["npz"]), 430), "yd_train")
            with c2:
                ui.fig_title("Forecast error", locked=True)
                show(_bigfont(yourdata.error_fig(fc["npz"]), 430), "yd_err")
            with c3:
                ui.fig_title("How precisely known")
                f = ui.precision_fig(uq)
                if f:
                    show(_bigfont(f), "yd_prec_row")
        story = res.get("story") or {}
        with st.expander("How it got there", icon=":material/psychology:"):
            tool_chips([e["name"] for e in job.events if e.get("type") == "tool"], title="tools it used")
            if story.get("headline"):
                st.markdown(story["headline"])
            for s_ in story.get("key_steps") or []:
                st.markdown(f"- **{s_.get('observation', s_.get('decision', ''))}** → {s_.get('decision', s_.get('outcome', ''))}")
        _meaning_expander(res)
        if video and png:
            with st.expander("🔍 Fit to the data the agents saw"):
                st.image(png, width="stretch")
    else:
        from eqdisc.sr import evaluate_expr
        names, target, expr = res["names"], res["target"], res.get("expr")
        if not expr:
            st.error("No law was found.")
            return
        df = pd.read_csv(res["csv"])
        _, _, X, y, _, _ = live.static_task_arrays(df, target)
        yh = evaluate_expr(expr, names, X)
        left, right = st.columns([1.35, 1], gap="large")
        with left:
            ui.fig_title("Predicted vs measured")
            show(_bigfont(viz.pred_vs_true(y, yh), 460), "yd_pvt")
        with right:
            ui.fig_title("The verdict")
            ui.verdict_box(uq)
            ui.fig_title("The law it found")
            ui.equations([f"{ui.expr_latex(target, [target])} = {ui.expr_latex(expr, names)}"], small=True)
            ui.next_box(uq)
        with st.expander("How it got there", icon=":material/psychology:"):
            tool_chips([e["name"] for e in job.events if e.get("type") == "tool"], title="tools it used")
            if (res.get("verdict") or {}).get("recommendation"):
                st.markdown(res["verdict"]["recommendation"])
        _meaning_expander(res)
    with st.expander("The checks behind the verdict", icon=":material/fact_check:"):
        st.markdown(ui.checks_md(uq) or "No checks available.")
        f = None if fc.get("npz") else ui.precision_fig(uq)       # already shown in the figure row
        if f:
            show(_bigfont(f), "yd_prec")
    with st.expander("Downloads", icon=":material/download:"):
        rp = Path(res["report"]) if res.get("report") else None
        if rp and rp.exists():
            st.download_button("Full report (HTML)", rp.read_bytes(), rp.name, "text/html")
        if video:
            st.download_button("Forecast video (MP4)", Path(video).read_bytes(), "forecast.mp4", "video/mp4")
        st.download_button("Result (JSON)", json.dumps(res, default=str, indent=1), "result.json")
        if res.get("rehearsal_note"):
            st.caption(res["rehearsal_note"])


def v3_yourdata():
    ui.page_title("Your Data", "Upload measurements → the equation, how sure it is, and where to measure next")
    env_fake = os.environ.get("EQDISC_DEMO_FAKE", "") not in ("", "0", "false")
    job = st.session_state.get("job")
    running = job is not None and not job.done
    c1, c2 = st.columns([1.35, 1], gap="large")
    with c1:
        ui.fig_title("Data")
        src = st.radio("Data", ["Upload a CSV"] + list(EXAMPLES) + ["Saved results"], horizontal=True,
                       disabled=running, key="src", label_visibility="collapsed")
        df, fname, npz = None, None, None
        if src == "Saved results":
            saved = yourdata.list_saved(live.RUNS)
            if not saved:
                st.info("No saved results yet. Every finished run is saved automatically and shows up here.")
            else:
                pick = st.selectbox("Saved results (newest first)", [lab for lab, _ in saved], disabled=running)
                if st.button("Open", disabled=running):
                    st.session_state.job = yourdata.SavedJob(dict(saved)[pick])
                st.caption(f"Saved in {live.RUNS}: one folder per run, with the result, forecast video and report.")
        elif src == "Upload a CSV":
            up = st.file_uploader("CSV: a time column plus one column per variable, or one row per measurement. "
                                  "Or .npz: time and variable/field arrays",
                                  type=["csv", "tsv", "txt", "npz"], disabled=running)
            if up is not None and up.name.lower().endswith(".npz"):
                npz, fname = yourdata.read_npz(up.getvalue(), up.name, live.RUNS), Path(up.name).stem
                if npz.get("error"):
                    st.error(npz["error"])
                    npz = None
                else:
                    df = npz["df"]
                    if npz.get("warning"):
                        st.warning(npz["warning"])
                    st.caption(f".npz read as {'a time series' if npz['kind'] == 'ode' else 'a field over space'}: "
                               f"{', '.join(npz['variables'])}, shape {tuple(npz['shape'])}. {npz['summary']}"[:400])
            elif up is not None:
                df, fname = pd.read_csv(up, sep=None, engine="python"), Path(up.name).stem
        else:
            path, _ = EXAMPLES[src]
            if path.exists():
                df, fname = pd.read_csv(path), path.stem
        if df is not None:
            st.dataframe(df.head(6), hide_index=True, width="stretch")
        field = npz is not None and df is None          # .npz field over space: goes to discovery as the file itself
        ui.fig_title("Tell it about your data")
        prompt = st.text_area(
            "Prompt", key=f"prompt_{src}", height=130, disabled=running, label_visibility="collapsed",
            placeholder="Optional. e.g. ‘Positions and velocities of a pendulum, released from 4 angles. Angle in "
                        "radians. Is there damping?’  Leave empty for a blind, data-only run.")
        if prompt.strip():
            st.caption("The agents treat this as a strong hint and check it against the data; the result will say "
                       "a prompt was used.")
    with c2:
        ui.fig_title("Settings")
        mode = st.segmented_control("Kind of law", ["Auto", "Dynamics", "Static y = f(x)"], default="Auto",
                                    disabled=running, key="mode") or "Auto"
        resolved = mode
        if npz is not None:
            resolved = "Dynamics"                        # .npz holds time + states/fields
        elif df is not None and mode == "Auto":
            resolved = "Dynamics" if live.detect_mode(df) == "dynamics" else "Static y = f(x)"
        target = None
        if df is not None and resolved.startswith("Static"):
            num = list(df.select_dtypes("number").columns)
            target = st.selectbox("Predict which column", num, index=len(num) - 1, disabled=running)
        budget = st.segmented_control("Effort", ["Quick", "Full"], default="Quick", disabled=running, key="budget") or "Quick"
        holdout = None
        if df is None or resolved.startswith("Dyn"):
            hold = st.segmented_control("Forecast check", ["Off", "10%", "20%", "30%"], default="20%",
                                        disabled=running, key="holdout") or "Off"
            holdout = None if hold == "Off" else int(hold[:-1]) / 100
            st.caption(("Holds back the last snapshots. " if field else "Holds back the end of every run. ")
                       + "The agents never see it; the video shows the found law forecasting it." if holdout
                       else "Off: the agents see all the data; no forecast video.")
        fake = st.checkbox("Rehearsal (replays a saved run, no API cost)", value=env_fake, disabled=running, key="fake")
        ready = df is not None or field
        go_btn = st.button("Discover", type="primary", disabled=running or not ready, width="stretch")
        if df is not None:
            st.caption(f"{df.shape[0]} rows × {df.shape[1]} columns · detected: {resolved}")
    if go_btn and ready:
        run_dir = live.RUNS / time.strftime("%Y%m%d-%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        if field:
            csv_path = run_dir / Path(npz["path"]).name           # the framework's ingester reads .npz directly
            csv_path.write_bytes(Path(npz["path"]).read_bytes())
        else:
            csv_path = run_dir / f"{live.ident(fname or 'data')}.csv"
            df.to_csv(csv_path, index=False)
        quick = budget == "Quick"
        if resolved.startswith("Dyn"):
            job = live.Job(yourdata.dynamics_job, backend=live.fake_dynamics if fake else live.run_dynamics,
                           holdout=holdout, use_llm=not fake, csv_path=str(csv_path), run_dir=str(run_dir),
                           n_branches=2 if quick else 3, adversary=not quick, context=prompt.strip())
        else:
            job = live.Job(yourdata.static_job, backend=live.fake_static if fake else live.run_static,
                           use_llm=not fake, run_dir=str(run_dir), csv_path=str(csv_path), target=target,
                           context=prompt.strip(), n_sessions=2 if quick else 3)
        job.run_dir = str(run_dir)
        job.prompt = prompt.strip()
        st.session_state.job = job.start()
        running = True
    job = st.session_state.get("job")
    if job is None:
        return
    with st.status("The agents are working…" if not job.done else "Done", expanded=not job.done,
                   state="running" if not job.done else ("error" if job.error else "complete")) as status:
        timer = st.empty()
        logph = st.container(height=260).empty()

        def paint():
            lines = [live.fmt_event(e).replace("$", "\\$") for e in job.events]
            logph.markdown("\n\n".join(lines[-30:]) or "_starting…_", unsafe_allow_html=True)
            timer.caption(f"{job.elapsed:.0f} s · {sum(e.get('type') == 'tool' for e in job.events)} steps")
        while not job.done:
            job.drain()
            paint()
            time.sleep(0.4)
        job.drain()
        paint()
        status.update(label="Failed" if job.error else f"Done in {job.elapsed:.0f} s",
                      state="error" if job.error else "complete", expanded=False)
    if running:
        st.rerun()
    if job.error:
        st.error(job.error)
        return
    _yourdata_result(job)


V3_PAGES = {}


def main():
    ui.inject_css3()
    st.logo(str(BRAND / "logo-lorenz.svg"), size="large")
    env_fake = os.environ.get("EQDISC_DEMO_FAKE", "") not in ("", "0", "false")
    V3_PAGES.update({
        "home": st.Page(v3_home, title="Home", icon=":material/home:", default=True),
        "sat": st.Page(v3_satellite, title="Satellite", icon=":material/satellite_alt:", url_path="satellite"),
        "bulge": st.Page(v3_bulge, title="Big Bulge Orbit", icon=":material/public:", url_path="big-bulge-orbit"),
        "chaos": st.Page(v3_chaos, title="Blind Chaos (KS)", icon=":material/cyclone:", url_path="blind-chaos"),
        "rd": st.Page(v3_reaction, title="Reaction-Diffusion (Chemistry)", icon=":material/texture:",
                      url_path="reaction-diffusion"),
        "hidden": st.Page(v3_hidden, title="Hidden Oscillator", icon=":material/graphic_eq:", url_path="hidden-oscillator"),
        "lorenz": st.Page(v3_lorenz, title="Lorenz (In vs Out of Sample)", icon=":material/all_inclusive:", url_path="lorenz"),
        "evidence": st.Page(evidence.page, title="When Not to Trust It", icon=":material/gpp_maybe:",
                            url_path="when-not-to-trust-it"),
        "yours": st.Page(v3_yourdata, title="Your Data", icon=":material/upload_file:", url_path="your-data"),
    })
    nav = st.navigation(list(V3_PAGES.values()), position="top")
    nav.run()

main()
