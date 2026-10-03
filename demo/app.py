"""eqdisc demo.   cd /Users/danield/eqdisc && PYTHONPATH=. streamlit run demo/app.py

Three out-of-sample cases (one screen each) + a live "Run on your data" page. Reads only demo/showcase/ and
demo/examples/ (build with demo/build_showcase.py). Rehearsal mode (EQDISC_DEMO_FAKE=1 or the sidebar toggle)
replays a scripted live run without API calls.
"""
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

st.set_page_config(page_title="eqdisc — equations that forecast", page_icon="🧭", layout="wide",
                   initial_sidebar_state="expanded")

import live  # noqa: E402
import ui  # noqa: E402
import viz  # noqa: E402

SHOW = DEMO / "showcase"
ui.inject_css()

PROTOCOL = ("Data only: the agent gets numbers with neutral names, no description of the system, no domain guidance, "
            "and its code cannot read anything else. Honest test: noisy training data → forecast an unseen future or "
            "held-out trajectory from a noisy observed state; coefficients refitted; a neural net trained on the same data.")
TOOL_LABEL = {"diagnose": "diagnose", "intuit": "intuition", "run_sindy": "SINDy", "weak_sindy": "weak SINDy",
              "run_pysr": "PySR", "fit_skeleton": "skeleton fit", "fit_flow": "flow-map fit",
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
         "⚙️ How it works", "🌍 Orbit with a big bulge"]


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
    ui.question("Given only a year of a real satellite's numbers, can the agent find its law of motion and forecast "
                "the next month?")
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        hero_video("lageos")
        st.caption("LAGEOS-1, real hourly positions: measured track vs forecasts over the first 3 unseen days.")
    with right:
        if pe_ag:
            ok = pe_ag["30d"] < min(n["N"], n["K"])
            ui.verdict_chip("VALIDATED" if ok else "NOT RECOVERED", "data-only agent, unseen month")
            ui.equations(lageos_latex(lageos_agent_rhs(ag)), small=True)
            st.caption("Its law, constants refitted on 2017 (r² = u1² + u2² + u3²). The agent was not told that "
                       "u1–u3 are position and u4–u6 velocity, and the units are random, so neither the gravity "
                       "constant nor Earth's radius equals 1.")
            ui.tiles([
                {"label": "Agent: 30-day error", "value": km(pe_ag["30d"]), "delta": f"1 day: {km(pe_ag['1d'])}",
                 "help": "Position error after 30 days of forecasting from the last training state."},
                {"label": "Neural net (MLP)", "value": km(n["N"]), "delta": "same data"},
                {"label": "Round-Earth gravity", "value": km(n["K"]), "delta": "textbook ellipse (Kepler)"},
                {"label": "Agent cost", "value": f"${ag.get('cost_usd', 0):.2f}",
                 "delta": f"{ag.get('n_tool_calls', '?')} tool calls"},
            ])
            tool_chips(ag.get("tools"))
        else:
            ui.verdict_chip("PENDING", "data-only agent run in progress")
            st.caption("Physics reference (hand-built Kepler + J₂, *not* discovered by the agent):")
            ui.equations([r"\ddot{\mathbf r} = -\frac{\mathbf r}{r^{3}}\left[1 + \tfrac{3}{2}J_2\,r^{-2}"
                          r"\left(1 - \tfrac{5z^{2}}{r^{2}}\right)\right]"], small=True)
            ui.tiles([{"label": "Reference: 30-day error", "value": km(n["J"]), "delta": f"Kepler {km(n['K'])}"},
                      {"label": "Neural net (MLP)", "value": km(n["N"]), "delta": "same data"}])
    errs = {m: a[f"err{i}"] for i, m in enumerate(info["models"])}
    if "err_agent" in a:
        errs["agent"] = a["err_agent"]
    c1, c2 = st.columns([1, 1.15], gap="large")
    with c1:
        st.markdown("**What the agent was given**: one year of hourly measurements (2017)")
        if "train_orbits" in a:
            show(viz.lageos_training(a["train_orbits"], info["train_dates"]), "lageos_train")
            st.caption("Real LAGEOS-1 positions, one orbit every two weeks (hourly samples, drawn smooth). Press ▶: "
                       "the orbit plane turns about Earth's axis over the year. The agent saw only six unnamed columns.")
    with c2:
        st.markdown("**Forecast error over the unseen month**")
        show(viz.lageos_errors(a["days"], errs), "lageos_err")
    st.markdown(LEGEND_ORBIT, unsafe_allow_html=True)
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
    ui.question("A satellite around a planet with a huge equatorial bulge: from 3 days of noisy positions, can the "
                "agent find the law and forecast the next 3 days?")
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        hero_video("orbit")
        st.caption("Synthetic data (orbit_discover workshop), bulge 500× Earth's (J₂ = 0.5). First unseen day shown.")
    with right:
        if pa:
            ok = pa["24h"] < min(pe["neural step model (MLP)"]["24h"], pe["Kepler"]["24h"])
            ui.verdict_chip("VALIDATED" if ok else "NOT RECOVERED", "data-only agent, unseen 3 days")
            ui.equations(lageos_latex(lageos_agent_rhs(ag)), small=True)
            st.caption("Its law with constants refitted on the 3 training days (r² = u1² + u2² + u3²); random units, "
                       "unnamed columns, 1% noise, no context.")
            ui.tiles([
                {"label": "eqdisc: error after 1 day", "value": km(pa["24h"]), "delta": f"after 3 days: {km(pa['72h'])}"},
                {"label": "Neural network", "value": km(pe["neural step model (MLP)"]["24h"]), "delta": "after 1 day"},
                {"label": "Round-Earth gravity", "value": km(pe["Kepler"]["24h"]), "delta": "after 1 day"},
                {"label": "True law (best possible)", "value": km(pe["Kepler + J2"]["24h"]), "delta": "after 1 day"},
            ])
            tool_chips(ag.get("tools"))
        else:
            ui.verdict_chip("PENDING", "data-only agent run in progress")
        if pa and not ok:
            st.markdown("<div class='small'><b>Why it failed.</b> The agent found the kinematics and the rotational "
                        "symmetry (conserved L<sub>z</sub>) but fitted a polynomial in r², z², v² instead of inverse-"
                        "square gravity plus a bulge term; its own diagnostics showed 25% derivative error and it "
                        "submitted anyway. A correct verdict would have been <i>not confident</i>.</div>",
                        unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**What the agent was given**: 3 days of noisy positions (and velocities)")
        show(viz.orbit_animation(a["t_train"], a["U_train"]), "orbit_anim")
        st.caption("Press ▶ to fly the first orbits. The faint tangle is all 3 training days: the orbit never closes on "
                   "itself, because the bulge keeps turning it.")
    with c2:
        if "kj_t" in a:
            st.markdown("**Why the bulge matters**: same start, two laws, ~6 orbits")
            show(viz.orbit_kepler_vs_j2(a["kj_t"], a["kj_disc"], a["kj_kep"]), "orbit_kj")
            st.caption("Round-Earth gravity (orange) repeats one ellipse in a fixed plane. The true law (blue) swings the "
                       "plane around the polar axis, as the data do. This is what the agent needed to find.")
    if "raan_disc" in a:
        st.markdown("**The tell-tale drift**, measured over all 6 days vs the forecasts of the unseen half")
        show(viz.orbit_elements(a["el_hrs"], a["raan_data"], a["argp_data"], a["hrs"], a["raan_disc"], a["argp_disc"],
                                a["raan_kep"], a["argp_kep"]), "orbit_el")
    errs = {m: a[f"err{i}"] for i, m in enumerate(info["models"])}
    errs = {("agent" if m == "data-only agent" else m): v for m, v in errs.items()}
    st.markdown("**Forecast error over the unseen 3 days**")
    show(viz.lageos_errors((a["hrs"] - a["hrs"][0]) / 24, errs, xlabel="days into the unseen second half"), "orbit_err")
    st.markdown(LEGEND_ORBIT.replace("Earth's equatorial bulge", "the planet's equatorial bulge").replace(
        "fitted by us; shown for reference only", "here the exact law that generated the data"), unsafe_allow_html=True)
    with st.expander("Details & caveats"):
        st.dataframe(pd.DataFrame({PLAIN_ROW.get(k, k): v for k, v in pe.items()}).T.map(lambda v: f"{v:,.3g}"),
                     width="content")
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
    ui.question("Can it forecast chaos it has never seen, from noisy data, with coefficients no textbook has?")
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        hero_video("ks")
        st.caption("Space (vertical) vs time (horizontal), revealed as the unseen future unfolds: truth, eqdisc's "
                   "equation, and a neural operator. The dark rows below show where each forecast is wrong; dashed "
                   "lines mark when it stops being useful. Chaos defeats every forecast eventually.")
    with right:
        v = info.get("verdict") or ag.get("verdict") or {}
        ui.verdict_chip(v.get("status"), "agent's own verdict")
        ui.equations(ui.rhs_latex(ag["refit"] if isinstance(ag["refit"], dict) else {"u": ag["refit"]}, pde=True, digits=5), small=True)
        st.caption("hidden truth: " + ", ".join(f"{x['truth']:.4g} {x['term'].replace('*', '·')}" for x in rows
                                                 if x["truth"] is not None))
        ui.tiles([
            {"label": "Agent forecast", "value": f"{vt['eqdisc agent (refit)']:.2f} λ",
             "delta": f"true PDE {vt['true PDE from noisy state']:.2f}",
             "help": "Lyapunov times until relative error exceeds 0.5. The true PDE itself cannot do better from a noisy state."},
            {"label": "FNO forecast", "value": f"{vt['FNO (same noisy data)']:.2f} λ", "delta": "same noisy data",
             "help": "Fourier neural operator trained on the same noisy window. λ = Lyapunov times."},
            {"label": "Truth in 90% CI", "value": f"{n_in}/{len(rows)}", "delta": "refitted coefficients"},
            {"label": "Agent cost", "value": f"${info.get('cost_usd') or 0:.2f}",
             "delta": f"{(info.get('wall_s') or 0) / 60:.0f} min" if info.get("wall_s") else None},
        ])
        tool_chips(info.get("tools"))
    errs = {lab: a[f"err{i}"] for i, lab in enumerate(info["labels"]) if i > 0}
    show(viz.ks_errors(a["t_lyap"], errs), "ks_err")
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
    ui.question("Can it forecast reaction–diffusion patterns from two noisy movies, against neural surrogates?")
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        if info.get("has_video") and hero_video("gray_scott"):
            st.caption(f"Held-out trajectory ('{info['regime']}' regime) vs forecasts from its noisy first frame.")
        elif "fb_data" in a:
            show(viz.gs_animation(a["fb_t"], a["fb_data"], a["fb_model"], GS_PENDING), "gs_anim")
            st.caption("Press ▶. Forecast video appears when the out-of-sample run finishes.")
    with right:
        if agent:
            ui.verdict_chip("VALIDATED", "held-out trajectory")
            ui.equations(ui.rhs_latex(agent["refit"], pde=True, digits=3), small=True)
        else:
            ui.verdict_chip("PENDING", "data-only agent run in progress · true law shown")
            ui.equations([r"\partial_t A = d_A \nabla^2 A - A B^2 + F(1-A)",
                          r"\partial_t B = d_B \nabla^2 B + A B^2 - (F+k)B"])
        best = "eqdisc agent (refit)" if "eqdisc agent (refit)" in vr else "weak SINDy (no LLM)"
        if vr:
            t = []
            for lab in (best, "FNO (same noisy data)", "true PDE from noisy frame"):
                if lab in vr:
                    t.append({"label": f"{SHORT.get(lab, lab)} VRMSE", "value": f"{vr[lab]['6-12']:.3g}",
                              "delta": f"steps 13–30: {vr[lab]['13-30']:.3g}", "help": f"{lab}; rollout steps 6–12"})
            t.append({"label": "Well paper best", "value": "0.29", "delta": "steps 13–30: 7.62",
                      "help": "CNextU-net in The Well paper, trained on hundreds of trajectories"})
            ui.tiles(t[:4])
        else:
            ui.tiles([{"label": "Training data", "value": "2 movies", "delta": "60 frames, 5% noise"},
                      {"label": "Test", "value": "held-out", "delta": "unseen trajectory"},
                      {"label": "Well paper best", "value": "0.29", "delta": "VRMSE, steps 6–12"},
                      {"label": "Our forecast", "value": "pending", "delta": "run in progress"}])
        if agent:
            tool_chips(info.get("tools"))
        else:
            ui.chips(["2 noisy training movies", "forecast held-out trajectory", "VRMSE as in The Well",
                      "FNO on same data"], title="protocol")
    if vr:
        show(viz.gs_vrmse(vr), "gs_vrmse")
    else:
        gal = {k[4:]: v for k, v in a.items() if k.startswith("gal_")}
        if gal:
            show(viz.gs_gallery(gal, info["regimes"]), "gs_gal")
            st.caption("Six pattern regimes from one two-term reaction law (only F and k change).")
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
            cards.append(("🌍 Orbit with a big bulge (synthetic)", "✗ not recovered",
                          f"an honest failure: after 1 day the agent is {km(ag_)} off, the neural net {km(nn_)}",
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
def main():
    env_fake = os.environ.get("EQDISC_DEMO_FAKE", "") not in ("", "0", "false")
    with st.sidebar:
        st.markdown("### 🧭 eqdisc")
        page = st.radio("Navigate", [PAGES[i] for i in (0, 1, 6, 2, 3, 4, 5)], key="nav", label_visibility="collapsed")
        with st.expander("⚙️", expanded=False):
            fake = st.toggle("Rehearsal mode (no API calls)", value=env_fake, key="fake")
    {"Home": page_home, PAGES[1]: page_lageos, PAGES[2]: page_ks, PAGES[3]: page_gs,
     PAGES[5]: page_how, PAGES[6]: page_orbit}.get(page, lambda: page_live(fake))()


main()
