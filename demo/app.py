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

PROTOCOL = ("Honest protocol: noisy training data → forecast unseen future or held-out trajectories, autoregressively "
            "from a noisy observed state; coefficients refitted, never rounded; a neural net trained on the same data.")
J2_ACCEPTED = 1.08263e-3
PAGES = ["Home", "🛰️ LAGEOS-1 satellite", "🔥 Chaos (KS)", "🌀 Gray–Scott patterns", "⚡ Run on your data",
         "⚙️ How it works"]


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


def agent_j2(ag):
    """J2 from the agent's submitted law: the coefficient c in (1 + c/r^2 (...)) equals (3/2) J2 (Re = 1)."""
    import re as _re
    m = _re.search(r"\(1 \+ ([0-9.eE+-]+)/\(x\*\*2", str((ag.get("submitted") or {}).get("vx", "")).replace(" ", "").replace("(1+", "(1 + "))
    try:
        return float(m.group(1)) / 1.5 if m else None
    except ValueError:
        return None


def page_lageos():
    info, a = load_case("lageos")
    if not info:
        return missing("lageos")
    n = lageos_numbers(info)
    ag = info.get("agent") or {}
    ui.question("Can it learn a real satellite's law of motion from one year of data, and forecast the next month?")
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        hero_video("lageos")
        st.caption("LAGEOS-1, real hourly positions. Measured track vs forecasts in the first 3 unseen days.")
    with right:
        ui.verdict_chip("VALIDATED", "30-day forecast of an unseen month")
        ui.equations([r"\ddot{\mathbf r} = -\frac{\mu\,\mathbf r}{r^{3}}\left[1 + \tfrac{3}{2}J_2\left(\tfrac{R_e}{r}\right)^{2}"
                      r"\left(1 - \tfrac{5z^{2}}{r^{2}}\right)\right]",
                      rf"J_2 = ({n['J2'] * 1e3:.5f} \pm {n['J2s'] * 1e3:.5f})\times 10^{{-3}}"])
        ui.tiles([
            {"label": "30-day error", "value": km(n["J"]), "delta": f"Kepler {km(n['K'])}",
             "help": "Position error after 30 days of autoregressive forecast; discovered law (Kepler + J₂) vs Kepler."},
            {"label": "Neural net", "value": km(n["N"]), "delta": "same data",
             "help": "MLP step model trained on the same 2017 data."},
            {"label": "J₂ fitted (×10⁻³)", "value": f"{n['J2'] * 1e3:.5f}", "delta": f"accepted {J2_ACCEPTED * 1e3:.5f}",
             "help": f"± {n['J2s'] * 1e3:.5f} (1σ). Earth's oblateness, fitted from the 2017 data only."},
            {"label": "Node drift (°/day)", "value": f"{n['node']['Kepler + J2']:.4f}",
             "delta": f"measured {n['node']['data']:.4f}", "help": "Orbit-plane precession; Kepler predicts 0."},
        ])
        aj = agent_j2(ag)
        if aj:
            st.caption(f"The eqdisc agent, on its own ({ag.get('n_tool_calls', '?')} tool calls, \\${ag.get('cost_usd', 0):.2f}): "
                       f"J₂ = {aj * 1e3:.5f}×10⁻³, 30-day error {km(ag['position_error_km']['30d'])}.")
        ui.chips([("sampling", "hourly → flow-map fit"), ("residuals", "Kepler error depends on z/r"),
                  ("zonal J₂ term", f"{n['one_step_gain']:.0f}× better"), ("check", "node drift matches")])
    errs = {m: a[f"err{i}"] for i, m in enumerate(info["models"])}
    pts = None
    if ag.get("position_error_km"):
        pts = [(int(k[:-1]), v) for k, v in ag["position_error_km"].items()]
    show(viz.lageos_errors(a["days"], errs, pts), "lageos_err")
    with st.expander("Details & caveats"):
        pe = info["results"]["position_error_km"]
        st.markdown("**Forecast position error (km), autoregressive from the last training state**")
        st.dataframe(pd.DataFrame(pe).T.map(lambda v: f"{v:,.3g}"), width="content")
        st.markdown("**Fitted parameters** (nondimensional, given μ and Rₑ)")
        st.dataframe(pd.DataFrame([{"model": m, **{f"{k}": f"{v:.9g} ± {info['results'][m]['param_sigma'][k]:.2g}"
                                                    for k, v in info["results"][m]["params"].items()},
                                    "held-out one-step error": f"{info['results'][m]['one_step_rel_err_heldout']:.3g}"}
                                   for m in ("Kepler", "Kepler + J2")]), hide_index=True)
        if ag:
            st.markdown(f"**eqdisc agent's own law** (cost ${ag.get('cost_usd', 0):.2f}, {ag.get('n_tool_calls', '?')} tool calls)")
            if ag.get("submitted"):
                ui.equations(ui.rhs_latex({k: v for k, v in ag["submitted"].items() if k.startswith("v")}))
        else:
            st.caption("The agent's own run on the 2017 data is pending; the law above is the Kepler + J₂ fit.")
        if (SHOW / "lageos" / "errors.png").exists():
            st.image(str(SHOW / "lageos" / "errors.png"), caption="Error vs time, and node drift over 6 years.")
        st.markdown(
            "- **Protocol.** Train on 2017 only (hourly, real). Forecast January 2018 from the last observed state. "
            "The neural step model is trained on the same year.\n"
            "- **Honesty.** The J₂ formula is textbook physics that the LLM knows; the *value* is fitted from data and "
            "lands within 0.03% of the accepted one.\n"
            "- **Data.** orbit_discover workshop repository (MIT); nondimensional units with the μ and Rₑ given there.")


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
        st.caption("Unseen future of a blinded Kuramoto–Sivashinsky field (2% noise in training). "
                   "Black: truth.")
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
        ui.chips([("intuit", "mean conserved → flux form"), ("weak SINDy", "3 terms found"),
                  ("assessment", "each term ΔBIC ≈ 1100"), ("repair", "extra terms rejected")])
    errs = {lab: a[f"err{i}"] for i, lab in enumerate(info["labels"]) if i > 0}
    show(viz.ks_errors(a["t_lyap"], errs), "ks_err")
    with st.expander("Details & caveats"):
        st.dataframe(pd.DataFrame([{**x, "90% CI": f"[{x['90% CI'][0]:.4g}, {x['90% CI'][1]:.4g}]"} for x in rows]),
                     hide_index=True)
        st.markdown(
            f"- **Protocol.** Real KS data, blinded by rescaling x, t and u, so the coefficients are not textbook values. "
            f"Add 2% noise, train on t ∈ [0, {r['train_window'][1]:.1f}], forecast t ∈ [{r['test_window'][0]:.1f}, "
            f"{r['test_window'][1]:.1f}] from the noisy last state. Lyapunov exponent {r['lyapunov_exponent']:.3f}.\n"
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
            ui.verdict_chip("PENDING", "agent run pending · target law shown")
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
        cards.append(("🛰️ LAGEOS-1 satellite (real)", f"{km(n['J'])} vs {km(n['K'])}",
                      "30-day forecast error: discovered law vs Kepler", "lageos", PAGES[1]))
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
    "Example: pendulum (time series)": (DEMO / "examples" / "pendulum.csv",
                                        "Angle (rad) and angular velocity (rad/s) of a pendulum; 4 releases; "
                                        "time-correlated sensor noise."),
    "Example: E. coli growth (static law)": (DEMO / "examples" / "ecoli_growth.csv",
                                             "Bacterial growth rate db vs population b, substrate s, temperature temp, pH."),
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
        context = st.text_input("Context (optional)", value=ctx_default, disabled=running, key=f"ctx_{src}")
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
  B [label="parallel Claude agents\\n(weak SINDy, PySR, skeleton fits,\\ninvariants, own Python, plots)"];
  T [label="tournament"]; A [label="red team", fillcolor="#fee2e2", color="#dc2626"];
  Q [label="assessment\\nCIs · ΔBIC · noise floor"]; V [label="verdict +\\nnext experiment", fillcolor="#dcfce7", color="#16a34a"];
  D->I->P->B->T->A->Q->V; }""")
    st.markdown(f"<div class='protocol'>✅ {PROTOCOL}</div>", unsafe_allow_html=True)
    st.markdown("| verdict | meaning |\n|---|---|\n"
                "| ✓ CONFIDENT | every term supported, error at the noise floor, predicts held-out data |\n"
                "| ◐ COLLECT MORE DATA | competing models remain; it names the experiment that separates them |\n"
                "| ? INCONCLUSIVE | the data cannot determine the model |")
    with st.expander("Earlier experiments (in-sample or less strict; not shown as results)"):
        st.markdown(
            "- Synthetic orbit with exaggerated J₂ = 0.5: recovered, but noise-free generator data.\n"
            "- Noisy pendulum: verdict COLLECT MORE DATA with a ranked next experiment.\n"
            "- LLM-SR E. coli growth: low error, but the equation structure is published (possible recall).\n"
            "- Unpublished oscillators: 3/4 exact vs PySR 0/4 (static symbolic regression).\n"
            "- Real KS, unblinded: textbook coefficients, so recall cannot be excluded.")


# ============================================================================= main
def main():
    env_fake = os.environ.get("EQDISC_DEMO_FAKE", "") not in ("", "0", "false")
    with st.sidebar:
        st.markdown("### 🧭 eqdisc")
        page = st.radio("Navigate", PAGES, key="nav", label_visibility="collapsed")
        with st.expander("⚙️", expanded=False):
            fake = st.toggle("Rehearsal mode (no API calls)", value=env_fake, key="fake")
    {"Home": page_home, PAGES[1]: page_lageos, PAGES[2]: page_ks, PAGES[3]: page_gs,
     PAGES[5]: page_how}.get(page, lambda: page_live(fake))()


main()
