"""eqdisc demo app.   streamlit run demo/app.py  (from the repo root)

Showcase: precomputed results from demo/showcase/ (build with demo/build_showcase.py).
Run on your data: live pipeline in a background thread with streamed progress; rehearsal mode
(EQDISC_DEMO_FAKE=1 or the sidebar toggle) replays a scripted run with no API calls.
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

st.set_page_config(page_title="eqdisc — autoresearch for governing equations", page_icon="🧭", layout="wide",
                   initial_sidebar_state="expanded")

import live  # noqa: E402
import ui  # noqa: E402
import viz  # noqa: E402

SHOW = DEMO / "showcase"
ui.inject_css()

CASES = {  # presentation order
    "orbit": ("🛰️ Satellite orbit + J2", "Recovers Earth's oblateness term; explains orbit-plane precession"),
    "gray_scott": ("🌀 Gray–Scott (The Well)", "2-D reaction–diffusion PDE from a NeurIPS 2024 dataset"),
    "pendulum": ("🧪 Noisy pendulum", "Not sure yet → tells you exactly where to measure next"),
    "osc": ("🆕 Fresh oscillators", "Unpublished laws: no LLM could have memorised them"),
    "ecoli": ("🦠 E. coli growth", "LLM-SR benchmark: ~190× lower out-of-distribution error than published methods"),
    "ks": ("🔥 Real Kuramoto–Sivashinsky", "Chaotic PDE from real data files, confident verdict"),
}


@st.cache_data(show_spinner=False)
def load_case(name):
    d = SHOW / name
    info = json.loads((d / "case.json").read_text())
    arr = {}
    if (d / "arrays.npz").exists():
        with np.load(d / "arrays.npz") as z:
            arr = {k: z[k] for k in z.files}
    return info, arr


@st.cache_data(show_spinner=False)
def fig_cached(kind, case, *args):
    """Build a (possibly heavy, animated) figure once per session server-wide."""
    info, a = load_case(case)
    if kind == "orbit_anim":
        return viz.orbit_animation(a["t"], a["U"])
    if kind == "orbit_kj2":
        return viz.orbit_kepler_vs_j2(a["tt"], a["S_disc"], a["S_kep"])
    if kind == "orbit_el":
        return viz.orbit_elements(a["td_s"], a["raan_data"], a["argp_data"], a["tt"], a["raan_disc"], a["argp_disc"],
                                  a["raan_kep"], a["argp_kep"])
    if kind == "ks_heat":
        return viz.ks_heatmaps(a["t"], a["x"], a["U"], a["Y"])
    if kind == "ks_anim":
        return viz.ks_animation(a["t"], a["x"], a["U"], a["Y"])
    if kind == "gs_anim":
        return viz.gs_animation(a["t"], a["data"], a["model"], 1,
                                "agent's discovered PDE" if info["model_source"] == "agent" else "true PDE (reference)")
    if kind == "gs_gal":
        return viz.gs_gallery({k[4:]: v for k, v in a.items() if k.startswith("gal_")}, info["regimes"])
    raise KeyError(kind)


def show(fig, key=None):
    st.plotly_chart(fig, key=key, config={"displaylogo": False})


def missing_showcase():
    st.error("Showcase data not found. Build it first:\n\n"
             "`PYTHONPATH=. python demo/build_showcase.py`")


def report_download(path, label="⬇️ Download the full HTML report", key=None):
    p = Path(path) if path else None
    if p and p.exists():
        st.download_button(label, p.read_bytes(), file_name=p.name, mime="text/html", key=key)


# ============================================================================= showcase cases
def case_orbit():
    info, a = load_case("orbit")
    d = info["discovery"]
    ui.result_layout(d["verdict"], d["final_model"], d["story"], d["insights"], d["assessment"], truth=info["truth"],
                     names=info["names"], cost=d.get("cost_usd"), wall=d.get("wall_s"), latex_lines=info["latex"],
                     truth_latex=info["truth_latex"])
    st.divider()
    st.markdown("### 🌍 The data: 6 days of a satellite's position and velocity")
    st.caption("Positions in Earth radii; time unit 806.8 s. Press ▶ to fly the first few orbits. "
               "The faint grey tangle is the full measured track: the orbit does not close on itself.")
    show(fig_cached("orbit_anim", "orbit"), "orbit_anim")

    st.markdown("### Why the extra term matters: the orbit plane precesses")
    r = info["raan_rate_deg_per_unit"]
    w = info["argp_rate_deg_per_unit"]
    per_day = 86400 / info["time_unit_s"]
    c1, c2 = st.columns([1, 1.25])
    with c1:
        show(fig_cached("orbit_kj2", "orbit"), "orbit_kj2")
        st.caption("Same starting state, two laws. **Orange:** pure Kepler gravity keeps a single fixed ellipse. "
                   "**Blue (light → dark = time):** the discovered law (Kepler + J2) swings the orbit plane around "
                   "Earth's spin axis, as the data does.")
    with c2:
        m = st.columns(3)
        m[0].metric("Ω drift: data", f"{r['data']:.2f}°", "per time unit", delta_color="off")
        m[1].metric("Ω drift: discovered", f"{r['disc']:.2f}°",
                    f"{100 * (r['disc'] - r['data']) / r['data']:+.1f}% vs data", delta_color="off")
        m[2].metric("Ω drift: Kepler", f"{abs(r['kep']):.2f}°", "fixed plane", delta_color="off")
        show(fig_cached("orbit_el", "orbit"), "orbit_el")
        st.caption(f"Ω = longitude of the ascending node: where the orbit plane cuts the equator. It drifts "
                   f"{r['data'] * per_day:.0f}°/day here (real Earth J2 gives a few °/day for low orbits). "
                   f"Perigee also rotates: {w['data']:.1f}°/unit in the data vs {w['disc']:.1f}°/unit for the "
                   f"discovered model, and 0 for Kepler. RAAN Ω = atan2(h_x, −h_y) with h = r × v; the data curve "
                   "is averaged over one orbit to remove noise.")
    st.info("**J2 = Earth's oblateness** (the equatorial bulge). The challenge's data generator exaggerated it to "
            "J2 = 0.5 so the effect is visible within days; the agents recovered 0.75 = 3/2·J2, i.e. **J2 = 0.5**, by "
            "trajectory shooting (0.75008 ± 0.0001). Honest note: the agents were told the context "
            "\"satellite orbiting Earth\", which made a multipole library a natural first try.", icon="🛰️")
    report_download(SHOW / "orbit" / "report.html", key="dl_orbit")


def case_ks():
    info, a = load_case("ks")
    d = info["discovery"]
    ui.result_layout(d["verdict"], d["final_model"], d["story"], d["insights"], d["assessment"], truth=info["truth"],
                     pde=True, names=info["names"], cost=d.get("cost_usd"), wall=d.get("wall_s"),
                     truth_title="Textbook KS equation")
    st.divider()
    st.markdown("### Space–time: data vs the discovered PDE rolled out from the first snapshot")
    show(fig_cached("ks_heat", "ks"), "ks_heat")
    c1, c2 = st.columns([1.6, 1])
    with c1:
        show(fig_cached("ks_anim", "ks"), "ks_anim")
    with c2:
        show(viz.ks_error(a["t"], a["rel_err"]), "ks_err")
        st.caption("The KS equation is chaotic, yet the discovered PDE tracks the whole 100-time-unit record "
                   "(error ≲ 1e-5 of the signal). These tutorial data were themselves produced by a spectral "
                   "solver, so an exact law reproduces them almost exactly.")
    report_download(SHOW / "ks" / "report.html", key="dl_ks")


def case_gray_scott():
    info, a = load_case("gray_scott")
    res = info.get("results")
    regime, noise = info["regime"], info["noise"]
    p = info["params"]
    row = (res or {}).get(f"{regime}|{noise}") or (res or {}).get(f"{regime}|{noise:g}") or {}
    ag = row.get("agent") or {}
    if ag.get("rhs"):
        f1 = ag.get("f1")
        ui.verdict_banner({"status": "BENCHMARK",
                           "headline": f"Regime '{regime}', {noise:.0%} noise: term F1 = {ui._fmt(f1, 2)}; "
                                       f"rollout VRMSE {ui._fmt(ag.get('vrmse_6-12'))} (steps 6–12), "
                                       f"{ui._fmt(ag.get('vrmse_13-30'))} (13–30).",
                           "recommendation": "One agent session (no skills, no memory, one-line context) on 2 noisy "
                                             "trajectories; scored on a held-out trajectory."})
    else:
        ui.verdict_banner({"status": "BENCHMARK",
                           "headline": "Benchmark running: results pending. Showing the data and the true PDE "
                                       "re-simulated from the held-out initial state.",
                           "recommendation": "Rebuild with `python demo/build_showcase.py gray_scott` once "
                                             "runs/well_gs/results.json exists; this card then shows the agent's PDE."})
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Discovered equation" if ag.get("rhs") else "#### Discovered equation (pending)")
        if ag.get("rhs"):
            ui.show_equations(ag["rhs"], pde=True)
        else:
            st.caption("The agent is working on it…")
    with c2:
        st.markdown("#### Ground truth (The Well, App. C)")
        st.latex(r"\partial_t A = d_A \nabla^2 A - A B^2 + F(1-A)")
        st.latex(r"\partial_t B = d_B \nabla^2 B + A B^2 - (F+k) B")
        st.caption(f"regime '{regime}': F = {p['F']}, k = {p['k']}, d_A = {p['dA']:g}, d_B = {p['dB']:g}")
    st.markdown("### Six pattern regimes, one equation")
    show(fig_cached("gs_gal", "gray_scott"), "gs_gal")
    st.caption("Field B, first frame of each regime. Same two-term reaction law; only F and k change.")
    st.markdown(f"### Held-out trajectory, regime '{regime}' ({noise:.0%} noise in training data)")
    show(fig_cached("gs_anim", "gray_scott"), "gs_anim")
    vr = [v for v in info["vrmse_per_frame"] if v is not None]
    own = {"vrmse_6-12": float(np.mean(info["vrmse_per_frame"][6:13])), "vrmse_13-30": float(np.mean(info["vrmse_per_frame"][13:31]))} if len(vr) == 31 else None
    st.markdown("### Rollout error vs noise, against The Well's neural surrogates")
    show(viz.gs_vrmse_chart(res, regime, own), "gs_vrmse")
    st.caption("Dashed lines: published VRMSE of neural surrogates on this dataset (The Well, Ohana et al., NeurIPS 2024; "
               "one-step VRMSE FNO 0.1365, TFNO 0.3633, U-net 0.2252, CNextU-net 0.1761). Those networks are trained "
               "on hundreds of trajectories; the equation-discovery arms see 2 noisy trajectories of 60 frames. Our rollout "
               "windows start at snapshot 50 of a held-out trajectory, so the comparison is indicative, not like-for-like.")
    if res:
        rows = []
        for k, v in res.items():
            r_, n_ = k.split("|")
            for arm in ("sindy", "weak_sindy", "agent"):
                s = v.get(arm) or {}
                rows.append({"regime": r_, "noise": float(n_), "method": viz.METHOD_LABEL[arm], "term F1": s.get("f1"),
                             "VRMSE 6–12": s.get("vrmse_6-12"), "VRMSE 13–30": s.get("vrmse_13-30"),
                             "err F": s.get("err_F"), "err d_A": s.get("err_dA"), "error": s.get("error")})
        with st.expander("All regimes × noise levels"):
            st.dataframe(pd.DataFrame(rows), hide_index=True)
    else:
        st.info("Results pending: the benchmark (6 regimes × 4 noise levels; SINDy, weak SINDy, agent) is still "
                "running. Only the true-PDE reference point is plotted.", icon="⏳")


def case_pendulum():
    info, a = load_case("pendulum")
    d = info["discovery"]
    ui.result_layout(d["verdict"], d["final_model"], d["story"], d["insights"], d["assessment"], truth=info["truth"],
                     names=info["names"], cost=d.get("cost_usd"), wall=d.get("wall_s"))
    st.divider()
    st.markdown("### Where to measure next")
    n = min(3, a["fans"].shape[0])
    pick = st.segmented_control("Recommended start", list(range(n)), default=0, format_func=lambda i: f"★ #{i + 1}",
                                key="pend_pick") or 0
    show(viz.pendulum_fan(a["U"], a["ics"], a["fans"], a["tt"], pick), f"pend_fan_{pick}")
    ex = d["assessment"]["experiments"]["ranked"][pick]
    pins = ", ".join(f"{c['coefficient']} ({c['info_gain_vs_existing']:.0f}×)" for c in ex.get("informs_coefficients", [])[:3])
    st.caption(f"Each blue curve is a model whose coefficients are drawn from the bootstrap 90% intervals; all of them "
               f"fit the existing data. From start #{pick + 1} (θ = {ex['initial_condition'][0]:.2f}, "
               f"ω = {ex['initial_condition'][1]:.2f}) they diverge, so one swing from there is "
               f"{ex.get('gain_vs_existing_data', 0):.1f}× more informative than repeating your old conditions. "
               f"It mainly pins down: {pins}. Data: 4 trajectories with 5% red (time-correlated) noise. "
               "The damping coefficient is the weak point (±24%).")
    report_download(SHOW / "pendulum" / "report.html", key="dl_pend")


def _restoring(expr, names, xs):
    from eqdisc.sr import evaluate_expr
    X = np.zeros((len(xs), len(names)))
    X[:, names.index("x")] = xs
    y = evaluate_expr(expr, names, X)
    X0 = np.zeros((1, len(names)))
    return y - evaluate_expr(expr, names, X0)[0]


def _exact(m):
    return m["ID"] < 1e-20 and m["OOD"] < 1e-20


def case_osc():
    info, a = load_case("osc")
    V = info["variants"]
    n_exact = sum(_exact(P["agent"]) for P in V.values())
    n_pysr = sum(_exact(P["pysr"]) for P in V.values())
    partial = [k for k, P in V.items() if not _exact(P["agent"])]
    extra = ""
    if partial:
        P = V[partial[0]]
        extra = (f" On {partial[0].replace('osc_', '')} it is partial, yet still "
                 f"{P['pysr']['OOD'] / P['agent']['OOD']:,.0f}× better out-of-distribution than PySR.")
    ui.verdict_banner({"status": "BENCHMARK",
                       "headline": f"{len(V)} oscillator laws written for this test and never published: the agent recovers "
                                   f"{n_exact} EXACTLY (NMSE ~1e-31 in and out of distribution); PySR (300 s each) {n_pysr}/{len(V)}." + extra,
                       "recommendation": "Each problem: 10,000 noise-free samples of (t,) x, v → acceleration a, and a one-line "
                                         "description. Out-of-distribution test = larger amplitudes than training."})
    rows = []
    for k, P in V.items():
        vd = (P["agent"].get("verdict") or {}).get("status", "—")
        rows.append({"problem": k.replace("osc_", ""), "hidden truth": P["truth"],
                     "agent": "✅ exact" if _exact(P["agent"]) else "◐ partial", "agent verdict": vd,
                     "agent OOD": P["agent"]["OOD"], "PySR OOD": P["pysr"]["OOD"], "sparse OOD": P["sparse"]["OOD"],
                     "agent ID": P["agent"]["ID"], "PySR ID": P["pysr"]["ID"]})
    nf = st.column_config.NumberColumn(format="%.2g")
    st.dataframe(pd.DataFrame(rows), hide_index=True,
                 column_config={c: nf for c in ("agent OOD", "PySR OOD", "sparse OOD", "agent ID", "PySR ID")})
    show(viz.nmse_bars({k.replace("osc_", ""): V[k] for k in V}), "osc_bars")
    st.caption("NMSE floors at ~1e-31 (machine precision) when the recovered law is exact.")

    v = st.segmented_control("Problem", list(V), default=info["featured"], key="osc_pick",
                             format_func=lambda k: k.replace("osc_", "")) or info["featured"]
    P = V[v]
    names = P.get("names", ["t", "x", "v"])
    st.markdown(f"### Problem {v.replace('osc_', '')}: the agent's own verdict")
    ui.verdict_banner(P["agent"].get("verdict") or {"status": "RESULT", "headline": "No assessment stored."})
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Hidden truth")
        st.latex(r"\ddot x = " + ui.expr_latex(P["truth"], names, 3))
    with c2:
        st.markdown(f"#### eqdisc agent (\\${P['agent'].get('cost_usd') or 0:.2f})")
        st.latex(r"\ddot x = " + ui.expr_latex(P["agent"]["expr"], names, 4))
    st.caption(f"PySR found: `{P['pysr']['expr']}`" + (f" · description given to all methods: “{P['description']}”"
                                                         if P.get("description") else ""))
    if P["agent"].get("assessment"):
        t2, t3 = st.tabs(["📏 Confidence", "🧪 Next experiments"])
        with t2:
            ui.confidence(P["agent"]["assessment"], static=True)
        with t3:
            ui.experiments(P["agent"]["assessment"])
    sets = {}
    for split, lab in (("test_id", "in-distribution test"), ("test_ood", "out-of-distribution test (larger amplitudes)")):
        sets[lab] = (a[f"{v}_{split}_y"], {mm: a[f"{v}_{split}_{mm}"] for mm in ("sparse", "pysr", "agent")})
    show(viz.pred_vs_true(sets), f"osc_pvt_{v}")
    Xtr, Xood = a[f"{v}_train_X"], a[f"{v}_test_ood_X"]
    lim = float(np.abs(Xood[:, 1]).max()) * 1.15
    xs = np.linspace(-lim, lim, 400)
    curves = {"hidden truth": (_restoring(P["truth"], names, xs), viz.VIOLET, "solid"),
              "eqdisc agent": (_restoring(P["agent"]["expr"], names, xs), viz.BLUE, "dash"),
              "PySR": (_restoring(P["pysr"]["expr"], names, xs), viz.ORANGE, "dot")}
    st.markdown("#### Restoring force, inside and outside the training range")
    show(viz.restoring_force(xs, curves, (float(Xtr[:, 1].min()), float(Xtr[:, 1].max())),
                             (float(Xood[:, 1].min()), float(Xood[:, 1].max()))), f"osc_rf_{v}")
    if _exact(P["agent"]):
        st.caption("Exact recovery: the agent's curve lies on the truth everywhere, including far outside the training range.")
    else:
        st.caption("Partial recovery: damping and forcing are right, but the restoring force is written as a polynomial that "
                   "matches the truth inside the training range and drifts outside it. The agent's own verdict flags this "
                   "(COLLECT MORE DATA, with the missing-term evidence).")
    dev = info.get("dev_batch") or {}
    if dev:
        with st.expander("Honesty note: development batch A"):
            st.markdown("These four problems (batch B) were held out. An earlier batch A was used while developing the "
                        "static mode; there the agent beat PySR out-of-distribution on 3 of 4 but wrote rational restoring "
                        "forces as Taylor polynomials, which motivated the assessment stage.")
            st.dataframe(pd.DataFrame([{"problem": k.replace("osc_", ""), "hidden truth": d["truth"],
                                        "agent OOD": d["agent"]["OOD"], "PySR OOD": d["pysr"]["OOD"],
                                        "sparse OOD": d["sparse"]["OOD"]} for k, d in dev.items()]),
                         hide_index=True, column_config={c: nf for c in ("agent OOD", "PySR OOD", "sparse OOD")})


def case_ecoli():
    info, a = load_case("ecoli")
    ours = info["comparison"][0]
    best_other = min(r["OOD"] for r in info["comparison"][1:])
    ui.verdict_banner({"status": "BENCHMARK",
                       "headline": f"Test NMSE {ours['ID']:.2g} (in-distribution) / {ours['OOD']:.2g} (out-of-distribution): "
                                   f"{best_other / ours['OOD']:.0f}× lower OOD error than the best published LLM-SR result.",
                       "recommendation": "Multiplicative law: Monod substrate uptake × temperature optimum × pH window."})
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.markdown("#### Discovered law")
        st.latex(info["latex"])
        cand = (info.get("candidates") or [{}])[0]
        if cand.get("rationale"):
            st.caption("Agent's rationale: " + cand["rationale"])
        st.caption(f"cost \\${info['cost_usd']:.2f} · {info['wall_s'] / 60:.0f} min")
    with c2:
        df = pd.DataFrame(info["comparison"])
        st.dataframe(df, hide_index=True, column_config={"ID": st.column_config.NumberColumn("NMSE ID", format="%.3g"),
                                                         "OOD": st.column_config.NumberColumn("NMSE OOD", format="%.3g")})
        show(viz.bars_simple([r["method"] for r in info["comparison"]], [r["OOD"] for r in info["comparison"]],
                             [viz.BLUE] + ["rgba(128,128,128,.6)"] * (len(info["comparison"]) - 1), "OOD NMSE (log)"), "eco_bars")
    st.markdown("### Each factor, seen in the data")
    st.caption("Divide each measured growth rate by the model's other factors; if the law is right, the points collapse onto "
               "the discovered single-variable factor.")
    tr = pd.concat([pd.read_csv(SHOW / "ecoli" / f) for f in ("train.csv", "test_id.csv")])
    b, s, T, pH, y = (tr[c].to_numpy() for c in ("b", "s", "temp", "pH", "db"))
    fS = s / (1 + s)
    fT = 1 / (1 + ((T - 35) / 3.76) ** 4)
    fP = np.exp(-np.abs(pH - 7)) * np.sin(np.pi * (pH - 2) / 10) ** 2
    cols = st.columns(3)
    for col, (x, other, f, lab, xs_fn) in zip(cols, [
            (T, 0.5 * b * fS * fP, fT, "temperature (°C)", lambda xs: 1 / (1 + ((xs - 35) / 3.76) ** 4)),
            (pH, 0.5 * b * fS * fT, fP, "pH", lambda xs: np.exp(-np.abs(xs - 7)) * np.sin(np.pi * (xs - 2) / 10) ** 2),
            (s, 0.5 * b * fT * fP, fS, "substrate S", lambda xs: xs / (1 + xs))]):
        ok = other > 0.15 * np.nanmax(other)
        xs = np.linspace(np.nanmin(x[ok]), np.nanmax(x[ok]), 300)
        with col:
            show(viz.ecoli_collapse(x[ok], y[ok] / other[ok], xs, xs_fn(xs), lab), f"eco_{lab}")
    te = {}
    for split, lab in (("test_id", "in-distribution test"), ("test_ood", "out-of-distribution test")):
        d = pd.read_csv(SHOW / "ecoli" / f"{split}.csv")
        te[lab] = (d["db"].to_numpy(), {"agent": a[f"{split}_pred"]})
    show(viz.pred_vs_true(te), "eco_pvt")
    st.markdown("<div class='footnote'>⚠️ LLM-SR's paper publishes this equation's structure, so recall cannot be excluded; "
                "see the unpublished oscillators for a contamination-free test.<br>Data: LLM-SR benchmark "
                "(Shojaee et al., ICLR 2025; github.com/deep-symbolic-mathematics/LLM-SR, MIT License). Baseline numbers "
                "from the LLM-SR paper.</div>", unsafe_allow_html=True)


RENDER = {"orbit": case_orbit, "ks": case_ks, "gray_scott": case_gray_scott, "pendulum": case_pendulum,
          "osc": case_osc, "ecoli": case_ecoli}


def page_showcase():
    st.markdown("## Give it data. Get back the equation, how it was found, how sure it is, and what to measure next.")
    avail = [c for c in CASES if (SHOW / c / "case.json").exists()]
    if not avail:
        missing_showcase()
        return
    pick = st.pills("Case", avail, default=avail[0], format_func=lambda c: CASES[c][0], key="case",
                    label_visibility="collapsed") or avail[0]
    st.caption(CASES[pick][1])
    RENDER[pick]()


# ============================================================================= live tab
EXAMPLES = {
    "Example: pendulum (4 noisy trajectories)": (DEMO / "examples" / "pendulum.csv",
                                                 "Angle (rad) and angular velocity (rad/s) of a swinging pendulum; "
                                                 "4 releases from different angles. Sensor noise is time-correlated."),
    "Example: E. coli growth (static law)": (DEMO / "examples" / "ecoli_growth.csv",
                                             "Bacterial growth rate db as a function of population density b, substrate "
                                             "concentration s, temperature temp (°C) and pH."),
}


def _render_log(events, box):
    lines = [live.fmt_event(e).replace("$", "\\$") for e in events]
    box.markdown("\n\n".join(lines[-60:]) or "_starting…_", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _dyn_figures(dataset_path, rhs_json, assessment_json, out_dir):
    from eqdisc.evaluate import load
    from eqdisc.plots import plot_model, plot_recommendations
    meta, data = load(dataset_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    figs = []
    try:
        figs.append(("Model vs held-out trajectory", plot_model(meta, data, json.loads(rhs_json), out / "fig_model.png")))
    except Exception as e:  # noqa: BLE001
        figs.append(("plot_model failed", str(e)))
    try:
        p = plot_recommendations(meta, data, json.loads(assessment_json), out / "fig_recs.png")
        if p:
            figs.append(("Recommended experiments", p))
    except Exception as e:  # noqa: BLE001
        figs.append(("plot_recommendations failed", str(e)))
    return meta["kind"], meta["variables"], figs


def render_live_result(job):
    res = job.result
    if res.get("rehearsal"):
        st.caption("🎭 Rehearsal result (scripted backend, no API calls).")
    if res.get("rehearsal_note"):
        st.info(res["rehearsal_note"])
    if res["kind"] == "dynamics":
        run_dir = Path(job.run_dir)
        kind, names, figs = "ode", None, []
        if res.get("dataset_path") and res.get("final_model"):
            kind, names, figs = _dyn_figures(res["dataset_path"], json.dumps(res["final_model"]),
                                             json.dumps(res.get("assessment") or {}), str(run_dir / "demo_figs"))
        ui.result_layout(res.get("verdict"), res.get("final_model"), res.get("story"), res.get("insights"),
                         res.get("assessment"), pde=kind == "pde", names=names, cost=res.get("cost_usd"),
                         wall=job.elapsed)
        for title, p in figs:
            st.markdown(f"#### {title}")
            if str(p).endswith(".png") and Path(p).exists():
                st.image(p)
            else:
                st.caption(p)
        c1, c2 = st.columns(2)
        with c1:
            report_download(res.get("report"), key="dl_live_report")
        with c2:
            st.download_button("⬇️ Result JSON", json.dumps(res, default=str, indent=1), "discovery.json", key="dl_live_json")
    else:
        from eqdisc.sr import evaluate_expr, nmse
        names, target, expr = res["names"], res["target"], res.get("expr")
        if not expr:
            st.error("No expression was found.")
            return
        v = res.get("verdict") or {"status": "RESULT",
                                   "headline": f"Best validated law for {target}: validation NMSE {res.get('val_nmse', float('nan')):.3g}.",
                                   "recommendation": "Check the residuals below for structure before trusting it outside "
                                                     "the observed range."}
        ui.verdict_banner(v)
        st.markdown("#### Discovered law")
        st.latex(f"{ui.expr_latex(target, [target])} = {ui.expr_latex(expr, names)}")
        m = st.columns(3)
        m[0].metric("Validation NMSE", f"{res.get('val_nmse', float('nan')):.3g}")
        m[1].metric("Cost", f"${res.get('cost_usd') or 0:.2f}")
        m[2].metric("Wall time", f"{job.elapsed:.0f} s")
        if res.get("assessment"):
            t2, t3 = st.tabs(["📏 Confidence", "🧪 Next experiments"])
            with t2:
                ui.confidence(res["assessment"], static=True)
            with t3:
                ui.experiments(res["assessment"], names)
        df = pd.read_csv(res["csv"])
        nm2, tg2, X, y, tr, va = live.static_task_arrays(df, target)
        yh = evaluate_expr(expr, names, X)
        c1, c2 = st.columns(2)
        with c1:
            show(viz.pred_vs_true({"all rows": (y, {"agent": yh})}), "live_pvt")
        with c2:
            import plotly.graph_objects as go
            fig = go.Figure(go.Scattergl(x=yh, y=y - yh, mode="markers", marker=dict(size=4, color=viz.BLUE, opacity=0.5)))
            fig.add_hline(y=0, line=dict(color="gray", dash="dash"))
            fig.update_layout(xaxis_title="predicted", yaxis_title="residual (true − predicted)")
            show(viz._layout(fig, 420), "live_resid")
        if res.get("candidates"):
            with st.expander("All submitted candidates"):
                st.dataframe(pd.DataFrame(res["candidates"]), hide_index=True)
        st.download_button("⬇️ Result JSON", json.dumps(res, default=str, indent=1), "sr_result.json", key="dl_live_sr")


def page_live(fake):
    st.markdown("## Run on your data")
    st.caption("Upload a CSV. Time series: a time column, one column per variable, optional trajectory-id column. "
               "Static law: one row per measurement.")
    job = st.session_state.get("job")
    running = job is not None and not job.done
    src = st.radio("Data", ["Upload a CSV"] + list(EXAMPLES), horizontal=True, disabled=running, key="src")
    df, ctx_default, fname = None, "", None
    if src == "Upload a CSV":
        up = st.file_uploader("CSV file", type=["csv", "tsv", "txt"], disabled=running)
        if up is not None:
            df = pd.read_csv(up, sep=None, engine="python")
            fname = Path(up.name).stem
    else:
        path, ctx_default = EXAMPLES[src]
        if path.exists():
            df, fname = pd.read_csv(path), path.stem
        else:
            st.error(f"{path} missing: run demo/build_showcase.py")
    if df is not None:
        with st.expander(f"Preview: {df.shape[0]} rows × {df.shape[1]} columns", expanded=False):
            st.dataframe(df.head(12), hide_index=True)
    c1, c2 = st.columns([1.4, 1])
    with c1:
        context = st.text_area("Domain context (optional): what the variables are, units, anything known",
                               value=ctx_default, height=100, disabled=running, key=f"ctx_{src}")
    with c2:
        mode = st.segmented_control("Mode", ["Auto", "Dynamics (ODE/PDE)", "Static law y = f(x)"], default="Auto",
                                    disabled=running, key="mode") or "Auto"
        resolved = mode
        if df is not None and mode == "Auto":
            resolved = "Dynamics (ODE/PDE)" if live.detect_mode(df) == "dynamics" else "Static law y = f(x)"
            st.caption(f"Auto → **{resolved}** ({'time column found' if resolved.startswith('Dyn') else 'no time column'})")
        target = None
        if df is not None and resolved.startswith("Static"):
            num = list(df.select_dtypes("number").columns)
            target = st.selectbox("Target column", num, index=len(num) - 1, disabled=running)
        budget = st.segmented_control("Budget", ["Quick", "Full"], default="Quick", disabled=running, key="budget") or "Quick"
        st.caption("Quick: 2 branches, no red team (≈ \\$0.5–1, 2–4 min). Full: 3 branches + adversary (≈ \\$1–3, 4–8 min)."
                   if resolved.startswith("Dyn") else "Quick: 2 agent sessions. Full: 3 sessions.")
    go_btn = st.button("🚀 Discover", type="primary", disabled=running or df is None)
    if fake:
        st.caption("🎭 Rehearsal mode is ON: scripted backend, no API calls, no cost.")

    if go_btn and df is not None:
        run_dir = live.RUNS / time.strftime("%Y%m%d-%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        csv_path = run_dir / f"{live.ident(fname or 'data')}.csv"
        df.to_csv(csv_path, index=False)
        quick = budget == "Quick"
        if resolved.startswith("Dyn"):
            fn = live.fake_dynamics if fake else live.run_dynamics
            job = live.Job(fn, csv_path=str(csv_path), run_dir=str(run_dir), n_branches=2 if quick else 3,
                           adversary=not quick, context=context)
        else:
            fn = live.fake_static if fake else live.run_static
            job = live.Job(fn, csv_path=str(csv_path), target=target, context=context, n_sessions=2 if quick else 3)
        job.run_dir = str(run_dir)
        st.session_state.job = job.start()
        running = True

    job = st.session_state.get("job")
    if job is None:
        return
    st.divider()
    label = "Agents at work…" if not job.done else ("Failed" if job.error else "Done")
    with st.status(label, expanded=not job.done, state="running" if not job.done else ("error" if job.error else "complete")) as status:
        timer = st.empty()
        box = st.container(height=380)
        logph = box.empty()
        while not job.done:
            job.drain()
            _render_log(job.events, logph)
            timer.caption(f"⏱️ {job.elapsed:.0f} s · {sum(e.get('type') == 'tool' for e in job.events)} tool calls")
            time.sleep(0.4)
        job.drain()
        _render_log(job.events, logph)
        timer.caption(f"⏱️ {job.elapsed:.0f} s · {sum(e.get('type') == 'tool' for e in job.events)} tool calls")
        status.update(label="Failed" if job.error else f"Done in {job.elapsed:.0f} s",
                      state="error" if job.error else "complete", expanded=False)
    if running:
        st.rerun()
    if job.error:
        st.error(job.error)
        return
    render_live_result(job)
    if st.button("Clear result"):
        st.session_state.pop("job", None)
        st.rerun()


# ============================================================================= how it works
def page_how():
    st.markdown("## How it works")
    st.graphviz_chart("""
digraph G { rankdir=LR; bgcolor="transparent"; node [shape=box, style="rounded,filled", fillcolor="#eef2ff",
  color="#6366f1", fontname="Helvetica", fontsize=11]; edge [color="#888888"];
  D [label="your data\\n(csv, mat, npz, h5…)"]; I [label="ingest\\n+ data card"];
  P [label="intuition\\npre-analysis"]; B1 [label="branch:\\nstructure-first"]; B2 [label="branch:\\nsparse regression"];
  B3 [label="branch:\\nsymbolic"]; T [label="tournament\\nCV error · BIC · rollouts"]; A [label="adversary\\n(red team)", fillcolor="#fee2e2", color="#dc2626"];
  Q [label="assessment\\nUQ · noise floor · OED"]; V [label="verdict + next\\nexperiments + report", fillcolor="#dcfce7", color="#16a34a"];
  D->I->P; P->B1; P->B2; P->B3; B1->T; B2->T; B3->T; T->A->Q->V; }""")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**Parallel Claude agents**, each with a different strategy and a shared scientific toolbox: weak-form "
                "SINDy, PySR with templates, skeleton fits, invariants, symmetries, coordinate transforms, their own "
                "Python, and plots they look at.")
    c2.markdown("**A red-team agent** attacks the tournament winner: structured residuals, missing terms, simpler rivals. "
                "A challenger must win the same held-out tournament to replace it.")
    c3.markdown("**Assessment** gives per-term bootstrap intervals and ΔBIC, the noise floor, competing models, the "
                "predictability horizon, and simulated experiments ranked by how much plausible models disagree.")
    st.markdown("| verdict | meaning |\n|---|---|\n"
                "| ✅ **CONFIDENT** | every term supported, error at the noise floor, predicts held-out data |\n"
                "| 🔷 **CONFIDENT IN PREDICTIONS** | all plausible models predict the same; exact terms not unique |\n"
                "| 🧪 **COLLECT MORE DATA (here)** | competing models remain; we tell you which experiment separates them |\n"
                "| ❔ **INCONCLUSIVE** | the data cannot determine the model; we say what is missing |")


# ============================================================================= main
def main():
    env_fake = os.environ.get("EQDISC_DEMO_FAKE", "") not in ("", "0", "false")
    with st.sidebar:
        st.markdown("# 🧭 eqdisc")
        st.markdown("**autoresearch for governing equations**")
        st.caption("Data in → the ODE/PDE or law, the key steps that led to it, how sure we are, and what to measure next.")
        page = st.radio("Navigate", ["🏆 Showcase", "⚡ Run on your data", "⚙️ How it works"], label_visibility="collapsed")
        st.divider()
        st.caption("Pipeline: ingest → intuition → parallel Claude agents → tournament → red team → assessment → report")
        with st.expander("⚙️", expanded=False):
            fake = st.toggle("Rehearsal mode (no API calls)", value=env_fake, key="fake")
    if page.startswith("🏆"):
        page_showcase()
    elif page.startswith("⚡"):
        page_live(fake)
    else:
        page_how()


main()
