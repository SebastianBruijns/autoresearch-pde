"""Self-contained HTML report for one discovery session (the autoscience deliverable).

    python -m eqdisc.report runs/agent_xxx            # writes runs/agent_xxx/report.html
"""
import base64
import html
import os
import json
import sys
from pathlib import Path

import sympy as sp

from . import plots
from .evaluate import load
from .solvers import parse

CSS = """
:root{--bg:#fbfaf7;--fg:#1d1d1f;--mut:#6b6b70;--card:#fff;--line:#e6e3dc;--acc:#2f6f5e;--bad:#b4462f}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe8;--mut:#a3a19b;--card:#1f1f1d;--line:#34332f;--acc:#7cc4ad;--bad:#e07a62}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,Segoe UI,Inter,sans-serif}
main{max-width:1040px;margin:0 auto;padding:28px 16px 80px}
h1{font-size:26px;margin:0 0 4px;overflow-wrap:anywhere} h2{font-size:18px;margin:34px 0 10px;border-bottom:1px solid var(--line);padding-bottom:6px}
.sub{color:var(--mut)} .card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:10px 0}
table{border-collapse:collapse;width:100%;font-size:13px} td,th{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
th{color:var(--mut);font-weight:600} code{font-size:12.5px;word-break:break-word}
.kpi{display:flex;gap:12px;flex-wrap:wrap}.kpi div{flex:1;min-width:130px;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px}
.kpi b{display:block;font-size:22px}.kpi span{color:var(--mut);font-size:12px}
img{max-width:100%;border-radius:8px;border:1px solid var(--line);background:#fff}
.eq{font-size:17px;overflow-x:auto} .good{color:var(--acc)} .bad{color:var(--bad)}
"""


def _txt(x):
    """Narration fields may be strings or lists of strings."""
    return "; ".join(map(str, x)) if isinstance(x, (list, tuple)) else str(x or "")


def _latex(v, e, names):
    try:
        ex = parse(e, names)
        ex = ex.xreplace({a: sp.Float(float(a), 4) for a in ex.atoms(sp.Float)})
        return rf"\[\dot{{{sp.latex(sp.Symbol(v))}}} = {sp.latex(ex)}\]"
    except Exception:  # noqa: BLE001
        return f"<code>d{html.escape(v)}/dt = {html.escape(str(e))}</code>"


def _img(path):
    return f'<img src="data:image/png;base64,{base64.b64encode(Path(path).read_bytes()).decode()}">'


def _fmt(x):
    if isinstance(x, float):
        return f"{x:.3g}"
    return html.escape(str(x))


def build_report(run_dir, dataset=None):
    run_dir = Path(run_dir)
    res = json.loads((run_dir / "result.json").read_text())
    log = json.loads((run_dir / "transcript.json").read_text())
    dataset = Path(dataset or res.get("dataset_path") or Path("datasets") / res["dataset"])
    meta, data = load(dataset)
    names = list(meta["allowed_symbols"]) + list(meta["variables"])
    sub = res.get("submitted") or {}
    he = res.get("hidden_eval") or {}
    parts = [f"<h1>Equation discovery report</h1><div class='sub'>dataset <code>{html.escape(meta['name'])}</code>"
             f" · {meta['kind'].upper()} · variables {', '.join(meta['variables'])} · {res.get('n_tool_calls')} tool calls"
             f" · {res.get('wall_s')} s"
             + (f" · ${res['cost_usd']:.2f}" if res.get("cost_usd") is not None else "") + "</div>"]

    # discovered model
    parts.append("<h2>Discovered model</h2><div class='card eq'>"
                 + "".join(_latex(v, e, names) for v, e in (sub.get("rhs") or {}).items()) + "</div>")
    if sub.get("rhs_active"):
        parts.append(f"<div class='sub'>fitted in coordinates <b>{html.escape(sub.get('submitted_in', ''))}</b>: "
                     + " ; ".join(f"<code>d{k}/dt = {html.escape(v)}</code>" for k, v in sub["rhs_active"].items()) + "</div>")
    if sub.get("rationale"):
        parts.append(f"<div class='card'><b>Agent's rationale.</b> {html.escape(sub['rationale'])}</div>")
    if res.get("critic"):
        parts.append("<div class='card'><b>Critic review.</b><br>" + "<br>".join(html.escape(c) for c in res["critic"]) + "</div>")

    # scores
    if he and "score" in he and "error" not in he:
        kp = [("hidden score", he["score"]), ("vector-field NRMSE", he.get("vf_nrmse")),
              ("rollout NRMSE", he.get("rollout_nrmse")), ("valid fraction", he.get("valid_frac")),
              ("terms", he.get("n_terms"))]
        if "f1" in he:
            kp.append(("term F1 vs truth", he["f1"]))
        if res.get("judge"):
            kp.append(("symbolically equivalent", "yes" if res["judge"]["equivalent"] else "no"))
        parts.append("<h2>Evaluation on hidden data (unseen initial conditions)</h2><div class='kpi'>"
                     + "".join(f"<div><b>{_fmt(v)}</b><span>{k}</span></div>" for k, v in kp) + "</div>")
        if he.get("truth"):
            parts.append("<div class='card eq'><span class='sub'>ground truth</span>"
                         + "".join(_latex(v, e, names) for v, e in he["truth"].items()) + "</div>")

    # confidence & next steps (assessment)
    a = res.get("assessment") or {}
    if a and "confidence" in a:
        c = a["confidence"]
        cls = {"high": "good", "medium": "", "low": "bad"}.get(c["level"], "")
        parts.append(f"<h2>Confidence and next steps</h2><div class='card'><b class='{cls}' style='font-size:20px'>"
                     f"{c['level'].upper()} confidence</b><ul>" + "".join(f"<li>{html.escape(r)}</li>" for r in c["reasons"])
                     + f"</ul><span class='sub'>statistics basis: {html.escape(a.get('statistics_basis', ''))}</span></div>")
        rows = "".join(f"<tr><td>d{html.escape(t['var'])}/dt</td><td><code>{html.escape(t['term'])}</code></td><td>{_fmt(t['coef'])}</td>"
                       f"<td>{_fmt(t.get('ci90'))}</td><td>{'yes' if t.get('significant') else '<b class=bad>no</b>'}</td>"
                       f"<td>{_fmt(t.get('dBIC_if_removed'))}</td></tr>" for t in a.get("terms", []))
        parts.append("<div class='card'><b>Per-term evidence</b><table><tr><th>equation</th><th>term</th><th>coef</th>"
                     "<th>90% CI</th><th>significant</th><th>ΔBIC if removed</th></tr>" + rows + "</table></div>")
        if a.get("model_ambiguity"):
            parts.append(f"<div class='card'><b>Competing models.</b> {html.escape(a['model_ambiguity']['verdict'])}</div>")
        pr = a.get("predictability") or {}
        if pr:
            parts.append(f"<div class='card'><b>Predictability horizon:</b> {_fmt(pr.get('horizon'))}. {html.escape(pr.get('meaning', ''))}</div>")
        ex = (a.get("experiments") or {}).get("ranked", [])
        if ex:
            er = ""
            for i, e in enumerate(ex, 1):
                what = e.get("initial_condition") or e.get("description")
                if isinstance(what, list):
                    what = "start at (" + ", ".join(f"{x:.3g}" for x in what) + ")" + ("" if e.get("inside_data_range") else " · outside current data")
                pins = "; ".join(f"{x['coefficient']} ×{x['info_gain_vs_existing']}" for x in e.get("informs_coefficients", []))
                er += (f"<tr><td>{i}</td><td>{html.escape(str(what))}</td><td>{_fmt(e.get('score'))}</td>"
                       f"<td>{_fmt(e.get('gain_vs_existing_data'))}</td><td>{html.escape(pins)}</td></tr>")
            parts.append("<div class='card'><b>Recommended next experiments</b> <span class='sub'>(where plausible models "
                         "disagree most, relative to noise; ×N = information about that coefficient vs. re-measuring "
                         "existing conditions)</span><table><tr><th>#</th><th>experiment</th><th>score</th><th>× existing</th>"
                         "<th>pins down</th></tr>" + er + "</table></div>")
        if a.get("data_advice"):
            parts.append("<div class='card'><b>Data advice</b><ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in a["data_advice"]) + "</ul></div>")
        if a.get("questions_for_human"):
            parts.append("<div class='card'><b>Questions for the scientist</b><ul>" + "".join(
                f"<li>{html.escape(x)}</li>" for x in a["questions_for_human"]) + "</ul></div>")
    if res.get("human_log"):
        parts.append("<h2>Human-in-the-loop exchanges</h2>" + "".join(
            f"<div class='card'><span class='sub'>{html.escape(h['question'][:300])}</span><br><b>{html.escape(h['answer'])}</b></div>"
            for h in res["human_log"]))

    # figures
    figs = []
    p = run_dir / "fig_data.png"
    plots.plot_data(meta, data, p)
    figs.append(("Data overview", p))
    if sub.get("rhs"):
        p = run_dir / "fig_model.png"
        try:
            plots.plot_model(meta, data, sub["rhs"], p)
            figs.append(("Model vs held-out public trajectory", p))
        except Exception:  # noqa: BLE001
            pass
    parts.append("<h2>Figures</h2>" + "".join(f"<div class='card'><b>{t}</b><br>{_img(f)}</div>" for t, f in figs))

    # research trail
    rows = []
    for ev in log:
        if ev["type"] != "tool":
            continue
        out = ev["output"] if isinstance(ev["output"], dict) else {}
        v = out.get("validation_original", out.get("validation", {})) or {}
        found = out.get("rhs_original", out.get("rhs", ""))
        extra = ""
        if ev["name"] == "find_invariants":
            extra = "; ".join(i["H"][:80] for i in out.get("invariants", []) if i.get("rel_variation", 1) < 0.05)
        if ev["name"] == "diagnose":
            extra = f"noise≈{ {k: round(x, 3) for k, x in (out.get('noise_rel_estimate') or {}).items()} }"
        rows.append(f"<tr><td>{html.escape(ev['name'])}</td><td><code>{html.escape(json.dumps(ev['input'])[:220])}</code></td>"
                    f"<td><code>{html.escape(json.dumps(found)[:220]) if found else html.escape(extra)}</code></td>"
                    f"<td>{_fmt(v.get('deriv_nrmse', ''))}</td><td>{_fmt(v.get('rollout_valid_time', ''))}</td></tr>")
    parts.append("<h2>Research trail</h2><table><tr><th>tool</th><th>input</th><th>result</th>"
                 "<th>deriv err</th><th>valid time</th></tr>" + "".join(rows) + "</table>")
    notes = [ev["text"] for ev in log if ev["type"] == "text" and ev["text"].strip()]
    if notes:
        parts.append("<h2>Agent notes</h2>" + "".join(f"<div class='card'>{html.escape(n[:1500])}</div>" for n in notes[:30]))
    if res.get("lessons"):
        parts.append("<h2>Lessons written to memory</h2><ul>" + "".join(f"<li>{html.escape(l)}</li>" for l in res["lessons"]) + "</ul>")

    doc = (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>Discovery report</title><style>{CSS}</style>"
           "<script>window.MathJax={tex:{inlineMath:[['$','$']]}};</script>"
           "<script src='https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js' async></script></head>"
           f"<body><main>{''.join(parts)}</main></body></html>")
    out = run_dir / "report.html"
    out.write_text(doc)
    return str(out)


if __name__ == "__main__":
    print(build_report(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))


def _checks_card(res):
    """'Data and model checks' card: fired evidence-layer findings in plain language, repairs, and the valid range."""
    from .audit import describe
    a = res.get("assessment") or {}
    ev = res.get("evidence") or {}
    findings = a.get("findings") if a.get("findings") is not None else \
        (ev.get("data_findings") or []) + (ev.get("final_findings") or [])
    fired = [f for f in findings if f.get("fired")]
    rank = {"critical": 0, "warn": 1, "info": 2}
    fired.sort(key=lambda f: (bool(f.get("resolved")), rank.get(f.get("severity"), 3)))
    if not fired:
        body = "<div class='good'><b>All checks passed.</b></div><div class='sub'>Data (glitches, gaps, sampling) and model " \
               "(same coefficients on every slice, residual is only noise) checks found nothing.</div>"
    else:
        items = []
        for f in fired:
            sev = f.get("severity", "")
            cls = "bad" if sev == "critical" and not f.get("resolved") else "sub" if f.get("resolved") or sev == "info" else ""
            items.append(f"<li class='{cls}'>{html.escape(describe(f))} <span class='sub'>[{html.escape(str(f.get('id')))}]</span></li>")
        body = "<ul>" + "".join(items) + "</ul>"
    for r in ev.get("data_repairs") or []:
        body += f"<div class='sub'>Data repair applied before fitting: {html.escape(r.get('note', ''))}.</div>"
    vr = (res.get("verdict") or {}).get("valid_range")
    if vr:
        body += "<div><b>Valid range.</b> " + html.escape("; ".join(
            f"{v} in [{_fmt(lo)}, {_fmt(hi)}]" for v, (lo, hi) in vr.items())) + " (no data beyond).</div>"
    return "<h2>Data and model checks</h2><div class='card'>" + body + "</div>"


def build_discovery_report(out_dir, res, meta, data):
    """Top-level report for eqdisc.discover: verdict first, then the story, then the evidence."""
    out_dir = Path(out_dir)
    names = list(meta["allowed_symbols"]) + list(meta["variables"])
    v = res["verdict"]
    color = {"CONFIDENT": "good", "CONFIDENT IN PREDICTIONS": "good", "COLLECT MORE DATA": "", "INCONCLUSIVE": "bad"}.get(v["status"], "")
    st = res.get("story") or {}
    P = [f"<h1>{html.escape(meta['name'])}: discovered model</h1><div class='sub'>{meta['kind'].upper()} · variables "
         f"{', '.join(meta['variables'])} · {len(res['branches'])} parallel branches"
         f"{' + adversary' if res.get('adversary') else ''} · {res['wall_s']} s · ${res['cost_usd']:.2f}</div>"]
    P.append(f"<div class='card' style='border-width:2px'><div class='{color}' style='font-size:22px;font-weight:700'>"
             f"{html.escape(v['status'])}</div><div style='font-size:16px;margin:6px 0'>{html.escape(v['headline'])}</div>"
             f"<div><b>Recommendation.</b> {html.escape(v['recommendation'])}</div></div>")
    if res.get("final_model"):
        P.append("<div class='card eq'>" + "".join(_latex(k, e, names) for k, e in res["final_model"].items()) + "</div>")
    P.append(_checks_card(res))
    if st.get("headline"):
        P.append(f"<div class='card'><b>Summary.</b> {html.escape(st['headline'])}"
                 + (f"<br><br><b>Physical interpretation.</b> {html.escape(st.get('physical_interpretation', ''))}" if st.get("physical_interpretation") else "")
                 + "</div>")
    # key steps
    steps = st.get("key_steps") or []
    if steps:
        P.append("<h2>Key steps</h2><table><tr><th>#</th><th>observation</th><th>decision</th><th>outcome</th></tr>" + "".join(
            f"<tr><td>{i}</td><td>{html.escape(_txt(s_.get('observation', '')))}</td><td>{html.escape(_txt(s_.get('decision', '')))}</td>"
            f"<td>{html.escape(_txt(s_.get('outcome', '')))}</td></tr>" for i, s_ in enumerate(steps, 1)) + "</table>")
    P.append("<h2>Findings along the way</h2><table><tr><th>step</th><th>finding</th><th>why it matters</th></tr>" + "".join(
        f"<tr><td>{html.escape(i['step'])}</td><td>{html.escape(i['finding'])}</td><td class='sub'>{html.escape(i['why_it_matters'])}</td></tr>"
        for i in res["insights"]) + "</table>")
    # confidence & next experiments
    a = res.get("assessment") or {}
    if a:
        P.append("<h2>Confidence</h2><div class='card'><b>" + a["confidence"]["level"].upper() + "</b><ul>"
                 + "".join(f"<li>{html.escape(r)}</li>" for r in a["confidence"]["reasons"]) + "</ul></div>")
        rows = "".join(f"<tr><td>d{html.escape(t['var'])}/dt</td><td><code>{html.escape(t['term'])}</code></td><td>{_fmt(t['coef'])}</td>"
                       f"<td>{_fmt(t.get('ci90'))}</td><td>{'yes' if t.get('significant') else '<b class=bad>no</b>'}</td>"
                       f"<td>{_fmt(t.get('dBIC_if_removed'))}</td></tr>" for t in a.get("terms", []))
        P.append("<table><tr><th>equation</th><th>term</th><th>coef</th><th>90% CI</th><th>significant</th><th>ΔBIC if removed</th></tr>"
                 + rows + "</table>")
        try:
            from .plots import plot_recommendations
            f = plot_recommendations(meta, data, a, out_dir / "fig_recommendations.png")
            if f:
                P.append("<h2>Where to measure next</h2><div class='card'>" + _img(f) + "</div>")
        except Exception:  # noqa: BLE001
            pass
        ex = (a.get("experiments") or {}).get("ranked", [])
        if ex:
            P.append("<table><tr><th>#</th><th>experiment</th><th>score</th><th>× existing</th><th>pins down</th></tr>" + "".join(
                f"<tr><td>{i}</td><td>{html.escape(str(e.get('initial_condition') or e.get('description')))}</td><td>{_fmt(e.get('score'))}</td>"
                f"<td>{_fmt(e.get('gain_vs_existing_data'))}</td><td>{html.escape('; '.join(c['coefficient'] + ' ×' + str(c['info_gain_vs_existing']) for c in e.get('informs_coefficients', [])))}</td></tr>"
                for i, e in enumerate(ex, 1)) + "</table>")
        if a.get("data_advice"):
            P.append("<div class='card'><b>Data advice</b><ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in a["data_advice"]) + "</ul></div>")
        if a.get("questions_for_human"):
            P.append("<div class='card'><b>Questions for you</b><ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in a["questions_for_human"]) + "</ul></div>")
    if st.get("caveats"):
        P.append("<div class='card'><b>Caveats</b><ul>" + "".join(f"<li>{html.escape(c)}</li>" for c in st["caveats"]) + "</ul></div>")
    # intuition, branches, tournament, adversary
    P.append("<h2>Intuition (pre-analysis, before any fitting)</h2><ul>" + "".join(
        f"<li><b>[{h['confidence']}]</b> {html.escape(h['hypothesis'])}</li>" for h in res["intuition"]["hypotheses"][:8]) + "</ul>")
    rows = ""
    for k, b in res["branches"].items():
        sv = b.get("self_validation") or {}
        link = f"<a href='{html.escape(os.path.relpath(b['report'], out_dir))}'>report</a>" if b.get("report") else ""
        rows += (f"<tr><td>{html.escape(k)}{' 🏆' if k == res['winner_branch'] else ''}</td><td><code>{html.escape(json.dumps(b.get('model'))[:260])}</code></td>"
                 f"<td>{_fmt(sv.get('rollout_valid_time', ''))}</td><td>{_fmt(b.get('cost_usd'))}</td><td>{link}</td></tr>")
    P.append("<h2>Parallel branches</h2><table><tr><th>strategy</th><th>model</th><th>valid time</th><th>$</th><th></th></tr>" + rows + "</table>")
    P.append(f"<div class='card'><b>Tournament.</b> {html.escape(res['tournament'].get('verdict', ''))}</div>")
    adv = res.get("adversary") or {}
    if adv:
        duel = adv.get("duel") or {}
        P.append(f"<div class='card'><b>Adversary (red team).</b> challenger: <code>{html.escape(json.dumps(adv.get('challenger'))[:300])}</code><br>"
                 f"{html.escape(adv.get('rationale', '')[:800])}<br><b>Result:</b> "
                 f"{html.escape(duel.get('verdict', 'the adversary submitted the incumbent unchanged'))}</div>")
    if res.get("benchmark"):
        b = res["benchmark"]
        P.append(f"<h2>Benchmark only (hidden ground truth)</h2><div class='card'>final hidden score {_fmt(b['final'].get('score'))}, "
                 f"term F1 {_fmt(b['final'].get('f1'))}; per branch: {html.escape(json.dumps({k: (round(x, 2) if x else x) for k, x in b['branches'].items()}))}"
                 + "".join(_latex(k, e, names) for k, e in (b["final"].get("truth") or {}).items()) + "</div>")
    p = out_dir / "fig_data.png"
    from .plots import plot_data
    plot_data(meta, data, p)
    P.append("<h2>Data</h2><div class='card'>" + _img(p) + "</div>")
    doc = (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"<title>Discovery report</title><style>{CSS}</style>"
           "<script src='https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js' async></script></head>"
           f"<body><main>{''.join(P)}</main></body></html>")
    f = out_dir / "report.html"
    f.write_text(doc)
    return str(f)
