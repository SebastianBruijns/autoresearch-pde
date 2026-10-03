"""Data in -> discovered equation + the story of how it was found + confidence + what to do next.

    python -m eqdisc.discover PATH [--branches 3] [--no-adversary] [--human] [--context "..."]

Pipeline
  1. ingest (any file -> dataset + data card), 2. intuition pre-analysis (hypotheses, recommended config),
  3. PARALLEL agent branches with different strategies seeded by the hypotheses,
  4. TOURNAMENT: branch models compared on public data (cross-validated error, BIC, rollout, parsimony),
  5. ADVERSARY: a red-team agent tries to break the winner (find structured residuals, regions of failure, a
     better or simpler rival); the challenger must win the same tournament to replace the incumbent,
  6. ASSESSMENT of the final model -> verdict (CONFIDENT / COLLECT MORE DATA / ...) + next experiments,
  7. one HTML report: verdict, key steps, intuition, branches, tournament, adversary, model, UQ, experiments.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import uq
from .agent import Usage, _jsonable, make_client, run_agent
from .assess import assess, brief_markdown
from .evaluate import evaluate, load
from .insights import extract_insights, narrate, verdict
from .intuition import intuit

STRATEGIES = {
    "structure-first": ("Strategy for this branch: STRUCTURE FIRST. Before any wide regression, establish structure: "
                        "conserved quantities (find_invariants), symmetries (detect_symmetries) and natural coordinates "
                        "(transform). Fit small libraries in the best coordinates. Judge every transform by validation_original."),
    "sparse-regression": ("Strategy for this branch: SPARSE REGRESSION. Use weak_sindy / run_sindy with the recommended "
                          "configuration, then ensemble_sindy for robustness and repair for near-misses. Keep the library "
                          "as small as the hypotheses allow."),
    "symbolic": ("Strategy for this branch: SYMBOLIC STRUCTURE. Use the non-polynomial hypotheses: propose parametrised "
                 "skeletons (fit_skeleton) and PySR with templates and model_selection='rollout'. Prefer physically "
                 "interpretable closed forms."),
    "minimalist": ("Strategy for this branch: MINIMALIST. Find the simplest model (fewest terms) whose error is at the "
                   "noise floor (compare_models). Remove every term the data do not demand."),
}

ADVERSARY = """You are the ADVERSARY (red team). Another team proposes this model for the data:
{incumbent}
Its assessment: {assessment}
Other candidates found: {others}

Your job is to BREAK it. Look for structured residuals (plot_model, run_python), trajectories or regions where it fails, terms
that the data demand but it lacks (repair, weak_sindy with a wider library), spurious terms, or a different structure (other
coordinates, non-polynomial forms) that is better or equally good and simpler. Be rigorous: a challenger must beat the
incumbent on held-out validation and compare_models, not only in-sample. Submit your best challenger. If after a genuine
attack you cannot beat it, submit the incumbent unchanged and say in the rationale what you tried."""


def _intuition_summary(intu):
    lines = [f"- [{h['confidence']}] {h['hypothesis']}" + (f"  (suggested: {h['try']['tool']} {json.dumps(h['try'].get('args', {}))})"
                                                             if h.get("try") else "")
             for h in intu["hypotheses"][:8]]
    return ("Pre-analysis hypotheses (data-driven 'intuition'; verify, do not trust blindly):\n" + "\n".join(lines)
            + f"\nRecommended starting configuration: {json.dumps(intu.get('recommended_config', {}))}")


def _pick_strategies(intu, n):
    order = ["sparse-regression", "structure-first", "symbolic", "minimalist"]
    hyp = " ".join(h["hypothesis"] for h in intu["hypotheses"][:6]).lower()
    if any(w in hyp for w in ("conserved", "polar", "log(", "rotate", "symmetric")):
        order.remove("structure-first")
        order.insert(0, "structure-first")
    if any(w in hyp for w in ("non-polynomial", "saturating", "sin(", "nonlinear")):
        order.remove("symbolic")
        order.insert(1, "symbolic")
    return order[:n]


def tournament(meta, data, candidates):
    cands = {k: v for k, v in candidates.items() if v}
    if len(cands) < 2:
        k = next(iter(cands), None)
        return {"winner": k, "verdict": "single candidate", "ranking": [k] if k else []}
    cmp = uq.compare_models(meta, data, cands)
    return {"winner": cmp["preferred"], "verdict": cmp["verdict"], "ranking": cmp["ranking"],
            "details": {k: {kk: cmp["candidates"][k].get(kk) for kk in ("n_terms", "cv_deriv_nrmse", "rollout_valid_frac_mean",
                                                                       "dbic", "z_vs_best")} for k in cmp["candidates"]}}


def discover(path, n_branches=3, adversary=True, human=None, context=None, model="claude-opus-5-5", effort="high",
             max_tools=20, out_dir=None, workers=3, verbose=True):
    t0 = time.time()
    client = make_client()
    path = Path(path)
    data_card = None
    if path.is_file():
        from .ingest import ingest
        path, data_card = ingest(path)
        path = Path(path)
    elif (path / "data_card.json").exists():
        data_card = json.loads((path / "data_card.json").read_text())
    out = Path(out_dir or f"runs/discover_{path.name}_{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    meta, data = load(path)
    say = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)

    say(f"[1/6] intuition pre-analysis on {meta['name']}")
    intu = intuit(meta, data)
    for h in intu["hypotheses"][:6]:
        say(f"      [{h['confidence']}] {h['hypothesis'][:130]}")
    base_ctx = (context + "\n\n" if context else "") + _intuition_summary(intu)

    strategies = _pick_strategies(intu, n_branches)
    say(f"[2/6] {len(strategies)} parallel branches: {strategies}")

    def branch(name):
        try:
            r = run_agent(path, model=model, effort=effort, max_tools=max_tools, client=client, verbose=False,
                          out_dir=out / f"branch_{name}", context=base_ctx + "\n\n" + STRATEGIES[name],
                          final_assessment=False, report=True, human=None)
            say(f"      branch {name}: {json.dumps((r.get('submitted') or {}).get('rhs'))[:150]}  (${r['cost_usd']:.2f})")
            return name, r
        except Exception as e:  # noqa: BLE001
            say(f"      branch {name} failed: {e}")
            return name, {"error": str(e)}

    with ThreadPoolExecutor(min(workers, len(strategies))) as ex:
        branches = dict(ex.map(branch, strategies))
    cands = {k: (r.get("submitted") or {}).get("rhs") for k, r in branches.items() if isinstance(r, dict)}

    say("[3/6] tournament")
    tour = tournament(meta, data, cands)
    incumbent = tour["winner"]
    say(f"      winner: {incumbent}. {tour['verdict'][:200]}")

    adv = None
    if adversary and incumbent:
        say("[4/6] adversary (red team) attacks the winner")
        a0 = assess(meta, data, cands[incumbent], {k: v for k, v in cands.items() if k != incumbent and v})
        prompt = ADVERSARY.format(incumbent=json.dumps(cands[incumbent]), assessment=json.dumps(a0["confidence"]),
                                  others=json.dumps({k: v for k, v in cands.items() if k != incumbent}))
        try:
            ra = run_agent(path, model=model, effort=effort, max_tools=max_tools, client=client, verbose=False,
                           out_dir=out / "adversary", context=base_ctx + "\n\n" + prompt, final_assessment=False, report=True)
            chal = (ra.get("submitted") or {}).get("rhs")
            adv = {"run": ra, "challenger": chal, "rationale": (ra.get("submitted") or {}).get("rationale", "")}
            if chal and chal != cands[incumbent]:
                duel = tournament(meta, data, {"incumbent": cands[incumbent], "challenger": chal})
                adv["duel"] = duel
                if duel["winner"] == "challenger":
                    cands["adversary"] = chal
                    incumbent = "adversary"
                    say(f"      challenger WINS: {json.dumps(chal)[:150]}")
                else:
                    say("      incumbent survives the attack")
            else:
                say("      adversary could not beat the incumbent")
        except Exception as e:  # noqa: BLE001
            adv = {"error": str(e)}
    final = cands[incumbent] if incumbent else None

    # optional human checkpoint on the final model
    human_log = []
    say("[5/6] assessment of the final model")
    assessment = assess(meta, data, final, {k: v for k, v in cands.items() if k != incumbent and v}) if final else None
    if human is not None and assessment:
        ans = human("FINAL MODEL: " + json.dumps(final) + "\n\n" + brief_markdown(assessment)
                    + "\n\nReply 'accept' or give feedback / domain knowledge (the system will record it).")
        human_log.append({"question": "final review", "answer": ans})

    say("[6/6] write-up")
    usage = Usage(model)
    log_all = []
    for name, r in list(branches.items()) + ([("adversary", adv["run"])] if adv and "run" in adv else []):
        try:
            log_all += json.loads((Path(r["out_dir"]) / "transcript.json").read_text())
        except Exception:  # noqa: BLE001
            pass
    insights = extract_insights(log_all, assessment, data_card)
    try:
        story, resp = narrate(client, model, log_all, final, assessment, data_card)
        usage.add(resp)
    except Exception as e:  # noqa: BLE001
        story = {"headline": f"(narration unavailable: {e})"}
    v = verdict(assessment)
    res = {"dataset": meta["name"], "dataset_path": str(path), "final_model": final, "winner_branch": incumbent,
           "verdict": v, "story": story, "insights": insights, "intuition": intu, "tournament": tour,
           "adversary": {k: v_ for k, v_ in (adv or {}).items() if k != "run"} | (
               {"report": adv["run"].get("report"), "cost_usd": adv["run"].get("cost_usd")} if adv and "run" in adv else {}),
           "branches": {k: {"model": (r.get("submitted") or {}).get("rhs") if isinstance(r, dict) else None,
                            "rationale": (r.get("submitted") or {}).get("rationale", "") if isinstance(r, dict) else "",
                            "self_validation": r.get("self_validation") if isinstance(r, dict) else None,
                            "cost_usd": r.get("cost_usd") if isinstance(r, dict) else None,
                            "report": r.get("report") if isinstance(r, dict) else None,
                            "error": r.get("error") if isinstance(r, dict) else None} for k, r in branches.items()},
           "assessment": assessment, "brief": brief_markdown(assessment) if assessment else None, "human_log": human_log,
           "data_card": data_card, "wall_s": round(time.time() - t0, 1)}
    res["cost_usd"] = round(sum((b.get("cost_usd") or 0) for b in res["branches"].values())
                            + (res["adversary"].get("cost_usd") or 0) + usage.cost(), 3)
    if (path / "hidden" / "truth.json").exists() and final:          # benchmark datasets only
        res["benchmark"] = {"final": evaluate(path, {"rhs": final}, reveal=True),
                            "branches": {k: (evaluate(path, {"rhs": m}, reveal=True)["score"] if m else None)
                                         for k, m in cands.items()}}
    (out / "discovery.json").write_text(json.dumps(_jsonable(res), indent=2, default=str))
    from .report import build_discovery_report
    res["report"] = build_discovery_report(out, res, meta, data)
    say(f"\n{v['status']}: {v['headline']}\n{v['recommendation']}\nreport: {res['report']}  (cost ${res['cost_usd']:.2f}, {res['wall_s']} s)")
    return res


def reassess(run_dir):
    """Recompute assessment, verdict and report of a finished discover run (no LLM calls)."""
    run_dir = Path(run_dir)
    res = json.loads((run_dir / "discovery.json").read_text())
    meta, data = load(res["dataset_path"])
    alts = {k: b["model"] for k, b in res["branches"].items() if b.get("model") and b["model"] != res["final_model"]}
    res["assessment"] = assess(meta, data, res["final_model"], alts)
    res["brief"] = brief_markdown(res["assessment"])
    res["verdict"] = verdict(res["assessment"])
    (run_dir / "discovery.json").write_text(json.dumps(_jsonable(res), indent=2, default=str))
    from .report import build_discovery_report
    res["report"] = build_discovery_report(run_dir, res, meta, data)
    return res
