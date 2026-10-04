"""Data in -> discovered equation + the story of how it was found + confidence + what to do next.

    python -m eqdisc.discover PATH [--branches 3] [--no-adversary] [--human] [--context "..."]

Pipeline
  0. EVIDENCE CHECKS on the data (eqdisc.audit): outliers, gaps, ...; data repairs (split at gaps, despike) are
     applied before anything is fitted and the repaired dataset is saved as <out>/dataset_audited/,
  1. ingest (any file -> dataset + data card), 2. intuition pre-analysis (hypotheses, recommended config),
  3. PARALLEL agent branches with different strategies seeded by the hypotheses,
  4. TOURNAMENT: branch models compared on public data (cross-validated error, BIC, rollout, parsimony),
  5. ADVERSARY: a red-team agent tries to break the winner (find structured residuals, regions of failure, a
     better or simpler rival); the challenger must win the same tournament to replace the incumbent,
     The tournament winner and the final model are checked again (slice consistency, residual structure);
     fired findings move the grade and can veto a CONFIDENT verdict. Every finding, repair and tournament outcome
     is appended to <out>/ledger.jsonl.
  6. ASSESSMENT of the final model -> verdict (CONFIDENT / COLLECT MORE DATA / ...) + next experiments,
  7. one HTML report: verdict, key steps, intuition, branches, tournament, adversary, model, UQ, experiments.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import uq
from . import audit as _audit
from .audit import fired as fired_findings, summary as findings_summary
from .audit.repair import audit_and_repair, has_nan, repair_model, save_dataset
from .ledger import Ledger
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
             max_tools=20, out_dir=None, workers=3, verbose=True, on_event=None):
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
    _print = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)

    def say(*a):
        _print(*a)
        if on_event:
            on_event({"type": "stage", "text": " ".join(map(str, a)).strip()})

    original_path = path
    ledger = Ledger(out, path, config={"n_branches": n_branches, "adversary": adversary, "model": model, "effort": effort,
                                       "max_tools": max_tools, "context": context})
    say("[1/7] evidence checks on the data")
    meta, data, data_findings, data_repairs = audit_and_repair(meta, data, ledger=ledger)
    if has_nan(data) and _audit.enabled():
        msg = ("INCONCLUSIVE before fitting: values are missing in every part of the record, so no gap-free window is "
               "long enough to estimate derivatives or weak-form integrals. Collect complete snapshots (or longer "
               "gap-free stretches); eqdisc does not impute missing data.")
        ledger.append("verdict", status="INCONCLUSIVE", headline=msg)
        say("      " + msg)
        raise ValueError(msg)
    if any(a.get("ok") for a in data_repairs):
        path = save_dataset(meta, data, out / "dataset_audited", data_repairs, source=original_path)
        ledger.set_dataset(path)
        ledger.append("dataset", path=str(path), note="repaired dataset used for every later stage")
    for f in fired_findings(data_findings, "info"):
        say(f"      [{f['severity']}] {f['id']}: {(f.get('message') or '')[:130]}")
    for a in data_repairs:
        say(f"      repair {a['tool']} for {a['finding']}: {a['note'][:120]}")
    if not fired_findings(data_findings, "info"):
        say("      all data checks passed")
    evidence = {"data_findings": data_findings, "data_repairs": data_repairs,
                "dataset_audited_path": str(path) if path != original_path else None}

    say(f"[2/7] intuition pre-analysis on {meta['name']}")
    intu = intuit(meta, data)
    for h in intu["hypotheses"][:6]:
        say(f"      [{h['confidence']}] {h['hypothesis'][:130]}")
    base_ctx = (context + "\n\n" if context else "") + _intuition_summary(intu) + "\n\n" + findings_summary(data_findings)

    strategies = _pick_strategies(intu, n_branches)
    say(f"[3/7] {len(strategies)} parallel branches: {strategies}")

    def branch(name):
        try:
            r = run_agent(path, model=model, effort=effort, max_tools=max_tools, client=client, verbose=False,
                          out_dir=out / f"branch_{name}", context=base_ctx + "\n\n" + STRATEGIES[name],
                          final_assessment=False, report=True, human=None, on_event=on_event, tag=name)
            say(f"      branch {name}: {json.dumps((r.get('submitted') or {}).get('rhs'))[:150]}  (${r['cost_usd']:.2f})")
            return name, r
        except Exception as e:  # noqa: BLE001
            say(f"      branch {name} failed: {e}")
            return name, {"error": str(e)}

    with ThreadPoolExecutor(min(workers, len(strategies))) as ex:
        branches = dict(ex.map(branch, strategies))
    cands = {k: (r.get("submitted") or {}).get("rhs") for k, r in branches.items() if isinstance(r, dict)}

    say("[4/7] tournament")
    tour = tournament(meta, data, cands)
    incumbent = tour["winner"]
    say(f"      winner: {incumbent}. {tour['verdict'][:200]}")
    ledger.append("tournament", stage="branches", winner=incumbent, verdict=tour.get("verdict"), ranking=tour.get("ranking"),
                  candidates=cands)
    tour_findings = []
    if incumbent and cands.get(incumbent):
        tour_findings = _audit.audit_model(meta, data, cands[incumbent])
        ledger.findings(tour_findings, "model audit (tournament winner)")
        evidence["tournament_findings"] = tour_findings
        for f in fired_findings(tour_findings, "info"):
            say(f"      check [{f['severity']}] {f['id']}: {(f.get('message') or '')[:130]}")

    adv = None
    if adversary and incumbent:
        say("[5/7] adversary (red team) attacks the winner")
        a0 = assess(meta, data, cands[incumbent], {k: v for k, v in cands.items() if k != incumbent and v},
                    findings=data_findings + tour_findings)
        prompt = ADVERSARY.format(incumbent=json.dumps(cands[incumbent]), assessment=json.dumps(a0["confidence"]),
                                  others=json.dumps({k: v for k, v in cands.items() if k != incumbent}))
        if fired_findings(tour_findings, "info"):
            prompt += "\n\n" + findings_summary(tour_findings, "Automatic checks on the incumbent (deterministic)")
        try:
            ra = run_agent(path, model=model, effort=effort, max_tools=max_tools, client=client, verbose=False,
                           out_dir=out / "adversary", context=base_ctx + "\n\n" + prompt, final_assessment=False, report=True,
                           on_event=on_event, tag="adversary")
            chal = (ra.get("submitted") or {}).get("rhs")
            adv = {"run": ra, "challenger": chal, "rationale": (ra.get("submitted") or {}).get("rationale", "")}
            if chal and chal != cands[incumbent]:
                duel = tournament(meta, data, {"incumbent": cands[incumbent], "challenger": chal})
                adv["duel"] = duel
                ledger.append("tournament", stage="adversary duel", winner=duel.get("winner"), verdict=duel.get("verdict"))
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
    say("[6/7] assessment of the final model (with evidence checks)")
    assessment = None
    if final:
        # deterministic checks: reuse the tournament winner's findings when the adversary did not replace it
        final_findings = tour_findings if (tour_findings and final == cands.get(tour["winner"])) else \
            _audit.audit_model(meta, data, final)
        ledger.findings(final_findings, "model audit (final model)")
        evidence["final_findings"] = final_findings
        evidence["model_repairs"] = repair_model(meta, data, final, final_findings, ledger=ledger)
        for f in fired_findings(final_findings, "info"):
            say(f"      check [{f['severity']}] {f['id']}: {(f.get('message') or '')[:130]}")
        assessment = assess(meta, data, final, {k: v for k, v in cands.items() if k != incumbent and v},
                            findings=data_findings + final_findings)
    if human is not None and assessment:
        ans = human("FINAL MODEL: " + json.dumps(final) + "\n\n" + brief_markdown(assessment)
                    + "\n\nReply 'accept' or give feedback / domain knowledge (the system will record it).")
        human_log.append({"question": "final review", "answer": ans})

    say("[7/7] write-up")
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
    res = {"dataset": meta["name"], "dataset_path": str(original_path), "final_model": final, "winner_branch": incumbent,
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
           "data_card": data_card, "evidence": evidence, "wall_s": round(time.time() - t0, 1)}
    res["cost_usd"] = round(sum((b.get("cost_usd") or 0) for b in res["branches"].values())
                            + (res["adversary"].get("cost_usd") or 0) + usage.cost(), 3)
    if (original_path / "hidden" / "truth.json").exists() and final:          # benchmark datasets only
        res["benchmark"] = {"final": evaluate(original_path, {"rhs": final}, reveal=True),
                            "branches": {k: (evaluate(original_path, {"rhs": m}, reveal=True)["score"] if m else None)
                                         for k, m in cands.items()}}
    ledger.append("verdict", status=v["status"], headline=v["headline"], valid_range=v.get("valid_range"),
                  level=(assessment or {}).get("confidence", {}).get("level"),
                  points=(assessment or {}).get("confidence", {}).get("points"))
    (out / "discovery.json").write_text(json.dumps(_jsonable(res), indent=2, default=str))
    from .report import build_discovery_report
    res["report"] = build_discovery_report(out, res, meta, data)
    say(f"\n{v['status']}: {v['headline']}\n{v['recommendation']}\nreport: {res['report']}  (cost ${res['cost_usd']:.2f}, {res['wall_s']} s)")
    return res


def reassess(run_dir):
    """Recompute evidence findings, assessment, grade, verdict and report of a finished discover run (no LLM calls).

    Old runs (before the evidence layer) are re-scored too: the data are audited and repaired exactly as `discover`
    would, the final model is audited, and the new grade and verdict are written back. Appends to ledger.jsonl."""
    run_dir = Path(run_dir)
    res = json.loads((run_dir / "discovery.json").read_text())
    src = Path(res["dataset_path"])
    ledger = Ledger(run_dir, src, config={"reassess": True})
    meta, data = load(src)
    meta, data, data_findings, data_repairs = audit_and_repair(meta, data, ledger=ledger)
    if any(a.get("ok") for a in data_repairs):
        p = save_dataset(meta, data, run_dir / "dataset_audited", data_repairs, source=src)
        ledger.set_dataset(p)
    final = res.get("final_model")
    final_findings = _audit.audit_model(meta, data, final) if final else []
    ledger.findings(final_findings, "model audit (reassess)")
    res["evidence"] = {"data_findings": data_findings, "data_repairs": data_repairs, "final_findings": final_findings,
                       "model_repairs": repair_model(meta, data, final, final_findings, ledger=ledger) if final else [],
                       "dataset_audited_path": str(run_dir / "dataset_audited") if any(a.get("ok") for a in data_repairs) else None,
                       "reassessed": time.strftime("%Y-%m-%dT%H:%M:%S")}
    alts = {k: b["model"] for k, b in res.get("branches", {}).items() if b.get("model") and b["model"] != final}
    res["assessment"] = assess(meta, data, final, alts, findings=data_findings + final_findings) if final else None
    res["brief"] = brief_markdown(res["assessment"]) if res["assessment"] else None
    res["verdict"] = verdict(res["assessment"])
    ledger.append("verdict", phase="reassess", status=res["verdict"]["status"], headline=res["verdict"]["headline"],
                  valid_range=res["verdict"].get("valid_range"))
    (run_dir / "discovery.json").write_text(json.dumps(_jsonable(res), indent=2, default=str))
    from .report import build_discovery_report
    res["report"] = build_discovery_report(run_dir, res, meta, data)
    return res
