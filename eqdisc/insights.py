"""Turn a discovery session into the human-facing story: key insights, a verdict and a recommendation.

    extract_insights(log, assessment)      deterministic, from the tool trail (always available)
    verdict(assessment)                    CONFIDENT / COLLECT MORE DATA / INCONCLUSIVE + recommendation
    narrate(client, model, ...)            optional LLM write-up of the research story (JSON)
"""
import json


def _num(x, d=3):
    try:
        return f"{float(x):.{d}g}"
    except Exception:  # noqa: BLE001
        return str(x)


def extract_insights(log, assessment=None, data_card=None):
    ins = []

    def add(step, finding, why=""):
        ins.append({"step": step, "finding": finding, "why_it_matters": why})

    if data_card:
        warn = data_card.get("warnings", [])
        add("Read the data", data_card.get("summary", "")[:300],
            ("Warnings: " + "; ".join(w[:120] for w in warn[:3])) if warn else "No data-quality warnings.")
    best_before_transform = None
    for ev in log:
        if ev.get("type") != "tool":
            continue
        name, inp, out = ev["name"], ev.get("input", {}), ev.get("output", {})
        if not isinstance(out, dict) or "error" in out:
            continue
        if name == "diagnose":
            nz = out.get("noise_rel_estimate", {})
            if nz:
                n = max(nz.values())
                add("Diagnosed the data", f"noise about {n:.1%} of signal; "
                    + ("coarse sampling" if max((out.get("mean_change_per_step_rel") or {0: 0}).values()) > 0.15 else "adequate sampling"),
                    "Noise above ~1% means derivative-based fits are biased, so weak-form methods were preferred." if n > 0.01 else
                    "Low noise: derivative-based regression is reliable.")
        elif name == "find_invariants":
            good = [i for i in out.get("invariants", []) if i.get("rel_variation", 1) < 0.03]
            if good:
                g = good[0]
                add("Searched for conserved quantities", f"{g['H'][:120]} is conserved ({g['kind'].split(':')[0]})",
                    "A constraint removes a dimension; a first integral labels the orbits and suggests natural coordinates.")
            else:
                add("Searched for conserved quantities", "nothing conserved in the tested library", "")
        elif name == "detect_symmetries":
            c = out.get("continuous", {})
            gens = [g.get("interpretation") or g.get("name") for g in c.get("generators", []) if g.get("is_symmetry", True)]
            disc = [d.get("name") or d.get("description") for d in out.get("discrete", []) if d.get("is_symmetry")] \
                if isinstance(out.get("discrete"), list) else []
            msg = ", ".join(str(x) for x in (gens + disc)[:4] if x) or "no clear symmetry"
            add("Looked for symmetries", msg, "Symmetries restrict which terms can appear and suggest coordinates.")
        elif name == "transform":
            add("Changed coordinates", f"to {inp.get('name')}: {json.dumps(inp.get('forward'))}",
                "Dynamics are often simpler in natural coordinates (polar for rotation, log for growth, "
                "reduced for conservation laws).")
        elif name in ("run_sindy", "weak_sindy", "fit_skeleton", "run_pysr", "equivariant_sindy", "ensemble_sindy"):
            v = out.get("validation_original") or out.get("validation") or {}
            vt, hz = v.get("rollout_valid_time"), v.get("rollout_horizon")
            if out.get("coordinates") and vt is not None:
                coord = out["coordinates"]
                add(f"Fitted in {coord} coordinates ({name})",
                    f"held-out rollout valid for {_num(vt)} of {_num(hz)} time units"
                    + (f" (vs {_num(best_before_transform)} in the original coordinates)" if best_before_transform is not None else ""),
                    "Better than in the raw variables, so the transformed coordinates are more natural for this system."
                    if best_before_transform is not None and vt > best_before_transform * 1.2 else "")
            elif vt is not None:
                best_before_transform = max(best_before_transform or 0, vt)
        elif name == "repair" and out.get("top_edits"):
            e = out["top_edits"][0]
            if e.get("delta_bic", 0) < -10:
                add("Repaired the model", f"{e['edit']} {e['term']} in d{e['var']}/dt (ΔBIC {_num(e['delta_bic'])})",
                    "A strongly supported single-term edit: the earlier model was near-correct.")
        elif name == "compare_models":
            add("Compared candidate models", out.get("verdict", "")[:300], "")
        elif name == "ask_human":
            add("Asked the scientist", f"{inp.get('question', '')[:150]} → {out.get('answer', '')[:150]}", "")
    if assessment and "confidence" in assessment:
        c = assessment["confidence"]
        add("Assessed the final model", f"{c['level'].upper()} confidence: " + "; ".join(c["reasons"][:3]), "")
    return ins


def verdict(assessment):
    """Verdict + recommendation. The evidence layer (assessment["findings"]) can only make it more cautious:
    an unresolved fired critical finding rules out CONFIDENT / CONFIDENT IN PREDICTIONS and is named in the headline;
    fired scope findings add `valid_range` and a sentence to the recommendation. No fired finding -> unchanged."""
    v = _verdict(assessment)
    if not assessment:
        return v
    from .audit import range_sentence, unresolved, valid_range
    findings = assessment.get("findings") or []
    crit = unresolved(findings, "critical")
    if crit:
        names = "; ".join(f"{f['id']}: {f.get('message') or 'failed'}" for f in crit[:2])
        if v["status"].startswith("CONFIDENT"):
            v = {"status": "COLLECT MORE DATA" if all(f.get("stage") == "data" or f.get("response") == "scope"
                                                       for f in crit) else "INCONCLUSIVE",
                 "headline": f"Not confident: a critical check failed ({names}). The model may still be the best "
                             f"available, but this evidence contradicts it.",
                 "recommendation": v["recommendation"] if v["status"] != "CONFIDENT" else
                 "Do not use the model as confirmed until the failed check is explained or new data remove it."}
        else:
            v = dict(v, headline=v["headline"].rstrip(".") + f". Critical check failed: {names}.")
        v["failed_checks"] = [f["id"] for f in crit]
    # Warnings: two independent model checks against the model, or one plus a low grade, also rule out confidence.
    # (residual_white is excluded: it restates the noise-floor ratio the grade already uses.)
    warns = [f for f in unresolved(findings, "warn") if f.get("severity") == "warn" and f.get("stage") == "model"
             and f.get("id") != "residual_white"]
    level = (assessment.get("confidence") or {}).get("level")
    if v["status"].startswith("CONFIDENT") and (len(warns) >= 2 or (warns and level == "low")):
        names = "; ".join(f"{f['id']}: {f.get('message') or 'warning'}" for f in warns[:2])
        v = {"status": "COLLECT MORE DATA",
             "headline": f"Best current model, not confirmed: independent checks disagree with it ({names}).",
             "recommendation": v["recommendation"], "failed_checks": [f["id"] for f in warns]}
    vr = valid_range(findings)
    if vr:
        v["valid_range"] = vr
        v["recommendation"] = (v["recommendation"] + " " + range_sentence(vr)).strip()
    return v


def _verdict(assessment):
    if not assessment or "confidence" not in assessment:
        return {"status": "INCONCLUSIVE", "headline": "No assessment available.", "recommendation": ""}
    c = assessment["confidence"]["level"]
    ex = (assessment.get("experiments") or {}).get("ranked", [])
    p = assessment.get("predictability") or {}
    strong_add = [m for m in assessment.get("missing_term_evidence", []) if (m.get("dBIC_if_added") or 0) < -10
                  and (m.get("error_reduction") is None or m["error_reduction"] >= 0.02)
                  and (m.get("rollout_improvement") is None or m["rollout_improvement"] > 0.10)]
    weak_terms = [t for t in assessment.get("terms", []) if not t.get("significant")]

    def where(e):
        w = e.get("initial_condition") or e.get("description")
        return ("start at (" + ", ".join(f"{x:.3g}" for x in w) + ")") if isinstance(w, list) else str(w)

    if c == "high" and not strong_add and not weak_terms:
        rec = ("The model is supported term by term, its remaining error is consistent with the noise, and it predicts "
               "held-out data. Use it." + (f" Predictions are reliable for about {p['horizon']} time units from a new state."
                                           if p and not p.get("horizon_is_full_data_span") else ""))
        if ex:
            rec += f" To validate it independently, the most demanding test would be: {where(ex[0])}."
        return {"status": "CONFIDENT", "headline": "We are confident this is your equation.", "recommendation": rec}
    informative = ex and (ex[0].get("score") or 0) > 1 and (ex[0].get("gain_vs_existing_data") is None
                                                            or ex[0]["gain_vs_existing_data"] > 1.2)
    has_issues = bool(strong_add or weak_terms or len((assessment.get("model_ambiguity") or {}).get("indistinguishable", [])) > 1)
    if c in ("high", "medium") and not has_issues and not informative:
        return {"status": "CONFIDENT", "headline": "We are confident this is your equation: no term is in doubt and no "
                "feasible experiment is more informative than the data you already have.",
                "recommendation": "Use it." + (f" Predictions agree across plausible models for about {p['horizon']} time units."
                                              if p and not p.get("horizon_is_full_data_span") else "")}
    if informative and (has_issues or c != "high"):
        top = ex[0]
        pins = ", ".join(x["coefficient"] for x in top.get("informs_coefficients", [])[:2])
        rec = (f"Collect data at: {where(top)}" + (f". It would mainly pin down {pins}" if pins else "")
               + f" (about {top.get('gain_vs_existing_data') or 'several'}x more informative than repeating existing conditions).")
        if len(ex) > 1:
            rec += f" Next best: {where(ex[1])}."
        issues = []
        if strong_add:
            issues.append("the data favour terms the model lacks: " + ", ".join(f"{m['term']} in d{m['var']}/dt" for m in strong_add[:2]))
        if weak_terms:
            issues.append("some terms are not significant: " + ", ".join(f"{t['term']} in d{t['var']}/dt" for t in weak_terms[:2]))
        amb = (assessment.get("model_ambiguity") or {}).get("indistinguishable", [])
        if len(amb) > 1:
            issues.append(f"{len(amb) - 1} alternative structure(s) fit equally well")
        return {"status": "COLLECT MORE DATA",
                "headline": "Best current model, not yet confirmed: " + ("; ".join(issues) if issues else "coefficients are not yet pinned down") + ".",
                "recommendation": rec}
    rollout_ok = not (assessment.get("validation") or {}).get("rollout_blew_up") and p.get("horizon_is_full_data_span")
    if rollout_ok and not strong_add:
        amb = (assessment.get("model_ambiguity") or {}).get("indistinguishable", [])
        return {"status": "CONFIDENT IN PREDICTIONS",
                "headline": "All plausible models predict the same behaviour over the whole data range"
                            + (", but the exact terms are not unique" if len(amb) > 1 else "") + ".",
                "recommendation": ("No new experiment would separate the candidate structures, because they agree everywhere "
                                   "we can test. They differ only by an identity in the data, typically a conservation "
                                   "law. Choose between them with domain knowledge: " + "; ".join(assessment.get("questions_for_human", [])[:2]))
                                  if len(amb) > 1 else "Use the model for prediction within the observed range."}
    return {"status": "INCONCLUSIVE", "headline": "The data do not determine the model well, and no simulated experiment "
            "clearly separates the candidates.", "recommendation": " ".join(assessment.get("data_advice", [])[:2])
            or "Consider more trajectories, lower noise, or domain constraints."}


NARRATE_PROMPT = """You are writing the summary section of a scientific equation-discovery report for a domain scientist.
Below are the data card, the agent's tool trail (with results), its final model, and a statistical assessment.

Write JSON only:
{{"headline": "<one sentence: what the system is and the discovered equation in words>",
  "key_steps": [{{"observation": "...", "decision": "...", "outcome": "..."}}],   // 3-6 decisive steps, e.g. "the dynamics are simpler in polar coordinates"
  "physical_interpretation": "<2-3 sentences: what each term means physically, if a reasonable guess exists>",
  "caveats": ["..."],
  "recommendation": "<one or two sentences: use the model, or exactly what data to collect next and why>"}}

DATA CARD: {card}
TRAIL: {trail}
FINAL MODEL: {model}
ASSESSMENT: {assessment}"""


def narrate(client, model, log, final_rhs, assessment, data_card=None):
    trail = []
    for ev in log:
        if ev.get("type") == "text":
            trail.append({"agent_note": ev["text"][:400]})
        elif ev.get("type") == "tool":
            out = ev.get("output", {})
            v = (out.get("validation_original") or out.get("validation") or {}) if isinstance(out, dict) else {}
            trail.append({"tool": ev["name"], "input": json.dumps(ev.get("input"))[:200],
                          "rhs": (out.get("rhs_original") or out.get("rhs")) if isinstance(out, dict) else None,
                          "valid_time": v.get("rollout_valid_time"), "note": str(out)[:200] if not v else None})
    a = {k: assessment.get(k) for k in ("confidence", "terms", "missing_term_evidence", "model_ambiguity", "predictability",
                                        "data_advice", "questions_for_human")} if assessment else {}
    a["experiments"] = (assessment or {}).get("experiments", {}).get("ranked", [])[:3]
    prompt = NARRATE_PROMPT.format(card=json.dumps(data_card or {})[:2500], trail=json.dumps(trail, default=str)[:12000],
                                   model=json.dumps(final_rhs), assessment=json.dumps(a, default=str)[:6000])
    resp = client.beta.messages.create(model=model, max_tokens=6000, output_config={"effort": "medium"},
                                       betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                                       messages=[{"role": "user", "content": prompt}])
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    try:
        return json.loads(text[text.index("{"): text.rindex("}") + 1]), resp
    except Exception:  # noqa: BLE001
        return {"headline": text[:300]}, resp
