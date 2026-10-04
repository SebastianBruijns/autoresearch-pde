"""Evidence-layer benchmark: score eqdisc arms on the corruption benchmark (datasets/corrupt/index.json).

Arms
  A  plain eqdisc: discover(n_branches=2, adversary=False) with EQDISC_EVIDENCE=0          (LLM cost)
  B  no LLM: evidence data audit + repair -> autobase.auto_fit -> model audit -> assess -> verdict  (evidence on)
  C  arm A's run dirs copied to C/ and re-scored with evidence on via orchestrate.reassess   (no LLM)
  D  full discover with evidence on, same settings as A                                    (LLM cost)

Every case runs in its own subprocess (clean EQDISC_EVIDENCE, contained crashes, hard timeout). The parent scores
each run and appends one JSON line to runs/evidence_bench/outcomes.jsonl (the latest line per (arm, case) wins).

Outcome of a run
  right      the submitted model matches the BASE equation in hidden/truth.json:
             - clean, outliers, gaps_random, gaps_state, traj_coeffs: `judge.judge` says symbolically equivalent
               (sympy first, coefficients within 5%; LLM fallback, cached in runs/evidence_bench/judge_cache.json);
             - forcing_time, source_space ("structural"): every base term is present with relative coefficient error
               <= COEF_TOL (10%), and every extra term is forcing-like, i.e. contains no field variable or field
               derivative (only t, x and constants: sin(w*t), cos(k*x), 1, ...);
             - amp_term ("structural"): every base term present within COEF_TOL; the only extra term allowed is the
               cube of a field variable (u**3), whose inclusion is optional.
  confident  verdict status starts with "CONFIDENT" (CONFIDENT or CONFIDENT IN PREDICTIONS).
  flagged    not confident AND (verdict names a reason: failed_checks or a headline, OR a fired finding of
             severity >= warn exists).
  crashed    exception, timeout, or no output; counted separately (never "confident-wrong"). A run that returns no
             model because the evidence layer refused to fit (INCONCLUSIVE with a reason) is "abstained": wrong+flagged.
Categories: right+confident, right+cautious, wrong+flagged, wrong+confident (headline failure), wrong+unflagged
(not confident, no reason given; should be empty), crashed.

    python -m eqdisc.bench_evidence --arm B --split dev --seeds 0 1 2 [--systems burgers] [--corruptions clean]
    python -m eqdisc.bench_evidence --arm A --split report --seeds 10 --workers 4
    python -m eqdisc.bench_evidence --arm C --split report --seeds 10      # needs arm A run dirs
    python -m eqdisc.bench_evidence --table [--md runs/evidence_bench/table.md]
"""
import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .audit import fired
from .corrupt import CORRUPTIONS

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "runs" / "evidence_bench"
INDEX = REPO / "datasets" / "corrupt" / "index.json"
ARMS = ("A", "B", "C", "D")
STRUCTURAL = {"forcing_time": "forcing", "source_space": "forcing", "amp_term": "amp"}
COEF_TOL = 0.10
TIMEOUT = {"A": 900, "D": 900, "B": 600, "C": 600}
CATEGORIES = ("right+confident", "right+cautious", "wrong+flagged", "wrong+confident", "incomplete+cautious",
              "wrong+unflagged", "crashed")
_LOCK = threading.Lock()


# ----------------------------------------------------------------------------- cases
def load_index(path=INDEX):
    return json.loads(Path(path).read_text())


def case_name(e):
    return f"{e['split']}_{e['system']}_{e['corruption']}_s{e['seed']}"


def select(index, split=None, seeds=None, systems=None, corruptions=None):
    return [e for e in index if (split is None or e["split"] == split) and (not seeds or e["seed"] in seeds)
            and (not systems or e["system"] in systems) and (not corruptions or e["corruption"] in corruptions)]


# ----------------------------------------------------------------------------- correctness
def _judge_cached(truth, cand, use_llm=True, client=None):
    from .judge import judge
    key = hashlib.sha256(json.dumps([truth, cand], sort_keys=True).encode()).hexdigest()[:24]
    p = ROOT / "judge_cache.json"
    with _LOCK:
        cache = json.loads(p.read_text()) if p.exists() else {}
    if key in cache:
        return cache[key]
    j = judge(truth, cand, use_llm=use_llm, client=client)
    if use_llm or all(not str(v["how"]).startswith("llm unavailable") for v in j["per_var"].values()):
        with _LOCK:
            cache = json.loads(p.read_text()) if p.exists() else {}
            cache[key] = j
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(cache, indent=1))
    return j


def structural_right(truth_rhs, cand_rhs, names, mode, tol=COEF_TOL, fields=None):
    """Base terms all present within `tol`; extra terms allowed only if forcing-like (mode 'forcing') or a cubed field
    variable (mode 'amp'). Returns (right, details)."""
    import sympy as sp

    from .evaluate import terms
    from .solvers import parse
    fields = set(fields or [])
    field_syms = {n for n in names if n not in ("t", "x", "y", "z")}
    missing, bad_coef, bad_extra, ok_extra = [], [], [], []
    for v, e in truth_rhs.items():
        T = terms(parse(e, names))
        C = terms(parse(cand_rhs.get(v, "0"), names))
        for m, c in T.items():
            if m not in C:
                missing.append(f"{v}:{m}")
            elif abs(C[m] - c) > tol * abs(c):
                bad_coef.append(f"{v}:{m} {C[m]:.4g} vs {c:.4g}")
        for m in C.keys() - T.keys():
            syms = {str(s) for s in m.free_symbols}
            if mode == "forcing":
                allowed = not (syms & field_syms)
            else:
                allowed = (isinstance(m, sp.Pow) and m.exp == 3 and str(m.base) in (fields or field_syms))
            (ok_extra if allowed else bad_extra).append(f"{v}:{m}")
    right = not (missing or bad_coef or bad_extra)
    return right, {"missing": missing, "bad_coef": bad_coef, "bad_extra": bad_extra, "allowed_extra": ok_extra}


FREQ_TOL = 0.10


def _trig_arg_coef(m, var):
    """k for sin(k*var) / cos(k*var) monomials, else None."""
    import sympy as sp
    if isinstance(m, (sp.sin, sp.cos)):
        a = m.args[0]
        if {str(x) for x in a.free_symbols} == {var}:
            k = sp.Poly(a, *a.free_symbols).coeffs()
            return abs(float(k[0])) if len(k) == 1 else None
    return None


def full_right(truth, full_rhs, cand_rhs, corruption, tol=COEF_TOL, freq_tol=FREQ_TOL):
    """Does the candidate match the FULL generating equation (base + the corruption's extra term)?
    Base terms within `tol`; extra terms must be of the allowed kind, AND the corruption's term must be present:
      forcing_time  a sin/cos(w*t) term with w within freq_tol of the true w (phase/sin-cos mix allowed);
      source_space  a sin/cos(k*x) term with k within freq_tol of the true k;
      amp_term      the cubed field term u**3 with the same sign as the true coefficient.
    Returns (right, details)."""
    import sympy as sp

    from .evaluate import _names, terms
    from .solvers import parse
    names = _names(truth)
    mode = STRUCTURAL[corruption]
    base_ok, det = structural_right(truth["rhs"], cand_rhs, names, mode, tol=tol, fields=truth.get("variables"))
    if not base_ok:
        return False, dict(det, extra_term="base terms not recovered")
    present = True
    for v, e in full_rhs.items():
        T, B = terms(parse(e, names)), terms(parse(truth["rhs"].get(v, "0"), names))
        C = terms(parse(cand_rhs.get(v, "0"), names))
        extra_true = {m: c for m, c in T.items() if m not in B}
        for m, c in extra_true.items():
            if corruption == "amp_term":
                ok = m in C and np.sign(C[m]) == np.sign(c)
            else:
                var = "t" if corruption == "forcing_time" else "x"
                k = _trig_arg_coef(m, var)
                ks = [_trig_arg_coef(mc, var) for mc in C if mc not in B]
                ok = k is not None and any(kc is not None and abs(kc - k) <= freq_tol * k for kc in ks)
            present &= bool(ok)
    return present, dict(det, extra_term="present" if present else "missing (base-only / incomplete law)")


def correctness(truth, cand_rhs, corruption, use_llm=True, client=None, full_rhs=None):
    """{"right": bool, "mode": ..., "how": ..., "right_base": bool} for a candidate rhs.

    Non-dynamic corruptions: right = judge-equivalent to the base (truth.json); right_base = right.
    Dynamic corruptions (forcing_time, source_space, amp_term) with `full_rhs` (corruption.json rhs_test):
    right = the FULL generating equation is recovered (`full_right`); right_base = the base terms are recovered
    (`structural_right`, the previous definition, kept as a secondary view: right_mode "base_only")."""
    from .evaluate import _names
    if not cand_rhs:
        return {"right": False, "right_base": False, "mode": "none", "how": "no model"}
    names = _names(truth)
    if corruption in STRUCTURAL:
        try:
            ok_base, det = structural_right(truth["rhs"], cand_rhs, names, STRUCTURAL[corruption],
                                            fields=truth.get("variables"))
            ok = ok_base
            if full_rhs:
                ok, det = full_right(truth, full_rhs, cand_rhs, corruption)
        except Exception as e:  # noqa: BLE001
            ok = ok_base = False
            det = {"error": f"{type(e).__name__}: {e}"}
        return {"right": ok, "right_base": ok_base, "mode": "full_equation" if full_rhs else "base_only", "how": det}
    try:
        j = _judge_cached(truth["rhs"], cand_rhs, use_llm=use_llm, client=client)
    except Exception as e:  # noqa: BLE001
        return {"right": False, "right_base": False, "mode": "judge", "how": f"judge error {type(e).__name__}: {e}"}
    ok = bool(j["equivalent"])
    return {"right": ok, "right_base": ok, "mode": "judge", "how": {v: p["how"] for v, p in j["per_var"].items()}}


def term_metrics(truth, cand_rhs):
    from .evaluate import _names, structure_metrics
    if not cand_rhs:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "exact_structure": False}
    try:
        m = structure_metrics(cand_rhs, truth["rhs"], _names(truth))
        return {k: m[k] for k in ("f1", "precision", "recall", "exact_structure", "coef_rel_err")}
    except Exception as e:  # noqa: BLE001
        return {"f1": None, "error": str(e)[:200]}


# ----------------------------------------------------------------------------- classification
def is_confident(verdict):
    return str((verdict or {}).get("status", "")).startswith("CONFIDENT")


def is_flagged(verdict, findings):
    if is_confident(verdict):
        return False
    v = verdict or {}
    names_reason = bool(v.get("failed_checks")) or bool(str(v.get("headline", "")).strip()) and \
        v.get("headline") != "No assessment available."
    return names_reason or bool(fired(findings or [], "warn"))


def classify(rec, base_view=False):
    """Headline category. Dynamic corruptions: `right` means the FULL equation was recovered; a base-only model
    (`right_base` but not `right`) is an incomplete law: confident -> wrong+confident, not confident with a reason ->
    wrong+flagged, not confident without a reason -> incomplete+cautious. base_view=True: the previous definition
    (right = base terms recovered)."""
    if rec.get("crashed"):
        return "crashed"
    right = rec.get("right_base", rec["right"]) if base_view else rec["right"]
    if right:
        return "right+confident" if rec["confident"] else "right+cautious"
    if rec["confident"]:
        return "wrong+confident"
    if rec["flagged"]:
        return "wrong+flagged"
    return "incomplete+cautious" if rec.get("right_base") and not base_view else "wrong+unflagged"


# ----------------------------------------------------------------------------- provenance
def provenance():
    files = sorted((REPO / "eqdisc" / "audit").glob("*.py")) + [REPO / "eqdisc" / "audit" / "thresholds.json"] + \
        [REPO / "eqdisc" / f for f in ("insights.py", "assess.py", "orchestrate.py", "uq.py")]
    h = hashlib.sha256()
    for f in files:
        if f.exists():
            h.update(f.name.encode() + f.read_bytes())
    try:
        stat = subprocess.run(["git", "diff", "--stat", "eqdisc/audit", "eqdisc/insights.py", "eqdisc/assess.py"], cwd=REPO, capture_output=True, text=True,
                              timeout=20).stdout.strip()
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        stat, head = "", ""
    return {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_head": head, "audit_diff_stat": stat,
            "audit_hash": h.hexdigest()[:12]}


# ----------------------------------------------------------------------------- workers (run in a subprocess)
def _w_discover(path, out_dir):
    from .orchestrate import discover
    discover(path, n_branches=2, adversary=False, out_dir=out_dir, verbose=True)


def _w_auto(path, out_dir):
    """Arm B: deterministic pipeline mirroring discover's evidence stages, with auto_fit in place of the agents."""
    from . import audit as _audit
    from .assess import assess
    from .audit.repair import audit_and_repair
    from .autobase import auto_fit
    from .evaluate import evaluate, load
    from .insights import verdict
    t0 = time.time()
    out = Path(out_dir)
    meta, data = load(path)
    res = {"dataset_path": str(path), "final_model": None}
    try:
        meta, data, data_f, data_rep = audit_and_repair(meta, data)
    except ValueError as e:         # no NaN-free window long enough to fit
        res["verdict"] = {"status": "INCONCLUSIVE", "headline": f"{e} (no imputation)", "recommendation": "",
                          "abstained": True}
        data_f, data_rep = [], []
    res["evidence"] = {"data_findings": data_f, "data_repairs": data_rep}
    if "verdict" not in res:
        from .orchestrate import _revise
        fit = auto_fit(meta, data)
        rhs = fit.get("rhs")
        res["auto_config"] = fit.get("auto_config")
        model_f = _audit.audit_model(meta, data, rhs) if rhs else []
        if rhs:
            rhs, model_f, res["revisions"] = _revise(meta, data, rhs, model_f, None, lambda *a: None)
        res["final_model"] = rhs
        res["evidence"]["final_findings"] = model_f
        a = assess(meta, data, rhs, data_findings=data_f) if rhs else None
        res["verdict"] = verdict(a)
        res["findings"] = (a or {}).get("findings", data_f + model_f)
        res["confidence"] = (a or {}).get("confidence")
        if rhs:
            try:
                ev = evaluate(path, {"rhs": rhs}, reveal=True)
                res["benchmark"] = {"final": {k: ev.get(k) for k in ("score", "vf_nrmse", "rollout_nrmse", "valid_frac", "f1")}}
            except Exception as e:  # noqa: BLE001
                res["benchmark"] = {"error": str(e)[:300]}
    res["cost_usd"] = 0.0
    res["wall_s"] = round(time.time() - t0, 1)
    from .agent import _jsonable
    (out / "result.json").write_text(json.dumps(_jsonable(res), indent=1, default=str))


def _w_reassess(a_dir, out_dir):
    from .orchestrate import reassess
    out = Path(out_dir)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(a_dir, out, ignore=shutil.ignore_patterns("log.txt"))
    t0 = time.time()
    reassess(out)
    (out / "reassess_wall_s.txt").write_text(str(round(time.time() - t0, 1)))


WORKERS = {"discover": _w_discover, "auto": _w_auto, "reassess": _w_reassess}


def _spawn(kind, args, out_dir, evidence, timeout):
    """Run a worker in a fresh process group; returns (returncode or 'timeout', wall seconds)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, EQDISC_EVIDENCE="1" if evidence else "0", PYTHONPATH=str(REPO), PYTHONUNBUFFERED="1",
               MPLBACKEND="Agg")
    cmd = [sys.executable, "-m", "eqdisc.bench_evidence", "--_worker", kind, *map(str, args)]
    t0 = time.time()
    with open(out_dir / ("log.txt" if kind != "reassess" else "../" + out_dir.name + ".reassess.log"), "w") as log:
        p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            rc = "timeout"
    return rc, round(time.time() - t0, 1)


def _log_tail(path, n=1500):
    try:
        return Path(path).read_text()[-n:]
    except Exception:  # noqa: BLE001
        return ""


def _branch_models(run_dir):
    out = {}
    for r in sorted(Path(run_dir).glob("branch_*/result.json")):
        try:
            out[r.parent.name[7:]] = (json.loads(r.read_text()).get("submitted") or {}).get("rhs")
        except Exception:  # noqa: BLE001
            pass
    return out


def _partial_cost(run_dir):
    tot = 0.0
    for r in Path(run_dir).glob("*/result.json"):
        try:
            tot += float(json.loads(r.read_text()).get("cost_usd") or 0)
        except Exception:  # noqa: BLE001
            pass
    return round(tot, 3)


# ----------------------------------------------------------------------------- one case
def run_case(arm, e, force=False, timeout=None, use_llm_judge=True, prov=None):
    name = case_name(e)
    out = ROOT / arm / name
    path = REPO / e["path"]
    timeout = timeout or TIMEOUT[arm]
    rc, wall, err, report_error = 0, None, None, None
    if arm in ("A", "D"):
        attempted = (out / "log.txt").exists() and not (out / "discovery.json").exists()
        if attempted and not force:      # a crashed / timed-out / killed run: re-score its log, never re-spend
            rc = "previous attempt"
        elif force or not (out / "discovery.json").exists():
            if out.exists():
                shutil.rmtree(out)
            rc, wall = _spawn("discover", [path, out], out, evidence=(arm == "D"), timeout=timeout)
        res_file = out / "discovery.json"
    elif arm == "B":
        rc, wall = _spawn("auto", [path, out], out, evidence=True, timeout=timeout)
        res_file = out / "result.json"
    else:  # C
        a_dir = ROOT / "A" / name
        if not (a_dir / "log.txt").exists():          # arm A never ran this case (budget): nothing to re-score
            return {"arm": arm, "case": name, "skipped": "no arm-A run"}
        if not (a_dir / "discovery.json").exists():
            return _record(arm, e, None, crashed=True, error="no arm-A run to reassess" +
                           (f" (A log: {_log_tail(a_dir / 'log.txt', 400)})" if a_dir.exists() else ""), prov=prov)
        if out.exists():
            shutil.rmtree(out)
        rc, wall = _spawn("reassess", [a_dir, out], out, evidence=True, timeout=timeout)
        res_file = out / "discovery.json"
        if rc != 0:
            # reassess writes discovery.json before rendering the HTML report: accept a run whose re-scored result
            # was written (report-rendering crashes are recorded, not counted as a crash of the method)
            try:
                ok = bool((json.loads(res_file.read_text()).get("evidence") or {}).get("reassessed"))
            except Exception:  # noqa: BLE001
                ok = False
            if ok:
                rc = 0
                report_error = _log_tail(ROOT / arm / f"{name}.reassess.log", 300)
            else:
                res_file = out / "__missing__"
    if not res_file.exists() or rc != 0:
        err = f"rc={rc}: " + _log_tail(out / "log.txt" if arm != "C" else ROOT / arm / f"{name}.reassess.log")
        rec = _record(arm, e, None, crashed=True, error=err, wall=wall,
                      cost=_partial_cost(out) if arm in ("A", "D") else 0.0, prov=prov)
        if arm in ("A", "D"):          # what the agents had found before the pipeline crashed
            rec["branch_models"] = _branch_models(out)
        return rec
    res = json.loads(res_file.read_text())
    rec = _record(arm, e, res, wall=wall, use_llm_judge=use_llm_judge, prov=prov)
    if arm == "C" and report_error:
        rec["report_error"] = report_error
    return rec


def _findings_of(res):
    a = res.get("assessment") or {}
    if a.get("findings"):
        return a["findings"]
    if res.get("findings"):
        return res["findings"]
    ev = res.get("evidence") or {}
    return (ev.get("data_findings") or []) + (ev.get("final_findings") or [])


def _record(arm, e, res, crashed=False, error=None, wall=None, cost=None, use_llm_judge=True, prov=None):
    truth = json.loads((REPO / e["path"] / "hidden" / "truth.json").read_text())
    rec = {"arm": arm, "case": case_name(e), "path": e["path"], "system": e["system"], "corruption": e["corruption"],
           "seed": e["seed"], "split": e["split"], "truth": truth["rhs"], "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "provenance": prov}
    if crashed or res is None:
        rec.update({"crashed": True, "error": (error or "")[-1500:], "right": False, "confident": False, "flagged": False,
                    "model": None, "cost_usd": cost or 0.0, "wall_s": wall, "f1": 0.0, "right_base": False})
        rec["category"] = rec["category_base"] = classify(rec)
        return rec
    model = res.get("final_model")
    v = res.get("verdict") or {}
    findings = _findings_of(res)
    corr = correctness(truth, model, e["corruption"], use_llm=use_llm_judge, full_rhs=_full_rhs(e))
    tm = term_metrics(truth, model)
    bench = (res.get("benchmark") or {}).get("final") or {}
    rec.update({"crashed": False, "model": model, "verdict_status": v.get("status"), "headline": v.get("headline"),
                "failed_checks": v.get("failed_checks"), "valid_range": v.get("valid_range"),
                "fired": sorted({f["id"] for f in fired(findings or [], "warn")}),
                "fired_critical": sorted({f["id"] for f in findings if f.get("fired") and f.get("severity") == "critical"}),
                "right": corr["right"], "right_base": corr["right_base"], "right_mode": corr["mode"],
                "right_how": corr["how"],
                "confident": is_confident(v), "flagged": is_flagged(v, findings),
                "abstained": model is None, "f1": tm.get("f1"), "exact_structure": tm.get("exact_structure"),
                "coef_rel_err": tm.get("coef_rel_err"), "score": bench.get("score"),
                "cost_usd": float(res.get("cost_usd") or 0.0),
                "wall_s": res.get("wall_s") if arm in ("A", "B", "D") else wall,
                "level": ((res.get("assessment") or {}).get("confidence") or res.get("confidence") or {}).get("level")})
    if arm == "C":                      # re-scoring is free; the LLM cost belongs to arm A
        rec["reassess_wall_s"] = wall
        rec["a_cost_usd"], rec["cost_usd"] = rec["cost_usd"], 0.0
    rec["category"] = classify(rec)
    rec["category_base"] = classify(rec, base_view=True)
    return rec


def _full_rhs(e):
    """Generating rhs of the hidden test set (corruption.json rhs_test) for dynamic corruptions, else None."""
    if e["corruption"] not in STRUCTURAL:
        return None
    try:
        return json.loads((REPO / e["path"] / "hidden" / "corruption.json").read_text())["rhs_test"]
    except Exception:  # noqa: BLE001
        return None


def rescore(rec, use_llm_judge=True):
    """Re-classify a stored outcome row from its stored model (no re-run). Used after definition changes."""
    r = dict(rec)
    if r.get("crashed"):
        r["right_base"] = False
        r["category"] = r["category_base"] = "crashed"
        return r
    truth = json.loads((REPO / r["path"] / "hidden" / "truth.json").read_text())
    corr = correctness(truth, r.get("model"), r["corruption"], use_llm=use_llm_judge, full_rhs=_full_rhs(r))
    r.update({"right": corr["right"], "right_base": corr["right_base"], "right_mode": corr["mode"],
              "right_how": corr["how"], "rescored_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    if r["arm"] == "C" and r.get("cost_usd"):
        r["a_cost_usd"], r["cost_usd"] = r["cost_usd"], 0.0
    r["category"] = classify(r)
    r["category_base"] = classify(r, base_view=True)
    return r


def append_outcome(rec, path=None):
    path = Path(path or ROOT / "outcomes.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, open(path, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def load_outcomes(path=None):
    """Latest record per (arm, case); malformed lines are skipped."""
    path = Path(path or ROOT / "outcomes.jsonl")
    out = {}
    for line in path.read_text().splitlines() if path.exists() else []:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        out[(r["arm"], r["case"])] = r
    return list(out.values())


def run_arm(arm, cases, workers=4, force=False, budget=None, use_llm_judge=True, outcomes=None):
    prov = provenance()
    (ROOT / arm).mkdir(parents=True, exist_ok=True)
    (ROOT / f"provenance_{arm}_{prov['timestamp'].replace(':', '')}.json").write_text(json.dumps(prov, indent=1))
    spent = [0.0]
    results = []

    def job(e):
        if budget is not None and arm in ("A", "D") and spent[0] >= budget:
            return {"arm": arm, "case": case_name(e), "skipped": "budget"}
        try:
            r = run_case(arm, e, force=force, use_llm_judge=use_llm_judge, prov=prov)
            if r.get("skipped"):
                return r
        except Exception:  # noqa: BLE001
            r = _record(arm, e, None, crashed=True, error="harness: " + traceback.format_exc()[-1200:], prov=prov)
        spent[0] += r.get("cost_usd") or 0.0
        return r

    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(job, e) for e in cases]
        for fu in as_completed(futs):
            r = fu.result()
            if r.get("skipped"):
                print(f"[{arm}] skipped {r['case']} ({r['skipped']})", flush=True)
                continue
            append_outcome(r, outcomes)
            results.append(r)
            print(f"[{arm}] {r['case']}: {r['category']} status={r.get('verdict_status')} right={r['right']} "
                  f"f1={r.get('f1')} ${r.get('cost_usd', 0):.2f} {r.get('wall_s')}s spent=${spent[0]:.2f}"
                  + (f" ERR {r.get('error', '')[-200:]!r}" if r.get("crashed") else ""), flush=True)
    return results


# ----------------------------------------------------------------------------- table
def _rate(n, d):
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "-"


GAPS = ("gaps_random", "gaps_state")
SHORT = {"right+confident": "RC", "right+cautious": "Rc", "wrong+flagged": "WF", "wrong+confident": "WC",
         "incomplete+cautious": "IC", "wrong+unflagged": "WU", "crashed": "X"}


def _counts(rs, key="category"):
    return {c: sum(r.get(key, r["category"]) == c for r in rs) for c in CATEGORIES}


def table(records):
    """Markdown: headline per arm x split x subset (all / non-gap / gaps), per arm x corruption, the base-only view,
    and term F1 / cost."""
    import statistics as st
    a_cases = {r["case"] for r in records if r["arm"] == "A"}
    records = [r for r in records if not (r["arm"] == "C" and r["case"] not in a_cases)]   # C needs an A run
    groups = {}
    for r in records:
        groups.setdefault((r["arm"], r["split"]), []).append(r)
    lines = ["### Headline: outcomes per arm, split and subset", "",
             "| arm | split | subset | n | " + " | ".join(CATEGORIES) + " | confident-wrong rate | confident-wrong (excl. crashed) |",
             "|---|---|---|---|---|" + "---|" * len(CATEGORIES) + "---|"]
    for (arm, split), rs0 in sorted(groups.items()):
        for sub, rs in (("all", rs0), ("non-gap", [r for r in rs0 if r["corruption"] not in GAPS]),
                        ("gaps", [r for r in rs0 if r["corruption"] in GAPS])):
            if not rs:
                continue
            cnt = _counts(rs)
            nc = len(rs) - cnt["crashed"]
            lines.append(f"| {arm} | {split} | {sub} | {len(rs)} | " + " | ".join(str(cnt[c]) for c in CATEGORIES)
                         + f" | {_rate(cnt['wrong+confident'], len(rs))} | {_rate(cnt['wrong+confident'], nc)} |")
    arms = sorted({r["arm"] for r in records})
    corrs = [c for c in CORRUPTIONS if any(r["corruption"] == c for r in records)]
    legend = ", ".join(f"{v} = {k}" for k, v in SHORT.items())
    for title, key in (("Per corruption (headline definition; splits pooled)", "category"),
                       ("Per corruption, secondary base-only view (dynamic corruptions: right = base terms recovered)",
                        "category_base")):
        lines += ["", f"### {title}", "", legend, "",
                  "| corruption | " + " | ".join(f"arm {a}" for a in arms) + " |", "|---|" + "---|" * len(arms)]
        for c in corrs:
            cells = []
            for a in arms:
                rs = [r for r in records if r["arm"] == a and r["corruption"] == c]
                cnt = _counts(rs, key)
                cells.append((" ".join(f"{SHORT[k]}{cnt[k]}" for k in CATEGORIES if cnt[k]) + f" (n={len(rs)})")
                             if rs else "")
            lines.append(f"| {c} | " + " | ".join(cells) + " |")
    lines += ["", "### Secondary: right rates, term F1, cost, wall time", "",
              "| arm | split | n | right (headline) | right (base-only) | confident | mean term F1 | exact structure | "
              "total cost $ | median wall s |", "|---|---|---|---|---|---|---|---|---|---|"]
    for (arm, split), rs in sorted(groups.items()):
        f1 = [r["f1"] for r in rs if r.get("f1") is not None]
        walls = [r["wall_s"] for r in rs if isinstance(r.get("wall_s"), (int, float))]
        lines.append(f"| {arm} | {split} | {len(rs)} | {sum(bool(r['right']) for r in rs)} | "
                     f"{sum(bool(r.get('right_base')) for r in rs)} | {sum(bool(r['confident']) for r in rs)} | "
                     f"{(st.mean(f1) if f1 else float('nan')):.2f} | {sum(bool(r.get('exact_structure')) for r in rs)} | "
                     f"{sum(r.get('cost_usd') or 0 for r in rs):.2f} | {(st.median(walls) if walls else float('nan')):.0f} |")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- CLI
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--_worker", nargs="+", help=argparse.SUPPRESS)
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--split", choices=("dev", "report", "blind"))
    p.add_argument("--seeds", type=int, nargs="*")
    p.add_argument("--systems", nargs="*")
    p.add_argument("--corruptions", nargs="*")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--force", action="store_true", help="arms A/D: re-run even if the case was already attempted "
                   "(by default a finished run is only re-scored and a crashed one keeps its crash record)")
    p.add_argument("--budget", type=float, help="arms A/D: stop launching new runs after this many $")
    p.add_argument("--no-llm-judge", action="store_true", help="sympy judge only (undecided -> not equivalent)")
    p.add_argument("--rescore", action="store_true", help="re-classify every stored outcome row (no re-runs)")
    p.add_argument("--table", action="store_true", help="aggregate outcomes.jsonl into markdown tables")
    p.add_argument("--md", help="also write the table to this file")
    a = p.parse_args(argv)
    if a._worker:
        kind, *args = a._worker
        WORKERS[kind](*args)
        return
    if a.arm:
        cases = select(load_index(), a.split, a.seeds, a.systems, a.corruptions)
        print(f"arm {a.arm}: {len(cases)} cases", flush=True)
        run_arm(a.arm, cases, workers=a.workers, force=a.force, budget=a.budget, use_llm_judge=not a.no_llm_judge)
    if a.rescore:
        rows = load_outcomes()
        for r in rows:
            append_outcome(rescore(r, use_llm_judge=not a.no_llm_judge))
        print(f"re-classified {len(rows)} rows")
    if a.table:
        t = table(load_outcomes())
        print(t)
        if a.md:
            Path(a.md).write_text(t + "\n")


if __name__ == "__main__":
    main()
