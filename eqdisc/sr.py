"""Static symbolic regression mode: discover y = f(x1, ..., xn) from tabular data with a Claude tool-use agent.

Same philosophy as the dynamics agent (intuition first, structure proposals fitted by variable projection, PySR
with templates, validation-driven selection, parallel sessions + pick-by-validation), adapted to y = f(x).

    from eqdisc.sr import SRTask, solve
    task = SRTask(X_train, y_train, X_val, y_val, names=["x0", "x1"], description="optional context")
    res = solve(task, n_sessions=2)          # -> {"expr", "val_nmse", "sessions": [...], "cost_usd"}

The test set is never passed to the agent; score `res["expr"]` on held-out data yourself.
"""
import itertools
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import sympy as sp

from .fitting import split_params, varpro_fit
from .solvers import parse

FUNCS = {"sin", "cos", "tan", "exp", "log", "sqrt", "tanh", "sinh", "cosh", "asin", "acos", "atan", "Abs", "pi", "E"}


@dataclass
class SRTask:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    names: list
    description: str = ""
    target: str = "y"
    extra: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- numerics
def nmse(y, yhat):
    with np.errstate(all="ignore"):
        v = float(np.mean((y - yhat) ** 2) / (np.var(y) + 1e-300))
    return v if np.isfinite(v) else float("inf")


def r2(y, yhat):
    return 1.0 - nmse(y, yhat)


def evaluate_expr(expr, names, X):
    e = parse(expr, list(names) + [f"p{i}" for i in range(20)])
    f = sp.lambdify([sp.Symbol(n) for n in names], e, modules=["numpy"])
    with np.errstate(all="ignore"):
        out = np.asarray(f(*[X[:, i] for i in range(X.shape[1])]), float)
    return np.broadcast_to(out, (X.shape[0],)).astype(float)


def complexity(expr, names):
    try:
        return int(sp.count_ops(parse(expr, list(names)), visual=False)) + len(parse(expr, list(names)).free_symbols)
    except Exception:  # noqa: BLE001
        return 999


def _r(x, s=4):
    try:
        return float(f"{float(x):.{s}g}")
    except Exception:  # noqa: BLE001
        return None


# ----------------------------------------------------------------------------- tools
def describe(task):
    X, y, names = task.X_train, task.y_train, task.names
    out = {"n_train": int(len(y)), "n_val": int(len(task.y_val)), "target": task.target,
           "variables": {}, "target_stats": {"min": _r(y.min()), "max": _r(y.max()), "mean": _r(y.mean()),
                                             "std": _r(y.std()), "always_positive": bool(y.min() > 0)}}
    pos_all = bool(y.min() > 0)
    for i, n in enumerate(names):
        x = X[:, i]
        rec = {"min": _r(x.min()), "max": _r(x.max()), "positive": bool(x.min() > 0),
               "decades": _r(np.log10(x.max() / x.min())) if x.min() > 0 else None,
               "corr_with_y": _r(np.corrcoef(x, y)[0, 1])}
        out["variables"][n] = rec
    # power-law exponents (log-log regression) if everything positive -> suggests monomial structure
    Xp = X[:, [i for i, n in enumerate(names) if X[:, i].min() > 0]]
    if pos_all and Xp.shape[1] == X.shape[1]:
        A = np.column_stack([np.ones(len(y)), np.log(X)])
        c, *_ = np.linalg.lstsq(A, np.log(y), rcond=None)
        res = np.log(y) - A @ c
        out["log_log_fit"] = {"exponents": {n: _r(c[i + 1]) for i, n in enumerate(names)},
                              "log_constant": _r(c[0]),
                              "rel_residual": _r(res.std() / (np.log(y).std() + 1e-300)),
                              "note": "rel_residual ~ 0 means y is a pure power-law monomial; exponents near simple "
                                      "fractions (1, 2, 1/2, -1, 3/2...) suggest the structure"}
    # additive vs multiplicative separability probes (first-order partial dependence via binning)
    sub = np.random.default_rng(0).choice(len(y), min(4000, len(y)), replace=False)
    try:
        from .intuition import _additive_components, _shape_fit
        G, _ = _additive_components(X[sub], y[sub])
        add_unexpl = float(np.var(y[sub] - y[sub].mean() - G.sum(1)) / (np.var(y[sub]) + 1e-300))
        out["additive_model_unexplained"] = _r(add_unexpl)
        if pos_all:
            Gl, _ = _additive_components(X[sub], np.log(y[sub]))
            ly = np.log(y[sub])
            out["multiplicative_model_unexplained"] = _r(np.var(ly - ly.mean() - Gl.sum(1)) / (np.var(ly) + 1e-300))
        shapes = {}
        target = np.log(y[sub]) if pos_all and out.get("multiplicative_model_unexplained", 1) < add_unexpl else y[sub]
        Gs, _ = _additive_components(X[sub], target)
        for j, n in enumerate(names):
            share = float(Gs[:, j].var() / (target.var() + 1e-300))
            if share > 0.01:
                name, K, score, rr = _shape_fit(X[sub, j], Gs[:, j])
                shapes[n] = {"variance_share": _r(share), "shape": name, **({"K": _r(K)} if K is not None else {})}
        out["single_variable_shapes"] = {"in": "log(y)" if target is not y[sub] else "y", "shapes": shapes,
                                         "note": "approximate; additive model in y (or log y if multiplicative fits better)"}
    except Exception as e:  # noqa: BLE001
        out["shape_error"] = str(e)
    # pairwise symmetric-combination probes: does y depend on xi - xj or xi + xj or xi/xj only?
    probes = []
    for i, j in itertools.combinations(range(len(names)), 2):
        for kind, comb in (("difference", X[:, i] - X[:, j]), ("ratio", X[:, i] / np.where(X[:, j] == 0, np.nan, X[:, j]))):
            if not np.all(np.isfinite(comb)):
                continue
            others = [k for k in range(len(names)) if k not in (i, j)]
            Z = np.column_stack([comb] + [X[:, k] for k in others])
            Zs = Z[sub]
            try:
                Gz, _ = _additive_components(Zs, target)
                rz = float(np.var(target - target.mean() - Gz.sum(1)) / (np.var(target) + 1e-300))
                probes.append((rz, f"{names[i]} {'-' if kind == 'difference' else '/'} {names[j]}"))
            except Exception:  # noqa: BLE001
                pass
    if probes:
        probes.sort()
        out["best_two_variable_combinations"] = [{"combination": c, "unexplained_when_used": _r(r)} for r, c in probes[:3]]
    return out


def fit_skeleton(task, expr_with_params, n_restarts=8, max_rows=20000, seed=0):
    names = task.names
    pnames = sorted({str(s) for s in parse(expr_with_params, names + [f"p{i}" for i in range(20)]).free_symbols
                     if str(s).startswith("p") and str(s)[1:].isdigit()}, key=lambda s: int(s[1:]))
    allnames = names + pnames
    e = parse(expr_with_params, allnames)
    bad = {str(s) for s in e.free_symbols} - set(allnames)
    if bad:
        return {"error": f"unknown symbols {sorted(bad)}; variables are {names}, parameters p0..p19"}
    rows = np.random.default_rng(seed).choice(len(task.y_train), min(max_rows, len(task.y_train)), replace=False)
    F = {n: task.X_train[rows, i] for i, n in enumerate(names)}
    vals, rel = (varpro_fit({"y": e}, ["y"], F, [task.y_train[rows]], pnames, None, n_restarts, seed)
                 if pnames else ({}, None))
    fitted = str(e.subs({sp.Symbol(k): v for k, v in vals.items()}))
    return {"expr": fitted, "params": vals, **validate(task, fitted)}


def sparse_fit(task, terms, thresholds=(1e-4, 1e-3, 1e-2, 3e-2, 0.1), max_rows=20000):
    """Linear-in-coefficients regression y ~ sum c_k * term_k with sequential thresholding; picks by validation."""
    from .baselines import stlsq
    names = task.names
    rows = np.random.default_rng(0).choice(len(task.y_train), min(max_rows, len(task.y_train)), replace=False)
    cols, ok = [], []
    for t in terms:
        try:
            v = evaluate_expr(t, names, task.X_train[rows])
            if np.all(np.isfinite(v)):
                cols.append(v)
                ok.append(t)
        except Exception:  # noqa: BLE001
            pass
    if not cols:
        return {"error": "no usable terms"}
    A = np.stack(cols, 1)
    best = None
    for th in thresholds:
        c = stlsq(A, task.y_train[rows], th)
        expr = " + ".join(f"({ci:.8g})*({t})" for ci, t in zip(c, ok) if ci != 0) or "0"
        v = validate(task, expr)
        if best is None or v["val_nmse"] < best[1]["val_nmse"] * 0.98:
            best = (expr, v)
    return {"expr": best[0], **best[1]}


def validate(task, expr):
    try:
        yv = evaluate_expr(expr, task.names, task.X_val)
        yt = evaluate_expr(expr, task.names, task.X_train[:5000])
    except Exception as e:  # noqa: BLE001
        return {"error": f"cannot evaluate: {e}"}
    return {"val_nmse": _r(nmse(task.y_val, yv)), "train_nmse": _r(nmse(task.y_train[:5000], yt)),
            "val_r2": _r(r2(task.y_val, yv)), "complexity": complexity(expr, task.names)}


def sr_pysr(meta, data, binary_operators=("+", "-", "*", "/"), unary_operators=("sin", "cos", "exp", "log", "sqrt"),
            maxsize=25, niterations=60, timeout=150, n_samples=2000, template=None):
    """PySR on (X, y); called in an isolated process."""
    from pysr import PySRRegressor
    X, y, names = data["X"], data["y"], meta["names"]
    sub = np.random.default_rng(0).choice(len(y), min(n_samples, len(y)), replace=False)
    safe = [f"z{i}" for i in range(len(names))]
    kw = dict(niterations=niterations, maxsize=maxsize, binary_operators=list(binary_operators),
              unary_operators=list(unary_operators), timeout_in_seconds=timeout, model_selection="best",
              verbosity=0, progress=False, random_state=0, deterministic=True, parallelism="serial")
    model = PySRRegressor(**kw)
    model.fit(X[sub], y[sub], variable_names=safe)

    def back(e):
        e = str(e)
        for j in reversed(range(len(names))):
            e = e.replace(f"z{j}", f"({names[j]})")
        return e
    front = [{"complexity": int(r.complexity), "loss": float(r.loss), "expr": back(r.sympy_format)}
             for r in model.equations_.itertuples()]
    return {"pareto_front": front[-10:], "best": back(model.sympy())}


# ----------------------------------------------------------------------------- agent
SYSTEM = """You are an expert at symbolic regression: discovering a closed-form expression y = f({vars}) from tabular data.
Work only through the tools. The final answer is scored on HELD-OUT test data (normalised MSE and exact symbolic
recovery), so prefer the simplest expression that reaches the noise floor on validation data; physically
meaningful, dimensionally sensible forms generalise best, often far outside the training range.

Strategy that works:
1. describe: scales, power-law exponents (log-log fit), additive vs multiplicative separability, single-variable shapes,
   and which two-variable combinations (differences or ratios) the target depends on. Use these to propose structures.
2. Propose explicit structures with free constants p0, p1, ... and fit them with fit_skeleton (linear constants are solved
   exactly; nonlinear ones by multi-start). Try several distinct hypotheses, not just one.
3. Use run_pysr (slow: ~3 min) when the structure is unclear, then distil its best expressions into clean skeletons.
4. Use sparse_fit for sums of candidate terms. Use run_python for any custom analysis (data in X, y, names).
5. Compare candidates by val_nmse and complexity. A tiny val_nmse inside the data range does NOT prove the law: a Taylor
   polynomial can mimic a rational, exponential or periodic term. Before submitting, call assess on your best candidate:
   if the data favour adding a term you lack, refit with that term (replacing its polynomial imitation) and prefer the
   closed form that keeps validation error at least as low; it extrapolates. Do not keep spurious tiny terms.
You have {budget} tool calls. Submit before the budget runs out. Write constants as numbers, with no parameters left."""


def _obj(props, required=()):
    return {"type": "object", "properties": props, "required": list(required), "additionalProperties": False}


_s, _i, _n = {"type": "string"}, {"type": "integer"}, {"type": "number"}
TOOLS = [
    {"name": "describe", "description": "Statistics and structural probes of the data (see system prompt).",
     "input_schema": _obj({})},
    {"name": "fit_skeleton", "description": "Fit constants p0..p19 in a proposed expression by least squares "
     "(variable projection). Returns the fitted expression with val_nmse, val_r2, complexity.",
     "input_schema": _obj({"expr": _s}, ["expr"])},
    {"name": "sparse_fit", "description": "Fit y as a sparse linear combination of the given candidate terms "
     "(expressions in the variables).", "input_schema": _obj({"terms": {"type": "array", "items": _s}}, ["terms"])},
    {"name": "run_pysr", "description": "Run PySR symbolic regression (about 3 minutes). Returns the Pareto front.",
     "input_schema": _obj({"binary_operators": {"type": "array", "items": _s}, "unary_operators": {"type": "array", "items": _s},
                           "maxsize": _i, "niterations": _i})},
    {"name": "validate", "description": "Validate a fully numeric expression.", "input_schema": _obj({"expr": _s}, ["expr"])},
    {"name": "run_python", "description": "Run Python analysis code. Preloaded: X (train inputs, n x d), y, X_val, y_val, "
     "names, np, sp. print() results. 60 s limit.", "input_schema": _obj({"code": _s}, ["code"])},
    {"name": "assess", "description": "Assess a candidate before submitting: per-term bootstrap intervals and dBIC, terms the "
     "data favour ADDING (validation-gated), the noise floor, alternatives that fit equally well, and input regions where "
     "plausible models disagree (extrapolation risk). If the data favour a term you lack (e.g. a rational or periodic form "
     "instead of a Taylor polynomial), refit with it before submitting.",
     "input_schema": _obj({"expr": _s, "alternatives": {"type": "array", "items": _s}}, ["expr"])},
    {"name": "submit", "description": "Submit the final expression (numeric constants, variables only).",
     "input_schema": _obj({"expr": _s, "rationale": _s}, ["expr"])},
]


def _run_python(task, code, timeout=60):
    import pickle
    import subprocess
    import sys
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "d.pkl").write_bytes(pickle.dumps((task.X_train, task.y_train, task.X_val, task.y_val, task.names)))
        (tmp / "s.py").write_text("import pickle, numpy as np, sympy as sp\nX, y, X_val, y_val, names = pickle.load(open(%r,'rb'))\n"
                                  % str(tmp / "d.pkl") + code)
        try:
            p = subprocess.run([sys.executable, str(tmp / "s.py")], capture_output=True, text=True, timeout=timeout)
            return {"stdout": p.stdout[-5000:], "stderr": p.stderr[-2000:] if p.returncode else ""}
        except subprocess.TimeoutExpired:
            return {"error": f"timeout after {timeout}s"}


def session(task, client=None, model="claude-opus-5-5", effort="high", max_tools=20, seed_note="", verbose=False,
            on_event=None, tag=""):
    from .agent import Usage, _jsonable, make_client
    from .isolate import run_isolated
    client = client or make_client()
    usage = Usage(model)
    system = SYSTEM.replace("{vars}", ", ".join(task.names)).replace("{budget}", str(max_tools))
    intro = f"Variables: {task.names}. Target: {task.target}."
    if task.description:
        intro += f"\nProblem description: {task.description}"
    if seed_note:
        intro += f"\n{seed_note}"
    messages = [{"role": "user", "content": intro + "\nDiscover the expression."}]
    log, best, submitted, n = [], None, None, 0
    while submitted is None and n < max_tools + 2:
        resp = client.beta.messages.create(model=model, max_tokens=16000, system=system, tools=TOOLS, messages=messages,
                                           thinking={"type": "adaptive"}, output_config={"effort": effort},
                                           betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        usage.add(resp)
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason == "refusal":
            break
        uses = [b for b in resp.content if b.type == "tool_use"]
        if not uses:
            messages.append({"role": "user", "content": "Continue with the tools; call submit when done."})
            n += 1
            continue
        results = []
        for u in uses:
            n += 1
            a = dict(u.input)
            try:
                if u.name == "describe":
                    out = describe(task)
                elif u.name == "fit_skeleton":
                    out = fit_skeleton(task, a["expr"])
                elif u.name == "sparse_fit":
                    out = sparse_fit(task, a["terms"])
                elif u.name == "validate":
                    out = {"expr": a["expr"], **validate(task, a["expr"])}
                elif u.name == "assess":
                    full = assess_sr(task, a["expr"], {f"alt{i}": x for i, x in enumerate(a.get("alternatives") or [])})
                    out = {k: full.get(k) for k in ("confidence", "terms", "missing_term_evidence", "noise_floor",
                                                    "model_ambiguity", "data_advice")}
                    out["experiments"] = (full.get("experiments") or {}).get("ranked", [])[:3]
                elif u.name == "run_pysr":
                    out = run_isolated("eqdisc.sr", "sr_pysr", {"names": task.names},
                                       {"X": task.X_train, "y": task.y_train}, a, timeout=420)
                    if "pareto_front" in out:
                        for r in out["pareto_front"]:
                            r.update(validate(task, r["expr"]))
                elif u.name == "run_python":
                    out = _run_python(task, a["code"])
                elif u.name == "submit":
                    out = {"expr": a["expr"], **validate(task, a["expr"])}
                    if "error" not in out:
                        submitted = {"expr": a["expr"], "rationale": a.get("rationale", ""), **out}
                else:
                    out = {"error": "unknown tool"}
            except Exception as e:  # noqa: BLE001
                out = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-600:]}
            out = _jsonable(out)
            if isinstance(out, dict) and out.get("expr") and out.get("val_nmse") is not None:
                if best is None or out["val_nmse"] < best["val_nmse"]:
                    best = {"expr": out["expr"], "val_nmse": out["val_nmse"], "complexity": out.get("complexity")}
            s = json.dumps(out, default=str)
            s = s[:10000] + ("...(truncated)" if len(s) > 10000 else "")
            left = max_tools - n
            results.append({"type": "tool_result", "tool_use_id": u.id,
                            "content": s + f"\n[tool calls left: {left}]" + (" Submit now." if left <= 2 else "")})
            log.append({"tool": u.name, "input": a, "output": out})
            if on_event:
                on_event({"type": "tool", "branch": tag, "n": n, "name": u.name, "input": json.dumps(a, default=str)[:300],
                          "expr": out.get("expr") if isinstance(out, dict) else None,
                          "val_nmse": out.get("val_nmse") if isinstance(out, dict) else None})
            if verbose:
                print(f"  [{n}] {u.name} {json.dumps(a)[:100]} -> {str(out.get('expr', ''))[:80]} {out.get('val_nmse', '')}",
                      flush=True)
        messages.append({"role": "user", "content": results})
    final = submitted or (best and {**best, "rationale": "budget exhausted; best validated candidate"})
    return {"final": final, "best_seen": best, "log": log, "usage": usage.as_dict()}


def solve(task, n_sessions=2, max_tools=20, effort="high", model="claude-opus-5-5", client=None, verbose=False,
          on_event=None, assess_result=True):
    """Parallel sessions; pick the submitted expression with the best validation NMSE (ties -> lower complexity)."""
    from .agent import make_client
    client = client or make_client()
    notes = ["", "Approach this independently: start from the describe probes and test several structurally different "
                 "hypotheses before refining any one."]
    t0 = time.time()
    with ThreadPoolExecutor(n_sessions) as ex:
        sess = list(ex.map(lambda k: session(task, client, model, effort, max_tools, notes[k % len(notes)], verbose,
                                             on_event, f"session {k + 1}"), range(n_sessions)))
    cands = [s["final"] for s in sess if s.get("final")]
    if not cands:
        return {"expr": None, "sessions": sess, "cost_usd": sum(s["usage"]["cost_usd"] for s in sess)}
    best_v = min(c["val_nmse"] for c in cands)
    near = [c for c in cands if c["val_nmse"] <= best_v * 1.5 + 1e-14]
    pick = min(near, key=lambda c: (c.get("complexity") or 999, c["val_nmse"]))
    out = {"expr": pick["expr"], "val_nmse": pick["val_nmse"], "complexity": pick.get("complexity"),
           "candidates": cands, "sessions": sess, "wall_s": round(time.time() - t0, 1),
           "cost_usd": round(sum(s["usage"]["cost_usd"] for s in sess), 3)}
    if assess_result:
        try:
            from .insights import verdict
            alts = {f"session {i + 1}": c["expr"] for i, c in enumerate(cands) if c["expr"] != pick["expr"]}
            for s in sess:                                   # other good candidates the sessions validated
                for ev in s["log"]:
                    o = ev.get("output") or {}
                    if isinstance(o, dict) and o.get("expr") and o.get("val_nmse") is not None and len(alts) < 8:
                        alts.setdefault(f"{ev['tool']} #{len(alts)}", o["expr"])
            out["assessment"] = assess_sr(task, pick["expr"], alts)
            out["verdict"] = verdict(out["assessment"])
        except Exception as e:  # noqa: BLE001
            out["assessment"] = {"error": f"{type(e).__name__}: {e}"}
    return out


# ============================================================================ assessment (static mode)
POOL_TEMPLATES = ["{v}", "{v}**2", "{v}**3", "sin({v})", "cos({v})", "exp(-{v}**2)", "{v}/(1 + {v}**2)", "{v}*Abs({v})"]


def _terms_of(expr, names):
    e = parse(expr, list(names))
    out = []
    for t in sp.Add.make_args(e):
        c, m = t.as_coeff_Mul()
        out.append((float(c) if c.is_number else 1.0, m if c.is_number else t))
    return out


def _cols(structs, names, X):
    f = [sp.lambdify([sp.Symbol(n) for n in names], s, "numpy") for s in structs]
    with np.errstate(all="ignore"):
        return np.stack([np.broadcast_to(np.asarray(g(*[X[:, i] for i in range(X.shape[1])]), float), (X.shape[0],))
                         for g in f], 1)


def _noise_floor(task, k=1):
    """Irreducible noise from nearest-neighbour differences in standardised input space (smooth f, dense data)."""
    from scipy.spatial import cKDTree
    mu, sd = task.X_train.mean(0), task.X_train.std(0) + 1e-300
    Z = (task.X_train - mu) / sd
    n = min(len(Z), 20000)
    idx = np.random.default_rng(0).choice(len(Z), n, replace=False)
    tree = cKDTree(Z[idx])
    dist, nb = tree.query(Z[idx], k=2)
    dy = task.y_train[idx] - task.y_train[idx][nb[:, 1]]
    return float(np.var(dy) / 2 / (np.var(task.y_train) + 1e-300))      # upper bound if f varies between neighbours


def assess_sr(task, expr, alternatives=None, n_boot=100, n_draws=12, seed=0):
    """Confidence + next measurements for a static law y = f(x). Same layout as eqdisc.assess.assess, so
    eqdisc.insights.verdict and the reports work unchanged."""
    from .assess import grade, questions
    rng = np.random.default_rng(seed)
    names = task.names
    X, y, Xv, yv = task.X_train, task.y_train, task.X_val, task.y_val
    n = len(y)
    terms = _terms_of(expr, names)
    structs = [m for _, m in terms]
    A = _cols(structs, names, X)
    ok = np.all(np.isfinite(A), axis=0)
    res = {"model": expr, "statistics_basis": "least squares on the data (static regression)"}

    def fit(cols, yy=y):
        c, *_ = np.linalg.lstsq(cols, yy, rcond=None)
        return c, float(np.sum((cols @ c - yy) ** 2))

    def bic(rss, k):
        return n * np.log(rss / n + 1e-300) + k * np.log(n)

    c0, rss0 = fit(A[:, ok]) if ok.any() else (np.zeros(0), float(np.sum(y ** 2)))
    boots = np.array([fit(A[idx][:, ok], y[idx])[0] for idx in (rng.integers(0, n, n) for _ in range(n_boot))]) if ok.any() else None
    term_rows, coefs = [], {}
    j = 0
    for (cg, m), good in zip(terms, ok):
        if not good:
            continue
        lo, hi = np.percentile(boots[:, j], [5, 95])
        others = [k for k in range(A.shape[1]) if ok[k] and k != list(np.where(ok)[0])[j]]
        rss_r = fit(A[:, others])[1] if others else float(np.sum(y ** 2))
        term_rows.append({"var": task.target, "term": str(m), "coef": _r(c0[j]), "submitted_coef": _r(cg),
                          "ci90": [_r(lo), _r(hi)], "rel_uncertainty": _r((hi - lo) / 2 / (abs(c0[j]) + 1e-300)),
                          "significant": bool(lo > 0 or hi < 0), "dBIC_if_removed": _r(bic(rss_r, len(others)) - bic(rss0, A[:, ok].shape[1]))})
        coefs[str(m)] = {"fit": float(c0[j]), "ci90": [float(lo), float(hi)], "struct": m}
        j += 1
    res["terms"] = term_rows
    # candidate additions (validation-gated)
    have = {str(m) for m in structs}
    pool = sorted({tpl.format(v=v) for v in names for tpl in POOL_TEMPLATES} |
                  {f"{a}*{b}" for i, a in enumerate(names) for b in names[i + 1:]} | {"1"})
    base_val = nmse(yv, _cols(structs, names, Xv)[:, ok] @ c0) if ok.any() else 1.0
    adds = []
    for t in pool:
        st = parse(t, names)
        if str(st) in have:
            continue
        col = _cols([st], names, X)
        if not np.all(np.isfinite(col)):
            continue
        Aa = np.hstack([A[:, ok], col])
        ca, rssa = fit(Aa)
        with np.errstate(all="ignore"):
            va = nmse(yv, np.hstack([_cols(structs, names, Xv)[:, ok], _cols([st], names, Xv)]) @ ca)
        er = 1 - va / (base_val + 1e-300)
        # meaningful only above floating-point noise, and by a large factor when the fit is already near-exact
        meaningful = base_val > 1e-25 and (er >= 0.5 if base_val < 1e-8 else er >= 0.02)
        adds.append({"var": task.target, "term": t, "dBIC_if_added": _r(bic(rssa, Aa.shape[1]) - bic(rss0, A[:, ok].shape[1])),
                     "error_reduction": _r(er) if meaningful else 0.0, "raw_error_reduction": _r(er)})
    adds.sort(key=lambda e: e["dBIC_if_added"])
    res["missing_term_evidence"] = adds[:4]
    # noise floor
    floor = _noise_floor(task)
    res["noise_floor"] = {"floor": _r(floor), "error_to_floor_ratio": _r(base_val / (floor + 1e-300)),
                          "note": "floor from nearest-neighbour differences (an upper bound where f varies between neighbours)"}
    # competing models
    alts = {k: v for k, v in (alternatives or {}).items() if v and v != expr}
    amb = []
    for k, a in alts.items():
        try:
            va = nmse(yv, evaluate_expr(a, names, Xv))
        except Exception:  # noqa: BLE001
            continue
        if va <= max(1.5 * base_val, 2 * min(floor, base_val), 1e-12):   # floor is only an upper bound on the noise
            amb.append((k, a, va))
    res["model_ambiguity"] = {"indistinguishable": ["submitted"] + [k for k, _, _ in amb],
                              "verdict": ("alternative structures fit the validation data equally well: "
                                          + "; ".join(f"{k}: {a[:80]} (val NMSE {va:.2g})" for k, a, va in amb)) if amb else
                              "no alternative structure fits comparably"}
    res["validation"] = {"val_nmse": _r(base_val), "val_r2": _r(1 - base_val)}
    # plausible model set: coefficient draws + alternatives
    models = {"submitted": lambda Z: _cols(structs, names, Z)[:, ok] @ c0}
    for i in range(n_draws):
        cd = np.array([rng.normal(coefs[str(m)]["fit"], (coefs[str(m)]["ci90"][1] - coefs[str(m)]["ci90"][0]) / 3.29 * 2)
                       for m in structs if str(m) in coefs])
        models[f"draw{i}"] = (lambda cdd: (lambda Z: _cols(structs, names, Z)[:, ok] @ cdd))(cd)
    for k, a, _ in amb:
        models[k] = (lambda aa: (lambda Z: evaluate_expr(aa, names, Z)))(a)
    # rival models: the current structure plus each term the data favour (refit) -- these typically agree in range
    # and diverge outside it, which is what the experiment design should expose
    for e in [e for e in adds if (e["dBIC_if_added"] or 0) < -10 and (e["error_reduction"] or 0) > 0][:3]:
        st = parse(e["term"], names)
        Aa = np.hstack([A[:, ok], _cols([st], names, X)])
        ca, _ = fit(Aa)
        models[f"+{e['term']}"] = (lambda cc, sst: (lambda Z: np.hstack([_cols(structs, names, Z)[:, ok],
                                                                          _cols([sst], names, Z)]) @ cc))(ca, st)
        res["model_ambiguity"]["indistinguishable"].append(f"with {e['term']}")
    # measurement noise cannot exceed what the fitted model leaves unexplained (the NN floor is only an upper bound)
    noise_sd = np.sqrt(max(min(floor, base_val), 1e-14) * np.var(y))
    # candidate measurement points: Latin hypercube in an expanded box (positivity preserved)
    lo, hi = X.min(0), X.max(0)
    span = hi - lo
    lo2, hi2 = lo - 0.5 * span, hi + 0.5 * span
    lo2 = np.where(lo > 0, np.maximum(lo2, 0.2 * lo), lo2)
    m_pts = 400
    cut = (np.argsort(rng.random((m_pts, X.shape[1])), axis=0) + rng.random((m_pts, X.shape[1]))) / m_pts
    P = lo2 + cut * (hi2 - lo2)
    preds = []
    for k, f in models.items():
        try:
            v = np.asarray(f(P), float)
            if np.all(np.isfinite(v)):
                preds.append(v)
        except Exception:  # noqa: BLE001
            pass
    preds = np.array(preds)
    spread = preds.std(0) / (noise_sd + 1e-300)
    sub = X[rng.choice(n, min(400, n), replace=False)]
    pe = []
    for k, f in models.items():
        try:
            v = np.asarray(f(sub), float)
            if np.all(np.isfinite(v)):
                pe.append(v)
        except Exception:  # noqa: BLE001
            pass
    base_spread = float(np.mean(np.array(pe).std(0) / (noise_sd + 1e-300))) if pe else None
    order = np.argsort(-spread)
    ranked, seen = [], []
    Ttr = _cols(structs, names, sub)[:, ok]
    tr_info = np.mean(Ttr ** 2, axis=0) + 1e-300
    for i in order:
        p = P[i]
        if any(np.linalg.norm((p - q) / (span + 1e-300)) < 0.25 for q in seen):
            continue
        seen.append(p)
        Tp = _cols(structs, names, p[None, :])[0, ok]
        info = sorted([(float(Tp[jj] ** 2 / tr_info[jj]), str(structs[k])) for jj, k in enumerate(np.where(ok)[0])], reverse=True)
        inside = bool(np.all((p >= lo) & (p <= hi)))
        ranked.append({"description": "measure at " + ", ".join(f"{nm}={v:.3g}" for nm, v in zip(names, p))
                                      + ("" if inside else " (outside the current data range)"),
                       "inputs": {nm: float(v) for nm, v in zip(names, p)}, "inside_data_range": inside,
                       "score": _r(float(spread[i]) ** 2), "max_snr": _r(float(spread[i])),
                       "gain_vs_existing_data": _r(float(spread[i]) / base_spread) if base_spread else None,
                       "informs_coefficients": [{"coefficient": f"{s} in {task.target}", "info_gain_vs_existing": _r(g)}
                                                for g, s in info[:2]]})
        if len(ranked) == 5:
            break
    res["experiments"] = {"ranked": ranked, "existing_data_discrimination": _r(base_spread ** 2) if base_spread else None,
                          "note": "score = (spread of plausible models / noise)^2 at that input; inputs where plausible "
                                  "models disagree are where a measurement discriminates between them"}
    res["coverage"] = {"ranges": {nm: [_r(a), _r(b)] for nm, a, b in zip(names, lo, hi)}, "n": int(n)}
    adv = []
    if res["noise_floor"]["error_to_floor_ratio"] and res["noise_floor"]["error_to_floor_ratio"] > 2:
        adv.append("Validation error is well above the noise floor: structure is still missing.")
    if ranked and not ranked[0]["inside_data_range"] and (ranked[0]["gain_vs_existing_data"] or 0) > 3:
        adv.append("Plausible models agree inside the data but diverge outside it: the law is only pinned down within the "
                   "measured range. Measure beyond it (see experiments) before trusting extrapolation.")
    res["data_advice"] = adv
    res["confidence"] = grade(res)
    res["questions_for_human"] = questions({"kind": "static", "variables": names}, res)
    return json.loads(json.dumps(res, default=lambda o: _r(o) if isinstance(o, (float, np.floating)) else str(o)))
