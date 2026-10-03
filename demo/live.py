"""Background jobs for the live "Run on your data" tab: real backends (eqdisc.orchestrate.discover / eqdisc.sr.solve)
and a scripted rehearsal backend that makes NO API calls (EQDISC_DEMO_FAKE=1 or the sidebar toggle)."""
import json
import queue
import re
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

DEMO = Path(__file__).resolve().parent
SHOW = DEMO / "showcase"
RUNS = DEMO / "_live_runs"

TOOL_ICON = {"intuit": "💡", "weak_sindy": "🧮", "run_sindy": "🧮", "ensemble_sindy": "🧮", "sparse_fit": "🧮",
             "run_pysr": "🧬", "fit_skeleton": "🦴", "find_invariants": "⚖️", "detect_symmetries": "🪞",
             "equivariant_sindy": "🪞", "transform": "🔄", "compare_models": "🏆", "coefficient_uncertainty": "📏",
             "assess_model": "📋", "run_python": "🐍", "plot_data": "📈", "plot_model": "📈", "load_skill": "📚",
             "submit": "📤", "repair": "🔧", "describe": "🔍", "validate": "✔️", "ask_human": "🙋",
             "request_experiment": "🧪"}


def fmt_event(ev):
    """One log line (markdown) for an on_event dict."""
    typ = ev.get("type")
    br = f"`{ev['branch']}` " if ev.get("branch") else ""
    if typ == "stage":
        txt = (ev.get("text") or "").strip()
        return f"**▶ {txt}**" if re.match(r"^\[\d/\d\]", txt) else f"&nbsp;&nbsp;&nbsp;{txt}"
    if typ == "note":
        return f"💭 {br}_{(ev.get('text') or '')[:220]}_"
    if typ == "tool":
        name = ev.get("name", "?")
        line = f"{TOOL_ICON.get(name, '🔧')} {br}**{name}**"
        inp = (ev.get("input") or "").strip()
        if inp and inp not in ("{}",):
            line += f" <small>{inp[:90]}</small>"
        res = ev.get("rhs") or ev.get("expr")
        if res:
            s = json.dumps(res) if isinstance(res, dict) else str(res)
            line += f" → `{s[:110]}`"
        if ev.get("valid_time") is not None:
            line += f" · valid time {float(ev['valid_time']):.3g}"
        if ev.get("val_nmse") is not None:
            line += f" · val NMSE {float(ev['val_nmse']):.3g}"
        if ev.get("error"):
            line += f" · ⚠️ {str(ev['error'])[:80]}"
        return line
    return str(ev)[:200]


class Job:
    def __init__(self, fn, **kwargs):
        self.fn, self.kwargs = fn, kwargs
        self.q = queue.Queue()
        self.events, self.result, self.error = [], None, None
        self.t0 = time.time()
        self.t1 = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def push(self, ev):
        self.q.put(ev)

    def _run(self):
        try:
            self.result = self.fn(on_event=self.push, **self.kwargs)
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}"
        finally:
            self.t1 = time.time()

    def start(self):
        self.thread.start()
        return self

    def drain(self):
        while True:
            try:
                self.events.append(self.q.get_nowait())
            except queue.Empty:
                break
        return self.events

    @property
    def done(self):
        return not self.thread.is_alive()

    @property
    def elapsed(self):
        return (self.t1 or time.time()) - self.t0


# ----------------------------------------------------------------------------- data helpers
def ident(c):
    s = re.sub(r"\W+", "_", str(c)).strip("_") or "col"
    return ("v_" + s) if s[0].isdigit() else s


def detect_mode(df):
    cols = [c.lower() for c in df.columns]
    return "dynamics" if any(c in ("t", "time", "times", "t_s", "time_s") for c in cols) else "static"


def static_task_arrays(df, target, seed=0):
    df = df.select_dtypes("number").dropna()
    df = df.rename(columns={c: ident(c) for c in df.columns})
    target = ident(target)
    names = [c for c in df.columns if c != target]
    X, y = df[names].to_numpy(float), df[target].to_numpy(float)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y))
    n_val = max(1, len(y) // 10)
    va, tr = perm[:n_val], perm[n_val:]
    return names, target, X, y, tr, va


# ----------------------------------------------------------------------------- real backends
def run_dynamics(csv_path, run_dir, n_branches, adversary, context, on_event):
    from eqdisc.ingest import ingest
    from eqdisc.orchestrate import discover
    run_dir = Path(run_dir)
    on_event({"type": "stage", "text": "[0/6] ingest: inferring the layout and writing a data card"})
    ds, card = ingest(csv_path, out_dir=str(run_dir / "dataset"))
    on_event({"type": "stage", "text": (card or {}).get("summary", "")[:200]})
    res = discover(ds, n_branches=n_branches, adversary=adversary, context=context or None, verbose=False,
                   on_event=on_event, out_dir=str(run_dir / "run"))
    res["data_card"] = res.get("data_card") or card
    return {"kind": "dynamics", **res}


def run_static(csv_path, target, context, n_sessions, on_event):
    from eqdisc.sr import SRTask, solve
    df = pd.read_csv(csv_path)
    names, target, X, y, tr, va = static_task_arrays(df, target)
    on_event({"type": "stage", "text": f"[1/3] static law: {target} = f({', '.join(names)}); "
                                       f"{len(tr)} training rows, {len(va)} validation rows (90/10 split)"})
    task = SRTask(X[tr], y[tr], X[va], y[va], names, description=context or "", target=target)
    on_event({"type": "stage", "text": f"[2/3] {n_sessions} parallel agent sessions"})
    res = solve(task, n_sessions=n_sessions, on_event=on_event)
    on_event({"type": "stage", "text": "[3/3] picked the best validated expression"})
    return {"kind": "static", "names": names, "target": target, "csv": str(csv_path),
            **{k: v for k, v in res.items() if k != "sessions"}}


# ----------------------------------------------------------------------------- rehearsal (scripted, no API)
def _sleep(s):
    time.sleep(s)


def fake_dynamics(csv_path, run_dir, n_branches, adversary, context, on_event, speed=1.0):
    from eqdisc.ingest import ingest
    from eqdisc.evaluate import load
    run_dir = Path(run_dir)
    on_event({"type": "stage", "text": "[0/6] ingest: inferring the layout and writing a data card"})
    ds, card = ingest(csv_path, out_dir=str(run_dir / "dataset"))          # real, local, no API
    on_event({"type": "stage", "text": (card or {}).get("summary", "")[:200]})
    case = json.loads((SHOW / "pendulum" / "case.json").read_text())
    disc = case["discovery"]
    branches = ["structure-first", "symbolic", "sparse-regression"][:n_branches]
    script = [
        ({"type": "stage", "text": "[1/6] intuition pre-analysis"}, 0.4),
        ({"type": "stage", "text": "[medium] non-polynomial dependence detected: sin(theta) in domega/dt"}, 0.3),
        ({"type": "stage", "text": "[medium] energy-like quantity decays slowly: weak dissipation"}, 0.3),
        ({"type": "stage", "text": f"[2/6] {len(branches)} parallel branches: {branches}"}, 0.3),
        ({"type": "tool", "branch": branches[0], "name": "intuit", "input": "{}"}, 0.3),
        ({"type": "tool", "branch": branches[-1], "name": "weak_sindy", "input": '{"poly_degree": 1, "custom_terms": ["sin(theta)"]}',
          "rhs": {"theta": "1.0*omega", "omega": "-9.79*sin(theta)"}, "valid_time": 2.1}, 0.5),
        ({"type": "note", "branch": branches[0], "text": "Undamped model drifts on rollouts; testing a linear damping term."}, 0.4),
        ({"type": "tool", "branch": branches[0], "name": "fit_skeleton", "input": '{"omega": "-p0*sin(theta) - p1*omega"}',
          "rhs": {"theta": "omega", "omega": "-9.794*sin(theta) - 0.106*omega"}, "valid_time": 4.26}, 0.5),
        ({"type": "tool", "branch": branches[-1], "name": "coefficient_uncertainty", "input": "{}"}, 0.3),
        ({"type": "tool", "branch": branches[0], "name": "compare_models", "input": '{"damped", "undamped", "quadratic drag"}'}, 0.4),
        ({"type": "tool", "branch": branches[0], "name": "submit", "input": "{}", "rhs": disc["final_model"]}, 0.3),
        ({"type": "stage", "text": "[3/6] tournament"}, 0.3),
        ({"type": "stage", "text": "winner: structure-first; 2 alternatives indistinguishable"}, 0.3),
    ]
    if adversary:
        script += [({"type": "stage", "text": "[4/6] adversary (red team) attacks the winner"}, 0.3),
                   ({"type": "tool", "branch": "adversary", "name": "repair", "input": '{"add": ["omega*Abs(omega)", "sign(omega)"]}'}, 0.4),
                   ({"type": "stage", "text": "incumbent survives the attack"}, 0.2)]
    script += [({"type": "stage", "text": "[5/6] assessment of the final model"}, 0.5),
               ({"type": "stage", "text": "[6/6] write-up"}, 0.3)]
    for ev, dt in script:
        on_event(ev)
        _sleep(dt * speed)
    meta, _ = load(ds)
    use_upload = set(meta["variables"]) == set(disc["final_model"])
    res = {"kind": "dynamics", **{k: disc[k] for k in ("verdict", "final_model", "story", "insights", "assessment",
                                                       "branches", "winner_branch", "tournament")},
           "dataset_path": str(ds if use_upload else SHOW / "pendulum" / "dataset"),
           "report": str(SHOW / "pendulum" / "report.html"), "data_card": card, "cost_usd": 0.0,
           "rehearsal": True, "rehearsal_note": None if use_upload else
           "Rehearsal mode replays the precomputed pendulum result; your file's variables differ, so plots use the pendulum data."}
    on_event({"type": "stage", "text": f"\n{res['verdict']['status']}: {res['verdict']['headline']}"})
    return res


def fake_static(csv_path, target, context, n_sessions, on_event, speed=1.0):
    df = pd.read_csv(csv_path)
    names, target, X, y, tr, va = static_task_arrays(df, target)
    from eqdisc.sr import evaluate_expr, nmse
    on_event({"type": "stage", "text": f"[1/3] static law: {target} = f({', '.join(names)}); "
                                       f"{len(tr)} training rows, {len(va)} validation rows (90/10 split)"})
    _sleep(0.4 * speed)
    on_event({"type": "stage", "text": f"[2/3] {n_sessions} parallel agent sessions"})
    if {"b", "s", "temp", "pH"} <= set(names) and target == "db":
        expr = json.loads((SHOW / "ecoli" / "case.json").read_text())["expr"]
        steps = [("describe", "{}", None), ("fit_skeleton", '{"expr": "p0*b*s/(p1+s)"}', "0.31*b*s/(1.0+s)"),
                 ("run_python", '{"code": "partial dependence on temp, pH"}', None),
                 ("fit_skeleton", '{"expr": "... /(1+((temp-p2)/p3)**4) ..."}', "0.49*b*s/(1+s)/(1+((temp-35.1)/3.7)**4)"),
                 ("fit_skeleton", '{"expr": "... *exp(-abs(pH-p4))*sin(pi*(pH-p5)/p6)**2"}', expr)]
    else:  # generic: least-squares linear law
        A = np.column_stack([X[tr], np.ones(len(tr))])
        c = np.linalg.lstsq(A, y[tr], rcond=None)[0]
        expr = " + ".join(f"{ci:.4g}*{n}" for ci, n in zip(c[:-1], names)) + f" + {c[-1]:.4g}"
        steps = [("describe", "{}", None), ("sparse_fit", json.dumps({"terms": names}), expr)]
    for k, (nm, inp, e) in enumerate(steps):
        v = nmse(y[va], evaluate_expr(e, names, X[va])) if e else None
        on_event({"type": "tool", "branch": f"session {k % n_sessions + 1}", "name": nm, "input": inp, "expr": e,
                  "val_nmse": v})
        _sleep(0.8 * speed)
    v = nmse(y[va], evaluate_expr(expr, names, X[va]))
    on_event({"type": "tool", "branch": "session 1", "name": "submit", "input": "{}", "expr": expr, "val_nmse": v})
    on_event({"type": "stage", "text": "[3/3] picked the best validated expression; assessing it (local, no API)"})
    out = {"kind": "static", "names": names, "target": target, "csv": str(csv_path), "expr": expr, "val_nmse": v,
           "candidates": [{"expr": expr, "val_nmse": v}], "cost_usd": 0.0, "rehearsal": True}
    try:                                   # the real static assessment is pure numerics: run it for realism
        from eqdisc.insights import verdict
        from eqdisc.sr import SRTask, assess_sr
        task = SRTask(X[tr], y[tr], X[va], y[va], names, description=context or "", target=target)
        out["assessment"] = assess_sr(task, expr)
        out["verdict"] = verdict(out["assessment"])
    except Exception as e:  # noqa: BLE001  (older eqdisc without assess_sr)
        out["assessment_error"] = str(e)
    return out
