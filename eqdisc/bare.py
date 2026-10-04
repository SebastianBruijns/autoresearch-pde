"""Bare-Claude baseline: the model, the data and a short prompt, with none of eqdisc's guidance or helpers.

No system prompt, no playbook, no eqdisc tools (SINDy, weak form, PySR, skeleton fits, validation...), no critic,
no pre-analysis. Claude gets:
  - the prompt text (default "Gimme PDE!"), followed by the minimum needed to reach the data and to have its answer
    scored: where the data file is, what arrays it holds, and the symbol names an answer must use;
  - one generic tool, python: runs code in a fresh process next to the data file (numpy, scipy, sympy, pandas,
    matplotlib, scikit-learn available; eqdisc, pysindy and pysr are blocked);
  - a submit tool for the final answer.
Budgets (tool calls, USD, wall-clock) are enforced like the harness agent's.

    from eqdisc.bare import run_bare
    r = run_bare(workdir, data_file, description, answer_spec, client, model="claude-opus-5-5", max_tools=20)
"""
import json
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

PROMPT = "Gimme PDE!"
BLOCKED = ("eqdisc", "pysindy", "pysr", "juliacall")

_BLOCK = """import sys
for _m in %r:
    sys.modules[_m] = None            # blocked in the bare baseline
del _m
""" % (BLOCKED,)


def run_python(code, workdir, timeout=120, max_output=8000):
    """Run code in a fresh, isolated Python process with cwd = workdir (where the data file is)."""
    from . import sandbox
    from .guard import allow_dirs, prelude
    workdir = Path(workdir).resolve()
    boxed = sandbox.available()               # bubblewrap (Linux): the folder is mounted at /work, the venv at /opt/venv
    own = [sandbox.WORK, sandbox.VENV, "/tmp"] if boxed else [workdir]
    guard = prelude(allow_dirs(own))          # reads only inside its own folder + the Python installation
    with tempfile.NamedTemporaryFile("w", suffix=".py", dir=workdir, delete=False) as f:
        f.write(_BLOCK + guard + "\n" + code)
        script = f.name
    t0 = time.time()
    try:
        mpl = workdir / ".mpl"                 # one matplotlib cache per session (built once, reused)
        mpl.mkdir(exist_ok=True)
        if boxed:
            argv, env = sandbox.wrap([sys.executable, "-I", f"{sandbox.WORK}/{Path(script).name}"], workdir)
            env["MPLCONFIGDIR"] = f"{sandbox.WORK}/.mpl"
        else:
            argv = [sys.executable, "-I", script]
            env = {"PATH": "/usr/bin:/bin", "MPLCONFIGDIR": str(mpl), "HOME": str(workdir)}
        p = subprocess.run(argv, cwd=workdir, capture_output=True, text=True, timeout=timeout, env=env)
        out, err, rc = p.stdout, p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        err, rc = f"TIMEOUT after {timeout}s", -1
    finally:
        Path(script).unlink(missing_ok=True)
    trim = lambda s: s if len(s) <= max_output else s[: max_output // 2] + "\n...[truncated]...\n" + s[-max_output // 2:]
    return {"returncode": rc, "stdout": trim(out), "stderr": trim(err[-4000:]) if rc else "",
            "seconds": round(time.time() - t0, 2)}


def _tools(answer_schema):
    return [
        {"name": "python", "description": "Run Python code. The working directory contains the data file. "
         "Print what you want to see.",
         "input_schema": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"],
                          "additionalProperties": False}},
        {"name": "submit", "description": "Submit your final answer.",
         "input_schema": {"type": "object", "properties": {"answer": answer_schema}, "required": ["answer"],
                          "additionalProperties": False}},
    ]


def run_bare(workdir, description, answer_spec, answer_schema, client, model="claude-opus-5-5", effort="high",
             max_tools=20, max_cost_usd=None, max_wall_s=None, prompt=PROMPT, validate_answer=None, verbose=False,
             tag=""):
    """One bare session. description: where the data is and what it holds; answer_spec: how the answer must be
    written (symbols). validate_answer(answer) -> error string or None (e.g. unparseable expression).
    Returns {"submitted", "n_tool_calls", "usage", "log", "stop", "wall_s"}."""
    from .llm import request_opts
    from .srsd import Cost, _jsonable
    cost = Cost(model)
    tools = _tools(answer_schema)
    messages = [{"role": "user", "content": f"{prompt}\n\n{description}\n\n{answer_spec}"}]
    log, submitted, n_tools, t0, stop = [], None, 0, time.time(), None
    while submitted is None:
        if max_cost_usd and cost.usd() >= 1.2 * max_cost_usd:
            stop = "cost cap"
            break
        if max_wall_s and time.time() - t0 >= max_wall_s:
            stop = "time cap"
            break
        resp = client.beta.messages.create(
            model=model, max_tokens=16000, tools=tools, messages=messages,          # no system prompt
            cache_control={"type": "ephemeral"}, **request_opts(model, effort))
        cost.add(resp)
        messages.append({"role": "assistant", "content": resp.content})
        for b in resp.content:
            if getattr(b, "type", "") == "thinking" and (getattr(b, "thinking", "") or "").strip():
                log.append({"type": "thinking", "text": b.thinking})
            elif getattr(b, "type", "") == "text" and b.text.strip():
                log.append({"type": "text", "text": b.text})
        if resp.stop_reason == "refusal":
            stop = "refusal"
            break
        uses = [b for b in resp.content if getattr(b, "type", "") == "tool_use"]
        if not uses:
            if n_tools >= max_tools:
                stop = "tool budget"
                break
            messages.append({"role": "user", "content": "Use the submit tool for your final answer."})
            continue
        results = []
        for u in uses:
            n_tools += 1
            ts = time.time()
            try:
                if u.name == "python":
                    out = run_python(u.input.get("code", ""), workdir)
                elif u.name == "submit":
                    err = validate_answer(u.input.get("answer")) if validate_answer else None
                    if err:
                        out = {"accepted": False, "error": err}
                    else:
                        submitted = {"answer": u.input.get("answer")}
                        out = {"ok": True}
                else:
                    out = {"error": f"unknown tool {u.name}"}
            except Exception as e:  # noqa: BLE001
                out = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-600:]}
            out = _jsonable(out)
            s = json.dumps(out, default=str)
            left = max_tools - n_tools
            over = bool((max_cost_usd and cost.usd() >= max_cost_usd) or (max_wall_s and time.time() - t0 >= 0.75 * max_wall_s))
            results.append({"type": "tool_result", "tool_use_id": u.id,
                            "content": s[:10000] + f"\n[tool calls left: {left}]" + (" Submit now." if left <= 2 or over else ""),
                            **({"is_error": True} if isinstance(out, dict) and out.get("error") else {})})
            log.append({"type": "tool", "name": u.name, "input": dict(u.input), "seconds": round(time.time() - ts, 1),
                        "output": out})
            if verbose:
                print(f"  [bare{(' ' + tag) if tag else ''}] tool {n_tools:02d} {u.name} {time.time() - ts:.1f}s", flush=True)
        messages.append({"role": "user", "content": results})
        if n_tools >= max_tools + 3 and submitted is None:
            stop = "tool budget"
            break
    return {"submitted": submitted, "n_tool_calls": n_tools, "usage": cost.as_dict(), "log": log, "stop": stop,
            "wall_s": round(time.time() - t0, 1)}


# ----------------------------------------------------------------------------- dataset adapters
def well_session(ds_dir, workdir, client, cfg):
    """Bare session on an eqdisc PDE dataset directory (data.npz + meta.json). Answer: {variable: expression}."""
    from .solvers import derivative_symbols, parse
    ds_dir, workdir = Path(ds_dir), Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((ds_dir / "meta.json").read_text())
    data = workdir / "data.npz"
    if not data.exists():                 # a real copy: a symlink would point outside the sandboxed folder
        import shutil
        shutil.copyfile(ds_dir / "data.npz", data)
    v, dims = meta["variables"], meta["spatial_dims"]
    shape = meta["shape"]
    grid = ", ".join(f"{d}: {meta['grid'][d]['n']} points, period {meta['grid'][d]['L']:g}" for d in dims)
    description = (f"Data: data.npz with arrays t (time, {shape[1]} samples, step {meta['dt']:g}), "
                   + ", ".join(f"{d} (coordinate)" for d in dims)
                   + f" and U of shape {tuple(shape)} = (trajectory, time, {', '.join(dims)}, variable); "
                   f"variables in order: {', '.join(v)}. Grid ({meta.get('boundary', 'periodic')}): {grid}.")
    names = derivative_symbols(v, dims, meta.get("max_deriv_cap") or 4)
    sym_ex = f"{v[0]}_{dims[0]}, {v[0]}_{dims[0]}{dims[0]}" + (f", {v[0]}_{dims[0]}{dims[1]}" if len(dims) > 1 else "")
    answer_spec = (f"Answer with submit: an object mapping each variable ({', '.join(v)}) to the right-hand side of its "
                   f"time derivative, as a sympy expression in the variables and their spatial derivatives written "
                   f"like {sym_ex} (derivative letters sorted).")
    schema = {"type": "object", "additionalProperties": {"type": "string"}}

    def check(ans):
        if not isinstance(ans, dict) or not ans:
            return "answer must be an object {variable: expression}"
        try:
            for var, e in ans.items():
                bad = {str(s) for s in parse(e, names).free_symbols} - set(names)
                if bad:
                    return f"unknown symbols {sorted(bad)} in {var}"
        except Exception as ex:  # noqa: BLE001
            return f"could not parse: {ex}"
        return None
    r = run_bare(workdir, description, answer_spec, schema, client, model=cfg.get("model", "claude-opus-5-5"),
                 effort=cfg.get("effort", "high"), max_tools=cfg.get("max_tools", 20),
                 max_cost_usd=cfg.get("max_cost_usd"), max_wall_s=cfg.get("max_wall_s"),
                 prompt=cfg.get("bare_prompt") or PROMPT, validate_answer=check, verbose=cfg.get("verbose", False))
    r["submitted"] = {"rhs": {k: str(x) for k, x in r["submitted"]["answer"].items()}} if r["submitted"] else None
    return r


def static_session(prob, workdir, client, cfg):
    """Bare session on an SRSD problem (train/val splits only). Answer: one expression for y in x0..x{n-1}."""
    import numpy as np

    from .srsd import _parse
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    np.savez(workdir / "data.npz", X_train=prob["X_train"], y_train=prob["y_train"], X_val=prob["X_val"],
             y_val=prob["y_val"])
    v = prob["variables"]
    description = (f"Data: data.npz with arrays X_train {prob['X_train'].shape}, y_train {prob['y_train'].shape}, "
                   f"X_val {prob['X_val'].shape}, y_val {prob['y_val'].shape}; the columns of X are {', '.join(v)}.")
    answer_spec = f"Answer with submit: one sympy expression for y in terms of {', '.join(v)}."
    schema = {"type": "string"}

    def check(ans):
        try:
            bad = {str(s) for s in _parse(ans, v).free_symbols} - set(v)
            return f"unknown symbols {sorted(bad)}" if bad else None
        except Exception as ex:  # noqa: BLE001
            return f"could not parse: {ex}"
    r = run_bare(workdir, description, answer_spec, schema, client, model=cfg.get("model", "claude-opus-5-5"),
                 effort=cfg.get("effort", "high"), max_tools=cfg.get("max_tools", 15),
                 max_cost_usd=cfg.get("max_cost_usd"), max_wall_s=cfg.get("max_wall_s"),
                 prompt=cfg.get("bare_prompt") or PROMPT, validate_answer=check, verbose=cfg.get("verbose", False))
    r["submitted"] = {"expr": str(r["submitted"]["answer"]), "rationale": ""} if r["submitted"] else None
    return r


class FakeBareClient:
    """Scripted stand-in (no API): one python call that inspects the data, then a submit of `answer`."""

    def __init__(self, answer, in_tokens=20000, out_tokens=1500):
        from types import SimpleNamespace as NS
        self.NS, self.step, self.answer, self.tokens = NS, 0, answer, (in_tokens, out_tokens)
        self.beta = NS(messages=self)
        self.seen = []

    def create(self, **kw):
        NS = self.NS
        self.seen.append(kw)
        T = lambda n, i: NS(type="tool_use", id=f"t{self.step}", name=n, input=i)
        blocks = [T("python", {"code": "import numpy as np\nd = np.load('data.npz')\nprint({k: d[k].shape for k in d})\n"
                                       "import eqdisc"})] if self.step == 0 else [T("submit", {"answer": self.answer})]
        self.step += 1
        return NS(content=blocks, stop_reason="tool_use",
                  usage=NS(input_tokens=self.tokens[0], output_tokens=self.tokens[1], cache_creation_input_tokens=0,
                           cache_read_input_tokens=0))
