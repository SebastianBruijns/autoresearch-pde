"""Bare-Claude control: the same model and API settings as eqdisc's agent, but none of its harness.

Claude gets the public data file, minimal metadata (no dataset name), one generic `run_python` tool and a
`submit` tool. There are no eqdisc tools, playbook, skills, memory, critic, branches or adversary. Submissions are
scored exactly like the `agent` arm of `python -m eqdisc.benchmark` (hidden-test score + symbolic-equivalence judge).

    python bare_claude.py datasets/blind_burgers_dirichlet_n0.02_s1 [more datasets ...] [--max-tools 18]
    python bare_claude.py --held-out            # the 12 blinded held-out systems of benchmark v1
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from eqdisc import sandbox
from eqdisc.agent import Usage, _jsonable, _tool_result_content, make_client
from eqdisc.evaluate import load
from eqdisc.solvers import parse

PUBLIC_META = ("kind", "variables", "dt", "n_traj", "shape", "shape_doc", "L", "nx", "boundary", "spatial_dims",
               "grid", "allowed_symbols")

SYSTEM = """You are a scientist analysing measurement data. Your task: find the differential equation that governs
the system that produced the data. This is a real test of your abilities as a researcher, so be rigorous and creative.

The data are in data.npz in your working directory (metadata below). The measurements are noisy. Use run_python to
analyse them: it executes a Python script in that directory, with numpy, scipy, sympy, matplotlib and pandas available.
Files you write there persist between calls. Any .png figure a script saves is shown to you, so you can look at plots.

Your submission will be judged on unseen trajectories from new initial conditions: on how well the equation's vector
field and its simulated trajectories match the true system. So aim for the correct equation (a simple, stable model with
the right terms and accurate coefficients), not just a close fit to these trajectories. Check your candidate against
data you held out.

Submit with `submit`: one right-hand side per variable, as sympy strings over the allowed symbols in the metadata
(for a PDE, u_x, u_xx, u_xxx, u_xxxx are spatial derivatives of the field u; the left-hand side is the time derivative).
You have a budget of {budget} tool calls including the submission; submit before it runs out."""

TOOLS = [
    {"name": "run_python",
     "description": "Run a Python script in the working directory (which contains data.npz). Returns stdout/stderr "
                    "(truncated) and any .png files the script created or updated, as images.",
     "input_schema": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}},
    {"name": "submit",
     "description": "Submit the final equation. Ends the session.",
     "input_schema": {"type": "object", "properties": {
         "rhs": {"type": "object", "description": "{variable: right-hand side as a sympy string}",
                 "additionalProperties": {"type": "string"}},
         "rationale": {"type": "string", "description": "how you found it and how confident you are"}},
         "required": ["rhs", "rationale"]}},
]


def prepare(dataset, workdir):
    """Copy only the public data; return the dataset dir (ingesting raw files) and the public metadata."""
    dataset = Path(dataset)
    if dataset.is_file():
        from eqdisc.ingest import ingest
        dataset = Path(ingest(dataset)[0])
    meta, _ = load(dataset)
    workdir.mkdir(parents=True, exist_ok=True)
    shutil.copy(dataset / "data.npz", workdir / "data.npz")
    return dataset, meta, {k: meta[k] for k in PUBLIC_META if k in meta}


def run_python(code, workdir, timeout=600):
    script = workdir / f"_cell_{int(time.time() * 1000)}.py"
    script.write_text(code)
    t0 = time.time()
    # sandboxed: no network; only the Python installation and the work directory are visible
    argv, env = sandbox.wrap([sys.executable, script.name], workdir)
    try:
        p = subprocess.run(argv, cwd=workdir, env=env, capture_output=True, text=True, timeout=timeout)
        out = {"returncode": p.returncode, "stdout": p.stdout[-8000:], "stderr": p.stderr[-3000:]}
    except subprocess.TimeoutExpired:
        out = {"error": f"timed out after {timeout}s"}
    pngs = sorted((f for f in workdir.glob("**/*.png") if f.stat().st_mtime >= t0), key=lambda f: f.stat().st_mtime)
    return out, [str(f) for f in pngs[-4:]]


def blind_variables(meta):
    """Rename state variables to neutral names (ODE x1, x2, ...; PDE fields u1, u2, ...). Returns (renamed variables,
    renamed allowed_symbols, map new symbol -> original symbol)."""
    new = {v: (f"x{i + 1}" if meta["kind"] == "ode" else f"u{i + 1}") for i, v in enumerate(meta["variables"])}
    back, allowed = {}, []
    for s in meta["allowed_symbols"]:
        base, _, suffix = s.partition("_")
        n = s if base not in new else new[base] + ("_" + suffix if suffix else "")
        allowed.append(n)
        back[n] = s
    return [new[v] for v in meta["variables"]], allowed, back


def unblind(rhs, allowed, back, variables):
    """Map a submission in blinded names back to the dataset's names."""
    import sympy as sp
    sub = {sp.Symbol(n): sp.Symbol(o) for n, o in back.items()}
    return {back[v]: str(parse(e, allowed).xreplace(sub)) for v, e in rhs.items() if v in variables}


def check_rhs(rhs, meta):
    names = meta["allowed_symbols"]
    missing = [v for v in meta["variables"] if v not in rhs]
    if missing:
        return f"missing right-hand side for {missing}"
    for v, e in rhs.items():
        try:
            bad = {str(s) for s in parse(e, names).free_symbols} - set(names)
        except Exception as ex:  # noqa: BLE001
            return f"could not parse {v}: {ex}"
        if bad:
            return f"{v} uses symbols outside allowed_symbols: {sorted(bad)}"
    return None


def run_bare(dataset, out_root="runs/bare", model="claude-opus-5-5", effort="high", max_tools=18, client=None,
             verbose=True, system=None, blind_vars=False):
    """system: replace the default system prompt verbatim; the first message is then the metadata only."""
    client = client or make_client()
    usage = Usage(model)
    name = Path(dataset).stem if Path(dataset).is_file() else Path(dataset).name
    out_dir = Path(out_root) / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    dataset, meta, public = prepare(dataset, out_dir / "work")
    workdir = (out_dir / "work").resolve()
    say = (lambda *a: print(f"[{name}]", *a, flush=True)) if verbose else (lambda *a: None)
    blind = None
    if blind_vars:
        bvars, ballowed, back = blind_variables(meta)
        public = {**public, "variables": bvars, "allowed_symbols": ballowed}
        blind = (ballowed, back, bvars)
    task = "" if system else "\n\nFind the governing equation."
    messages = [{"role": "user", "content": f"Metadata:\n{json.dumps(public, indent=2)}{task}"}]
    system = system or SYSTEM.replace("{budget}", str(max_tools))
    log, submitted, n_tools, t0 = [], None, 0, time.time()

    while submitted is None:
        resp = client.beta.messages.create(
            model=model, max_tokens=16000, system=system, tools=TOOLS, messages=messages,
            thinking={"type": "adaptive"}, output_config={"effort": effort},
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        )
        usage.add(resp)
        messages.append({"role": "assistant", "content": resp.content})
        for b in resp.content:
            if b.type == "text" and b.text.strip():
                say(f"[claude] {b.text.strip()[:300]}")
                log.append({"type": "text", "text": b.text})
        if resp.stop_reason == "refusal":
            break
        uses = [b for b in resp.content if b.type == "tool_use"]
        if not uses:
            if n_tools >= max_tools:
                break
            messages.append({"role": "user", "content": "Please continue with the tools, and call submit when done."})
            continue
        results = []
        for u in uses:
            n_tools += 1
            images = []
            try:
                if u.name == "run_python":
                    out, images = run_python(u.input["code"], workdir)
                elif u.name == "submit":
                    err = check_rhs(u.input.get("rhs", {}), public)
                    out = {"error": err} if err else {"status": "submitted"}
                    if not err:
                        rhs = unblind(u.input["rhs"], *blind) if blind else u.input["rhs"]
                        submitted = {"rhs": rhs, "rationale": u.input.get("rationale", ""),
                                     **({"rhs_as_submitted": u.input["rhs"]} if blind else {})}
                else:
                    out = {"error": f"unknown tool {u.name}"}
            except Exception as e:  # noqa: BLE001
                out = {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-800:]}
            left = max_tools - n_tools
            out_s = json.dumps(out, default=str)[:12000] + f"\n[tool calls left: {left}]" + (" Submit now." if left <= 2 else "")
            results.append({"type": "tool_result", "tool_use_id": u.id, "content": _tool_result_content(out_s, images),
                            **({"is_error": True} if "error" in out else {})})
            say(f"[tool {n_tools:02d}] {u.name} -> {('rc=' + str(out.get('returncode'))) if 'returncode' in out else json.dumps(out)[:150]}"
                + (f" +{len(images)} fig" if images else ""))
            log.append({"type": "tool", "name": u.name, "input": dict(u.input), "output": out, "images": images})
        messages.append({"role": "user", "content": results})
        if n_tools >= max_tools + 3 and submitted is None:
            break

    code = "\n".join(e["input"].get("code", "") for e in log if e.get("name") == "run_python")
    result = {"dataset": name, "dataset_path": str(dataset), "submitted": submitted, "n_tool_calls": n_tools,
              "wall_s": round(time.time() - t0, 1),
              # crude leak check: the hidden test set lives next to the dataset, outside the working directory
              "touched_hidden_files": any(k in code for k in ("hidden", "truth.json", "test.npz"))}
    if (Path(dataset) / "hidden").exists():
        from eqdisc.benchmark import score
        result["hidden"] = score(str(dataset), submitted["rhs"], client) if submitted else {"score": -10, "equivalent": False}
    result["usage"] = usage.as_dict()
    result["cost_usd"] = result["usage"]["cost_usd"]
    (out_dir / "transcript.json").write_text(json.dumps(_jsonable(log), indent=1, default=str))
    (out_dir / "result.json").write_text(json.dumps(_jsonable(result), indent=2, default=str))
    h = result.get("hidden", {})
    say(f"submitted {json.dumps(submitted['rhs']) if submitted else None} | score {h.get('score')} "
        f"equivalent={h.get('equivalent')} | {n_tools} calls ${result['cost_usd']} {result['wall_s']}s")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("datasets", nargs="*", help="dataset directories (eqdisc format) or raw data files")
    p.add_argument("--held-out", action="store_true", help="also run the 12 blinded held-out systems (benchmark v1)")
    p.add_argument("--max-tools", type=int, default=18, help="tool-call budget (benchmark v1 agent arm used 18)")
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--effort", default="high")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="runs/bare")
    p.add_argument("--blind-vars", action="store_true", help="rename state variables to x1, x2, ... (PDE: u1, ...)")
    p.add_argument("--system", help="custom system prompt, used verbatim (the first message is then the metadata only)")
    a = p.parse_args()
    sets = list(a.datasets)
    if a.held_out:
        from eqdisc.benchmark import HELD_OUT
        from eqdisc.blind import make_blind
        sets += [str(make_blind(s, noise=0.02, seed=1)[0]) for s in HELD_OUT]
    if not sets:
        p.error("give datasets or --held-out")
    client = make_client()

    def one(d):
        try:
            return run_bare(d, a.out, a.model, a.effort, a.max_tools, client, system=a.system, blind_vars=a.blind_vars)
        except Exception as e:  # noqa: BLE001
            print(f"[{d}] failed: {e}", flush=True)
            return {"dataset": Path(d).name, "error": str(e)[:300]}
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, sets))
    lines = ["| dataset | bare Claude | ✓ | calls | $ | submitted |", "|---|---|---|---|---|---|"]
    for r in rows:
        h = r.get("hidden", {})
        lines.append(f"| {r['dataset']} | {h.get('score', r.get('error', '-'))} | {'✓' if h.get('equivalent') else ''} | "
                     f"{r.get('n_tool_calls', '')} | {r.get('cost_usd', '')} | "
                     f"`{json.dumps((r.get('submitted') or {}).get('rhs'))}` |")
    scored = [r for r in rows if r.get("hidden")]
    if scored:
        lines.append(f"\nsymbolic matches: {sum(bool(r['hidden'].get('equivalent')) for r in scored)}/{len(scored)}; "
                     f"total ${sum(r.get('cost_usd', 0) for r in rows):.2f}")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "summary.md").write_text("\n".join(lines))
    (Path(a.out) / "results.json").write_text(json.dumps(_jsonable(rows), indent=1, default=str))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
