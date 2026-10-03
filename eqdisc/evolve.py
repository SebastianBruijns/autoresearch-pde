"""AlphaEvolve-style loop: an LLM mutates a *discovery program*, the evaluator scores it.

The evolved artefact is a Python file defining  discover(meta, data) -> {"rhs": {...}}.
Fitness = mean evaluator score over a set of training datasets ("generalizer mode":
a program that discovers many systems, not one equation). Held-out datasets measure
whether the evolved pipeline generalises.

Programs only ever see a copy of the *public* files (data.npz, meta.json).

    export ANTHROPIC_API_KEY=...
    python -m eqdisc.evolve --train datasets/lorenz* datasets/kdv* --test datasets/burgers* \
        --generations 10 --children 4
    python -m eqdisc.evolve --train datasets/pendulum* --dry-run   # plumbing test, no LLM
    python -m eqdisc.evolve --target playbook --train datasets/{pendulum,kdv}_n0.05* --children 2
        # evolves the agent's strategy document (playbook.md); each fitness eval runs the agent
"""
import argparse
import json
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .evaluate import evaluate

ROOT = Path(__file__).resolve().parent.parent
SEED = Path(__file__).with_name("seed_program.py")

SYSTEM_PROMPT = """You are an expert in data-driven discovery of governing equations (SINDy, weak-form SINDy, \
symbolic regression, sparse regression, numerical differentiation, denoising, model selection).

You are improving a Python *discovery program* inside an evolutionary search. The program defines

    def discover(meta: dict, data: dict) -> {"rhs": {var: "sympy expression string"}}

meta: kind ("ode"|"pde"), variables, dt, allowed_symbols, and for PDEs L, nx, boundary="periodic".
data: t (nt,), U (n_traj, nt, n_vars) for ODEs or (n_traj, nt, nx, n_fields) for PDEs, x (nx,) for PDEs.
Expressions may only use meta["allowed_symbols"] (PDE derivatives are written u_x, u_xx, u_xxx, u_xxxx),
numbers and sympy functions (sin, cos, exp, log, ...). The data are noisy.

Available: numpy, scipy, sympy, sklearn, pysindy; eqdisc.solvers (spectral_derivs, integrate_ode,
integrate_pde, make_ode_rhs, make_pde_rhs) and eqdisc.baselines (time_derivative, lowpass, stlsq,
select, pde_library, to_expr) and eqdisc.toolbox (diagnose, run_sindy with custom_terms/hyperparameters,
run_pysr, fit_skeleton for parametrised structures, validate = internal held-out validation on public data;
each returns {"rhs", "validation"}) and eqdisc.coordinates (find_invariants: conserved quantities / constraints;
make_coords + transform_data + map_back: fit in new coordinates such as polar, log or reduced, and map the
model back exactly). A good program can try several strategies and pick by
toolbox.validate. You may reimplement anything. Each run has a time limit of {timeout}s per
dataset, so keep the cost reasonable. The program must be deterministic and must not read other files.

The hidden evaluator simulates the returned model from unseen initial conditions and compares
(1) the vector field on clean held-out states, (2) rollouts over a horizon, (3) parsimony.
score = 0.5*(-log10 vf_nrmse) + 0.5*(-log10 rollout_nrmse) + 0.5*valid_frac - 0.02*n_terms (higher is better).

Respond with a short diagnosis of what limits the current program (2-5 sentences), then the COMPLETE
new program in a single ```python block. Make one or two focused, well-motivated changes per child rather
than rewriting everything; programs that crash score -10."""


PLAYBOOK_SYSTEM = """You are improving the *playbook* (strategy document) that a tool-using equation-discovery
agent follows. The agent has tools: diagnose, find_invariants, transform/set_coordinates (fit in new coordinates and map back),
run_sindy (library/derivative/threshold options), run_pysr, fit_skeleton (fit p0,p1,... in a proposed
structure), validate, submit. It is scored on hidden data
from unseen initial conditions (vector field, rollout, parsimony).

You see the current playbook, how the agent performed on each training dataset with it (hidden score,
the tools it called, what it submitted) and other playbooks. Write a better playbook. Be concrete:
decision rules, thresholds and settings that worked, failure modes to avoid, and when to switch tools.
Keep it under 900 words. Respond with a short diagnosis, then the full playbook in a single ```markdown block."""


# ----------------------------------------------------------------------------- evaluation
def run_program(code, dataset, timeout):
    """Execute a program on a public-only copy of a dataset; return evaluator result dict."""
    dataset = Path(dataset)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pub = tmp / "public"
        pub.mkdir()
        for f in ("data.npz", "meta.json"):
            shutil.copy(dataset / f, pub / f)
        (tmp / "program.py").write_text(code)
        try:
            p = subprocess.run([sys.executable, "-m", "eqdisc.run_program", str(tmp / "program.py"),
                                str(pub), str(tmp / "out.json")],
                               cwd=ROOT, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"dataset": dataset.name, "score": -10.0, "error": f"timeout after {timeout}s"}
        if p.returncode != 0:
            return {"dataset": dataset.name, "score": -10.0, "error": p.stderr.strip()[-1500:]}
        cand = json.loads((tmp / "out.json").read_text())
    r = evaluate(dataset, cand)
    r["runtime_s"] = cand.get("runtime_s")
    return r


def run_playbook(playbook, dataset, agent_kw):
    from .agent import run_agent
    try:
        r = run_agent(dataset, playbook, verbose=False, **agent_kw)
    except Exception as e:  # noqa: BLE001
        return {"dataset": Path(dataset).name, "score": -10.0, "error": f"agent crashed: {e}"}
    h = r["hidden_eval"]
    tools = []
    for ev in json.loads((Path(r["out_dir"]) / "transcript.json").read_text()):
        if ev["type"] == "tool":
            v = ev["output"].get("validation", {}) if isinstance(ev["output"], dict) else {}
            tools.append(f"{ev['name']}({json.dumps(ev['input'])[:200]}) -> valid_t={v.get('rollout_valid_time')}")
    return {**h, "dataset": Path(dataset).name, "tool_trace": tools,
            "rationale": (r["submitted"] or {}).get("rationale", "")}


def score_program(code, datasets, timeout, workers, target="program", agent_kw=None):
    if target == "playbook":
        with ThreadPoolExecutor(workers) as ex:
            results = list(ex.map(lambda d: run_playbook(code, d, agent_kw or {}), datasets))
        return float(np.mean([r["score"] for r in results])), results
    with ThreadPoolExecutor(workers) as ex:
        results = list(ex.map(lambda d: run_program(code, d, timeout), datasets))
    return float(np.mean([r["score"] for r in results])), results


# ----------------------------------------------------------------------------- LLM
def make_llm(model, effort):
    import anthropic
    client = anthropic.Anthropic()

    def call(system, user):
        with client.beta.messages.stream(
            model=model,
            max_tokens=64000,
            system=system,
            thinking={"type": "adaptive"},
            output_config={"effort": effort},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",                      # re-route on a refusal instead of stopping
            messages=[{"role": "user", "content": user}],
        ) as stream:
            msg = stream.get_final_message()
        if msg.stop_reason == "refusal":
            raise RuntimeError("model declined the request")
        return "".join(b.text for b in msg.content if b.type == "text")
    return call


def extract_code(text, lang="python"):
    blocks = re.findall(rf"```{lang}\n(.*?)```", text, re.S)
    return max(blocks, key=len) if blocks else None


def fmt_results(results):
    lines = []
    for r in results:
        if "error" in r:
            lines.append(f"- {r['dataset']}: ERROR {r['error'][-600:]}")
        else:
            rhs = "; ".join(f"d{k}/dt = {v}" for k, v in r["rhs"].items())
            lines.append(f"- {r['dataset']}: score={r['score']:.2f} vf_nrmse={r['vf_nrmse']:.2e} "
                         f"rollout_nrmse={r['rollout_nrmse']:.2e} valid_frac={r['valid_frac']:.2f} "
                         f"n_terms={r['n_terms']} runtime={r.get('runtime_s') or 0:.1f}s\n    found: {rhs[:600]}")
            if r.get("tool_trace"):
                lines.append("    tools: " + "\n           ".join(r["tool_trace"][:25]))
                lines.append(f"    agent rationale: {r.get('rationale', '')[:400]}")
    return "\n".join(lines)


def build_prompt(parent, inspirations):
    lang = "markdown" if parent.get("target") == "playbook" else "python"
    kind = "playbook" if lang == "markdown" else "program"
    s = [f"## Current {kind} (mean score {parent['score']:.3f})\n```{lang}\n{parent['code']}\n```",
         f"## Its results per training dataset\n{fmt_results(parent['results'])}"]
    for i, p in enumerate(inspirations):
        s.append(f"## Inspiration {kind} {i + 1} (mean score {p['score']:.3f}; per dataset: "
                 + ", ".join(f"{r['dataset']}={r['score']:.2f}" for r in p["results"])
                 + f")\n```{lang}\n{p['code']}\n```")
    s.append(f"Write an improved {kind}.")
    return "\n\n".join(s)


# ----------------------------------------------------------------------------- database
class Database:
    def __init__(self, run_dir):
        self.dir = run_dir
        (run_dir / "programs").mkdir(parents=True, exist_ok=True)
        self.programs = []

    def add(self, prog):
        prog["id"] = len(self.programs)
        self.programs.append(prog)
        ext = "md" if prog.get("target") == "playbook" else "py"
        (self.dir / "programs" / f"{prog['id']:04d}.{ext}").write_text(prog["code"])
        with open(self.dir / "db.jsonl", "a") as f:
            f.write(json.dumps({k: v for k, v in prog.items() if k != "code"}, default=str) + "\n")
        return prog

    def best(self):
        return max(self.programs, key=lambda p: p["score"])

    def sample(self, n_insp=2, temp=0.5):
        """Parent: softmax over score (exploit) with temperature (explore).
        Inspirations: best programs from *other* lineages for diversity."""
        ok = [p for p in self.programs if p["score"] > -10]
        pool = ok or self.programs
        s = np.array([p["score"] for p in pool])
        w = np.exp((s - s.max()) / temp)
        parent = pool[np.random.choice(len(pool), p=w / w.sum())]
        others = sorted([p for p in pool if p["id"] != parent["id"]], key=lambda p: -p["score"])
        insp = others[:1] + random.sample(others[1:], min(n_insp - 1, len(others[1:]))) if others else []
        return parent, insp


# ----------------------------------------------------------------------------- main loop
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", nargs="+", required=True, help="dataset dirs used for fitness")
    p.add_argument("--test", nargs="*", default=[], help="held-out dataset dirs (reported, never optimised)")
    p.add_argument("--target", choices=["program", "playbook"], default="program",
                   help="evolve a discovery program, or the playbook of the tool-using agent")
    p.add_argument("--seed-program", help="seed program (.py) or playbook (.md)")
    p.add_argument("--agent-model", default="claude-opus-5-5")
    p.add_argument("--agent-effort", default="medium")
    p.add_argument("--agent-max-tools", type=int, default=15)
    p.add_argument("--generations", type=int, default=5)
    p.add_argument("--children", type=int, default=4, help="LLM proposals per generation (run in parallel)")
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--timeout", type=int, default=120, help="seconds per program per dataset")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--run-dir")
    p.add_argument("--dry-run", action="store_true", help="no LLM: children are copies of the parent")
    a = p.parse_args()

    run_dir = Path(a.run_dir or f"runs/{time.strftime('%Y%m%d-%H%M%S')}")
    db = Database(run_dir)
    pb = a.target == "playbook"
    system = PLAYBOOK_SYSTEM if pb else SYSTEM_PROMPT.replace("{timeout}", str(a.timeout))
    seed_path = a.seed_program or (Path(__file__).with_name("playbook.md") if pb else SEED)
    agent_kw = {"model": a.agent_model, "effort": a.agent_effort, "max_tools": a.agent_max_tools}
    score = lambda code, ds: score_program(code, ds, a.timeout, a.workers, a.target, agent_kw)
    llm = None if a.dry_run else make_llm(a.model, a.effort)
    log = lambda *m: (print(*m, flush=True), open(run_dir / "log.txt", "a").write(" ".join(map(str, m)) + "\n"))

    code = Path(seed_path).read_text()
    s0, results = score(code, a.train)
    db.add({"code": code, "score": s0, "results": results, "parent": None, "gen": 0, "note": "seed",
            "target": a.target})
    log(f"[gen 0] seed score={s0:.3f}")

    def child(_):
        parent, insp = db.sample()
        if llm is None:
            text, new = "dry run", parent["code"]
        else:
            try:
                text = llm(system, build_prompt(parent, insp))
            except Exception as e:  # noqa: BLE001
                return None, f"llm error: {e}"
            new = extract_code(text, "markdown" if pb else "python")
            if not new:
                return None, "no code block in response"
        sc, res = score(new, a.train)
        return {"code": new, "score": sc, "results": res, "parent": parent["id"], "target": a.target,
                "note": text.split("```")[0].strip()[:1500]}, None

    for gen in range(1, a.generations + 1):
        with ThreadPoolExecutor(a.children) as ex:
            outs = list(ex.map(child, range(a.children)))
        for prog, err in outs:
            if err:
                log(f"[gen {gen}] child failed: {err}")
                continue
            prog["gen"] = gen
            db.add(prog)
            log(f"[gen {gen}] #{prog['id']} (parent #{prog['parent']}) score={prog['score']:.3f}  "
                + " ".join(f"{r['dataset'].split('_n')[0]}={r['score']:.2f}" for r in prog["results"]))
        b = db.best()
        log(f"[gen {gen}] best #{b['id']} score={b['score']:.3f}")

    best = db.best()
    (run_dir / ("best_playbook.md" if pb else "best_program.py")).write_text(best["code"])
    summary = {"best_id": best["id"], "train_score": best["score"],
               "train": {r["dataset"]: r["score"] for r in best["results"]}}
    if a.test:
        seed_test, _ = score(db.programs[0]["code"], a.test)
        best_test, res = score(best["code"], a.test)
        summary.update({"test_score_seed": seed_test, "test_score_best": best_test,
                        "test": {r["dataset"]: r["score"] for r in res}})
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
