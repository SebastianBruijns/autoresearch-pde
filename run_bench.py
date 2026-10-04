"""Benchmark the eqdisc agents on SRSD-Feynman (static symbolic regression) or The Well (PDEs), locally or on Modal.

Pick the benchmark with --benchmark:
    srsd:hard | srsd:hard_dummy | srsd:medium | srsd:easy        SRSD-Feynman problems (Hugging Face)
    well:<name>  or a Hugging Face link                           any dataset from The Well (polymathic-ai/*), e.g.
                                                                   well:shear_flow or
                                                                   https://huggingface.co/datasets/polymathic-ai/shear_flow
Data are read from Hugging Face where the agent runs: on Modal that is inside the container, so nothing is downloaded
to this machine, and the machine launching the run needs only `modal` installed (problem listing runs remotely too). The Well's ~1-2 GB HDF5 files are read by byte ranges: only the trajectories and frames used.

    modal secret create anthropic-api-key ANTHROPIC_API_KEY=sk-ant-...                               # once
    modal run run_bench.py --args "--benchmark srsd:hard --n 10 --total-budget 8"
    modal run run_bench.py --args "--benchmark well:shear_flow --n 2 --skills off --total-budget 6"
    modal run run_bench.py --args "--benchmark https://huggingface.co/datasets/polymathic-ai/shear_flow \\
        --well-params Reynolds_1e4_Schmidt_2e-1 --n-train 2 --t-end 4"
    python run_bench.py --list well:shear_flow             # list parameter settings (metadata only)

Inference uses your Claude API credits. --skills on|off switches the PDE agent's domain skills (eqdisc/skills/*.md,
the load_skill tool). There are no built-in skills (data-only rule); --skills-dir DIR opts in to your own .md skill
files, and only then do the agents get a load_skill tool. --context tells the agent what the
variables mean (default: blind). Results stream to runs/<run>/dashboard.html (refreshes itself while running).
Hard USD caps: --max-cost-per-problem and --total-budget.
"""
import argparse
import datetime as dt
import html
import json
import os
import random
import re
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRSD = {"hard": "yoshitomo-matsubara/srsd-feynman_hard", "hard_dummy": "yoshitomo-matsubara/srsd-feynman_hard_dummy",
        "medium": "yoshitomo-matsubara/srsd-feynman_medium", "easy": "yoshitomo-matsubara/srsd-feynman_easy"}


# ----------------------------------------------------------------------------- CLI
def parse_args(argv=None, read_skills=True):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", default="srsd:hard", help="srsd:<set> | well:<name> | Hugging Face dataset link")
    p.add_argument("--list", metavar="BENCHMARK", help="list the problems / parameter settings of a benchmark and exit")
    p.add_argument("--problems", help="SRSD: comma-separated problem names")
    p.add_argument("--well-params", help="The Well: comma-separated substrings selecting parameter files, "
                                         "e.g. Reynolds_1e4_Schmidt_2e-1")
    p.add_argument("--n", type=int, default=10, help="problems (SRSD) / parameter settings (Well) to sample; 0 = all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skills", choices=["on", "off"], default="off",
                   help="'on' needs --skills-dir (there are no built-in skills: data-only rule)")
    p.add_argument("--skills-dir", help="folder of skill .md files to give the PDE agent INSTEAD of eqdisc/skills "
                                        "(your own baseline skill); implies --skills on")
    p.add_argument("--context", action="store_true", help="tell the agent what the variables mean (default: blind)")
    p.add_argument("--agent", choices=["harness", "bare"], default="harness",
                   help="harness = eqdisc agent (playbook + tools); bare = Claude with only the data, one generic "
                        "python tool and the prompt text (no system prompt, playbook, eqdisc tools or critic)")
    p.add_argument("--bare-prompt", default="Gimme PDE!", help="prompt text for --agent bare")
    p.add_argument("--bare-exact", action="store_true", help="SRSD ablation: append the exactness instruction "
                   "(eqdisc.srsd.EXACT_NOTE) to the bare prompt")
    p.add_argument("--disguise", action="store_true", help="MHD_64: hide the dataset's identity without new "
                   "simulation (neutral field names q1..q7 in shuffled order, permuted axes, rescaled space/time/fields, "
                   "shifted clock; seeded by --seed). See eqdisc.hf_well.disguise")
    p.add_argument("--harness-prompt", choices=["full", "tools", "tools+exact"], default="full",
                   help="SRSD ablation: harness system prompt. full = playbook; tools = minimal task frame only; "
                        "tools+exact = minimal frame + the exactness instruction")
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    p.add_argument("--max-tools", type=int, default=None, help="default 15 (SRSD) / 20 (Well)")
    p.add_argument("--max-cost-per-problem", type=float, default=None, help="USD; default 1.0 (SRSD) / 2.0 (Well)")
    p.add_argument("--total-budget", type=float, default=10.0, help="USD; hard cap across the run")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-minutes", type=float, default=45.0,
                   help="per-problem wall-clock budget for the agent (asked to submit at 75%%); the Modal container "
                        "limit is 6 h")
    p.add_argument("--quiet", action="store_true", help="don't stream the agent's tool calls from the containers")
    p.add_argument("--critic", choices=["on", "off"], default="on", help="Well: the PDE agent's submission critic")
    # The Well: what to fetch per parameter setting
    p.add_argument("--n-train", type=int, default=2, help="Well: training trajectories")
    p.add_argument("--n-test", type=int, default=1, help="Well: hidden test trajectories (from the test split)")
    p.add_argument("--t-end", type=float, default=None,
                   help="Well: last time (simulation units); default per dataset (shear_flow 4.0, MHD_64 0.3, else all)")
    p.add_argument("--stride", type=int, default=1, help="Well: keep every k-th frame")
    p.add_argument("--coarsen", type=int, default=2, help="Well: spatial coarsening factor per axis (spectral)")
    p.add_argument("--pysr-timeout", type=int, default=300, help="SRSD: cap on one PySR call (s)")
    p.add_argument("--dry-run", action="store_true", help="scripted agent, no API calls (costs shown are nominal)")
    p.add_argument("--allow-local-download", action="store_true",
                   help="local backend only: allow fetching benchmark data to this machine (off by default)")
    p.add_argument("--out", help="output dir (default runs/<run id>)")
    a = p.parse_args(argv)
    a.kind, a.target = resolve_benchmark(a.benchmark)
    if a.disguise and "MHD" not in a.target:
        raise SystemExit("--disguise is implemented for MHD_64 only")
    if a.kind != "srsd" and (a.bare_exact or a.harness_prompt != "full"):
        raise SystemExit("--bare-exact / --harness-prompt are SRSD-only (prompt ablation)")
    if a.bare_exact and a.agent != "bare" or a.harness_prompt != "full" and a.agent == "bare":
        raise SystemExit("--bare-exact goes with --agent bare; --harness-prompt with the harness")
    a.skill_files = None
    if a.skills_dir and not read_skills:     # e.g. inside a Modal container, where the local folder doesn't exist
        a.skills = "on"
        a.skills_label = Path(a.skills_dir).name
    elif a.skills_dir:
        d = Path(a.skills_dir).expanduser()
        if not d.is_absolute():
            d = (Path.cwd() / d) if (Path.cwd() / d).is_dir() else ROOT / d
        files = sorted(d.glob("*.md")) if d.is_dir() else []
        if not files:
            raise SystemExit(f"--skills-dir {a.skills_dir}: no .md files found")
        a.skill_files = {f.stem: f.read_text() for f in files}   # sent to the container as text
        a.skills = "on"
        a.skills_label = d.name + ":" + ",".join(a.skill_files)
    else:
        if a.skills == "on":
            raise SystemExit("--skills on needs --skills-dir DIR: there are no built-in skills (data-only rule)")
        a.skills_label = a.skills
    is_well = a.kind == "well"
    a.max_tools = a.max_tools or (20 if is_well else 15)
    a.max_cost_per_problem = a.max_cost_per_problem or (2.0 if is_well else 1.0)
    return a


def resolve_benchmark(b):
    """-> ("srsd", hf repo) or ("well", 'polymathic-ai/<name>')."""
    b = b.strip()
    if b.startswith("srsd:"):
        key = b.split(":", 1)[1]
        if key not in SRSD and "/" not in key:
            raise SystemExit(f"unknown SRSD set {key!r}; choose from {sorted(SRSD)} or give a repo id")
        return "srsd", SRSD.get(key, key)
    if "srsd-feynman" in b:
        m = re.search(r"([^/\s]+/srsd-feynman[^/?#\s]*)", b)
        return "srsd", m.group(1)
    # same rule as eqdisc.hf_well.parse_ref, kept stdlib-only so the Modal client side needs nothing but `modal`
    r = re.sub(r"^well:", "", b.rstrip("/"))
    m = re.search(r"huggingface\.co/datasets/([^/]+/[^/?#]+)", r)
    return "well", (m.group(1) if m else (r if "/" in r else f"polymathic-ai/{r}"))


def enumerate_problems(a):
    """Problem specs, from Hugging Face metadata only (no data files are downloaded here)."""
    from huggingface_hub import HfApi
    rng = random.Random(a.seed)
    if a.kind == "srsd":
        items = HfApi().list_repo_tree(a.target, repo_type="dataset", path_in_repo="train")
        allp = sorted(Path(i.path).stem for i in items if i.path.endswith(".txt"))
        if a.problems:
            names = [x.strip() for x in a.problems.split(",") if x.strip()]
            bad = [x for x in names if x not in allp]
            if bad:
                raise SystemExit(f"unknown problems {bad}; available: {allp}")
        else:
            names = allp if a.n <= 0 or a.n >= len(allp) else sorted(rng.sample(allp, a.n))
        return [{"benchmark": "srsd", "repo": a.target, "name": n, "id": n} for n in names]
    from eqdisc.hf_well import list_param_files
    files = list_param_files(a.target, "train")
    if not files:
        raise SystemExit(f"{a.target}: no files under data/train")
    if a.well_params:
        keys = [k.strip() for k in a.well_params.split(",") if k.strip()]
        files = [f for f in files if any(k in f for k in keys)]
        if not files:
            raise SystemExit(f"no parameter file matches {keys}")
    elif 0 < a.n < len(files):
        files = sorted(rng.sample(files, a.n))
    return [{"benchmark": "well", "ref": a.target, "param_file": f, "id": Path(f).stem, "n_train": a.n_train,
             "n_test": a.n_test, "t_end": a.t_end, "stride": a.stride, "coarsen": a.coarsen, "seed": a.seed,
             **({"disguise": a.seed} if a.disguise else {})}
            for f in files]


# ----------------------------------------------------------------------------- one problem (runs where the agent runs)
def solve_spec(spec, cfg):
    """One problem in a fresh, neutrally named work folder that is deleted afterwards (a reused container must not
    expose an earlier problem's files). Downloaded caches are deleted before the agent starts."""
    import shutil
    import uuid
    work = Path(f"/tmp/{uuid.uuid4().hex[:12]}")
    cfg = {**cfg, "workdir": str(work)}
    try:
        if spec["benchmark"] == "srsd":
            from eqdisc.srsd import FakeClient, load_problem, solve_problem
            cache = work / "c"
            prob = load_problem(spec["name"], spec["repo"], cache=cache)      # test split + truth -> memory
            shutil.rmtree(cache, ignore_errors=True)                           # ...and off the disk
            return solve_problem(prob, cfg, client=FakeClient() if cfg.get("dry_run") else None)
        from eqdisc.hf_well import solve_well
        return solve_well(spec, cfg)
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ----------------------------------------------------------------------------- provenance
def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def provenance(argv):
    sha = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain"))
    return {"code_commit": sha + ("-dirty" if dirty else ""), "code_branch": _git("branch", "--show-current"),
            "code_repo": "autoresearch-pde", "run_command": "run_bench.py " + " ".join(shlex.quote(x) for x in argv),
            "uncommitted_changes": dirty}


# ----------------------------------------------------------------------------- run state + outputs
class Run:
    def __init__(self, a, specs, argv, backend):
        self.a, self.specs, self.backend = a, specs, backend
        self.names = [s["id"] for s in specs]
        tag = a.target.split("/")[-1].replace("srsd-feynman_", "srsd_")
        skill_tag = "_noskills" if a.skills == "off" else (f"_skills-{'-'.join(a.skill_files)}" if a.skill_files else "")
        model_tag = "" if a.model == "claude-opus-5-5" else "_" + a.model.replace("claude-", "").replace("-", "")
        # seed + a short hash of the options keep ids unique when many runs start in the same second
        import hashlib
        opt = hashlib.sha1(" ".join(argv).encode()).hexdigest()[:4]
        agent_tag = "_bare" if a.agent == "bare" else ""
        if a.agent == "bare" and getattr(a, "bare_exact", False):
            agent_tag += "-exact"
        if getattr(a, "disguise", False):
            agent_tag += "_disg"
        if a.agent != "bare" and getattr(a, "harness_prompt", "full") != "full":
            agent_tag += "_hp-" + a.harness_prompt.replace("+", "-")
        self.id = f"{tag}_{time.strftime('%Y%m%d-%H%M%S')}" + agent_tag + skill_tag + model_tag + f"_s{a.seed}_{opt}" \
            + ("_dry" if a.dry_run else "")
        self.out = Path(a.out or ROOT / "runs" / self.id)
        (self.out / "problems").mkdir(parents=True, exist_ok=True)
        self.status = {n: "queued" for n in self.names}
        self.results = {}
        self.spent = 0.0
        self.state = "running"
        self.started = dt.datetime.now()
        self.prov = provenance(argv)
        self.lock = threading.Lock()
        self.write()

    def record(self, name, res):
        with self.lock:
            self.results[name] = res
            self.status[name] = "error" if res.get("error") else "done"
            self.spent += float(res.get("cost_usd") or 0.0)
            (self.out / "problems" / f"{name}.json").write_text(json.dumps(res, indent=1, default=str))
            try:
                (self.out / "problems" / f"{name}.html").write_text(attempt_page(self, name, res))
            except Exception as ex:  # noqa: BLE001  (a report bug must not lose the result)
                print(f"  attempt page for {name} failed: {ex}", flush=True)
            with open(self.out / "results.jsonl", "a") as f:
                f.write(json.dumps({k: v for k, v in res.items() if k != "log"}, default=str) + "\n")
            self.write()

    def summary(self):
        rs = [r for r in self.results.values() if not r.get("error")]
        ev = [r.get("eval") or {} for r in rs]
        err = sorted(e["test"]["rel_err_median"] for e in ev if (e.get("test") or {}).get("rel_err_median") is not None)
        return {"done": len(self.results), "total": len(self.names),
                "errors": sum(1 for r in self.results.values() if r.get("error")),
                "symbolic": sum(bool(e.get("symbolic_match")) for e in ev),
                "numeric": sum(bool(e.get("numeric_exact")) for e in ev),
                "median_err": err[len(err) // 2] if err else None, "spent": round(self.spent, 3),
                "fallbacks": sum(bool(r.get("fallback_submission")) for r in rs)}

    def write(self):
        (self.out / "dashboard.html").write_text(dashboard_html(self))
        (self.out / "summary.json").write_text(json.dumps({"run": self.id, "state": self.state, **self.summary(),
                                                           "config": vars(self.a), **self.prov}, indent=1, default=str))


def _fmt(x, nd=3):
    if x is None:
        return "–"
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    return str(x)


def _labels(kind):
    if kind == "video":
        return ("match", "numeric", "error", "No ground truth: this page logs the process only.")
    if kind == "well":
        return ("all closed equations match", "residual ≈ truth's", "weak residual",
                "Well: 'match' = every equation that closes in the stored fields is symbolically equivalent to the true "
                "one (constants within 5%); 'weak residual' = mismatch between the two sides of the equation after both "
                "are averaged against smooth space-time bumps on the hidden test trajectories, with the time derivative "
                "moved onto the bumps so the data are never differentiated in time; averaged over the scored equations. "
                "The true equations' value is the floor; 'residual ≈ truth's' = within 1.5x (or +0.02) of it.")
    return ("symbolic match", "numerically exact", "median rel. err",
            "SRSD: symbolic match = same structure as the truth with constants within 0.1%; numerically exact = "
            "median test rel. error < 1e-4 and p95 < 1e-3.")


def dashboard_html(run):
    a, s = run.a, run.summary()
    esc = html.escape
    l_match, l_num, l_err, l_note = _labels(a.kind)
    rows = []
    for n in run.names:
        r, st = run.results.get(n), run.status[n]
        if r is None:
            rows.append(f"<tr><td><code>{esc(n)}</code></td><td><span class='pill {st}'>{st}</span></td>" + "<td></td>" * 8 + "</tr>")
            continue
        e = r.get("eval") or {}
        t = e.get("test") or {}
        sub = (r.get("submitted") or {}).get("expr", "")
        mk = lambda v: "✓" if v else ("✗" if e else "–")
        sym, num = mk(e.get("symbolic_match")), mk(e.get("numeric_exact"))
        stop = r.get("error") or r.get("stop") or ""
        if r.get("fallback_submission"):
            stop = (stop + "; " if stop else "") + "auto-submitted best"
        extra = ""
        w = e.get("well")
        if w and w.get("scored"):
            extra = "<br>" + " · ".join(f"{esc(v)}: {'✓' if p['equivalent'] else '✗'} F1 {p['f1']:.2f} res {_fmt(p['test_residual'], 2)} "
                                        f"(truth {_fmt(p['truth_residual'], 2)})" for v, p in w["per_var"].items())
        status = "error" if r.get("error") else "done"
        rows.append(
            f"<tr><td><a href='problems/{esc(n)}.html'><code>{esc(n)}</code></a><br><span class='note'>"
            f"<a href='problems/{esc(n)}.html'>process ›</a></span></td><td><span class='pill {status}'>{status}</span></td>"
            f"<td class='c {'ok' if sym == '✓' else ''}'>{sym}</td><td class='c {'ok' if num == '✓' else ''}'>{num}</td>"
            f"<td class='r'>{_fmt(t.get('rel_err_median'))}</td><td class='r'>{r.get('n_tool_calls', '–')}"
            f"<div class='seq'>{esc(tool_sequence(r.get('log')))}</div></td>"
            f"<td class='r'>{_fmt(r.get('cost_usd'), 3)}</td><td class='r'>{_fmt(r.get('wall_s'), 4)}</td>"
            f"<td class='expr'><code>{esc(sub[:400])}</code><details><summary>truth</summary><code>{esc(str(r.get('truth', '')))}"
            f"</code></details><span class='note'>{extra}</span></td><td class='note'>{esc(str(stop)[:200])}</td></tr>")
    refresh = "<meta http-equiv='refresh' content='10'>" if run.state == "running" else ""
    ok_n = s["done"] - s["errors"]
    frac = lambda k: f"{s[k]}/{ok_n}" if ok_n else "–"
    cfg = (f"{esc(a.target)} · model {esc(a.model)} ({esc(a.effort)}) · ≤{a.max_tools} tools · ≤${a.max_cost_per_problem}/problem · "
           f"backend {esc(run.backend)} · agent {esc(a.agent)}{(' (prompt: ' + esc(a.bare_prompt) + (' + exactness instruction' if getattr(a, 'bare_exact', False) else '') + ')') if a.agent == 'bare' else ''}{(' (system prompt: ' + esc(a.harness_prompt) + ')') if a.agent != 'bare' and getattr(a, 'harness_prompt', 'full') != 'full' else ''} · skills {esc(a.skills_label)} · "
           f"{'with variable context' if a.context else 'blind'}"
           + (f" · {a.n_train} train / {a.n_test} test traj, t ≤ {a.t_end if a.t_end is not None else 'default'}, coarsen {a.coarsen}" if a.kind == "well" else "")
           + (" · <b>DRY RUN: scripted agent, costs are nominal</b>" if a.dry_run else ""))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
{refresh}<title>Benchmark {esc(run.id)}</title><style>
:root{{--bg:#fafaf9;--fg:#1c1917;--muted:#78716c;--card:#fff;--line:#e7e5e4;--ok:#15803d;--bad:#b91c1c;--run:#a16207}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0c0a09;--fg:#e7e5e4;--muted:#a8a29e;--card:#1c1917;--line:#292524;--ok:#4ade80;--bad:#f87171;--run:#facc15}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,sans-serif}}
main{{max-width:1300px;margin:0 auto;padding:20px 16px}} h1{{font-size:20px;margin:0 0 4px}} .sub{{color:var(--muted);margin-bottom:16px}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:18px}}
.tile{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}}
.tile .k{{color:var(--muted);font-size:12px}} .tile .v{{font-size:22px;font-weight:600;font-variant-numeric:tabular-nums}}
.wrap{{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:8px}}
table{{border-collapse:collapse;width:100%;min-width:1000px}} th,td{{padding:7px 9px;border-bottom:1px solid var(--line);vertical-align:top;text-align:left}}
th{{font-size:12px;color:var(--muted);font-weight:600}} td.r{{text-align:right;font-variant-numeric:tabular-nums}} td.c{{text-align:center}}
td.ok{{color:var(--ok);font-weight:700}} .seq{{color:var(--muted);font-size:11px;text-align:left;max-width:240px;margin-top:3px}} .expr code{{font-size:12px;word-break:break-all}} details{{margin-top:3px;color:var(--muted)}}
.note{{color:var(--muted);font-size:12px}} td.note{{max-width:220px}} .pill{{font-size:11px;padding:1px 7px;border-radius:9px;border:1px solid currentColor}}
.pill.done{{color:var(--ok)}} .pill.error{{color:var(--bad)}} .pill.running{{color:var(--run)}} .pill.queued,.pill.skipped{{color:var(--muted)}}
footer{{color:var(--muted);font-size:12px;margin-top:14px}}</style></head><body><main>
<h1>Benchmark <code>{esc(run.id)}</code> — {esc(run.state)}</h1><div class="sub">{cfg}</div>
<div class="tiles">
<div class="tile"><div class="k">Problems done</div><div class="v">{s['done']}/{s['total']}</div></div>
<div class="tile"><div class="k">{esc(l_match.capitalize())}</div><div class="v">{frac('symbolic')}</div></div>
<div class="tile"><div class="k">{esc(l_num.capitalize())}</div><div class="v">{frac('numeric')}</div></div>
<div class="tile"><div class="k">Median {esc(l_err)}</div><div class="v">{_fmt(s['median_err'], 2)}</div></div>
<div class="tile"><div class="k">Spent{' (nominal)' if a.dry_run else ''}</div><div class="v">${s['spent']:.2f}</div><div class="k">of ${a.total_budget:.2f} cap</div></div>
<div class="tile"><div class="k">Errors · auto-submits</div><div class="v">{s['errors']} · {s['fallbacks']}</div></div></div>
<div class="wrap"><table><thead><tr><th>problem</th><th>status</th><th>{esc(l_match)}</th><th>{esc(l_num)}</th><th>{esc(l_err)}</th>
<th>tools (sequence)</th><th>$</th><th>time s</th><th>submitted (truth on click)</th><th>stop / error</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div>
<footer>Click a problem for its step-by-step process (tool calls, results, reasoning summaries). Started {run.started:%Y-%m-%d %H:%M}. {esc(l_note)} Code {esc(run.prov['code_commit'][:16])} on
{esc(run.prov['code_branch'])}. Per-problem transcripts in <code>problems/</code>.{' Refreshes every 10 s.' if refresh else ''}</footer>
</main></body></html>"""


# ----------------------------------------------------------------------------- per-attempt process report
def _as_obj(x):
    if isinstance(x, str):
        try:
            return json.loads(x)
        except ValueError:
            return x
    return x


def tool_sequence(log):
    """Compressed tool sequence, e.g. 'diagnose → fit_skeleton ×3 → submit'."""
    names = [ev["name"] for ev in log or [] if ev.get("type") == "tool"]
    out = []
    for nm in names:
        if out and out[-1][0] == nm:
            out[-1][1] += 1
        else:
            out.append([nm, 1])
    return " → ".join(nm + (f" ×{k}" if k > 1 else "") for nm, k in out)


def _key_result(name, out):
    """One line summarising a tool result: the model it produced and how it validated, or the error."""
    o = _as_obj(out)
    if not isinstance(o, dict):
        return str(o)[:220]
    if o.get("error"):
        return "error: " + str(o["error"])[:200]
    if name == "submit":
        return "accepted" if o.get("ok") or o.get("rhs_original") else json.dumps(o)[:200]
    if name == "load_skill":
        return f"loaded ({len(str(o.get('skill', '')))} chars)"
    model = o.get("expr") or o.get("rhs_original") or o.get("rhs") or o.get("consensus_rhs")
    v = o.get("validation_original") or o.get("validation") or {}
    v = v if isinstance(v, dict) else {}
    bits = []
    if model:
        bits.append(str(model if isinstance(model, str) else json.dumps(model))[:220])
    for k, lab in (("rel_err_median", "val rel err"), ("r2", "R²"), ("deriv_nrmse", "deriv err"),
                   ("rollout_valid_time", "valid time")):
        if v.get(k) is not None:
            bits.append(f"{lab} {_fmt(v[k], 3)}")
    if name == "compare_models" and o.get("preferred"):
        bits.append(f"preferred: {o['preferred']} — {str(o.get('verdict', ''))[:140]}")
    if name in ("diagnose", "intuit", "dependence", "detect_symmetries", "find_invariants") and not bits:
        keys = ", ".join(list(o)[:6])
        bits.append(f"report ({keys})")
    return " · ".join(bits) or json.dumps(o, default=str)[:220]


def attempt_page(run, name, res):
    """Standalone HTML page for one attempt: outcome, process at a glance, and the full step-by-step timeline."""
    esc = html.escape
    a = run.a
    log = res.get("log") or []
    e = res.get("eval") or {}
    l_match, l_num, l_err, l_note = _labels(a.kind)
    tools = [ev for ev in log if ev.get("type") == "tool"]
    counts = {}
    for ev in tools:
        counts[ev["name"]] = counts.get(ev["name"], 0) + 1
    secs = {}
    for ev in tools:
        secs[ev["name"]] = secs.get(ev["name"], 0.0) + float(ev.get("seconds") or 0)
    steps, k = [], 0
    for ev in log:
        t = ev.get("type")
        if t == "thinking":
            txt = ev.get("text", "")
            steps.append(f"<div class='ev think'><div class='lab'>reasoning (summary)</div><div class='txt'>{esc(txt)}</div></div>")
        elif t == "text":
            steps.append(f"<div class='ev say'><div class='lab'>agent note</div><div class='txt'>{esc(ev.get('text', ''))}</div></div>")
        elif t == "tool":
            k += 1
            inp = ev.get("input")
            inp_s = inp if isinstance(inp, str) else json.dumps(inp, indent=1, default=str)
            out_s = ev.get("output")
            out_s = out_s if isinstance(out_s, str) else json.dumps(out_s, indent=1, default=str)
            dur = f" · {ev['seconds']} s" if ev.get("seconds") is not None else ""
            steps.append(
                f"<div class='ev tool'><div class='lab'>step {k}: <b>{esc(ev['name'])}</b>{dur}</div>"
                f"<div class='res'>{esc(_key_result(ev['name'], ev.get('output')))}</div>"
                f"<details><summary>input</summary><pre>{esc(str(inp_s)[:3000])}</pre></details>"
                f"<details><summary>output</summary><pre>{esc(str(out_s)[:3000])}</pre></details></div>")
    sub = res.get("submitted") or {}
    w = e.get("well") or {}
    per_var = ""
    if w.get("scored"):
        per_var = "<table><tr><th>equation</th><th>match</th><th>term F1</th><th>weak residual</th><th>truth's weak</th><th>pointwise residual</th><th>truth's pointwise</th><th>submitted</th><th>truth</th></tr>" + "".join(
            f"<tr><td><code>{esc(v)}</code>{(' <span class=sub>(provisional; ' + ('closes' if q.get('closes') else 'does not close, not in headline') + ')</span>') if q.get('provisional') else ''}</td>"
            f"<td>{'✓' if q['equivalent'] else '✗'}</td><td>{q['f1']:.2f}</td>"
            f"<td>{_fmt(q['test_residual'], 3)}</td><td>{_fmt(q['truth_residual'], 3)}</td>"
            f"<td>{_fmt(q.get('strong_residual'), 3)}</td><td>{_fmt(q.get('truth_strong_residual'), 3)}</td><td><code>{esc(q['submitted'])}</code></td>"
            f"<td><code>{esc(q['truth'])}</code></td></tr>" for v, q in w["per_var"].items()) + "</table>"
    critic = "".join(f"<pre>{esc(str(c)[:2000])}</pre>" for c in res.get("critic") or [])
    glance = " · ".join(f"{esc(nm)} ×{c} ({secs.get(nm, 0):.0f} s)" for nm, c in sorted(counts.items(), key=lambda x: -x[1]))
    n_think = sum(1 for ev in log if ev.get("type") == "thinking")
    mk = lambda v: "✓" if v else "✗"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(name)} attempt</title><style>
:root{{--bg:#fafaf9;--fg:#1c1917;--muted:#78716c;--card:#fff;--line:#e7e5e4;--ok:#15803d;--think:#f5f3ff;--tool:#f0f9ff}}
@media (prefers-color-scheme:dark){{:root{{--bg:#0c0a09;--fg:#e7e5e4;--muted:#a8a29e;--card:#1c1917;--line:#292524;--ok:#4ade80;--think:#1e1b2e;--tool:#0f1e2a}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,sans-serif}}
main{{max-width:1000px;margin:0 auto;padding:20px 16px}} h1{{font-size:20px;margin:0 0 6px}} h2{{font-size:16px;margin:22px 0 8px}}
.sub{{color:var(--muted)}} .card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px;margin:10px 0}}
code,pre{{font-size:12px}} pre{{white-space:pre-wrap;word-break:break-word;margin:6px 0}} table{{border-collapse:collapse;width:100%}}
td,th{{border-bottom:1px solid var(--line);padding:5px 7px;text-align:left;vertical-align:top}} th{{color:var(--muted);font-size:12px}}
.ev{{border-left:3px solid var(--line);padding:6px 10px;margin:8px 0;border-radius:0 6px 6px 0}}
.ev.think{{background:var(--think)}} .ev.tool{{background:var(--tool);border-left-color:#0284c7}} .ev.say{{border-left-color:#a16207}}
.lab{{font-size:12px;color:var(--muted)}} .txt{{white-space:pre-wrap}} .res{{font-family:ui-monospace,monospace;font-size:12px;word-break:break-word}}
a{{color:inherit}}</style></head><body><main>
<p class="sub"><a href="../dashboard.html">← run dashboard</a></p>
<h1><code>{esc(name)}</code></h1>
<div class="sub">{esc(a.target)} · skills: {esc(str(res.get('skills', a.skills_label)))} · model {esc(a.model)} ({esc(a.effort)}) ·
{res.get('n_tool_calls', '–')} tool calls · ${_fmt(res.get('cost_usd'), 3)} · {_fmt(res.get('wall_s'), 4)} s · stop: {esc(str(res.get('stop') or 'submitted'))}</div>
<div class="card"><b>Outcome.</b> {esc(l_match)}: {mk(e.get('symbolic_match'))} · {esc(l_num)}: {mk(e.get('numeric_exact'))} ·
{esc(l_err)}: {_fmt((e.get('test') or {}).get('rel_err_median'), 3)}{' · <b>error:</b> ' + esc(res['error']) if res.get('error') else ''}
<p><b>Submitted:</b> <code>{esc(str(sub.get('expr', '–')))}</code></p>
<p><b>Agent's rationale:</b> {esc(str(sub.get('rationale', '–')))}</p>
<details><summary>truth</summary><code>{esc(str(res.get('truth', '')))}</code></details>{per_var}</div>
<h2>Process at a glance</h2>
<div class="card"><p><b>Sequence:</b> {esc(tool_sequence(log)) or '–'}</p><p><b>Tool use:</b> {glance or '–'}</p>
<p class="sub">{n_think} reasoning summaries recorded. Each step below shows the tool, the key result (model found and its
validation score), and the full input/output on click.</p></div>
<h2>Step by step</h2>
{''.join(steps) or '<p class="sub">No transcript recorded.</p>'}
{'<h2>Critic review</h2><div class="card">' + critic + '</div>' if critic else ''}
<p class="sub">{esc(l_note)}</p>
</main></body></html>"""

# ----------------------------------------------------------------------------- driver
def drive(a, argv, solve, backend, specs=None):
    specs = enumerate_problems(a) if specs is None else specs
    run = Run(a, specs, argv, backend)
    cfg = {"model": a.model, "effort": a.effort, "max_tools": a.max_tools, "max_cost_usd": a.max_cost_per_problem,
           "context": a.context, "pysr_timeout_cap": a.pysr_timeout, "dry_run": a.dry_run, "skills": a.skills == "on", "skill_files": a.skill_files,
           "critic": a.critic == "on", "verbose": not a.quiet, "max_wall_s": a.max_minutes * 60,
           "agent": a.agent, "bare_prompt": a.bare_prompt, "harness_prompt": a.harness_prompt,
           "bare_exact": a.bare_exact}
    print(f"run {run.id}: {len(specs)} problems from {a.target}, backend {backend}, budget ${a.total_budget}")
    print(f"dashboard: {run.out / 'dashboard.html'}", flush=True)
    queue, running = list(specs), {}
    try:
        with ThreadPoolExecutor(max(1, a.workers)) as ex:
            while queue or running:
                # budget gate: start a problem only if we stay under the total even if every running one hits its cap
                while queue and len(running) < a.workers and \
                        run.spent + (len(running) + 1) * a.max_cost_per_problem <= a.total_budget + 1e-9:
                    spec = queue.pop(0)
                    run.status[spec["id"]] = "running"
                    run.write()
                    running[ex.submit(solve, spec, dict(cfg))] = spec["id"]   # solve_spec picks a neutral work folder
                if not running:
                    for sp in queue:
                        run.status[sp["id"]] = "skipped"
                    run.state = "stopped: budget cap"
                    print(f"budget cap reached: {len(queue)} problems not started", flush=True)
                    queue = []
                    break
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for fut in done:
                    name = running.pop(fut)
                    try:
                        res = fut.result()
                    except Exception as e:  # noqa: BLE001
                        res = {"problem": name, "error": f"{type(e).__name__}: {e}", "cost_usd": 0.0}
                    run.record(name, res)
                    e = res.get("eval") or {}
                    print(f"  {name}: match={e.get('symbolic_match')} numeric={e.get('numeric_exact')} "
                          f"${res.get('cost_usd', 0):.3f}  (spent ${run.spent:.2f})"
                          + (f" ERROR {res['error'][:160]}" if res.get("error") else ""), flush=True)
        if run.state == "running":
            run.state = "finished"
    except KeyboardInterrupt:
        run.state = "interrupted"
        print("interrupted; writing partial results", flush=True)
    run.write()
    s = run.summary()
    print(f"\n{run.state}: match {s['symbolic']}/{s['done'] - s['errors']}, numeric {s['numeric']}, errors {s['errors']}, "
          f"spent ${s['spent']:.2f}\ndashboard: {run.out / 'dashboard.html'}")
    return run


def list_benchmark(b):
    kind, target = resolve_benchmark(b)
    a = argparse.Namespace(kind=kind, target=target, problems=None, well_params=None, n=0, seed=0, n_train=0, n_test=0,
                           t_end=0, stride=1, coarsen=1)
    for s in enumerate_problems(a):
        print(s["id"])


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--list")
    known, _ = pre.parse_known_args(argv)
    if known.list:
        return list_benchmark(known.list)
    a = parse_args(argv)
    if not a.allow_local_download:
        raise SystemExit("The local backend reads benchmark data onto this machine. Run it on Modal instead:\n"
                         f"  modal run run_bench.py --args {shlex.quote(' '.join(argv))}\n"
                         "or pass --allow-local-download to run here anyway.")
    if not a.dry_run:
        from eqdisc.agent import make_client
        try:
            make_client()
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"Claude API credentials not found ({e}). Set ANTHROPIC_API_KEY, put it in .env, "
                             "or run `ant auth login`; or use --dry-run.")
    return drive(a, argv, solve_spec, "local")


# ----------------------------------------------------------------------------- Modal backend
try:
    import modal
except ImportError:          # Modal is optional; the local backend does not need it
    modal = None

if modal is not None:
    _secret = os.environ.get("EQDISC_MODAL_SECRET", "anthropic-api-key")
    image = (modal.Image.debian_slim(python_version="3.11")
             .pip_install("numpy>=1.26", "scipy>=1.11", "sympy>=1.12", "matplotlib>=3.8", "pandas>=2.0",
                          "pysindy>=2.0", "anthropic>=1.11", "pysr>=1.0", "h5py", "huggingface_hub", "fsspec")
             # install Julia + SymbolicRegression.jl and precompile at build time, not in every container
             .run_commands('python -c "import numpy as np; from pysr import PySRRegressor; '
                           'PySRRegressor(niterations=1, progress=False, verbosity=0).fit(np.random.rand(30, 2), np.random.rand(30))"')
             # ship eqdisc's non-Python files too (playbook.md, skills/*.md): the default uploads only *.py
             .add_local_python_source("eqdisc", ignore=["**/*.pyc", "**/tests/**"]))
    app = modal.App("eqdisc-bench", image=image)

    @app.function(secrets=[modal.Secret.from_name(_secret)] if _secret else [], timeout=6 * 3600, cpu=4.0, memory=16384)
    def solve_remote(spec, cfg):
        return solve_spec(spec, cfg)

    @app.function(timeout=300)
    def list_remote(argv):
        """Enumerate problems in the cloud too, so the local side needs only `modal` (no huggingface_hub)."""
        return enumerate_problems(parse_args(argv, read_skills=False))   # skill files are read locally only

    @app.local_entrypoint()
    def modal_main(args: str = ""):
        argv = shlex.split(args)
        a = parse_args(argv)
        drive(a, argv, lambda spec, cfg: solve_remote.remote(spec, cfg), "modal", specs=list_remote.remote(argv))


if __name__ == "__main__":
    main()
