"""Benchmark protocol (anti-overfitting): arms x datasets, with held-out blinded systems reported separately.

Arms: plain SINDy (fixed defaults) | auto (intuition config, no LLM) | agent (single session, no skills / memory /
context) | discover (full pipeline, subset). Development systems (used while building the tool) are labelled 'dev';
held-out blinded systems (renamed variables, perturbed coefficients, never used for tuning) are labelled 'held-out'.
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import toolbox as tb
from .agent import make_client, run_agent
from .autobase import auto_fit
from .evaluate import evaluate, load
from .judge import judge


def score(d, rhs, client):
    r = evaluate(d, {"rhs": rhs}, reveal=True)
    j = judge(r["truth"], r["rhs"], use_llm=True, client=client)
    return {"score": round(r["score"], 2), "equivalent": j["equivalent"], "rhs": r["rhs"]}


def run(dev, held, discover_subset=(), out="runs/benchmark_v1", workers=4, max_tools=18):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    client = make_client()
    sets = [(d, "dev") for d in dev] + [(d, "held-out") for d in held]
    rows = {}
    for d, split in sets:
        m, D = load(d)
        rows[d] = {"dataset": Path(d).name, "split": split,
                   "plain": score(d, tb.run_sindy(m, D)["rhs"], client),
                   "auto": score(d, auto_fit(m, D)["rhs"], client)}
        print(f"baselines done {Path(d).name}: plain {rows[d]['plain']['score']} auto {rows[d]['auto']['score']}", flush=True)

    def agent_arm(d):
        try:
            r = run_agent(d, client=client, verbose=False, max_tools=max_tools, use_memory=False,
                          out_dir=out / f"agent_{Path(d).name}", final_assessment=False, report=True)
            if not r.get("submitted"):
                return d, {"score": -10, "equivalent": False, "cost": r["cost_usd"]}
            s = score(d, r["submitted"]["rhs"], client)
            s.update({"cost": r["cost_usd"], "tool_calls": r["n_tool_calls"]})
            return d, s
        except Exception as e:  # noqa: BLE001
            return d, {"score": None, "error": str(e)[:200]}

    with ThreadPoolExecutor(workers) as ex:
        for d, s in ex.map(agent_arm, [d for d, _ in sets]):
            rows[d]["agent"] = s
            print(f"agent done {Path(d).name}: {s.get('score')} equivalent={s.get('equivalent')} ${s.get('cost')}", flush=True)
    from .orchestrate import discover
    for d in discover_subset:
        try:
            r = discover(d, n_branches=3, adversary=True, max_tools=max_tools, out_dir=out / f"discover_{Path(d).name}",
                         verbose=False)
            s = score(d, r["final_model"], client)
            s.update({"cost": r["cost_usd"], "verdict": r["verdict"]["status"]})
            rows[d]["discover"] = s
            print(f"discover done {Path(d).name}: {s['score']} {s['verdict']} ${s['cost']}", flush=True)
        except Exception as e:  # noqa: BLE001
            rows[d]["discover"] = {"score": None, "error": str(e)[:200]}
    (out / "results.json").write_text(json.dumps(rows, indent=1, default=str))
    lines = ["| split | dataset | plain SINDy | auto (no LLM) | agent | discover | agent $ |", "|---|---|---|---|---|---|---|"]

    def cell(x):
        if not x:
            return ""
        if x.get("score") is None:
            return "error"
        return f"{x['score']:.2f}{' ✓' if x.get('equivalent') else ''}"
    for r in rows.values():
        lines.append(f"| {r['split']} | {r['dataset']} | {cell(r['plain'])} | {cell(r['auto'])} | {cell(r.get('agent'))} | "
                     f"{cell(r.get('discover'))} | {r.get('agent', {}).get('cost', '')} |")
    for split in ("dev", "held-out"):
        rs = [r for r in rows.values() if r["split"] == split]
        if rs:
            def acc(arm):
                v = [r.get(arm) for r in rs if r.get(arm) and r[arm].get("score") is not None]
                return f"{sum(bool(x.get('equivalent')) for x in v)}/{len(v)}" if v else "-"
            lines.append(f"\n**{split}: symbolic matches (✓)**: plain {acc('plain')}, auto {acc('auto')}, agent {acc('agent')}, "
                         f"discover {acc('discover')}")
    total = sum((r.get("agent", {}).get("cost") or 0) + (r.get("discover", {}).get("cost") or 0) for r in rows.values())
    lines.append(f"\nTotal API cost ${total:.2f}. ✓ = symbolically equivalent to the hidden truth (sympy, else LLM judge).")
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return rows


HELD_OUT = ["strogatz_bacterial_respiration", "strogatz_bar_magnets", "strogatz_glider", "strogatz_lv_competition",
            "strogatz_predator_prey", "strogatz_shear_flow", "strogatz_damped_oscillator", "strogatz_growth",
            "rossler", "fisher_kpp_neumann", "burgers_dirichlet", "advection_diffusion_dirichlet"]
DEV = [("lorenz", 0.05, 1), ("kdv", 0.05, 2), ("kuramoto_sivashinsky", 0.05, 1), ("burgers", 0.05, 1)]


def build_sets(noise=0.02, seed=1):
    """Dev datasets (systems used while building the tool) + blinded held-out datasets (never tuned on)."""
    from .blind import make_blind
    from .datagen import generate
    dev = [str(generate(s, noise=n, dt_mult=k, plot=False)) for s, n, k in DEV]
    held = [str(make_blind(s, noise=noise, seed=seed)[0]) for s in HELD_OUT]
    return dev, held


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="runs/benchmark")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-discover", action="store_true", help="skip the (more expensive) full-pipeline arm")
    a = p.parse_args()
    dev, held = build_sets()
    sub = [] if a.no_discover else [d for d in held if any(k in d for k in ("glider", "predator_prey", "burgers_dirichlet"))]
    run(dev, held, sub, out=a.out, workers=a.workers)


if __name__ == "__main__":
    main()
