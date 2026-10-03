"""Benchmark the discovery agent across datasets (parallel sessions) and summarise.

    python -m eqdisc.bench datasets/{pendulum,sir,hopf}_n0.01* --workers 3 --max-tools 15
Writes runs/bench_<time>/summary.{json,md} plus one report.html per session.
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .agent import make_client, run_agent


def run_bench(datasets, out=None, workers=3, **agent_kw):
    out = Path(out or f"runs/bench_{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    client = make_client()

    def one(d):
        name = Path(d).name
        try:
            r = run_agent(d, client=client, out_dir=out / name, verbose=False, **agent_kw)
        except Exception as e:  # noqa: BLE001
            return {"dataset": name, "error": f"{type(e).__name__}: {e}"}
        h = r.get("hidden_eval") or {}
        row = {"dataset": name, "score": h.get("score"), "f1": h.get("f1"), "vf_nrmse": h.get("vf_nrmse"),
               "rollout_nrmse": h.get("rollout_nrmse"), "n_terms": h.get("n_terms"),
               "equivalent": (r.get("judge") or {}).get("equivalent"), "tool_calls": r["n_tool_calls"],
               "cost_usd": r["cost_usd"], "wall_s": r["wall_s"], "submitted": (r.get("submitted") or {}).get("rhs"),
               "truth": h.get("truth"), "report": r.get("report")}
        print(f"  done {name}: score={row['score']} equivalent={row['equivalent']} ${row['cost_usd']}", flush=True)
        return row

    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(one, datasets))
    (out / "summary.json").write_text(json.dumps(rows, indent=2, default=str))
    ok = [r for r in rows if r.get("score") is not None]
    lines = ["| dataset | score | sym. equiv. | F1 | terms | tool calls | cost $ |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['dataset']} | ERROR: {r['error'][:60]} | | | | | |")
        else:
            lines.append(f"| {r['dataset']} | {r['score']:.2f} | {r['equivalent']} | {r['f1']:.2f} | {r['n_terms']} | "
                         f"{r['tool_calls']} | {r['cost_usd']:.2f} |")
    if ok:
        lines.append(f"\n**symbolic accuracy {sum(bool(r['equivalent']) for r in ok)}/{len(ok)}**, mean score "
                     f"{sum(r['score'] for r in ok) / len(ok):.2f}, total cost ${sum(r['cost_usd'] for r in ok):.2f}")
    (out / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))
    return rows, out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("datasets", nargs="+")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--max-tools", type=int, default=20)
    p.add_argument("--effort", default="high")
    p.add_argument("--model", default="claude-opus-5-5")
    p.add_argument("--learn", action="store_true")
    p.add_argument("--no-critic", action="store_true")
    a = p.parse_args()
    run_bench(a.datasets, workers=a.workers, max_tools=a.max_tools, effort=a.effort, model=a.model,
              learn=a.learn, critic=not a.no_critic)


if __name__ == "__main__":
    main()
