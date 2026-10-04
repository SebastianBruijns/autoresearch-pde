"""Static symbolic-regression benchmarks for the eqdisc SR agent.

    python -m eqdisc.sr_bench llmsr  --data PATH_TO/LLM-SR/data --specs PATH_TO/LLM-SR/specs
    python -m eqdisc.sr_bench srsd   --data PATH_TO/srsd-feynman_hard --srsd-repo PATH_TO/srsd-benchmark [--limit N]

LLM-SR (Shojaee et al., ICLR 2025): 4 problems (oscillator1/2, bactgrow, stressstrain); metric NMSE on in-domain (ID)
and out-of-domain (OOD) test sets. The problem description from LLM-SR's own spec files is given, as in their protocol.
Their train.csv is split 90/10 into train/validation for our agent.

SRSD-Feynman (Matsubara et al.): train/val/test text files, last column = target, variables x0..x{n-1}. The agent
gets NO variable descriptions or units (the published baselines don't). Metrics: R2 > 0.999 accuracy on test,
solution rate (SRBench: estimate - truth or estimate / truth simplifies to a constant), and normalised tree edit
distance (NED), computed with the official srsd-benchmark code.
"""
import argparse
import json
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp

from .sr import SRTask, evaluate_expr, nmse, r2, solve

LLMSR_PROBLEMS = {"oscillator1": "y", "oscillator2": "y", "bactgrow": "y", "stressstrain": "y"}


# ----------------------------------------------------------------------------- LLM-SR
def llmsr_task(data_dir, specs_dir, name, val_frac=0.1, seed=0):
    d = Path(data_dir) / name
    tr = pd.read_csv(d / "train.csv")
    names, target = list(tr.columns[:-1]), tr.columns[-1]
    X, y = tr[names].values.astype(float), tr[target].values.astype(float)
    idx = np.random.default_rng(seed).permutation(len(y))
    nv = int(len(y) * val_frac)
    desc = ""
    spec = Path(specs_dir) / f"specification_{name}_numpy.txt"
    if spec.exists():
        desc = spec.read_text().split('"""')[1].strip()
    task = SRTask(X[idx[nv:]], y[idx[nv:]], X[idx[:nv]], y[idx[:nv]], names, desc, target)
    tests = {}
    for split in ("test_id", "test_ood"):
        te = pd.read_csv(d / f"{split}.csv")
        tests[split] = (te[names].values.astype(float), te[target].values.astype(float))
    return task, tests


def run_llmsr(data_dir, specs_dir, out, n_sessions=2, max_tools=20, problems=None):
    rows = {}
    for name in problems or LLMSR_PROBLEMS:
        task, tests = llmsr_task(data_dir, specs_dir, name)
        t0 = time.time()
        res = solve(task, n_sessions=n_sessions, max_tools=max_tools)
        row = {"problem": name, "expr": res["expr"], "cost_usd": res["cost_usd"], "wall_s": round(time.time() - t0)}
        for split, (Xt, yt) in tests.items():
            row[split] = nmse(yt, evaluate_expr(res["expr"], task.names, Xt)) if res["expr"] else None
        rows[name] = row
        print(json.dumps(row), flush=True)
        (out / f"llmsr_{name}.json").write_text(json.dumps({**row, "candidates": res.get("candidates")}, default=str, indent=1))
    return rows


# ----------------------------------------------------------------------------- SRSD
def srsd_task(root, key):
    root = Path(root)
    load = lambda s: np.loadtxt(root / s / f"{key}.txt")
    tr, va, te = load("train"), load("val"), load("test")
    n = tr.shape[1] - 1
    names = [f"x{i}" for i in range(n)]
    task = SRTask(tr[:, :n], tr[:, n], va[:, :n], va[:, n], names, "", "y")
    return task, (te[:, :n], te[:, n])


def srsd_truth(root, key):
    with open(Path(root) / "true_eq" / f"{key}.pkl", "rb") as f:
        return pickle.load(f)


def _round_floats(e, sig=6):
    e = sp.sympify(e).evalf()
    return e.xreplace({f: sp.Float(f, sig) for f in e.atoms(sp.Float)})


def solution_check(est_str, gt, names, X=None, sig=6):
    """SRBench solution criterion (symbolic): est - gt or est / gt simplifies to a constant. Floats in both are
    rounded to `sig` significant figures first, so that e.g. 1/299792458 and 3.33564095198152e-9 match. There is
    no numeric fallback: an expression that is only numerically indistinguishable over the sampled range (e.g. a
    small-angle limit) is NOT a solution."""
    syms = {n: sp.Symbol(n, real=True) for n in names}
    try:
        est = _round_floats(sp.sympify(est_str, locals=syms), sig)
        g = _round_floats(sp.sympify(str(gt), locals=syms), sig)
        for expr in (sp.simplify(est - g), sp.simplify(est / g)):
            expr = _round_floats(expr, sig - 2)
            if not (expr.free_symbols & set(syms.values())):
                return True
    except Exception:  # noqa: BLE001
        pass
    return False


def _srsd_canonical(eq):
    """Exactly srsd-benchmark's load_eq_as_tree normalisation (eq_comparator.py)."""
    eq = sp.sympify(str(eq))
    eq = eq.subs(sp.pi, sp.pi.evalf()).evalf().factor().simplify().subs(1.0, 1)
    return sp.sympify(str(eq))


def _zss_tree(eq):
    """srsd-benchmark's sympy2zss_module: constants -> 'Const', symbols by name, functions by class name."""
    from sympy.utilities.misc import func_name
    from zss import Node
    label = "Const" if eq.is_number else (str(eq) if isinstance(eq, sp.Symbol) else func_name(eq))
    node = Node(label)
    for c in eq.args:
        node.addkid(_zss_tree(c))
    return node


def _count(n):
    return 1 + sum(_count(c) for c in n.children)


def _ned_worker(est_str, gt_pkl_path, q):
    from zss import simple_distance
    with open(gt_pkl_path, "rb") as f, sp.evaluate(False):
        gt = pickle.load(f)
    gt_t = _zss_tree(_srsd_canonical(gt))
    names = sorted({str(x) for x in sp.sympify(est_str).free_symbols})
    est = sp.sympify(est_str, locals={n: sp.Symbol(n, real=True) for n in names})
    est_t = _zss_tree(_srsd_canonical(est))
    n_gt = _count(gt_t)
    q.put(min(simple_distance(est_t, gt_t), n_gt) / n_gt)


def ned(est_str, gt_pkl_path, srsd_repo=None, timeout=120):
    """Normalised tree edit distance, reimplementing srsd-benchmark's eq_comparator (Zhang-Shasha via `zss`, same
    sympy canonicalisation, normalised by the true tree size, capped at 1; 1.0 on failure or timeout)."""
    import multiprocessing as mp
    q = mp.Queue()
    p = mp.Process(target=_ned_worker, args=(est_str, str(gt_pkl_path), q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        return 1.0
    return float(q.get()) if not q.empty() else 1.0


def run_srsd(root, srsd_repo, out, n_sessions=2, max_tools=20, limit=None, workers=3):
    keys = sorted(p.stem for p in (Path(root) / "true_eq").glob("*.pkl"))[:limit]

    def one(key):
        task, (Xt, yt) = srsd_task(root, key)
        gt = srsd_truth(root, key)
        t0 = time.time()
        try:
            res = solve(task, n_sessions=n_sessions, max_tools=max_tools)
        except Exception as e:  # noqa: BLE001
            return key, {"problem": key, "error": str(e)[:200]}
        row = {"problem": key, "expr": res["expr"], "truth": str(gt), "cost_usd": res["cost_usd"],
               "wall_s": round(time.time() - t0)}
        if res["expr"]:
            yhat = evaluate_expr(res["expr"], task.names, Xt)
            row["test_r2"] = r2(yt, yhat)
            row["acc_r2_0999"] = bool(row["test_r2"] > 0.999)
            row["solution"] = solution_check(res["expr"], gt, task.names, Xt)
            row["ned"] = ned(res["expr"], Path(root) / "true_eq" / f"{key}.pkl", srsd_repo)
        (out / f"srsd_{key}.json").write_text(json.dumps({**row, "candidates": res.get("candidates")}, default=str, indent=1))
        print(json.dumps({k: row.get(k) for k in ("problem", "test_r2", "acc_r2_0999", "solution", "ned", "cost_usd")}),
              flush=True)
        return key, row

    with ThreadPoolExecutor(workers) as ex:
        rows = dict(ex.map(one, keys))
    ok = [r for r in rows.values() if "error" not in r]
    summ = {"n": len(rows), "accuracy_r2>0.999": np.mean([bool(r.get("acc_r2_0999")) for r in ok]),
            "solution_rate": np.mean([bool(r.get("solution")) for r in ok]),
            "mean_NED": np.mean([r.get("ned", 1.0) for r in ok]), "cost_usd": sum(r.get("cost_usd", 0) for r in ok)}
    (out / "srsd_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print("SUMMARY", json.dumps(summ, default=float))
    return rows, summ


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bench", choices=["llmsr", "srsd"])
    p.add_argument("--data", required=True)
    p.add_argument("--specs", help="LLM-SR specs dir (problem descriptions)")
    p.add_argument("--srsd-repo", help="clone of github.com/omron-sinicx/srsd-benchmark (for NED)")
    p.add_argument("--out", default="runs/sr_bench")
    p.add_argument("--sessions", type=int, default=2)
    p.add_argument("--max-tools", type=int, default=20)
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--problems", nargs="*")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.bench == "llmsr":
        run_llmsr(a.data, a.specs, out, a.sessions, a.max_tools, a.problems)
    else:
        run_srsd(a.data, a.srsd_repo, out, a.sessions, a.max_tools, a.limit, a.workers)


if __name__ == "__main__":
    main()
