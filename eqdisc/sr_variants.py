"""Fresh, unpublished LLM-SR-style oscillator problems (contamination control for LLM-based SR).

Each variant is a driven, damped nonlinear oscillator x'' = f(t, x, v) built from a random combination of term
families (none of the exact equations appear in any paper), simulated and split as in LLM-SR:
train and in-domain test sampled from one time window, out-of-domain test from another window with larger amplitude.

    python -m eqdisc.sr_variants make --n 4 --seed 7      # -> datasets/osc_variants/<name>/{train,test_id,test_ood}.csv
    python -m eqdisc.sr_variants run                       # agent vs PySR vs sparse baseline, NMSE ID/OOD
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sympy as sp
from scipy.integrate import solve_ivp

RESTORING = ["x", "x**3", "sin(x)", "tanh(2*x)", "x/(1 + x**2)", "x*Abs(x)", "x*exp(-x**2)"]
DAMPING = ["v", "v**3", "v*Abs(v)", "x**2*v", "x*v", "v*cos(x)"]
FORCING = ["sin(w*t)", "cos(w*t)"]
DESC = ("Find the mathematical function skeleton that represents acceleration in a damped nonlinear oscillator "
        "system{forced}, given data on {inputs}.")


def make_variant(rng, idx):
    nr, nd = rng.integers(1, 3), rng.integers(1, 3)
    restoring = list(rng.choice(RESTORING, nr, replace=False))
    damping = list(rng.choice(DAMPING, nd, replace=False))
    forced = bool(rng.random() < 0.6)
    terms = []
    for t in restoring:
        terms.append(f"-{rng.uniform(0.3, 2.0):.3g}*{t}")
    for t in damping:
        terms.append(f"-{rng.uniform(0.05, 0.6):.3g}*{t}")
    if forced:
        f = rng.choice(FORCING).replace("w", f"{rng.uniform(0.5, 2.0):.3g}")
        terms.append(f"{rng.uniform(0.1, 0.5):.3g}*{f}")
    expr = " ".join(terms).replace(" -", " - ").replace("- -", "+ ")
    expr = " + ".join(terms).replace("+ -", "- ")
    return {"name": f"osc_v{idx}", "expr": expr, "forced": forced}


def simulate(expr, forced, rng):
    t, x, v = sp.symbols("t x v")
    f = sp.lambdify((t, x, v), sp.sympify(expr, locals={"Abs": sp.Abs}), "numpy")
    x0, v0 = rng.uniform(0.6, 1.2) * rng.choice([-1, 1]), rng.uniform(-0.5, 0.5)
    tt = np.linspace(0, 50, 50001)
    sol = solve_ivp(lambda s, y: [y[1], f(s, y[0], y[1])], (0, 50), [x0, v0], t_eval=tt, rtol=1e-10, atol=1e-12)
    X, V = sol.y
    A = np.array([f(a, b, c) for a, b, c in zip(tt, X, V)], dtype=float)
    return tt, X, V, A


def make(n=4, seed=7, out="datasets/osc_variants", max_tries=50):
    rng = np.random.default_rng(seed)
    out = Path(out)
    made = []
    idx = 0
    tries = 0
    while len(made) < n and tries < max_tries:
        tries += 1
        var = make_variant(rng, idx)
        tt, X, V, A = simulate(var["expr"], var["forced"], rng)
        if not np.all(np.isfinite(A)) or np.abs(X).max() > 20:
            continue
        late, early = tt >= 30, tt < 20
        # need non-trivial late dynamics and a clearly larger early (OOD) amplitude, as in LLM-SR
        if X[late].std() < 0.02 or np.ptp(X[early]) < 1.3 * np.ptp(X[late]):
            continue
        d = out / var["name"]
        d.mkdir(parents=True, exist_ok=True)
        cols = (["t"] if var["forced"] else []) + ["x", "v"]
        df = pd.DataFrame({"t": tt, "x": X, "v": V, "a": A})
        r = np.random.default_rng(seed + idx)
        late_df, early_df = df[late], df[early]
        tr_idx = r.permutation(len(late_df))
        late_df.iloc[tr_idx[:10000]][cols + ["a"]].to_csv(d / "train.csv", index=False)
        late_df.iloc[tr_idx[10000:20000]][cols + ["a"]].to_csv(d / "test_id.csv", index=False)
        early_df.sample(10000, random_state=seed + idx)[cols + ["a"]].to_csv(d / "test_ood.csv", index=False)
        desc = DESC.format(forced=" with driving force" if var["forced"] else "",
                           inputs=("time, " if var["forced"] else "") + "position, and velocity")
        (d / "description.txt").write_text(desc)
        (d / "truth.json").write_text(json.dumps(var, indent=1))
        made.append(var)
        print(var["name"], "|", var["expr"])
        idx += 1
    return made


def run(root="datasets/osc_variants", out="runs/sr_variants", sessions=2, max_tools=20, pysr_timeout=300):
    from .isolate import run_isolated
    from .sr import evaluate_expr, nmse, solve, sparse_fit
    from .sr_bench import llmsr_task
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for d in sorted(Path(root).iterdir()):
        if not (d / "train.csv").exists():
            continue
        task, tests = llmsr_task(root, root, d.name)
        task.description = (d / "description.txt").read_text()
        truth = json.loads((d / "truth.json").read_text())["expr"]

        def score(expr):
            if not expr:
                return {"ID": None, "OOD": None}
            return {k.replace("test_", "").upper(): nmse(yt, evaluate_expr(expr, task.names, Xt)) for k, (Xt, yt) in tests.items()}
        row = {"problem": d.name, "truth": truth}
        # 1) sparse baseline: generic polynomial (deg 3) + trig library, no LLM
        n = task.names
        lib = [f"{a}" for a in n] + [f"{a}*{b}" for i, a in enumerate(n) for b in n[i:]] + \
              [f"{a}*{b}*{c}" for i, a in enumerate(n) for j, b in enumerate(n[i:], i) for c in n[j:]] + \
              [f"sin({a})" for a in n] + [f"cos({a})" for a in n] + ["1"]
        sf = sparse_fit(task, lib)
        row["sparse"] = {"expr": sf.get("expr"), **score(sf.get("expr"))}
        # 2) PySR (same operator set as LLM-SR's PySR baseline: + - * / sin cos exp)
        pr = run_isolated("eqdisc.sr", "sr_pysr", {"names": task.names}, {"X": task.X_train, "y": task.y_train},
                          {"timeout": pysr_timeout, "niterations": 200, "unary_operators": ["sin", "cos", "exp"]},
                          timeout=pysr_timeout + 240)
        pe = pr.get("best") if isinstance(pr, dict) else None
        row["pysr"] = {"expr": pe, **score(pe)}
        # 3) our agent
        res = solve(task, n_sessions=sessions, max_tools=max_tools)
        row["agent"] = {"expr": res["expr"], **score(res["expr"]), "cost_usd": res["cost_usd"]}
        row["truth_score"] = score(truth)
        rows.append(row)
        print(json.dumps({"problem": d.name, "sparse": row["sparse"], "pysr": row["pysr"],
                          "agent": {k: row["agent"][k] for k in ("ID", "OOD", "cost_usd")}}, default=str), flush=True)
        (out / f"{d.name}.json").write_text(json.dumps(row, indent=1, default=str))
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["make", "run"])
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    make(a.n, a.seed) if a.cmd == "make" else run()
