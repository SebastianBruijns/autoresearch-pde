"""Tests / demo for eqdisc.uq (ensemble SINDy, CV, model comparison, coefficient bootstrap).

    python -m eqdisc.tests.test_uq            # full table over 5 datasets
    python eqdisc/tests/test_uq.py            # same
    pytest eqdisc/tests/test_uq.py            # assertions only

Hidden truth is read HERE ONLY (to build a 'truth' candidate and to score with
evaluate(..., reveal=True)); eqdisc.uq itself never touches hidden/.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from eqdisc import toolbox as tb  # noqa: E402
from eqdisc import uq  # noqa: E402
from eqdisc.evaluate import evaluate, load  # noqa: E402

DS = ROOT / "datasets"
DATASETS = {   # name -> library options shared by plain SINDy and the ensemble
    "lorenz_n0.05_dt1_s0": {},
    "sir_n0.05_dt1_s0": {},
    "pendulum_n0.05red_dt1_s0": {"include_trig": True},
    "kdv_n0.05_dt1_s0": {},
    "vanderpol_n0.1_dt1_s0": {},
}


def _truth(name):
    return json.loads((DS / name / "hidden" / "truth.json").read_text())["rhs"]


def _true_terms(meta, rhs):
    return {v: {tm for tm, _ in tl} for v, tl in uq._structure(meta, rhs).items()}


def _json_ok(x):
    json.dumps(x)
    return True


# ----------------------------------------------------------------------------- (a) lorenz inclusion
def test_lorenz_inclusion():
    name = "lorenz_n0.05_dt1_s0"
    meta, data = load(DS / name)
    t0 = time.time()
    r = uq.ensemble_sindy(meta, data, n_models=50)
    dt = time.time() - t0
    assert _json_ok(r) and dt < 30
    true = _true_terms(meta, _truth(name))
    # library term strings ('x*y') -> same canonical form as sympy ('x*y')
    import sympy as sp
    names = tb.symbols(meta)
    canon = lambda s: str(sp.expand(uq.parse(s, names)))
    incl = {v: {canon(tm): row[0] for tm, row in r["terms"][v].items()} for v in meta["variables"]}
    true_p = {(v, tm): incl[v].get(tm, 0.0) for v in true for tm in true[v]}
    spur_p = {(v, tm): p for v in incl for tm, p in incl[v].items() if tm not in true[v]}
    print(f"[a] lorenz E-SINDy ({dt:.2f}s): true-term inclusion {true_p}")
    print(f"    spurious terms with p>=0.1: {spur_p}")
    # 6 of the 7 true terms are found with p ~ 1; the weak -y term in y' (|contribution| ~ 1/28 of x)
    # is replaced by nothing at 5% noise (see report), every spurious term stays below 0.3.
    assert sum(p >= 0.9 for p in true_p.values()) >= 6
    assert all(p < 0.3 for p in spur_p.values())


# ----------------------------------------------------------------------------- (b)+(c) per dataset
def run_dataset(name, lib):
    meta, data = load(DS / name)
    row = {"dataset": name}
    t0 = time.time()
    plain = tb.run_sindy(meta, data, **lib)
    row["t_sindy"] = time.time() - t0
    over = tb.run_sindy(meta, data, thresholds=(1e-4,), **lib)          # deliberately over-fit
    t0 = time.time()
    ens = uq.ensemble_sindy(meta, data, n_models=50, **lib)
    row["t_ens"] = time.time() - t0
    assert _json_ok(ens)
    truth = _truth(name)
    cands = {"truth": truth, "overfit": over["rhs"], "sindy": plain["rhs"], "consensus": ens["consensus_rhs"]}
    t0 = time.time()
    cmp_ = uq.compare_models(meta, data, cands)
    row["t_cmp"] = time.time() - t0
    assert _json_ok(cmp_)
    row["preferred"] = cmp_["preferred"]
    row["verdict"] = cmp_["verdict"]
    row["floor_ratio"] = cmp_["error_to_floor_ratio"]
    rk = cmp_["ranking"]
    row["truth_beats_overfit"] = rk.index("truth") < rk.index("overfit")
    row["n_terms"] = {k: cmp_["candidates"][k]["n_terms"] for k in cands}
    row["cv"] = {k: cmp_["candidates"][k]["cv_deriv_nrmse"] for k in cands}
    # (c) hidden scores (test only)
    for k, rhs in (("sindy", plain["rhs"]), ("consensus", ens["consensus_rhs"])):
        ev = evaluate(str(DS / name), {"rhs": rhs}, reveal=True)
        row[f"score_{k}"] = ev["score"]
        row[f"f1_{k}"] = ev["f1"]
    # coefficient bootstrap on the true structure: do the 90% intervals cover the true values?
    cu = uq.coefficient_uncertainty(meta, data, truth, n_boot=100)
    assert _json_ok(cu)
    cov = [d["ci90"][0] <= d["given"] <= d["ci90"][1] for v in cu["coefs"].values() for d in v.values()]
    row["ci_cover"] = f"{sum(cov)}/{len(cov)}"
    row["max_rel_ci"] = max(d["rel_ci_halfwidth"] for v in cu["coefs"].values() for d in v.values())
    return row


def test_cross_validate_modes():
    meta, data = load(DS / "lorenz_n0.05_dt1_s0")
    truth = _truth("lorenz_n0.05_dt1_s0")
    fixed = uq.cross_validate(meta, data, truth)
    assert fixed["folds"] == "trajectory" and fixed["n_folds"] == 4 and _json_ok(fixed)
    fit = uq.cross_validate(meta, data, lambda m, d: tb.run_sindy(m, d))
    assert "term_stability" in fit and _json_ok(fit)
    # single trajectory -> time blocks, both modes
    m1 = dict(meta, n_traj=1, shape=[1] + meta["shape"][1:])
    d1 = {"U": data["U"][:1], "t": data["t"]}
    f1 = uq.cross_validate(m1, d1, truth)
    g1 = uq.cross_validate(m1, d1, lambda m, d: tb.run_sindy(m, d))
    assert f1["folds"] == "time_blocks" and g1["folds"] == "time_blocks"
    assert all("deriv_nrmse" in f for f in g1["per_fold"]), g1
    print(f"[cv] lorenz fixed truth: deriv {fixed['deriv_nrmse_mean']:.3f}, valid t "
          f"{fixed.get('rollout_valid_time_mean')}; fitter jaccard "
          f"{fit['term_stability']['mean_pairwise_jaccard']}, identical "
          f"{fit['term_stability']['identical_structure_all_folds']}; 1-traj fitter jaccard "
          f"{g1['term_stability']['mean_pairwise_jaccard']}")



def test_compare_models_prefers_truth():
    wins = 0
    for name in ("lorenz_n0.05_dt1_s0", "sir_n0.05_dt1_s0", "vanderpol_n0.1_dt1_s0"):
        meta, data = load(DS / name)
        over = tb.run_sindy(meta, data, thresholds=(1e-4,))
        c = uq.compare_models(meta, data, {"truth": _truth(name), "overfit": over["rhs"]}, rollouts=False)
        wins += c["preferred"] == "truth"
    assert wins >= 2


def main():
    test_lorenz_inclusion()
    test_cross_validate_modes()
    rows = []
    for name, lib in DATASETS.items():
        t0 = time.time()
        r = run_dataset(name, lib)
        r["t_total"] = time.time() - t0
        rows.append(r)
        print(f"\n== {name}  ({r['t_total']:.1f}s)\n   {r['verdict']}")
    hdr = ("dataset", "pref", "truth>over", "n_terms t/o/s/c", "floor_ratio", "score sindy", "score cons",
           "F1 sindy", "F1 cons", "CI cover", "t_ens", "t_cmp")
    print("\n| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
    for r in rows:
        nt = r["n_terms"]
        print(f"| {r['dataset']} | {r['preferred']} | {r['truth_beats_overfit']} | "
              f"{nt['truth']}/{nt['overfit']}/{nt['sindy']}/{nt['consensus']} | {r['floor_ratio']:.2f} | "
              f"{r['score_sindy']:.3f} | {r['score_consensus']:.3f} | {r['f1_sindy']:.2f} | "
              f"{r['f1_consensus']:.2f} | {r['ci_cover']} | {r['t_ens']:.1f}s | {r['t_cmp']:.1f}s |")
    n_ok = sum(r["preferred"] != "overfit" and r["truth_beats_overfit"] for r in rows)
    print(f"\ncompare_models prefers truth over over-fit SINDy on {n_ok}/{len(rows)} systems")
    assert n_ok >= 2


if __name__ == "__main__":
    main()
