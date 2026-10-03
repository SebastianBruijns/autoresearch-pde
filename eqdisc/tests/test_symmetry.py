"""Tests / demo for eqdisc.symmetry (symmetry detection, equivariant SINDy) and run_pysr templates.

    python -m eqdisc.tests.test_symmetry            # symmetry detection + equivariant SINDy table
    python -m eqdisc.tests.test_symmetry --pysr     # also the PySR template test (compiles Julia, ~2 min)
    pytest eqdisc/tests/test_symmetry.py            # assertions only (PySR test skipped unless EQDISC_PYSR=1)

Hidden truth is used HERE ONLY (evaluate(..., reveal=True)); eqdisc.symmetry never touches hidden/.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from eqdisc import symmetry as sym  # noqa: E402
from eqdisc import toolbox as tb  # noqa: E402
from eqdisc.evaluate import evaluate, load  # noqa: E402

DS = ROOT / "datasets"
KEYS = ["score", "vf_nrmse", "rollout_nrmse", "valid_frac", "n_terms", "f1", "coef_rel_err"]


def _ds(name):
    d = DS / name
    if not d.exists():
        system, rest = name.split("_n")
        noise = rest.split("_")[0]
        subprocess.run([sys.executable, "-m", "eqdisc.datagen", "--system", system, "--noise", noise],
                       cwd=ROOT, check=True)
    return load(d)


def _found(rep, text):
    return any(text in d["transform"] for d in rep["discrete"]["found"] if d.get("is_symmetry"))


def _recognised(rep):
    return [g["generator"] for g in rep["continuous"]["known_generators_in_symmetry_subspace"]]


def _short(e):
    return {k: (round(e[k], 4) if isinstance(e.get(k), float) else e.get(k)) for k in KEYS}


# ----------------------------------------------------------------------------- ODE detection
def test_hopf_rotation():
    rep = sym.detect_linear_symmetries(*_ds("hopf_n0.01_dt1_s0"))
    assert rep["continuous"]["n_symmetries"] == 1, rep["summary"]
    assert _recognised(rep) == ["rotation in (x, y)"], rep["summary"]
    assert _found(rep, "(-x, -y)")
    return rep


def test_lorenz_z_axis_flip():
    rep = sym.detect_linear_symmetries(*_ds("lorenz_n0.01_dt1_s0"))
    assert rep["continuous"]["n_symmetries"] == 0, rep["summary"]
    found = [d["transform"] for d in rep["discrete"]["found"] if d.get("is_symmetry")]
    assert found == ["sign flip ('x', 'y', 'z') -> (-x, -y, z)"], found
    return rep


def test_duffing_parity():
    rep = sym.detect_linear_symmetries(*_ds("duffing_n0.01_dt1_s0"))
    assert rep["continuous"]["n_symmetries"] == 0, rep["summary"]
    assert _found(rep, "(-x, -y)")
    # damping breaks the time-reversal symmetry (x, y) -> (x, -y), t -> -t
    assert not any(d.get("is_reversing_symmetry") for d in rep["discrete"]["found"])
    return rep


def test_lotka_volterra_no_hallucination():
    rep = sym.detect_linear_symmetries(*_ds("lotka_volterra_n0.01_dt1_s0"))
    assert rep["continuous"]["n_symmetries"] == 0, rep["summary"]
    assert rep["continuous"]["n_inconclusive"] == 0
    assert not rep["discrete"]["found"], rep["discrete"]["found"]
    rep_aff = sym.detect_linear_symmetries(*_ds("lotka_volterra_n0.01_dt1_s0"), affine=True, discrete=False)
    assert rep_aff["continuous"]["n_symmetries"] == 0, rep_aff["summary"]
    rep["affine_summary"] = rep_aff["summary"]
    return rep


# ----------------------------------------------------------------------------- PDE detection
def test_kdv():
    rep = sym.detect_pde_symmetries(*_ds("kdv_n0.01_dt1_s0"))
    assert rep["translation"]["translation_invariant"]
    refl = {r["transform"]: r["verdict"] for r in rep["reflection"]}
    assert refl["x -> -x, (u) -> (u)"].startswith("REVERSING"), refl
    gal = rep["shift_galilean"][0]
    assert gal["verdict"].startswith("GALILEAN"), gal
    assert 4.5 < abs(gal["frame_velocity_per_unit_shift"]) < 7.5, gal       # truth: 6
    assert rep["field_linear_generators"]["n_symmetries"] == 0                 # nonlinear
    return rep


def test_pde_controls():
    """Burgers: x->-x with u->-u and Galilean; advection-diffusion: linear (scaling) + shift."""
    b = sym.detect_pde_symmetries(*_ds("burgers_n0.01_dt1_s0"))
    assert any(r["verdict"] == "SYMMETRY" and r["transform"] == "x -> -x, (u) -> (-u)" for r in b["reflection"])
    assert b["shift_galilean"][0]["verdict"].startswith("GALILEAN")
    a = sym.detect_pde_symmetries(*_ds("advection_diffusion_n0.01_dt1_s0"))
    assert a["field_linear_generators"]["n_symmetries"] == 1
    assert a["shift_galilean"][0]["verdict"].startswith("SHIFT")
    return {"burgers": b["summary"], "advection_diffusion": a["summary"]}


# ----------------------------------------------------------------------------- equivariant SINDy
def compare_hopf(name="hopf_n0.05_dt1_s0"):
    meta, data = _ds(name)
    G = [sym.rotation_generator(2, 0, 1)]
    runs = {
        "run_sindy (default)": lambda: tb.run_sindy(meta, data),
        "equivariant_sindy, deriv selection": lambda: sym.equivariant_sindy(meta, data, generators=G),
        "unconstrained, rollout selection": lambda: sym.equivariant_sindy(meta, data, generators=[],
                                                                          selection="rollout"),
        "equivariant_sindy, rollout selection": lambda: sym.equivariant_sindy(meta, data, generators=G,
                                                                              selection="rollout"),
    }
    rows = {}
    for k, f in runs.items():
        t0 = time.time()
        r = f()
        e = evaluate(DS / name, {"rhs": r["rhs"]}, reveal=True)
        rows[k] = {"rhs": r["rhs"], **_short(e), "free_params": r.get("n_free_params"),
                   "seconds": round(time.time() - t0, 1)}
    return rows


def test_equivariant_sindy_hopf():
    rows = compare_hopf("hopf_n0.05_dt1_s0")
    eq = rows["equivariant_sindy, rollout selection"]
    assert eq["free_params"] == 4                      # SO(2)-equivariant cubic fields: (a + b r^2) z + (c + d r^2) iz
    assert eq["f1"] == 1.0 and eq["valid_frac"] == 1.0
    assert eq["score"] > rows["run_sindy (default)"]["score"] + 1.0
    rows10 = compare_hopf("hopf_n0.1_dt1_s0")
    assert rows10["equivariant_sindy, rollout selection"]["score"] > rows10["unconstrained, rollout selection"]["score"]
    return {"hopf_n0.05": rows, "hopf_n0.1": rows10}


def test_equivariant_constraint_exact():
    meta, data = _ds("lorenz_n0.1_dt1_s0")
    r = sym.equivariant_sindy(meta, data, discrete=[np.diag([-1.0, -1.0, 1.0])])
    assert r["n_free_params"] == 30 and r["constraint_residual"] < 1e-10
    e = evaluate(DS / "lorenz_n0.1_dt1_s0", r, reveal=True)
    p = tb.run_sindy(meta, data)
    ep = evaluate(DS / "lorenz_n0.1_dt1_s0", p, reveal=True)
    return {"equivariant": {"rhs": r["rhs"], **_short(e)}, "plain": {"rhs": p["rhs"], **_short(ep)}}


# ----------------------------------------------------------------------------- PySR template
def test_pysr_template_pendulum():
    if os.environ.get("EQDISC_PYSR") != "1" and "--pysr" not in sys.argv:
        return {"skipped": "set EQDISC_PYSR=1 or pass --pysr"}
    meta, data = _ds("pendulum_n0.01_dt1_s0")
    t0 = time.time()
    r = tb.run_pysr(meta, data, "omega", template="f(theta) + g(omega)", unary_operators=("sin", "cos", "exp"),
                    niterations=40, timeout=180, base_rhs={"theta": "omega"}, model_selection="rollout")
    e = evaluate(DS / "pendulum_n0.01_dt1_s0", r, reveal=True)
    out = {"rhs": r["rhs"], "template": r["template"], **_short(e), "seconds": round(time.time() - t0, 1),
           "selected_complexity": r.get("selected_complexity"),
           "pareto": [(q["complexity"], q.get("rollout_nrmse_full"), q["raw"]) for q in r["pareto_front"]]}
    assert "sin(theta)" in r["rhs"]["omega"] and "omega" in r["rhs"]["omega"], r["rhs"]     # damping kept
    assert e["vf_nrmse"] < 0.05 and e["valid_frac"] > 0.9, e
    return out


def main():
    results = {}
    for fn in [test_hopf_rotation, test_lorenz_z_axis_flip, test_duffing_parity, test_lotka_volterra_no_hallucination,
               test_kdv, test_pde_controls, test_equivariant_sindy_hopf, test_equivariant_constraint_exact,
               test_pysr_template_pendulum]:
        t0 = time.time()
        try:
            r = fn()
            status = "PASS"
        except AssertionError as ex:
            r, status = {"assertion": str(ex)[:500]}, "FAIL"
        name = fn.__name__
        print(f"\n=== {name}: {status} ({time.time() - t0:.1f}s)")
        if isinstance(r, dict) and "summary" in r:
            print("  summary:", json.dumps(r["summary"]))
            if "continuous" in r:
                print("  continuous rel. errors:", r["continuous"]["singular_values_rel"],
                      "| first generator:", {k: r["continuous"]["generators"][0][k]
                                             for k in ("A", "rel_equivariance_error", "noise_level", "nearest")})
            if "affine_summary" in r:
                print("  affine:", r["affine_summary"])
            if "shift_galilean" in r:
                print("  reflection:", [(x["transform"], x["rel_err_equivariant"], x["rel_err_time_reversal"],
                                         x["verdict"]) for x in r["reflection"]])
                print("  shift/galilean:", r["shift_galilean"])
        else:
            print(json.dumps(r, indent=1, default=str)[:3000])
        results[name] = status
    print("\n", results)


if __name__ == "__main__":
    main()
