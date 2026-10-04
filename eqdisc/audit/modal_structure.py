"""Run the structural calibration (calibrate_structure.run_one) on Modal: one container per
(system, noise, seed), each generating its own dataset. No LLM calls; only Modal compute.

    .venv/bin/modal run eqdisc/audit/modal_structure.py --seeds 0,1,2 --out runs/structure_calibration

    .venv/bin/modal run eqdisc/audit/modal_structure.py --split heldout --out runs/structure_heldout

dev:     the six registered systems (seeds 0-2). Thresholds and rules were developed on these only.
heldout: run ONCE, never tuned on: (a) four structures not used in development (HELDOUT_SYSTEMS, defined
         here so the shared systems registry is untouched) and (b) the six dev structures with every coefficient
         perturbed by an independent factor in [0.75, 1.25] (eqdisc.blind._perturb), on seeds 10-12.
"""
import json
from pathlib import Path

import modal

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("numpy==2.3.*", "scipy==1.16.*", "sympy==1.14.*", "matplotlib==3.10.*", "pandas", "pysindy")
         .add_local_python_source("eqdisc"))
app = modal.App("eqdisc-structure-calibration", image=image)

SYSTEMS = ("advection_diffusion", "burgers", "kdv", "kuramoto_sivashinsky", "allen_cahn", "fisher_kpp")
HELDOUT_SYSTEMS = ("kdv_burgers", "gardner", "swift_hohenberg", "airy_advection")
NOISES = (0.0, 0.02, 0.05)


def _register(system, seed, perturb):
    """Add a held-out system (or a coefficient-perturbed copy of a dev system) to the registry in this process."""
    import dataclasses
    import numpy as np
    import sympy as sp
    from eqdisc.blind import _perturb
    from eqdisc.solvers import parse, pde_symbols
    from eqdisc.systems import PDESystem, SYSTEMS, _fourier_ic, _kdv_solitons
    held = {
        "kdv_burgers": PDESystem("kdv_burgers", ["u"], {"u": "-u*u_x + 0.02*u_xx - 0.01*u_xxx"},
                                 L=2 * np.pi, nx=256, ic=_fourier_ic(1.0, 3), dt=0.02, t_end=4, dt_sim=2e-4,
                                 eval_horizon=4),
        "gardner": PDESystem("gardner", ["u"], {"u": "-6*u*u_x + 3*u**2*u_x - u_xxx"},
                             L=40.0, nx=256, ic=_kdv_solitons, dt=0.1, t_end=10, dt_sim=1e-3, eval_horizon=10),
        "swift_hohenberg": PDESystem("swift_hohenberg", ["u"], {"u": "0.2*u - u**3 - 2*u_xx - u_xxxx"},
                                     L=32 * np.pi, nx=256, ic=_fourier_ic(0.3, 12), dt=0.5, t_end=60, dt_sim=0.05,
                                     eval_horizon=20),
        "airy_advection": PDESystem("airy_advection", ["u"], {"u": "-0.5*u_x - 0.1*u_xxx"},
                                    L=2 * np.pi, nx=128, ic=_fourier_ic(1.0, 5), dt=0.02, t_end=4, dt_sim=1e-3,
                                    eval_horizon=4),
    }
    if system in held:
        SYSTEMS[system] = held[system]
        return system
    if not perturb:
        return system
    base = SYSTEMS[system]
    rng = np.random.default_rng(1000 + seed)
    names = pde_symbols(base.fields) + ["x"]
    rhs = {f: str(_perturb(sp.expand(parse(base.rhs[f], names)), rng, 0.25)) for f in base.fields}
    key = f"{system}_perturbed"
    SYSTEMS[key] = dataclasses.replace(base, name=key, rhs=rhs)
    return key


@app.function(cpu=2.0, memory=4096, timeout=1800)
def run_case(system: str, noise: float, seed: int, perturb: bool = False):
    import tempfile
    import warnings
    warnings.filterwarnings("ignore")
    from eqdisc.audit.calibrate_structure import run_one
    from eqdisc.datagen import generate
    system = _register(system, seed, perturb)
    d = generate(system, out_root=tempfile.mkdtemp(), noise=noise, seed=seed, plot=False)
    try:
        return run_one(str(d))
    except Exception as e:  # noqa: BLE001
        return {"dataset": f"{system}_n{noise:g}_s{seed}", "error": f"{type(e).__name__}: {e}"}


@app.local_entrypoint()
def main(split: str = "dev", seeds: str = "", out: str = "runs/structure_calibration", systems: str = ""):
    from eqdisc.audit.calibrate_structure import ARMS, summarise
    if split == "dev":
        ks = [int(k) for k in (seeds or "0,1,2").split(",")]
        assert all(k < 10 for k in ks), "dev seeds are 0-9"
        cases = [(s, n, k, False) for k in ks for s in (systems.split(",") if systems else SYSTEMS) for n in NOISES]
    else:
        ks = [int(k) for k in (seeds or "10,11,12").split(",")]
        cases = [(s, n, k, False) for k in ks for s in HELDOUT_SYSTEMS for n in NOISES] + \
                [(s, n, k, True) for k in ks for s in SYSTEMS for n in NOISES]
    rows, errors = [], []
    for r in run_case.starmap(cases, order_outputs=False):
        if r is None:
            continue
        if "error" in r:
            errors.append(r)
            print("ERROR", r)
            continue
        rows.append(r)
        print(f"{r['dataset']:40s} " + " ".join(f"{k}={r[k]['f1']:.2f}{'*' if r[k]['exact'] else ' '}"
                                                 for k in ARMS)
              + f" false_alarms={r['truth_false_alarms']} {r['seconds']}s", flush=True)
    rows.sort(key=lambda r: r["dataset"])
    s = summarise(rows) if rows else {}
    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    (o / "rows.json").write_text(json.dumps(rows + errors, indent=1, default=str))
    (o / "summary.json").write_text(json.dumps(s, indent=2))
    if split != "dev" and rows:
        s["by_group"] = {g: summarise([r for r in rows if (r["system"] in HELDOUT_SYSTEMS) == (g == "new_structures")])
                         for g in ("new_structures", "perturbed_dev_structures")}
    print(json.dumps(s, indent=2))
