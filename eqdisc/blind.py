"""Blinded benchmark variants, to measure DISCOVERY rather than recall of textbook equations.

For a registered system: rename variables (x1, x2, ... for ODEs; PDE fields to q, q2) and perturb every term
coefficient by an independent random factor in [1 - spread, 1 + spread] (structure unchanged), then generate
data with the usual hidden test split. Truth is stored as usual for scoring.

    python -m eqdisc.blind strogatz_glider --noise 0.02 --seed 1
"""
import argparse
import dataclasses
import hashlib

import numpy as np
import sympy as sp

from .datagen import generate
from .solvers import parse, pde_symbols
from .systems import SYSTEMS


def _perturb(expr, rng, spread):
    out = sp.Integer(0)
    for term in sp.Add.make_args(expr):
        c, m = term.as_coeff_Mul()
        if c.is_number and float(c) != 0.0:            # every coefficient, including +-1, gets its own factor
            c = sp.Float(float(c) * rng.uniform(1 - spread, 1 + spread), 4)
        out += c * m
    return out


def make_blind(system, noise=0.02, seed=0, spread=0.25, out_root="datasets", noise_type="gaussian"):
    sys = SYSTEMS[system]
    rng = np.random.default_rng(int(hashlib.sha1(f"{system}-{seed}".encode()).hexdigest()[:8], 16))
    if sys.kind == "ode":
        old = list(sys.variables)
        new = [f"x{i + 1}" for i in range(len(old))]
        names = old + ["t"]
    else:
        old = list(sys.fields)
        new = ["q"] if len(old) == 1 else [f"q{i + 1}" for i in range(len(old))]
        names = pde_symbols(old) + ["x", "y"]
    ren = {}
    for o, n in zip(old, new):
        ren[sp.Symbol(o)] = sp.Symbol(f"__{n}")
        if sys.kind == "pde":
            for s in names:
                if s.startswith(o + "_"):
                    ren[sp.Symbol(s)] = sp.Symbol(f"__{n}_{s.split('_', 1)[1]}")
    rhs = {}
    for o, n in zip(old, new):
        e = sp.expand(parse(sys.rhs[o], names)) if "/" not in sys.rhs[o] else parse(sys.rhs[o], names)
        e = _perturb(sp.expand(e), rng, spread)
        e = e.xreplace(ren)
        rhs[n] = str(e).replace("__", "")
    tag = hashlib.sha1(f"{system}{seed}{spread}".encode()).hexdigest()[:6]
    key = f"blind_{tag}"
    fields = {"rhs": rhs, "name": key}
    if sys.kind == "ode":
        fields["variables"] = new
    else:
        fields["fields"] = new
    SYSTEMS[key] = dataclasses.replace(sys, **fields)
    try:
        d = generate(key, out_root, noise=noise, noise_type=noise_type, seed=seed, blind=True,
                     name=f"blind_{system}_n{noise:g}_s{seed}", plot=False)
    finally:
        pass
    return d, rhs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("system")
    p.add_argument("--noise", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--spread", type=float, default=0.25)
    a = p.parse_args()
    d, rhs = make_blind(a.system, a.noise, a.seed, a.spread)
    print(d, rhs)


if __name__ == "__main__":
    main()
