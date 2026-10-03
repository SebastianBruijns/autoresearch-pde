"""Generate train / hidden-test datasets for ODE & PDE discovery.

Layout of a dataset directory:
    data.npz        public: t, U (+ x for PDEs). Noisy, possibly subsampled.
    meta.json       public: variable names, grid, shapes. No equations.
    hidden/test.npz clean trajectories from *unseen* initial conditions (scoring only)
    hidden/truth.json ground-truth equations + solver settings (scoring only)

U has shape (n_traj, nt, n_vars) for ODEs, (n_traj, nt, nx, n_fields) for 1-D PDEs and
(n_traj, nt, nx, ny, n_fields) for 2-D PDEs (coordinates x[, y] stored in data.npz; grid, boundary
type and allowed symbols in meta.json -- conventions documented in solvers.py).

Examples:
    python -m eqdisc.datagen --system lorenz --noise 0.01
    python -m eqdisc.datagen --system kdv --noise 0.05 --dt-mult 5 --blind
    python -m eqdisc.datagen --system pendulum --noise 0.05 --noise-type red
    python -m eqdisc.datagen --suite noise_ladder      # every system at 0, 1, 5, 10% noise
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .solvers import derivative_symbols, grid_coords, integrate_ode, integrate_pde_general
from .systems import SYSTEMS


def simulate(sys, n_traj, rng, t_end=None):
    t_end = sys.t_end if t_end is None else t_end
    t = np.round(np.arange(0, t_end + 1e-9, sys.dt), 10)
    trajs = []
    tries = 0
    while len(trajs) < n_traj:
        tries += 1
        if tries > 50 * n_traj:
            raise RuntimeError(f"{sys.name}: could not generate {n_traj} valid trajectories")
        if sys.kind == "ode":
            U = integrate_ode(sys.variables, sys.rhs, sys.ic(rng), t)
        else:
            lay = sys.layout()
            U = integrate_pde_general(sys.fields, sys.rhs, lay, sys.ic(rng, *grid_coords(lay)), t, sys.dt_sim,
                                      max_seconds=600.0)
        if np.all(np.isfinite(U)) and (sys.kind != "ode" or sys.valid is None or sys.valid(U)):
            trajs.append(U)
    return t, np.stack(trajs)


NOISE_TYPES = ("gaussian", "multiplicative", "red", "outliers")


def add_noise(U, level, rng, kind="gaussian"):
    """Measurement noise scaled to `level` x the per-variable std of the clean data.

    gaussian       white, additive
    multiplicative U * (1 + level*eps)   (keeps positive quantities mostly positive)
    red            additive AR(1) noise in time (rho=0.9): defeats naive smoothing
    outliers       gaussian + 1% of samples replaced by spikes of +-10 x level x std
    """
    if level <= 0:
        return U.copy()
    scale = U.reshape(-1, U.shape[-1]).std(axis=0)
    eps = rng.standard_normal(U.shape)
    if kind == "gaussian":
        return U + level * scale * eps
    if kind == "multiplicative":
        return U * (1 + level * eps)
    if kind == "red":
        rho = 0.9
        red = np.empty_like(eps)
        red[:, 0] = eps[:, 0]
        for i in range(1, U.shape[1]):
            red[:, i] = rho * red[:, i - 1] + np.sqrt(1 - rho ** 2) * eps[:, i]
        return U + level * scale * red
    if kind == "outliers":
        out = U + level * scale * eps
        mask = rng.random(U.shape) < 0.01
        return np.where(mask, out + 10 * level * scale * rng.choice([-1, 1], U.shape), out)
    raise ValueError(f"unknown noise type {kind!r}; choose from {NOISE_TYPES}")


def generate(system, out_root="datasets", noise=0.0, noise_type="gaussian", n_traj=None, n_test=None, t_end=None,
             dt_mult=1, seed=0, blind=False, name=None, plot=True):
    sys = SYSTEMS[system]
    rng = np.random.default_rng(seed)
    n_traj = n_traj or (4 if sys.kind == "ode" else 3)
    n_test = n_test or (4 if sys.kind == "ode" else 2)

    t, U = simulate(sys, n_traj, rng, t_end)
    t_test, U_test = simulate(sys, n_test, rng, t_end)
    U_obs = add_noise(U, noise, rng, noise_type)[:, ::dt_mult]
    t_obs = t[::dt_mult]

    nt_tag = "" if noise_type == "gaussian" or noise == 0 else noise_type[:4]
    tag = f"{system}_n{noise:g}{nt_tag}_dt{dt_mult}_s{seed}"
    if name is None:
        name = "mystery_" + hashlib.sha1(tag.encode()).hexdigest()[:6] if blind else tag
    d = Path(out_root) / name
    (d / "hidden").mkdir(parents=True, exist_ok=True)

    names = sys.variables if sys.kind == "ode" else sys.fields
    meta = {
        "name": name,
        "kind": sys.kind,
        "variables": names,
        "dt": float(t_obs[1] - t_obs[0]),
        "n_traj": int(U_obs.shape[0]),
        "shape": list(U_obs.shape),
        "shape_doc": "(n_traj, nt, n_vars)" if sys.kind == "ode" else "(n_traj, nt, nx, n_fields)",
        "system": None if blind else system,
    }
    arrays = {"t": t_obs, "U": U_obs}
    test = {"t": t_test, "U": U_test}
    if sys.kind == "pde":
        lay = sys.layout()
        dims = lay["spatial_dims"]
        coords = grid_coords(lay)
        if len(dims) > 1:
            meta["shape_doc"] = "(n_traj, nt, " + ", ".join(f"n{d}" for d in dims) + ", n_fields)"
        meta.update({"L": sys.L, "nx": sys.nx, "boundary": lay["boundary"],
                     "spatial_dims": dims, "grid": lay["grid"],
                     "allowed_symbols": derivative_symbols(sys.fields, dims, 4)})
        for dim, c in zip(dims, coords):
            arrays[dim] = test[dim] = c
    else:
        meta["allowed_symbols"] = list(sys.variables) + ["t"]

    np.savez_compressed(d / "data.npz", **arrays)
    np.savez_compressed(d / "hidden" / "test.npz", **test)
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    truth = {"system": system, "kind": sys.kind, "variables": names, "rhs": sys.rhs,
             "noise": noise, "noise_type": noise_type, "dt_mult": dt_mult, "seed": seed, "eval_horizon": sys.eval_horizon,
             "tags": list(sys.tags)}
    if sys.kind == "pde":
        truth.update({"L": sys.L, "dt_sim": sys.dt_sim, "nx": sys.nx, "spatial_dims": lay["spatial_dims"],
                      "boundary": lay["boundary"], "grid": lay["grid"]})
    (d / "hidden" / "truth.json").write_text(json.dumps(truth, indent=2))
    if plot:
        _plot(d, meta, arrays)
    return d


def _plot(d, meta, arrays):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    U, t = arrays["U"], arrays["t"]
    if meta["kind"] == "ode":
        fig, ax = plt.subplots(U.shape[-1], 1, figsize=(8, 1.8 * U.shape[-1]), sharex=True, squeeze=False)
        for i, v in enumerate(meta["variables"]):
            for j in range(U.shape[0]):
                ax[i, 0].plot(t, U[j, :, i], lw=0.8)
            ax[i, 0].set_ylabel(v)
        ax[-1, 0].set_xlabel("t")
    elif len(meta.get("spatial_dims", ["x"])) == 2:
        nf = U.shape[-1]
        times = np.linspace(0, len(t) - 1, 4).astype(int)
        fig, ax = plt.subplots(nf, len(times), figsize=(3.2 * len(times), 3 * nf), squeeze=False)
        for i, f in enumerate(meta["variables"]):
            for j, k in enumerate(times):
                ax[i, j].pcolormesh(arrays["x"], arrays["y"], U[0, k, :, :, i].T, shading="auto", cmap="RdBu_r")
                ax[i, j].set(title=f"{f}, t={t[k]:g}", aspect="equal")
    else:
        nf = U.shape[-1]
        fig, ax = plt.subplots(1, nf, figsize=(5 * nf, 3.5), squeeze=False)
        for i, f in enumerate(meta["variables"]):
            im = ax[0, i].pcolormesh(arrays["x"], t, U[0, :, :, i], shading="auto", cmap="RdBu_r")
            ax[0, i].set(xlabel="x", ylabel="t", title=f"{f}(x,t), traj 0")
            fig.colorbar(im, ax=ax[0, i])
    fig.suptitle(meta["name"])
    fig.tight_layout()
    fig.savefig(d / "preview.png", dpi=110)
    plt.close(fig)


SUITES = {
    # (system, noise, noise_type, dt_mult)
    "default": [(s, 0.01, "gaussian", 1) for s in SYSTEMS],
    "clean": [(s, 0.0, "gaussian", 1) for s in SYSTEMS],
    "noise_ladder": [(s, n, "gaussian", 1) for s in SYSTEMS for n in (0.0, 0.01, 0.05, 0.1)],
    "noise_types": [(s, 0.05, k, 1) for s in SYSTEMS for k in NOISE_TYPES],
    "hard": [("lorenz", 0.05, "gaussian", 2), ("pendulum", 0.05, "red", 1),
             ("michaelis_menten", 0.02, "multiplicative", 1), ("kdv", 0.05, "gaussian", 2),
             ("kuramoto_sivashinsky", 0.02, "outliers", 1), ("fitzhugh_nagumo", 0.05, "gaussian", 1)],
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--system", choices=sorted(SYSTEMS))
    p.add_argument("--suite", choices=sorted(SUITES))
    p.add_argument("--noise", type=float, default=0.0, help="relative noise level (x per-variable std)")
    p.add_argument("--noise-type", default="gaussian", choices=NOISE_TYPES)
    p.add_argument("--n-traj", type=int)
    p.add_argument("--n-test", type=int)
    p.add_argument("--t-end", type=float)
    p.add_argument("--dt-mult", type=int, default=1, help="keep every k-th snapshot (temporal subsampling)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--blind", action="store_true", help="hide the system name (mystery_xxxxxx)")
    p.add_argument("--out", default="datasets")
    p.add_argument("--list", action="store_true")
    a = p.parse_args()
    if a.list:
        for k, s in SYSTEMS.items():
            print(f"{k:22s} {s.kind}  {s.tags}\n    " + "\n    ".join(f"d{v}/dt = {e}" for v, e in s.rhs.items()))
        return
    jobs = SUITES[a.suite] if a.suite else [(a.system, a.noise, a.noise_type, a.dt_mult)]
    for system, noise, kind, dtm in jobs:
        d = generate(system, a.out, noise, kind, a.n_traj, a.n_test, a.t_end, dtm, a.seed, a.blind)
        print("wrote", d)


if __name__ == "__main__":
    main()
