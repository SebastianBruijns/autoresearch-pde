"""Corruption benchmark for the evidence layer (WS5).

Each case is a normal eqdisc dataset directory (works with `evaluate.load` / `evaluate.evaluate`) built from one of
four 1-D periodic base PDEs with 2% Gaussian noise and exactly one corruption:

    id            how                                                       expected detector(s)
    clean         none                                                      none
    outliers      datagen.add_noise(kind="outliers")                        outliers
    gaps_random   whole frames NaN in random time windows (~10% of time)    gaps
    gaps_state    NaN where local amplitude is in the top ~20% (censoring)  gaps_state_dependent
    forcing_time  rhs + A*sin(w*t)                                          residual_time_only, slice_time
    source_space  rhs + A*cos(2*pi*x/L)                                     residual_space_only, slice_space
    traj_coeffs   each trajectory's coefficients perturbed (blind._perturb)  slice_trajectory
    amp_term      rhs + c*u**3, IC amplitudes so it matters on the largest   slice_amplitude, residual_amplitude

Event corruptions (WS8, written to their own suite `datasets/corrupt_events/`, never mixed into the main suite):

    glitch        2-4 single-sample spikes per trajectory (single grid points) outliers (despike removes exactly them)
                  of 8-15 x the noise sd: measurement errors, no dynamics
    kick          at t_k (40-60% of the run) a smooth localized bump is added    external_shock or nothing; never
                  to the TRUE state and the same equation keeps integrating        outliers, nothing despiked
                  (a dynamical impulse the system remembers)

`hidden/truth.json` holds the BASE equation (what a correct discovery should recover). `hidden/corruption.json`
holds {"id", "params", "expected_detector", ...} plus the generating right-hand sides and diagnostics. The hidden test
set is simulated from unseen ICs with the corrupted dynamics where the corruption is in the dynamics (forcing_time,
source_space, amp_term), with the base dynamics for traj_coeffs, and is always clean (no noise, NaN or outliers).
The first frame of every training trajectory is never NaN'd (keeps an initial condition for rollouts).

Dataset names are opaque (`case_<hash>`) and meta["system"] is None, so nothing public names the system or the
corruption (same convention as datagen's blinded `mystery_<hash>` names). The mapping is in `<out_root>/index.json`.

    python -m eqdisc.corrupt --seeds 0 1 2            # dev suite (calibration)    -> datasets/corrupt/dev/
    python -m eqdisc.corrupt --seeds 10 11 12         # reporting suite            -> datasets/corrupt/report/
    python -m eqdisc.corrupt --seeds 10 11 12 --blind # blinded reporting variants -> datasets/corrupt/blind/
    python -m eqdisc.corrupt --events --seeds 0 1 2 10 11 12   # glitch / kick suite -> datasets/corrupt_events/
"""
import argparse
import dataclasses
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.ndimage import uniform_filter

from .blind import _perturb
from .datagen import add_noise
from .solvers import derivative_symbols, grid_coords, integrate_pde_general, make_pde_rhs, parse, pde_symbols
from .systems import SYSTEMS

BASE_SYSTEMS = ("advection_diffusion", "burgers", "kdv", "kuramoto_sivashinsky")
CORRUPTIONS = ("clean", "outliers", "gaps_random", "gaps_state", "forcing_time", "source_space", "traj_coeffs",
               "amp_term")
EXPECTED = {
    "clean": [],
    "outliers": ["outliers"],
    "gaps_random": ["gaps"],
    "gaps_state": ["gaps_state_dependent"],
    "forcing_time": ["residual_time_only", "slice_time"],
    "source_space": ["residual_space_only", "slice_space"],
    "traj_coeffs": ["slice_trajectory"],
    "amp_term": ["slice_amplitude", "residual_amplitude"],
}
EVENT_CORRUPTIONS = ("glitch", "kick")
EXPECTED.update({"glitch": ["outliers"], "kick": ["external_shock"]})
# what a correct audit does with them (hidden/corruption.json["expected_handling"])
HANDLING = {"glitch": {"outliers_fires": True, "despike_removes": "exactly the glitch samples"},
            "kick": {"outliers_fires": False, "despike_removes": "nothing",
                     "acceptable": ["external_shock", "nothing"]}}
DYNAMIC = ("forcing_time", "source_space", "amp_term")       # hidden test uses the corrupted dynamics

# Per-system corruption parameters. Forcing / source amplitudes are ~0.1-0.4 x rms of the base rhs (lower where the
# response is near-resonant: KdV, KS source at k=1); w gives 2-3 periods
# over the record. amp_term: c*u**3 with negative c (stabilising); IC amplitude scales so the term matters mainly on
# the largest training trajectory (KS forgets its IC amplitude, so there the term acts at the extremes of the attractor).
PARAMS = {
    "advection_diffusion": {"forcing_time": {"A": 0.2, "w": 3.1416}, "source_space": {"A": 0.2},
                            "amp_term": {"c": -0.5, "scales": [0.4, 0.5, 1.6], "test_scales": [0.6, 1.4]}},
    "burgers": {"forcing_time": {"A": 0.08, "w": 3.1416}, "source_space": {"A": 0.06},
                "amp_term": {"c": -0.12, "scales": [0.4, 0.5, 1.6], "test_scales": [0.6, 1.4]}},
    "kdv": {"forcing_time": {"A": 0.06, "w": 1.2566}, "source_space": {"A": 0.05},
            "amp_term": {"c": -0.4, "scales": [0.4, 0.5, 1.6], "test_scales": [0.6, 1.4]}},
    "kuramoto_sivashinsky": {"forcing_time": {"A": 0.1, "w": 0.1885}, "source_space": {"A": 0.05},
                             "amp_term": {"c": -0.01, "scales": [0.4, 0.5, 1.6], "test_scales": [0.6, 1.4]}},
}
COMMON = {"gaps_random": {"frac": 0.10, "n_windows": 3}, "gaps_state": {"quantile": 0.80, "window": [3, 5]},
          "traj_coeffs": {"spread": 0.20, "min_range": 0.15}, "outliers": {}, "clean": {},
          "glitch": {"n_per_traj": [2, 4], "amp_noise_sd": [8.0, 15.0], "min_sep": 10, "edge": 8},
          "kick": {"t_frac": [0.4, 0.6], "amp_std": [0.5, 1.0], "width_frac": 1 / 16}}
N_TRAIN, N_TEST, NOISE = 3, 2, 0.02


def case_params(system, corruption):
    return dict(PARAMS[system].get(corruption) or COMMON[corruption])


def _case_name(system, corruption, seed, blind):
    tag = f"corrupt|{system}|{corruption}|{seed}|{int(bool(blind))}"
    return "case_" + hashlib.sha1(tag.encode()).hexdigest()[:8]


def _blind_base(system, seed, spread=0.25):
    """Blinded base system as in blind.make_blind: field renamed to q, every coefficient perturbed."""
    sys = SYSTEMS[system]
    rng = np.random.default_rng(int(hashlib.sha1(f"{system}-{seed}".encode()).hexdigest()[:8], 16))
    old, new = list(sys.fields), ["q"] if len(sys.fields) == 1 else [f"q{i + 1}" for i in range(len(sys.fields))]
    names = pde_symbols(old)
    ren = {}
    for o, n in zip(old, new):
        ren[sp.Symbol(o)] = sp.Symbol(f"__{n}")
        for s in names:
            if s.startswith(o + "_"):
                ren[sp.Symbol(s)] = sp.Symbol(f"__{n}_{s.split('_', 1)[1]}")
    rhs = {n: str(_perturb(sp.expand(parse(sys.rhs[o], names)), rng, spread).xreplace(ren)).replace("__", "")
           for o, n in zip(old, new)}
    return dataclasses.replace(sys, fields=new, rhs=rhs, name=f"blind_{system}")


def _traj_rhs(base_rhs, fields, rng, spread, min_range, n):
    """n independently perturbed copies of the base rhs; redraw until some coefficient differs by >= min_range
    (relative) between trajectories, so the heterogeneity is actually present."""
    names = pde_symbols(fields)
    exprs = {f: sp.expand(parse(base_rhs[f], names)) for f in fields}
    for _ in range(100):
        out = [{f: str(_perturb(exprs[f], rng, spread)) for f in fields} for _ in range(n)]
        facs = []
        for f in fields:
            base_terms = {m: float(c) for c, m in (t.as_coeff_Mul() for t in sp.Add.make_args(exprs[f]))}
            for m, c0 in base_terms.items():
                cs = [float(sp.expand(parse(o[f], names)).coeff(m)) if m != 1 else 0.0 for o in out]
                facs.append((max(cs) - min(cs)) / abs(c0))
        if max(facs) >= min_range:
            return out
    return out


def _simulate(sys, rhs_list, ic_scales, rng, t):
    """One trajectory per entry of rhs_list (ICs drawn from sys.ic, times ic_scales). Redraws the IC of a
    trajectory that is non-finite or blows up (counted in `retries`)."""
    lay = sys.layout()
    coords = grid_coords(lay)
    trajs, retries = [], 0
    for rhs, sc in zip(rhs_list, ic_scales):
        for attempt in range(6):
            U0 = sc * sys.ic(rng, *coords)
            U = integrate_pde_general(sys.fields, rhs, lay, U0, t, sys.dt_sim, max_seconds=600.0)
            if np.all(np.isfinite(U)) and np.abs(U).max() < 50 * max(1.0, np.abs(U0).max()):
                break
            retries += 1
        else:
            raise RuntimeError(f"{sys.name}: unstable trajectory for rhs {rhs}")
        trajs.append(U)
    return np.stack(trajs), retries


def _add_glitches(U_obs, U, noise, rng, params):
    """In place: per trajectory 2-4 single-sample (single grid point) spikes of 8-15 x the noise sd, random sign,
    >= min_sep rows apart and >= edge rows from the ends. Returns [[traj, t_index, x_index, field, amp_sd], ...]."""
    nt, nx, nf = U.shape[1], U.shape[2], U.shape[-1]
    sd = noise * U.reshape(-1, nf).std(0)
    lo_n, hi_n = params["n_per_traj"]
    out = []
    for j in range(U.shape[0]):
        k = int(rng.integers(lo_n, hi_n + 1))
        for _ in range(1000):
            ti = np.sort(rng.choice(np.arange(params["edge"], nt - params["edge"]), k, replace=False))
            if k < 2 or np.diff(ti).min() >= params["min_sep"]:
                break
        for i in ti:
            xi, f = int(rng.integers(nx)), int(rng.integers(nf))
            amp = float(rng.uniform(*params["amp_noise_sd"]) * rng.choice([-1, 1]))
            U_obs[j, i, xi, f] += amp * sd[f]
            out.append([j, int(i), xi, f, amp])
    return out


def _simulate_kicked(sys, rhs, rng, t, params, n):
    """n trajectories of the base equation; at t_k (index k, 40-60% of the run) a smooth periodic Gaussian bump
    (amplitude 0.5-1 x the trajectory's std, width L * width_frac) is added to the true state, and the same equation
    keeps integrating from the kicked state. Row k is the state just before the kick. Returns (U, retries, kicks)."""
    lay = sys.layout()
    coords = grid_coords(lay)
    x = coords[0]
    trajs, kicks, retries = [], [], 0
    nt = len(t)
    for _ in range(n):
        for attempt in range(6):
            U0 = sys.ic(rng, *coords)
            k = int(round(rng.uniform(*params["t_frac"]) * (nt - 1)))
            A = integrate_pde_general(sys.fields, rhs, lay, U0, t[:k + 1], sys.dt_sim, max_seconds=600.0)
            amp = float(rng.uniform(*params["amp_std"]) * rng.choice([-1, 1]) * np.std(A))
            x0, w = float(rng.uniform(0, sys.L)), sys.L * params["width_frac"]
            dx = (x - x0 + sys.L / 2) % sys.L - sys.L / 2
            bump = amp * np.exp(-0.5 * (dx / w) ** 2)
            Uk = A[-1] + bump[:, None]
            B = integrate_pde_general(sys.fields, rhs, lay, Uk, t[k:], sys.dt_sim, max_seconds=600.0)
            U = np.concatenate([A, B[1:]], 0)
            if np.all(np.isfinite(U)) and np.abs(U).max() < 50 * max(1.0, np.abs(U0).max()):
                break
            retries += 1
        else:
            raise RuntimeError(f"{sys.name}: unstable kicked trajectory")
        trajs.append(U)
        kicks.append({"traj": len(trajs) - 1, "t_index": k, "t": float(t[k]), "x0": x0, "width": w,
                      "amplitude": amp, "amplitude_noise_sd": amp / (NOISE * float(np.std(U)) + 1e-300)})
    return np.stack(trajs), retries, kicks


def _term_ratio(fields, base_rhs, extra, L, U, x, t):
    """Per-trajectory rms(extra term) / rms(base rhs) on clean data (how strongly the corruption acts)."""
    fb = make_pde_rhs(fields, base_rhs, L)(U, x, t[None, :, None])
    fe = make_pde_rhs(fields, {f: extra for f in fields}, L)(U, x, t[None, :, None])
    red = tuple(range(1, U.ndim))
    return [float(v) for v in np.sqrt((fe ** 2).mean(red)) / np.sqrt((fb ** 2).mean(red))]


def make_case(system, corruption, seed, noise=NOISE, out_root="datasets/corrupt", blind=False, t_end=None):
    """Generate one corrupted dataset; returns its directory. `t_end` shortens the record (tests only)."""
    t_start = time.time()
    if system not in BASE_SYSTEMS or corruption not in CORRUPTIONS + EVENT_CORRUPTIONS:
        raise ValueError(f"unknown case {system}/{corruption}")
    sys = _blind_base(system, seed) if blind else SYSTEMS[system]
    fields, f0 = list(sys.fields), sys.fields[0]
    base = dict(sys.rhs)
    params = case_params(system, corruption)
    rng = np.random.default_rng(int(hashlib.sha1(f"corrupt-{system}-{corruption}-{seed}-{blind}".encode())
                                    .hexdigest()[:8], 16))
    t = np.round(np.arange(0, (t_end or sys.t_end) + 1e-9, sys.dt), 10)
    x = grid_coords(sys.layout())[0]

    extra = None
    if corruption == "forcing_time":
        extra = f"{params['A']}*sin({params['w']}*t)"
    elif corruption == "source_space":
        extra = f"{params['A']}*cos({2 * np.pi / sys.L:.10g}*x)"
    elif corruption == "amp_term":
        extra = f"{params['c']}*{f0}**3"
    gen = {f: f"{base[f]} + {extra}" for f in fields} if extra else dict(base)

    scales, test_scales = [1.0] * N_TRAIN, [1.0] * N_TEST
    if corruption == "amp_term":
        scales, test_scales = params["scales"], params["test_scales"]
    if corruption == "traj_coeffs":
        train_rhs = _traj_rhs(base, fields, rng, params["spread"], params["min_range"], N_TRAIN)
    else:
        train_rhs = [gen] * N_TRAIN
    test_rhs = [gen if corruption in DYNAMIC else base] * N_TEST

    if corruption == "kick":
        U, r1, kicks = _simulate_kicked(sys, base, rng, t, params, N_TRAIN)
    else:
        U, r1 = _simulate(sys, train_rhs, scales, rng, t)
    U_test, r2 = _simulate(sys, test_rhs, test_scales, rng, t)
    diagnostics = {"retries": r1 + r2, "max_abs_u_train": float(np.abs(U).max()),
                   "max_abs_u_test": float(np.abs(U_test).max())}
    if extra:
        diagnostics["term_rms_ratio_train"] = _term_ratio(fields, base, extra, sys.L, U, x, t)

    U_obs = add_noise(U, noise, rng, "outliers" if corruption == "outliers" else "gaussian")
    nan_mask = None
    if corruption == "gaps_random":
        nt = len(t)
        nan_mask = np.zeros(U.shape[:2], bool)
        for j in range(U.shape[0]):
            total, k = int(round(params["frac"] * nt)), params["n_windows"]
            lens = np.maximum(np.array([total // k + (i < total % k) for i in range(k)]), 1)
            for _ in range(1000):          # non-overlapping windows, frame 0 kept
                starts = np.sort(rng.integers(1, nt - lens.max(), params["n_windows"]))
                if np.all(np.diff(starts) > lens[:-1]):
                    break
            for s0, ln in zip(starts, lens):
                nan_mask[j, s0:s0 + ln] = True
        U_obs[nan_mask] = np.nan
    elif corruption == "gaps_state":
        amp = np.abs(U[..., 0] - U[..., 0].mean())
        loc = np.stack([uniform_filter(a, size=params["window"], mode=("nearest", "wrap")) for a in amp])
        thr = np.quantile(loc, params["quantile"])
        nan_mask = loc > thr
        nan_mask[:, 0] = False
        U_obs[nan_mask] = np.nan
        diagnostics["amplitude_threshold"] = float(thr)
    if corruption == "glitch":
        diagnostics["glitches"] = _add_glitches(U_obs, U, noise, rng, params)
    if corruption == "kick":
        diagnostics["kicks"] = kicks
    if nan_mask is not None:
        diagnostics["nan_fraction"] = float(nan_mask.mean() if nan_mask.ndim == 3 else nan_mask.mean())
        diagnostics["nan_fraction_per_traj"] = [float(m.mean()) for m in nan_mask]

    name = _case_name(system, corruption, seed, blind)
    d = Path(out_root) / name
    (d / "hidden").mkdir(parents=True, exist_ok=True)
    lay = sys.layout()
    dims = lay["spatial_dims"]
    meta = {"name": name, "kind": "pde", "variables": fields, "dt": float(t[1] - t[0]), "n_traj": int(U_obs.shape[0]),
            "shape": list(U_obs.shape), "shape_doc": "(n_traj, nt, nx, n_fields)", "system": None,
            "L": sys.L, "nx": sys.nx, "boundary": lay["boundary"], "spatial_dims": dims, "grid": lay["grid"],
            "allowed_symbols": derivative_symbols(fields, dims, 4)}
    np.savez_compressed(d / "data.npz", t=t, U=U_obs, x=x)
    np.savez_compressed(d / "hidden" / "test.npz", t=t, U=U_test, x=x)
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    truth = {"system": sys.name, "kind": "pde", "variables": fields, "rhs": base, "noise": noise,
             "noise_type": "outliers" if corruption == "outliers" else "gaussian", "dt_mult": 1, "seed": seed,
             "eval_horizon": min(sys.eval_horizon, float(t[-1])), "tags": list(sys.tags), "L": sys.L, "dt_sim": sys.dt_sim,
             "nx": sys.nx, "spatial_dims": dims, "boundary": lay["boundary"], "grid": lay["grid"]}
    (d / "hidden" / "truth.json").write_text(json.dumps(truth, indent=2))
    corr = {"id": corruption, "params": params, "expected_detector": EXPECTED[corruption],
            "base_system": system, "blind": bool(blind), "seed": seed, "noise": noise,
            "rhs_train": train_rhs, "rhs_test": test_rhs[0], "dynamics_corrupted": corruption in DYNAMIC + ("traj_coeffs",),
            "diagnostics": diagnostics, "wall_seconds": round(time.time() - t_start, 2)}
    if corruption in EVENT_CORRUPTIONS:
        corr["expected_handling"] = HANDLING[corruption]
        corr["events"] = diagnostics["glitches" if corruption == "glitch" else "kicks"]
    (d / "hidden" / "corruption.json").write_text(json.dumps(corr, indent=2))
    return d


def split_of(seed, blind=False):
    return "blind" if blind else ("dev" if seed < 10 else "report")


def _job(args):
    system, corruption, seed, out, blind = args
    try:
        d = make_case(system, corruption, seed, out_root=out, blind=blind)
        c = json.loads((d / "hidden" / "corruption.json").read_text())
        return {"path": str(d), "system": system, "corruption": corruption, "seed": seed,
                "split": split_of(seed, blind), "blind": blind, "expected_detector": EXPECTED[corruption],
                "wall_seconds": c["wall_seconds"]}
    except Exception as e:  # noqa: BLE001
        return {"system": system, "corruption": corruption, "seed": seed, "split": split_of(seed, blind),
                "blind": blind, "error": repr(e)}


def make_suite(seeds, out_root="datasets/corrupt", blind=False, systems=BASE_SYSTEMS, corruptions=CORRUPTIONS,
               workers=None):
    """All systems x corruptions x seeds into <out_root>/<split>/; updates <out_root>/index.json."""
    import os
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(s, c, seed, str(Path(out_root) / split_of(seed, blind)), blind)
            for seed in seeds for s in systems for c in corruptions]
    workers = workers or max(1, min(len(jobs), (os.cpu_count() or 2) - 1))
    if workers == 1:
        rows = [_job(j) for j in jobs]
    else:
        with ProcessPoolExecutor(workers) as ex:
            rows = list(ex.map(_job, jobs))
    idx_path = Path(out_root) / "index.json"
    old = json.loads(idx_path.read_text()) if idx_path.exists() else []
    key = lambda r: (r["system"], r["corruption"], r["seed"], r["blind"])  # noqa: E731
    new_keys = {key(r) for r in rows}
    index = [r for r in old if key(r) not in new_keys] + rows
    order = CORRUPTIONS + EVENT_CORRUPTIONS
    index.sort(key=lambda r: (r["split"], r["seed"], r["system"], order.index(r["corruption"])))
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    idx_path.write_text(json.dumps(index, indent=1))
    return [Path(r["path"]) for r in rows if "path" in r]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--blind", action="store_true", help="blinded variants (field renamed, base coefficients perturbed)")
    p.add_argument("--out", default="datasets/corrupt")
    p.add_argument("--systems", nargs="+", default=list(BASE_SYSTEMS), choices=BASE_SYSTEMS)
    p.add_argument("--corruptions", nargs="+", choices=CORRUPTIONS + EVENT_CORRUPTIONS)
    p.add_argument("--events", action="store_true",
                   help="glitch / kick suite (WS8) into datasets/corrupt_events unless --out is given")
    p.add_argument("--workers", type=int)
    a = p.parse_args()
    if a.events and a.out == "datasets/corrupt":
        a.out = "datasets/corrupt_events"
    a.corruptions = a.corruptions or list(EVENT_CORRUPTIONS if a.events else CORRUPTIONS)
    t0 = time.time()
    paths = make_suite(a.seeds, a.out, a.blind, a.systems, a.corruptions, a.workers)
    index = json.loads((Path(a.out) / "index.json").read_text())
    for r in index:
        if r["seed"] in a.seeds and r["blind"] == a.blind:
            print(f"{r['split']:6s} s{r['seed']:<3d} {r['system']:22s} {r['corruption']:13s} "
                  f"{r.get('path', 'ERROR ' + r.get('error', ''))}  {r.get('wall_seconds', '')}s")
    print(f"{len(paths)} cases in {time.time() - t0:.0f}s; index: {Path(a.out) / 'index.json'}")


if __name__ == "__main__":
    main()
