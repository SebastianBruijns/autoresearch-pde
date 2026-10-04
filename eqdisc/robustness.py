"""Robustness experiment: can eqdisc recover a system's own law when the data are hit by outside events or bad sensors?

Systems (from the hackathon material): Lorenz (PySINDy tutorial), KdV (kdv_discover), a satellite orbit (orbit_discover;
here Kepler + a moderate bulge so the clean case is solvable). Conditions, all with 2% measurement noise:

  clean     control
  forcing   2-3 localised top-hat kicks per run (random times; for KdV also a random patch of space): unmodelled outside
            input. The right answer is still the autonomous law, without a spurious forcing term.
  spikes    0.5% of readings replaced by +-8 sigma glitches (a noise shock)
  sensor    a lost/broken sensor: Lorenz loses z (x' still closes, y' does not: the right answer recovers x' and flags a
            missing variable); the orbit loses vz (same idea); KdV has stuck sensors (readings frozen at a few points)

Every dataset has clean, unforced held-out trajectories for scoring forecasts.

    python -m eqdisc.robustness make            # -> datasets/robust/<system>_<condition>/
    python -m eqdisc.robustness run --workers 4  # agent (data only) + non-LLM baseline, -> runs/robust/results.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

ROOT = Path("datasets/robust2")
NOISE = 0.02
CONDITIONS = ("clean", "forcing", "spikes", "sensor")

LORENZ = {"x": "10*(y - x)", "y": "x*(28 - z) - y", "z": "x*y - 8/3*z"}
J2C = 0.015                                     # 1.5 * J2 with J2 = 0.01 (mu = Re = 1)
_r = "(x**2 + y**2 + z**2)**0.5"
ORBIT = {"x": "vx", "y": "vy", "z": "vz",
         "vx": f"-x/{_r}**3 - {J2C}*x/{_r}**5*(1 - 5*z**2/{_r}**2)",
         "vy": f"-y/{_r}**3 - {J2C}*y/{_r}**5*(1 - 5*z**2/{_r}**2)",
         "vz": f"-z/{_r}**3 - {J2C}*z/{_r}**5*(3 - 5*z**2/{_r}**2)"}
KDV = {"u": "-6*u*u_x - u_xxx"}


# ----------------------------------------------------------------------------- simulation
def _ode(variables, rhs, y0, t, pulses=()):
    """pulses: [(var_index, amplitude, t_on, t_off)] top-hat forcing added to d(var)/dt."""
    syms = sp.symbols(variables)
    f = sp.lambdify(syms, [sp.sympify(rhs[v]) for v in variables], "numpy")

    def g(tt, y):
        d = np.array(f(*y), float)
        for i, a, t0, t1 in pulses:
            if t0 <= tt < t1:
                d[i] += a
        return d
    steps = [p[2] for p in pulses] + [p[3] for p in pulses]
    sol = solve_ivp(g, (t[0], t[-1]), y0, t_eval=t, rtol=1e-9, atol=1e-11, method="DOP853",
                    max_step=min(np.diff(t).min(), *(np.diff(sorted(steps)) if len(steps) > 1 else [1.0])) or None)
    return sol.y.T


def _kdv(u0, t, L=40.0, dt=1e-3, sources=()):
    """KdV u_t = -6 u u_x - u_xxx (+ top-hat sources [(amp, x0, x1, t0, t1)]), periodic, ETDRK4 with a full-circle
    contour (the linear operator i k^3 is imaginary)."""
    n = u0.size
    x = np.arange(n) * L / n
    k = 2 * np.pi * np.fft.rfftfreq(n, L / n)
    Lop = 1j * k ** 3
    M = 64
    r = np.exp(2j * np.pi * (np.arange(1, M + 1) - .5) / M)
    LR = dt * Lop[:, None] + r
    E, E2 = np.exp(dt * Lop), np.exp(dt * Lop / 2)
    Q = dt * np.mean((np.exp(LR / 2) - 1) / LR, 1)
    f1 = dt * np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3, 1)
    f2 = dt * np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR ** 3, 1)
    f3 = dt * np.mean((-4 - 3 * LR - LR ** 2 + np.exp(LR) * (4 - LR)) / LR ** 3, 1)
    mask = (np.arange(k.size) < n / 3)
    masks = [(a, ((x >= x0) & (x < x1)).astype(float), t0, t1) for a, x0, x1, t0, t1 in sources]

    def N(v, tt):
        u = np.fft.irfft(v, n)
        s = -3 * (1j * k) * np.fft.rfft(u * u)            # -6 u u_x = -3 (u^2)_x
        for a, m, t0, t1 in masks:
            if t0 <= tt < t1:
                s = s + np.fft.rfft(a * m)
        return s * mask
    v = np.fft.rfft(u0)
    out, tt, j = [u0.copy()], 0.0, 1
    nsteps = int(round((t[-1] - t[0]) / dt))
    every = int(round((t[1] - t[0]) / dt))
    for i in range(1, nsteps + 1):
        Nv = N(v, tt)
        a = E2 * v + Q * Nv
        Na = N(a, tt + dt / 2)
        b = E2 * v + Q * Na
        Nb = N(b, tt + dt / 2)
        c = E2 * a + Q * (2 * Nb - Nv)
        Nc = N(c, tt + dt)
        v = E * v + f1 * Nv + 2 * f2 * (Na + Nb) + f3 * Nc
        tt += dt
        if i % every == 0:
            out.append(np.fft.irfft(v, n))
    return np.array(out)


def _kdv_ic(rng, n=256, L=40.0):
    x = np.arange(n) * L / n
    u = np.zeros(n)
    for _ in range(2):
        c, x0 = rng.uniform(0.5, 2.0), rng.uniform(0, L)
        d = (x - x0 + L / 2) % L - L / 2
        u += c / 2 / np.cosh(np.sqrt(c) / 2 * d) ** 2
    return u


def _orbit_ic(rng):
    r = rng.uniform(1.3, 2.0)
    inc, node = rng.uniform(0.2, 1.3), rng.uniform(0, 2 * np.pi)
    v = np.sqrt(1 / r) * rng.uniform(0.95, 1.05)
    pos = r * np.array([np.cos(node), np.sin(node), 0.0])
    vd = np.array([-np.sin(node) * np.cos(inc), np.cos(node) * np.cos(inc), np.sin(inc)])
    return np.r_[pos, v * vd]


# ----------------------------------------------------------------------------- corruption
def _spikes(U, rng, frac=0.005, size=8.0):
    U = U.copy()
    sd = U.reshape(-1, U.shape[-1]).std(0)
    m = rng.random(U.shape) < frac
    U[m] += (rng.choice([-1, 1], m.sum()) * size * np.broadcast_to(sd, U.shape)[m])
    return U, int(m.sum())


def _blind(system, variables, rhs, rng):
    """Random rescaling so no textbook coefficient survives (recall cannot score). Returns (variable scales k,
    time scale s, space scale a or None, blinded law {var: expr})."""
    if system == "lorenz":
        k, s = rng.uniform(0.3, 0.8, 3), rng.uniform(0.4, 0.9)
    elif system == "orbit":
        Ls, Ts = rng.uniform(0.4, 0.8), rng.uniform(1.5, 3.0)
        k, s = np.array([Ls] * 3 + [Ls / Ts] * 3), Ts
    else:
        ku, a, s = rng.uniform(0.4, 1.6), rng.uniform(0.6, 1.5), rng.uniform(0.5, 1.5)
        k = np.array([ku])
        law = {"u": f"-{6 * a / (ku * s):.8g}*u*u_x - {a ** 3 / s:.8g}*u_xxx"}
        return k, float(s), float(a), law
    syms = sp.symbols(variables)
    sub = {sv: sv / float(kv) for sv, kv in zip(syms, k)}
    law = {v: str(sp.expand(sp.sympify(rhs[v], locals=dict(zip(variables, syms))).subs(sub, simultaneous=True)
                            * float(k[i]) / float(s)).evalf(9)) for i, v in enumerate(variables)}
    return k, float(s), None, law


def make(seed=7):
    rng = np.random.default_rng(seed)
    made = []
    for system in ("lorenz", "orbit", "kdv"):
        if system == "lorenz":
            variables, rhs, dt, T = ["x", "y", "z"], LORENZ, 0.01, 10.0
            ics = [rng.uniform([-15, -20, 5], [15, 20, 40]) for _ in range(6)]
        elif system == "orbit":
            variables, rhs, dt, T = ["x", "y", "z", "vx", "vy", "vz"], ORBIT, 0.05, 60.0
            ics = [_orbit_ic(rng) for _ in range(6)]
        else:
            variables, rhs, dt, T = ["u"], KDV, 0.1, 10.0
            ics = [_kdv_ic(rng) for _ in range(4)]
        t = np.round(np.arange(0, T + 1e-9, dt), 10)
        n_tr = len(ics) - 2 if system != "kdv" else 3
        sim = (lambda y0, ev=(): _ode(variables, rhs, y0, t, ev)) if system != "kdv" else \
              (lambda y0, ev=(): _kdv(y0, t, sources=ev)[..., None])
        clean = np.array([sim(y0) for y0 in ics])
        test = clean[n_tr:]
        # forcing amplitudes relative to the typical size of each variable's rate of change
        rate = np.abs(np.diff(clean[:n_tr], axis=1)).mean(axis=tuple(range(clean.ndim - 1))) / dt
        kb, sb, ab, blaw = _blind(system, variables, rhs, np.random.default_rng(seed + 991))
        tb, test_b = t * sb, test * kb
        L_b = 40.0 * ab if ab else None
        for cond in CONDITIONS:
            r2 = np.random.default_rng(seed * 100 + CONDITIONS.index(cond))
            events = []
            if cond == "forcing":
                trajs = []
                for j, y0 in enumerate(ics[:n_tr]):
                    ev = []
                    for _ in range(r2.integers(2, 4)):
                        t0 = r2.uniform(0.15, 0.8) * T
                        t1 = t0 + r2.uniform(0.02, 0.05) * T
                        if system == "kdv":
                            x0 = r2.uniform(0, 34)
                            ev.append((float(r2.choice([-1, 1]) * 0.6 * rate[0]), x0, x0 + r2.uniform(2, 5), t0, t1))
                        else:
                            i = int(r2.integers(len(variables)) if system == "lorenz" else r2.integers(3, 6))
                            ev.append((i, float(r2.choice([-1, 1]) * 0.6 * rate[i]), t0, t1))
                    events.append(ev)
                    trajs.append(sim(y0, ev))
                U = np.array(trajs)
            else:
                U = clean[:n_tr].copy()
            U = U * kb                                         # blinded units
            sd = U.reshape(-1, U.shape[-1]).std(0)
            U = U + NOISE * sd * r2.standard_normal(U.shape)
            obs_vars, info = list(variables), {}
            if cond == "spikes":
                U, info["n_spikes"] = _spikes(U, r2)
            if cond == "sensor":
                if system == "lorenz":
                    U, obs_vars = U[..., :2], ["x", "y"]
                    info["lost"] = "z"
                elif system == "orbit":
                    U, obs_vars = U[..., :5], variables[:5]
                    info["lost"] = "vz"
                else:
                    stuck = np.sort(r2.choice(U.shape[2], 8, replace=False))
                    U[:, :, stuck, 0] = U[:, :1, stuck, 0]       # frozen at the first reading
                    info["stuck_sensors"] = stuck.tolist()
            name = f"{system}_{cond}"
            d = ROOT / name
            (d / "hidden").mkdir(parents=True, exist_ok=True)
            meta = {"name": "dataset", "kind": "ode" if system != "kdv" else "pde", "variables": obs_vars,
                    "dt": float(dt * sb), "n_traj": int(U.shape[0]), "shape": list(U.shape), "system": None,
                    "allowed_symbols": obs_vars + (["t"] if system != "kdv" else
                                                   ["u_x", "u_xx", "u_xxx", "u_xxxx", "x"])}
            if system == "kdv":
                meta.update({"L": L_b, "nx": 256, "boundary": "periodic", "spatial_dims": ["x"]})
            for f_ in (d / "hidden").glob("*"):
                f_.unlink()
            (d / "meta.json").write_text(json.dumps(meta, indent=1))
            xg = {"x": np.arange(256) * L_b / 256} if system == "kdv" else {}
            np.savez_compressed(d / "data.npz", t=tb, U=U.astype(np.float64), **xg)
            lost = info.get("lost")
            score_rhs = {v: blaw[v] for v in obs_vars if not lost or (lost not in str(sp.sympify(blaw[v]).free_symbols)
                                                                       and v != {"vz": "z"}.get(lost))}
            np.savez_compressed(d / "hidden" / "test.npz", t=tb, U=test_b[..., :len(obs_vars)] if system != "kdv"
                                else test_b, **xg)
            score = {"system": system, "kind": meta["kind"], "variables": obs_vars, "rhs": score_rhs, "full_law": blaw,
                     "condition": cond, "events": events, "noise": NOISE, "scales": {"vars": kb.tolist(), "time": sb,
                     "space": ab}, **info}
            (d / "hidden" / "score.json").write_text(json.dumps(score, indent=1))
            if not lost:              # the pipeline's own hidden evaluation needs a complete law
                (d / "hidden" / "truth.json").write_text(json.dumps({
                    "system": None, "kind": meta["kind"], "variables": obs_vars, "rhs": blaw, "noise": NOISE,
                    "eval_horizon": float(T * sb / 4), "L": L_b, "dt_sim": 1e-3 * sb if system == "kdv" else None},
                    indent=1))
            made.append(name)
            print(name, U.shape, {k: v for k, v in info.items() if k != "stuck_sensors"}, flush=True)
    return made


# ----------------------------------------------------------------------------- scoring
def _terms(rhs_v, names):
    C = sp.Symbol("c")
    loc = {n: sp.Symbol(n) for n in names}
    e = sp.expand(sp.sympify(str(rhs_v), locals=loc))
    out = {}
    for term in sp.Add.make_args(e):
        c, m = term.as_coeff_Mul()
        out[sp.srepr(m)] = float(c)
    return out


def vf_error(d, truth_rhs, rhs):
    """Relative RMS difference between the found and the true law's right-hand sides on the clean held-out states
    (form-independent: equivalent expressions score 0). Per scored equation; None if it cannot be evaluated."""
    from . import solvers
    m = json.loads((Path(d) / "meta.json").read_text())
    te = np.load(Path(d) / "hidden" / "test.npz")
    U = te["U"][:, ::5]
    out = {}
    for v in truth_rhs:
        if not rhs or v not in rhs:
            out[v] = None
            continue
        try:
            if m["kind"] == "ode":
                full_t = {**{w: "0" for w in m["variables"]}, v: truth_rhs[v]}
                full_f = {**{w: "0" for w in m["variables"]}, v: rhs[v]}
                X = U.reshape(-1, U.shape[-1])
                i = m["variables"].index(v)
                a = solvers.make_ode_rhs(m["variables"], full_t)(X, np.zeros(len(X)))[:, i]
                b = solvers.make_ode_rhs(m["variables"], full_f)(X, np.zeros(len(X)))[:, i]
            else:
                lay = solvers.pde_layout(m)
                ft = solvers.make_pde_rhs_general(m["variables"], truth_rhs, lay)
                ff = solvers.make_pde_rhs_general(m["variables"], {v: rhs[v]}, lay)
                a = np.concatenate([ft(u).ravel() for u in U.reshape(-1, *U.shape[2:])])
                b = np.concatenate([ff(u).ravel() for u in U.reshape(-1, *U.shape[2:])])
            out[v] = float(np.sqrt(np.mean((a - b) ** 2)) / (np.sqrt(np.mean(a ** 2)) + 1e-30))
        except Exception as e:  # noqa: BLE001
            out[v] = None
    return out


def score_law(truth_rhs, rhs, names):
    """Per equation in truth_rhs: same terms (structure) and max relative coefficient error."""
    res = {}
    for v, tv in truth_rhs.items():
        if v not in (rhs or {}):
            res[v] = {"structure": False, "coef_err": None}
            continue
        try:
            a, b = _terms(tv, names), _terms(rhs[v], names)
            same = set(a) == set(b)
            err = max(abs(b[k] - a[k]) / abs(a[k]) for k in a) if same else None
            res[v] = {"structure": same, "coef_err": err}
        except Exception as e:  # noqa: BLE001
            res[v] = {"structure": False, "coef_err": None, "error": str(e)[:80]}
    return res


def run(workers=4, max_tools=20, names=None):
    from concurrent.futures import ThreadPoolExecutor
    from . import toolbox as tb
    from .agent import make_client, run_agent
    from .evaluate import load
    from .weakform import weak_sindy
    from . import insights
    out = Path("runs/robust2")
    out.mkdir(parents=True, exist_ok=True)
    cases = names or sorted(p.name for p in ROOT.iterdir() if (p / "meta.json").exists())
    client = make_client()

    def one(name):
        d = ROOT / name
        m, D = load(d)
        truth = json.loads((d / "hidden" / "score.json").read_text())
        names_ = m["variables"] + (["u_x", "u_xx", "u_xxx", "u_xxxx"] if m["kind"] == "pde" else [])
        row = {"case": name, "system": truth["system"], "condition": truth["condition"]}
        try:
            base = weak_sindy(m, D, poly_degree=2, max_deriv=3) if m["kind"] == "pde" else tb.run_sindy(m, D, poly_degree=2)
            row["baseline"] = {"rhs": base["rhs"], "law": score_law(truth["rhs"], base["rhs"], names_),
                               "vf_err": vf_error(d, truth["rhs"], base["rhs"])}
        except Exception as e:  # noqa: BLE001
            row["baseline"] = {"error": str(e)[:200]}
        try:
            r = run_agent(d, client=client, verbose=False, max_tools=max_tools, use_memory=False, context=None,
                          out_dir=out / name, final_assessment=True, report=True)
            rhs = (r.get("submitted") or {}).get("rhs")
            a = r.get("assessment") or {}
            row["agent"] = {"rhs": rhs, "law": score_law(truth["rhs"], rhs, names_), "vf_err": vf_error(d, truth["rhs"], rhs),
                            "verdict": (insights.verdict(a) if a else {}).get("status"),
                            "cost_usd": r.get("cost_usd"), "hidden_eval": {k: (r.get("hidden_eval") or {}).get(k)
                                                                            for k in ("score", "f1", "rollout_nrmse")}}
        except Exception as e:  # noqa: BLE001
            row["agent"] = {"error": str(e)[:300]}
        (out / f"{name}.json").write_text(json.dumps(row, indent=1, default=str))
        print(json.dumps({k: row[k] for k in ("case",)} | {"agent": {k: row.get("agent", {}).get(k) for k in
                          ("vf_err", "verdict", "cost_usd", "error")}, "baseline_vf": row.get("baseline", {}).get("vf_err")},
                         default=str), flush=True)
        return row
    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(one, cases))
    (out / "results.json").write_text(json.dumps(rows, indent=1, default=str))
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["make", "run"])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cases", nargs="*")
    a = p.parse_args()
    make() if a.cmd == "make" else run(a.workers, names=a.cases)
