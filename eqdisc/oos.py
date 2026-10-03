"""Honest out-of-sample (OOS) evaluation with autoregressive rollouts and videos.

Protocol (per case):
  * training data are NOISY and cover only a training window (or training trajectories);
  * the future window / held-out trajectories are never seen by any method;
  * every model is rolled out autoregressively from the NOISY observed state at the start of the test window;
  * the true PDE rolled out from the same noisy state is the reference (for chaotic systems even it diverges:
    that sets the predictability limit);
  * discovered models are used with REFITTED coefficients (no rounding to "nice" numbers);
  * baselines: weak SINDy (no LLM) and a Fourier Neural Operator trained on the same noisy window;
  * time is reported in Lyapunov times for chaotic systems (largest exponent measured with a twin experiment).

    python -m eqdisc.oos ks [--agent]      # blinded real Kuramoto-Sivashinsky data
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import solvers


# ----------------------------------------------------------------------------- generic pieces
def rel_err_t(Y, U):
    """Relative L2 error per time step, (nt,)."""
    red = tuple(range(1, U.ndim))
    with np.errstate(all="ignore"):
        e = np.sqrt(np.sum((Y - U) ** 2, axis=red) / np.sum(U ** 2, axis=red))
    return np.where(np.isfinite(e), e, np.inf)


def valid_time(err, t, thr=0.5):
    bad = np.where(err > thr)[0]
    return float(t[bad[0]] - t[0]) if bad.size else float(t[-1] - t[0])


def lyapunov_exponent(fields, rhs, meta, U0, t, eps=1e-7, seed=0):
    """Largest Lyapunov exponent of the TRUE system from a twin experiment (slope of log separation)."""
    rng = np.random.default_rng(seed)
    lay = solvers.pde_layout(meta)
    a = solvers.integrate_pde_general(fields, rhs, lay, U0, t)
    b = solvers.integrate_pde_general(fields, rhs, lay, U0 + eps * np.std(U0) * rng.standard_normal(U0.shape), t)
    sep = np.sqrt(np.mean((a - b) ** 2, axis=tuple(range(1, a.ndim))))
    ok = (sep > 10 * eps * np.std(U0)) & (sep < 0.1 * np.std(U0))
    if ok.sum() < 5:
        return None
    return float(np.polyfit(t[ok], np.log(sep[ok]), 1)[0])


def refit_rhs(meta, data, rhs):
    """Refit the linear coefficients of a discovered structure on the (noisy) training data, weak form if
    possible. Returns (refit_rhs, table of submitted vs refit coefficients)."""
    from .assess import weak_stats
    ws = weak_stats(meta, data, rhs)
    if ws is None:
        return rhs, []
    out, table = {}, []
    for v, terms in ws["coefs"].items():
        out[v] = " + ".join(f"({c['fit']:.6g})*({t})" for t, c in terms.items()) or "0"
        table += [{"var": v, "term": t, "submitted": c.get("given"), "refit": c["fit"], "ci90": c["ci90"]}
                  for t, c in terms.items()]
    return out, table


def write_dataset(d, meta, t, U_train, U_test, truth_rhs, x, extra_truth=None):
    d = Path(d)
    (d / "hidden").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "data.npz", t=t[: U_train.shape[1]], U=U_train, x=x)
    np.savez_compressed(d / "hidden" / "test.npz", t=t[U_train.shape[1] - 1:] - t[U_train.shape[1] - 1], U=U_test, x=x)
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    truth = {"system": meta["name"], "kind": meta["kind"], "variables": meta["variables"], "rhs": truth_rhs,
             "noise": meta.get("noise"), "eval_horizon": float(t[-1] - t[U_train.shape[1] - 1]),
             "L": meta.get("L"), "dt_sim": meta["dt"] / 8, **(extra_truth or {})}
    (d / "hidden" / "truth.json").write_text(json.dumps(truth, indent=2))
    return d


# ----------------------------------------------------------------------------- video
def video_1d(path, x, t, rows, title, fps=12, lyap=None):
    """rows: list of (label, Y (nt, nx), colour). Animated u(x) lines + error-vs-time panel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.animation as anim
    import matplotlib.pyplot as plt
    truth = rows[0][1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [2, 1]})
    lim = np.nanmax(np.abs(truth)) * 1.2
    lines = [a1.plot(x, Y[0], color=c, lw=2 if i == 0 else 1.4, label=lab, alpha=0.95 if i == 0 else 0.85)[0]
             for i, (lab, Y, c) in enumerate(rows)]
    a1.set(ylim=(-lim, lim), xlabel="x", title=title)
    a1.legend(loc="upper right", fontsize=8)
    tt = (t - t[0]) * (lyap if lyap else 1)
    errs = [rel_err_t(Y, truth) for _, Y, _ in rows[1:]]
    for (lab, _, c), e in zip(rows[1:], errs):
        a2.plot(tt, np.minimum(e, 2), color=c, label=lab)
    a2.axhline(0.5, color="grey", ls=":", lw=1)
    a2.set(xlabel="Lyapunov times" if lyap else "time", ylabel="relative error", ylim=(0, 1.6), title="error vs truth")
    cursor = a2.axvline(0, color="k", lw=1)
    stamp = a1.text(0.01, 0.95, "", transform=a1.transAxes, fontsize=10, va="top")

    def upd(i):
        for ln, (_, Y, _) in zip(lines, rows):
            ln.set_ydata(Y[i] if np.all(np.isfinite(Y[i])) else np.full_like(x, np.nan))
        cursor.set_xdata([tt[i], tt[i]])
        stamp.set_text(f"t = {tt[i]:.2f} {'Lyapunov times' if lyap else ''} after the last training snapshot")
        return lines + [cursor, stamp]
    fig.tight_layout()
    a = anim.FuncAnimation(fig, upd, frames=len(t), interval=1000 / fps, blit=False)
    a.save(path, writer=anim.FFMpegWriter(fps=fps, bitrate=2400))
    plt.close(fig)
    return str(path)


def spacetime_figure(path, x, t, rows, lyap=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    truth = rows[0][1]
    lim = np.nanmax(np.abs(truth))
    fig, axes = plt.subplots(2, len(rows), figsize=(3.6 * len(rows), 6.4), squeeze=False)
    tt = (t - t[0]) * (lyap if lyap else 1)
    for j, (lab, Y, _) in enumerate(rows):
        axes[0, j].pcolormesh(x, tt, np.where(np.isfinite(Y), Y, np.nan), cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
        axes[0, j].set_title(lab, fontsize=10)
        axes[1, j].pcolormesh(x, tt, np.where(np.isfinite(Y), Y - truth, np.nan), cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
        axes[1, j].set_title("error" if j else "", fontsize=9)
    axes[0, 0].set_ylabel("Lyapunov times" if lyap else "t")
    axes[1, 0].set_ylabel("Lyapunov times" if lyap else "t")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)


# ----------------------------------------------------------------------------- Kuramoto-Sivashinsky case
def ks_case(out="runs/oos_ks", noise=0.02, t_split=60.0, scale=(1.3, 0.7, 1.6), agent=False, seed=0,
            ks_path="examples/data/KS_data.mat", fno_minutes=6):
    """Real KS data (PySINDy tutorial), blinded by x' = a x, t' = t / b, u' = c u, so the true coefficients
    are not the textbook (-1, -1, -1):  u'_t' = -(a b / c) u' u'_x' - a^2 b u'_x'x' - a^4 b u'_x'x'x'x'
    (u'(x', t') = c u(x'/a, b t'); derived by the chain rule)."""
    from scipy.io import loadmat
    from .weakform import weak_sindy
    from .evaluate import load
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    a, b, c = scale
    m = loadmat(ks_path)
    u = np.asarray(m["uu"], float).T                     # (nt, nx)
    t = np.ravel(m["tt"]).astype(float) / b
    x = np.ravel(m["x"]).astype(float) * a
    u = u * c
    nx = u.shape[1]
    L = float(nx * (x[1] - x[0]))
    truth = {"u": f"-{a * b / c:.6g}*u*u_x - {a ** 2 * b:.6g}*u_xx - {a ** 4 * b:.6g}*u_xxxx"}
    k = int(np.searchsorted(t, t_split / b))
    rng = np.random.default_rng(seed)
    U = u[None, :, :, None]
    noisy = U + noise * U.std() * rng.standard_normal(U.shape)
    meta = {"name": "ks_blinded_oos", "kind": "pde", "variables": ["u"], "dt": float(t[1] - t[0]), "n_traj": 1,
            "shape": [1, k, nx, 1], "L": L, "nx": nx, "boundary": "periodic", "spatial_dims": ["x"],
            "allowed_symbols": ["u", "u_x", "u_xx", "u_xxx", "u_xxxx", "x"], "system": None, "noise": noise}
    d = write_dataset(out / "dataset", meta, t, noisy[:, :k], U[:, k - 1:], truth, x - x[0])
    m_, D = load(d)
    tt = t[k - 1:] - t[k - 1]
    U0 = noisy[0, k - 1]                                  # noisy observed state at the start of the test window
    Ut = U[0, k - 1:]
    lay = solvers.pde_layout(m_)
    res = {"truth": truth, "noise": noise, "train_window": [0, float(t[k - 1])], "test_window": [float(t[k - 1]), float(t[-1])]}
    lam = lyapunov_exponent(["u"], truth, m_, Ut[0], tt)
    res["lyapunov_exponent"] = lam
    print("truth:", truth, "| Lyapunov exponent", lam, flush=True)
    rows = [("truth (clean future)", Ut[..., 0], "k")]
    # reference: true PDE from the noisy state (chaos limit)
    Yt = solvers.integrate_pde_general(["u"], truth, lay, U0, tt)
    rows.append(("true PDE from noisy state", Yt[..., 0], "0.55"))
    # weak SINDy (no LLM)
    w = weak_sindy(m_, D, poly_degree=2, max_deriv=4)
    res["weak_sindy"] = {"rhs": w["rhs"]}
    Yw = solvers.integrate_pde_general(["u"], w["rhs"], lay, U0, tt)
    rows.append(("weak SINDy (no LLM)", Yw[..., 0], "tab:green"))
    # agent (LLM pipeline), refitted coefficients
    if agent:
        from .orchestrate import discover
        r = discover(d, n_branches=2, adversary=False, max_tools=16, out_dir=out / "discover", verbose=False)
        sub = r["final_model"]
        refit, table = refit_rhs(m_, D, sub)
        res["agent"] = {"submitted": sub, "refit": refit, "coefficients": table, "verdict": r["verdict"],
                        "cost_usd": r["cost_usd"], "report": r.get("report")}
        Ya = solvers.integrate_pde_general(["u"], refit, lay, U0, tt)
        rows.append(("eqdisc agent (refit)", Ya[..., 0], "tab:red"))
    # FNO trained on the same noisy training window
    from .fno import rollout_fno, train_fno
    t0 = time.time()
    model = train_fno(noisy[:, :k], epochs=400, modes=24, width=48, max_minutes=fno_minutes)
    Yf = rollout_fno(model, U0, len(tt) - 1)
    res["fno"] = {"train_minutes": round((time.time() - t0) / 60, 1)}
    rows.append(("FNO (same noisy data)", Yf[..., 0], "tab:blue"))
    # metrics
    res["valid_time_lyapunov"] = {}
    for lab, Y, _ in rows[1:]:
        e = rel_err_t(Y, Ut[..., 0])
        vt = valid_time(e, tt)
        res["valid_time_lyapunov"][lab] = round(vt * lam, 2) if lam else None
        res.setdefault("valid_time", {})[lab] = round(vt, 2)
    print(json.dumps({"valid_time_lyapunov": res["valid_time_lyapunov"], "weak": res["weak_sindy"]["rhs"]}), flush=True)
    xs = x - x[0]
    spacetime_figure(out / "ks_spacetime.png", xs, tt, rows, lyap=lam)
    video_1d(out / "ks_oos.mp4", xs, tt, rows, "Kuramoto–Sivashinsky (blinded, 2% noise): forecasting the unseen future", lyap=lam)
    np.savez_compressed(out / "rollouts.npz", t=tt, x=xs, **{f"Y{i}": Y for i, (_, Y, _) in enumerate(rows)},
                        labels=np.array([lab for lab, _, _ in rows]))
    (out / "results.json").write_text(json.dumps(res, indent=1, default=str))
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("case", choices=["ks"])
    p.add_argument("--agent", action="store_true")
    p.add_argument("--noise", type=float, default=0.02)
    a = p.parse_args()
    ks_case(agent=a.agent, noise=a.noise)


# ----------------------------------------------------------------------------- LAGEOS-1 (real satellite)
MU_E, RE_E = 3.986004414498200e14, 6.378136460000000e6      # values given with the orbit_discover challenge data
T_E = np.sqrt(RE_E ** 3 / MU_E)


def load_lageos(path):
    import pandas as pd
    df = pd.read_csv(path)
    X = np.hstack([df[["X", "Y", "Z"]].values / RE_E, df[["VX", "VY", "VZ"]].values / (RE_E / T_E)])
    t = np.arange(len(X)) * 3600 / T_E
    return t, X, pd.to_datetime(df.DATE)


def node_angle(X):
    h = np.cross(X[:, :3], X[:, 3:])
    return np.degrees(np.unwrap(np.arctan2(h[:, 0], -h[:, 1])))


def train_mlp_step(X, steps=4000, seed=0, max_minutes=4):
    """Neural baseline for an ODE: MLP mapping state_i -> state_{i+1} - state_i, rolled out autoregressively."""
    import time as _t
    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(seed)
    A, B = X[:-1], X[1:] - X[:-1]
    mu_a, sd_a, mu_b, sd_b = A.mean(0), A.std(0) + 1e-12, B.mean(0), B.std(0) + 1e-12
    xa = torch.tensor((A - mu_a) / sd_a, dtype=torch.float32, device=dev)
    xb = torch.tensor((B - mu_b) / sd_b, dtype=torch.float32, device=dev)
    net = torch.nn.Sequential(torch.nn.Linear(6, 256), torch.nn.GELU(), torch.nn.Linear(256, 256), torch.nn.GELU(),
                              torch.nn.Linear(256, 256), torch.nn.GELU(), torch.nn.Linear(256, 6)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    t0 = _t.time()
    for it in range(steps):
        i = torch.randint(0, len(xa), (512,), device=dev)
        loss = torch.mean((net(xa[i]) - xb[i]) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if _t.time() - t0 > max_minutes * 60:
            break
    net.eval()

    def roll(x0, n):
        out = [x0]
        x = x0.copy()
        with torch.no_grad():
            for _ in range(n):
                z = torch.tensor(((x - mu_a) / sd_a)[None], dtype=torch.float32, device=dev)
                x = x + net(z).cpu().numpy()[0] * sd_b + mu_b
                out.append(x.copy())
        return np.array(out)
    return roll


def lageos_case(path, out="runs/oos_lageos", train_days=365, test_days=30, long_years=6, agent=False):
    from .flow import fit_flow, make_flow_fn
    from scipy.integrate import solve_ivp
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t, X, dates = load_lageos(path)
    n_tr = 24 * train_days
    data = {"U": X[None, :n_tr], "t": t[:n_tr]}
    meta = {"name": "lageos1_train", "kind": "ode", "variables": ["x", "y", "z", "vx", "vy", "vz"], "dt": float(t[1] - t[0]),
            "allowed_symbols": ["x", "y", "z", "vx", "vy", "vz", "t"], "n_traj": 1, "shape": [1, n_tr, 6], "system": None,
            "units": "lengths in Earth radii, time in sqrt(Re^3/mu) = 806.8 s", "sampling": "hourly"}
    R = "sqrt(x**2+y**2+z**2)"
    kin = {"x": "vx", "y": "vy", "z": "vz"}
    skeletons = {
        "Kepler": {**kin, **{v: f"-p0*{c}/{R}**3" for v, c in (("vx", "x"), ("vy", "y"), ("vz", "z"))}},
        "Kepler + J2": {**kin,
                        "vx": f"-p0*x/{R}**3 - 1.5*p1*x/{R}**5*(1 - 5*z**2/{R}**2)",
                        "vy": f"-p0*y/{R}**3 - 1.5*p1*y/{R}**5*(1 - 5*z**2/{R}**2)",
                        "vz": f"-p0*z/{R}**3 - 1.5*p1*z/{R}**5*(3 - 5*z**2/{R}**2)"}}
    fits, res = {}, {"train_window": [str(dates[0].date()), str(dates[n_tr - 1].date())]}
    for name, sk in skeletons.items():
        f = fit_flow(meta, data, sk, max_pairs=800, init=[1.0] + ([1e-3] if "J2" in name else []))
        fits[name] = f
        res[name] = {k: f[k] for k in ("params", "param_sigma", "one_step_rel_err_heldout")}
        print(name, f["params"], f["param_sigma"], "%.2e" % f["one_step_rel_err_heldout"], flush=True)
    if agent:
        from .agent import run_agent
        d = write_ode_dataset(out / "dataset", meta, data)
        r = run_agent(d, max_tools=16, verbose=False, out_dir=out / "agent", final_assessment=False,
                      context="Hourly position and velocity of a satellite in an inertial frame, nondimensionalised "
                              "(lengths in Earth radii, time in sqrt(Re^3/mu)). Discover the equations of motion.")
        res["agent"] = {"submitted": (r.get("submitted") or {}).get("rhs"), "cost_usd": r["cost_usd"]}
    # out-of-sample forecasts from the first held-out observation
    i0 = n_tr
    n_te = 24 * test_days
    tt = t[i0:i0 + n_te + 1] - t[i0]
    truth = X[i0:i0 + n_te + 1]
    preds = {}
    for name, f in fits.items():
        rhs = make_flow_fn(meta["variables"], f["rhs"], [])
        sol = solve_ivp(lambda s, y: rhs(y[None], [])[0], (0, tt[-1]), X[i0], t_eval=tt, rtol=1e-11, atol=1e-13,
                        method="DOP853")
        preds[name] = sol.y.T
    roll = train_mlp_step(X[:n_tr])
    preds["neural step model (MLP)"] = roll(X[i0], n_te)
    km = lambda P: np.linalg.norm(P[:, :3] - truth[:, :3], axis=1) * RE_E / 1e3
    res["position_error_km"] = {name: {f"{d}d": float(km(P)[min(24 * d, n_te)]) for d in (1, 7, 30)} for name, P in preds.items()}
    print(json.dumps(res["position_error_km"], indent=1), flush=True)
    # long-term: secular node drift over the following years (J2 model propagated with a coarse output grid)
    n_long = 24 * 365 * long_years
    sel = np.arange(i0, min(i0 + n_long, len(X)), 24)
    tl = t[sel] - t[i0]
    node = {"data": node_angle(X[sel])}
    for name, f in fits.items():
        rhs = make_flow_fn(meta["variables"], f["rhs"], [])
        sol = solve_ivp(lambda s, y: rhs(y[None], [])[0], (0, tl[-1]), X[i0], t_eval=tl, rtol=1e-10, atol=1e-12,
                        method="DOP853")
        node[name] = node_angle(sol.y.T)
    days = tl * T_E / 86400
    res["node_rate_deg_per_day"] = {k: float(np.polyfit(days, v, 1)[0]) for k, v in node.items()}
    print(res["node_rate_deg_per_day"], flush=True)
    np.savez_compressed(out / "forecasts.npz", tt=tt, truth=truth, days_long=days,
                        **{f"P_{k}": v for k, v in preds.items()}, **{f"node_{k}": v for k, v in node.items()})
    (out / "results.json").write_text(json.dumps(res, indent=1, default=str))
    lageos_figures(out, tt, truth, preds, days, node, res)
    return res


def write_ode_dataset(d, meta, data):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "data.npz", t=data["t"], U=data["U"])
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    return d


def _densify(P, dt, per):
    """Fill each sampling interval of a (n, 6) nondimensional orbit (mu = 1) with `per` sub-steps by two-body
    propagation from both ends, blended linearly so the curve hits every sample exactly. Returns positions (m, 3)."""
    from scipy.integrate import solve_ivp

    def kepler(t, y):
        Y = y.reshape(-1, 6)
        r3 = np.linalg.norm(Y[:, :3], axis=1, keepdims=True) ** 3
        return np.hstack([Y[:, 3:], -Y[:, :3] / r3]).ravel()
    s = np.linspace(0, dt, per + 1)
    fw = solve_ivp(kepler, (0, dt), P[:-1].ravel(), t_eval=s, rtol=1e-10, atol=1e-12).y      # (6(n-1), per+1)
    bw = solve_ivp(kepler, (0, -dt), P[1:].ravel(), t_eval=-s, rtol=1e-10, atol=1e-12).y[:, ::-1]
    fw, bw = fw.reshape(len(P) - 1, 6, -1), bw.reshape(len(P) - 1, 6, -1)
    w = s / dt
    seg = (1 - w) * fw[:, :3] + w * bw[:, :3]                                                 # (n-1, 3, per+1)
    out = np.concatenate([seg[:, :, :-1].transpose(0, 2, 1).reshape(-1, 3), P[-1:, :3]])
    return out


def lageos_figures(out, tt, truth, preds, days, node, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.animation as anim
    import matplotlib.pyplot as plt
    colors = {"Kepler": "tab:orange", "Kepler + J2": "tab:red", "neural step model (MLP)": "tab:blue"}
    hrs = tt * T_E / 3600
    # 1) error vs time + node drift
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    for k, P in preds.items():
        a1.semilogy(hrs / 24, np.maximum(np.linalg.norm(P[:, :3] - truth[:, :3], axis=1) * RE_E / 1e3, 1e-3),
                    color=colors.get(k), label=k)
    a1.set(xlabel="days after the end of the training year", ylabel="position error (km)",
           title="LAGEOS-1: forecasting the unseen next month")
    a1.legend(fontsize=8)
    a2.plot(days / 365.25, node["data"] - node["data"][0], "k", lw=2.5, label="measured")
    for k in ("Kepler", "Kepler + J2"):
        a2.plot(days / 365.25, node[k] - node[k][0], color=colors[k], ls="--", label=f"{k} model")
    a2.set(xlabel="years after the training year", ylabel="orbit-plane node angle change (deg)",
           title="the orbit plane precesses: only J2 explains it")
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "lageos_errors.png", dpi=120)
    plt.close(fig)
    # 2) 3-D video: real track vs forecasts over the first 3 days, Earth drawn to scale. Hourly samples are only ~4
    # per orbit, so each hourly gap is filled by two-body propagation (forward from sample i and backward from sample
    # i+1, blended), which passes through every sample exactly; the interpolator's own bias over 1 h is a few km.
    n = min(len(tt), 24 * 3 + 1)
    per, sm = 8, 3                                   # frames per hour; line points per frame step
    fine_t = np.linspace(hrs[0], hrs[n - 1], per * (n - 1) + 1)
    tracks = {"measured": (_densify(truth[:n], tt[1] - tt[0], per * sm), "k")} | \
             {k: (_densify(P[:n], tt[1] - tt[0], per * sm), colors[k]) for k, P in preds.items()}
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    u_, v_ = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    ax.plot_surface(np.cos(u_) * np.sin(v_), np.sin(u_) * np.sin(v_), np.cos(v_), color="#3a6ea5", alpha=0.6, linewidth=0)
    ax.set_box_aspect((1, 1, 1))
    lim = 1.5
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim))
    ax.set_axis_off()
    lines, dots = {}, {}
    for k, (P, c) in tracks.items():
        lines[k], = ax.plot([], [], [], color=c, lw=2.5 if k == "measured" else 1.4, label=k)
        dots[k], = ax.plot([], [], [], "o", color=c, ms=6)
    ax.legend(loc="upper left", fontsize=8)
    title = ax.set_title("")
    truth_f = tracks["measured"][0]

    def upd(f):
        for k, (P, c) in tracks.items():
            seg = P[max(0, sm * (f - 10 * per)):sm * f + 1]
            lines[k].set_data(seg[:, 0], seg[:, 1])
            lines[k].set_3d_properties(seg[:, 2])
            dots[k].set_data([seg[-1, 0]], [seg[-1, 1]])
            dots[k].set_3d_properties([seg[-1, 2]])
        err = {k: np.linalg.norm(P[sm * f] - truth_f[sm * f]) * RE_E / 1e3 for k, (P, _) in tracks.items() if k != "measured"}
        title.set_text(f"LAGEOS-1, {fine_t[f]:.1f} h into the unseen future\n" +
                       "  ".join(f"{k}: {e:,.0f} km" for k, e in err.items()))
        ax.view_init(elev=20, azim=30 + 0.25 * f * 6 / per)
        return list(lines.values()) + list(dots.values())
    a = anim.FuncAnimation(fig, upd, frames=len(fine_t), interval=40)
    a.save(out / "lageos_forecast.mp4", writer=anim.FFMpegWriter(fps=25, bitrate=3000))
    plt.close(fig)


def lageos_agent(path, out="runs/oos_lageos", train_days=365, test_days=30, max_tools=16):
    """Agent arm for LAGEOS: the agent sees only the 2017 training year (nondimensional, inertial frame) and must find
    the law itself; its submitted model is then forecast out of sample exactly like the scripted fits."""
    from scipy.integrate import solve_ivp
    from .agent import run_agent
    from .flow import fit_flow, make_flow_fn
    out = Path(out)
    t, X, dates = load_lageos(path)
    n_tr = 24 * train_days
    meta = {"name": "satellite_hourly", "kind": "ode", "variables": ["x", "y", "z", "vx", "vy", "vz"],
            "dt": float(t[1] - t[0]), "allowed_symbols": ["x", "y", "z", "vx", "vy", "vz", "t"], "n_traj": 1,
            "shape": [1, n_tr, 6], "system": None}
    d = write_ode_dataset(out / "agent_dataset", meta, {"U": X[None, :n_tr], "t": t[:n_tr]})
    r = run_agent(d, max_tools=max_tools, verbose=True, out_dir=out / "agent", final_assessment=False,
                  context="Hourly measured position (x,y,z) and velocity (vx,vy,vz) of an Earth satellite in an inertial "
                          "frame, nondimensionalised: lengths in Earth radii, time in sqrt(Re^3/mu). Discover the "
                          "equations of motion.")
    sub = (r.get("submitted") or {}).get("rhs")
    res = {"submitted": sub, "cost_usd": r["cost_usd"], "n_tool_calls": r["n_tool_calls"],
           "tools": [ev["name"] for ev in json.loads((out / "agent" / "transcript.json").read_text()) if ev["type"] == "tool"]}
    if sub:
        rhs = make_flow_fn(meta["variables"], sub, [])
        n_te = 24 * test_days
        tt = t[n_tr:n_tr + n_te + 1] - t[n_tr]
        sol = solve_ivp(lambda s_, y: rhs(y[None], [])[0], (0, tt[-1]), X[n_tr], t_eval=tt, rtol=1e-11, atol=1e-13, method="DOP853")
        km = np.linalg.norm(sol.y.T[:, :3] - X[n_tr:n_tr + n_te + 1, :3], axis=1) * RE_E / 1e3
        res["position_error_km"] = {f"{dd}d": float(km[min(24 * dd, n_te)]) for dd in (1, 7, 30)}
    (out / "agent_results.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str)[:2000])
    return res


# ----------------------------------------------------------------------------- Gray-Scott (The Well)
def gs_case(regime="spirals", noise=0.05, out=None, agent_rhs=None, n_steps=30, fno_minutes=8):
    """Train on 2 noisy trajectories (60 snapshots each); forecast the held-out trajectory from its NOISY first
    frame; compare true PDE, weak SINDy (no LLM), the agent's PDE (refit) and an FNO trained on the same data."""
    from .evaluate import load
    from .weakform import weak_sindy
    from .well_gs import build
    out = Path(out or f"runs/oos_gs_{regime}_n{noise:g}")
    out.mkdir(parents=True, exist_ok=True)
    d = build(regime, noise=noise)
    m, D = load(d)
    truth = json.loads((d / "hidden" / "truth.json").read_text())
    te = np.load(d / "hidden" / "test.npz")
    Ut, t = te["U"][0][: n_steps + 1], te["t"][: n_steps + 1]
    rng = np.random.default_rng(1)
    scale = D["U"].reshape(-1, 2).std(0)
    U0 = Ut[0] + noise * scale * rng.standard_normal(Ut[0].shape)
    lay = solvers.pde_layout(m)
    roll = lambda rhs: solvers.integrate_pde_general(["A", "B"], rhs, lay, U0, t, dt_sim=1.0)
    rows = [("truth (held-out trajectory)", Ut), ("true PDE from noisy frame", roll(truth["rhs"]))]
    res = {"truth": truth["rhs"], "regime": regime, "noise": noise}
    w = weak_sindy(m, D, poly_degree=3, max_deriv=2)
    res["weak_sindy"] = w["rhs"]
    rows.append(("weak SINDy (no LLM)", roll(w["rhs"])))
    if agent_rhs:
        refit, table = refit_rhs(m, D, agent_rhs)
        res["agent"] = {"submitted": agent_rhs, "refit": refit, "coefficients": table}
        rows.append(("eqdisc agent (refit)", roll(refit)))
    from .fno import rollout_fno, train_fno
    model = train_fno(D["U"], epochs=300, modes=20, width=40, max_minutes=fno_minutes)
    rows.append(("FNO (same noisy data)", rollout_fno(model, U0, n_steps)))

    def vrmse(Y):
        v = []
        for i in range(len(t)):
            if not np.all(np.isfinite(Y[i])):
                v.append(np.inf)
                continue
            v.append(float(np.mean([np.sqrt(np.mean((Y[i, ..., f] - Ut[i, ..., f]) ** 2) /
                                            np.mean((Ut[i, ..., f] - Ut[i, ..., f].mean()) ** 2)) for f in range(2)])))
        return np.array(v)
    res["vrmse"] = {lab: {"1": float(vrmse(Y)[1]), "6-12": float(np.mean(vrmse(Y)[6:13])),
                          "13-30": float(np.mean(vrmse(Y)[13:31]))} for lab, Y in rows[1:]}
    print(json.dumps(res["vrmse"], indent=1), flush=True)
    np.savez_compressed(out / "rollouts.npz", t=t, **{f"Y{i}": Y.astype(np.float32) for i, (_, Y) in enumerate(rows)},
                        labels=np.array([lab for lab, _ in rows]))
    (out / "results.json").write_text(json.dumps(res, indent=1, default=str))
    gs_video(out / "gs_forecast.mp4", t, rows, f"Gray–Scott ({regime}, {int(noise * 100)}% noise): forecasting a held-out trajectory")
    return res


def gs_video(path, t, rows, title, field=1, fps=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.animation as anim
    import matplotlib.pyplot as plt
    show = [r for r in rows if not r[0].startswith("true PDE")]
    fig, axes = plt.subplots(1, len(show), figsize=(3.3 * len(show), 3.8))
    vmin, vmax = float(np.nanmin(rows[0][1][..., field])), float(np.nanmax(rows[0][1][..., field]))
    ims = []
    for ax, (lab, Y) in zip(axes, show):
        ims.append(ax.imshow(Y[0, ..., field].T, origin="lower", cmap="magma", vmin=vmin, vmax=vmax))
        ax.set_title(lab, fontsize=9)
        ax.set_axis_off()
    sup = fig.suptitle(f"{title}\nstep 0", fontsize=11)

    def upd(i):
        for im, (_, Y) in zip(ims, show):
            im.set_data(np.where(np.isfinite(Y[i, ..., field]), Y[i, ..., field], np.nan).T)
        sup.set_text(f"{title}\nstep {i} (Δt = 10 per step) after the noisy first frame")
        return ims
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    a = anim.FuncAnimation(fig, upd, frames=len(t), interval=1000 / fps)
    a.save(path, writer=anim.FFMpegWriter(fps=fps, bitrate=2400))
    plt.close(fig)
