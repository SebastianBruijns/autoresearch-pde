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


def lyapunov_benettin(fields, rhs, meta, U0, tau, n_renorm=80, n_skip=10, eps=1e-6, seed=0):
    """Largest Lyapunov exponent of the TRUE system by Benettin's method: integrate a reference and a perturbed copy
    for tau, renormalise the separation back to eps, repeat; average the log growth after n_skip transients."""
    rng = np.random.default_rng(seed)
    lay = solvers.pde_layout(meta)
    tt = np.array([0.0, tau / 2, tau])
    a = np.array(U0, float)
    d0 = eps * np.std(a)
    pert = rng.standard_normal(a.shape)
    b = a + d0 * pert / np.sqrt(np.mean(pert ** 2))
    logs = []
    for i in range(n_renorm):
        a = solvers.integrate_pde_general(fields, rhs, lay, a, tt)[-1]
        b = solvers.integrate_pde_general(fields, rhs, lay, b, tt)[-1]
        d = np.sqrt(np.mean((b - a) ** 2))
        if i >= n_skip:
            logs.append(np.log(d / d0))
        b = a + (b - a) * (d0 / d)
    return float(np.mean(logs) / tau), float(np.std(logs) / tau / np.sqrt(len(logs)))


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


def spacetime_video(path, x, t, truth, models, lyap=None, title="", fps=15, nx=256, valid=None):
    """'Creeping' space-time video: the field u(x, t) is revealed left to right as time advances, for the truth and
    each model (filled contours), with each model's error |model - truth| in the rows below.
    models: list of (label, Y); valid: optional {label: valid time in Lyapunov times} drawn as dashed lines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.animation as anim
    import matplotlib.pyplot as plt
    st = max(1, len(x) // nx)
    xs, U = x[::st], truth[:, ::st]
    T = (t - t[0]) * (lyap or 1.0)
    tlab = "Lyapunov times into the unseen future" if lyap else "time into the unseen future"
    vmax = float(np.abs(U).max())
    lv = np.linspace(-vmax, vmax, 31)
    E = [(lab, np.abs(Y[:, ::st] - U)) for lab, Y in models]
    emax = float(np.percentile(np.concatenate([e.ravel() for _, e in E]), 99))
    le = np.linspace(0, emax, 21)
    panels = [("truth", U, "field")] + [(lab, Y[:, ::st], "field") for lab, Y in models] + \
             [(f"error of {lab}", e, "err") for lab, e in E]
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 1.55 * len(panels) + 0.9), sharex=True)
    sup = fig.suptitle(title, fontsize=16, fontweight="bold")
    for ax, (lab, _, kind) in zip(axes, panels):
        ax.set_ylabel("x", fontsize=8)
        ax.set_xlim(T[0], T[-1])
        ax.set_ylim(xs[0], xs[-1])
        ax.tick_params(labelsize=7)
    axes[-1].set_xlabel(tlab, fontsize=13)
    m1 = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-vmax, vmax))
    m2 = plt.cm.ScalarMappable(cmap="magma", norm=plt.Normalize(0, emax))
    fig.colorbar(m1, ax=axes[:1 + len(models)], fraction=0.015, pad=0.01, label="u")
    fig.colorbar(m2, ax=axes[1 + len(models):], fraction=0.015, pad=0.01, label="|error|")

    def upd(i):
        k = max(i + 1, 2)
        for ax, (lab, Z, kind) in zip(axes, panels):
            for c in list(ax.collections):
                c.remove()
            for ln in list(ax.lines):
                ln.remove()
            if kind == "field":
                ax.contourf(T[:k], xs, Z[:k].T, levels=lv, cmap="RdBu_r", extend="both")
            else:
                ax.contourf(T[:k], xs, np.minimum(Z[:k], emax).T, levels=le, cmap="magma")
            ax.axvline(T[k - 1], color="k", lw=0.8)
            name = lab.replace("error of ", "")
            if valid and name in valid and valid[name] <= T[k - 1]:
                ax.axvline(valid[name], color="w" if kind == "err" else "k", ls="--", lw=1.2)
            ax.set_title(lab + (f"   · useful for {valid[name]:.1f} Lyapunov times" if valid and name in valid
                                and kind == "field" else ""), fontsize=13, loc="left", fontweight="bold")
        return []
    fig.tight_layout(rect=(0, 0, 0.93, 0.95))
    a = anim.FuncAnimation(fig, upd, frames=len(T), interval=1000 / fps)
    a.save(path, writer=anim.FFMpegWriter(fps=fps, bitrate=3000))
    plt.close(fig)


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
    meta = {"name": "dataset", "kind": "pde", "variables": ["u"], "dt": float(t[1] - t[0]), "n_traj": 1,
            "shape": [1, k, nx, 1], "L": L, "nx": nx, "boundary": "periodic", "spatial_dims": ["x"],
            "allowed_symbols": ["u", "u_x", "u_xx", "u_xxx", "u_xxxx", "x"], "system": None, "noise": noise}
    d = write_dataset(out / "dataset", meta, t, noisy[:, :k], U[:, k - 1:], truth, x - x[0])
    m_, D = load(d)
    tt = t[k - 1:] - t[k - 1]
    U0 = noisy[0, k - 1]                                  # noisy observed state at the start of the test window
    Ut = U[0, k - 1:]
    lay = solvers.pde_layout(m_)
    res = {"truth": truth, "noise": noise, "train_window": [0, float(t[k - 1])], "test_window": [float(t[k - 1]), float(t[-1])]}
    lam, lam_se = lyapunov_benettin(["u"], truth, m_, Ut[0], tau=5.0, n_renorm=100, n_skip=10)
    res["lyapunov_exponent"], res["lyapunov_se"] = lam, lam_se
    res["lyapunov_twin_crude"] = lyapunov_exponent(["u"], truth, m_, Ut[0], tt)
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
    ks_lab = {"eqdisc agent (refit)": "eqdisc: equation found from the data",
              "FNO (same noisy data)": "neural operator (FNO) trained on the same data"}
    mods = [(ks_lab[lab], Y) for lab, Y, _ in rows if lab in ks_lab]
    vt_ = res.get("valid_time_lyapunov", {})
    spacetime_video(out / "ks_spacetime.mp4", xs, tt, rows[0][1], mods, lyap=lam, valid={ks_lab[k]: v for k, v in vt_.items() if k in ks_lab},
                    title="Kuramoto–Sivashinsky (chaotic; blinded; trained on 2%-noise data): forecasting the unseen future")
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
    meta = {"name": "dataset", "kind": "ode", "variables": ["x", "y", "z", "vx", "vy", "vz"], "dt": float(t[1] - t[0]),
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
                      context=None)
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
    lageos_video(out / "lageos_forecast.mp4", tt, truth, {k: P for k, P in preds.items() if k != "Kepler + J2"})


LAGEOS_COLORS = {"data-only agent": "#16a34a", "Kepler + J2": "tab:red", "Kepler": "tab:orange",
                 "neural step model (MLP)": "#7c3aed"}
# plain-language names for the audience (legend) and short ones (running title)
PLAIN = {"data-only agent": "eqdisc", "neural step model (MLP)": "neural network", "Kepler": "round-Earth gravity",
         "Kepler + J2": "textbook law", "measured": "real satellite"}
SHORT = {"data-only agent": "eqdisc", "neural step model (MLP)": "neural net", "Kepler": "round Earth",
         "Kepler + J2": "textbook"}


def lageos_video(path, tt, truth, preds, days=3, colors=None, name="LAGEOS-1", title_where="Close-up"):
    """Left: 3-D view, Earth to scale, real track vs forecasts. Right: the same forecasts seen from the real
    satellite (offset in km, along-track vs out-of-plane; radial offsets are small for all models), so errors
    invisible at orbit scale become visible.
    Hourly samples are only ~4 per orbit, so each gap is filled by two-body propagation from both ends (exact at
    every sample; the interpolator's own bias over 1 h is a few km)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.animation as anim
    import matplotlib.pyplot as plt
    colors = {**LAGEOS_COLORS, **(colors or {})}
    hrs = tt * T_E / 3600
    dt_h = float(hrs[1] - hrs[0])
    if dt_h > 0.25:                                  # coarse (hourly) samples: densify by two-body propagation
        n = min(len(tt), int(round(24 * days / dt_h)) + 1)
        per, sm = 8, 3                               # frames per hour; line points per frame step
        fine_t = np.linspace(hrs[0], hrs[n - 1], per * (n - 1) + 1)
        dense = lambda P: _densify(P[:n], tt[1] - tt[0], per * sm)
    else:                                            # already dense: one frame every ~1/8 h of samples
        sm = max(1, int(round(0.125 / dt_h)))
        n = min(len(tt), int(round(24 * days / dt_h)) + 1)
        n = (n - 1) // sm * sm + 1
        per = 8
        fine_t = hrs[:n:sm]
        dense = lambda P: np.asarray(P[:n, :3])
    T = dense(truth)
    # truth velocity direction for the local frame, from the dense track (central differences)
    V = np.gradient(T, axis=0)
    R_ = T / np.linalg.norm(T, axis=1, keepdims=True)
    N_ = np.cross(T, V)
    N_ /= np.linalg.norm(N_, axis=1, keepdims=True)
    A_ = np.cross(N_, R_)
    tracks = {k: dense(P) for k, P in preds.items()}
    off = {k: np.stack([np.sum((P - T) * A_, 1), np.sum((P - T) * N_, 1)], 1) * RE_E / 1e3 for k, P in tracks.items()}
    lim = 1.15 * max(np.abs(o).max() for o in off.values())
    fig = plt.figure(figsize=(13, 6.2))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    bx = fig.add_subplot(1, 2, 2)
    u_, v_ = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    ax.plot_surface(np.cos(u_) * np.sin(v_), np.sin(u_) * np.sin(v_), np.cos(v_), color="#3a6ea5", alpha=0.6, linewidth=0)
    ax.set_box_aspect((1, 1, 1))
    ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), zlim=(-1.5, 1.5))
    ax.set_axis_off()
    lines, dots, olines, odots = {}, {}, {}, {}
    lines["measured"], = ax.plot([], [], [], color="k", lw=2.5, label=PLAIN["measured"])
    dots["measured"], = ax.plot([], [], [], "o", color="k", ms=6)
    for k in tracks:
        lines[k], = ax.plot([], [], [], color=colors.get(k), lw=1.4, label=PLAIN.get(k, k))
        dots[k], = ax.plot([], [], [], "o", color=colors.get(k), ms=6)
        olines[k], = bx.plot([], [], color=colors.get(k), lw=1.5, alpha=0.6)
        odots[k], = bx.plot([], [], "o", color=colors.get(k), ms=9, label=PLAIN.get(k, k))
    bx.plot([0], [0], "k+", ms=22, mew=3, label="real satellite")
    bx.set(xlim=(-lim, lim), ylim=(-lim, lim))
    bx.set_xlabel("ahead / behind (km)", fontsize=13)
    bx.set_ylabel("off the orbit plane (km)", fontsize=13)
    bx.set_title("Seen from the real satellite", fontsize=16, fontweight="bold")
    bx.set_aspect("equal")
    bx.grid(alpha=0.3)
    bx.legend(loc="upper left", fontsize=11)
    ax.legend(loc="upper left", fontsize=11)
    ax.set_title("The orbit", fontsize=16, fontweight="bold")
    title = fig.suptitle("", fontsize=15)

    def upd(f):
        g = sm * f
        a0 = max(0, sm * (f - 10 * per))
        for k, P in [("measured", T)] + list(tracks.items()):
            seg = P[a0:g + 1]
            lines[k].set_data(seg[:, 0], seg[:, 1])
            lines[k].set_3d_properties(seg[:, 2])
            dots[k].set_data([seg[-1, 0]], [seg[-1, 1]])
            dots[k].set_3d_properties([seg[-1, 2]])
        for k, o in off.items():
            seg = o[max(0, g - 6 * per * sm):g + 1]
            olines[k].set_data(seg[:, 0], seg[:, 1])
            odots[k].set_data([o[g, 0]], [o[g, 1]])
        err = {k: np.linalg.norm(tracks[k][g] - T[g]) * RE_E / 1e3 for k in tracks}
        title.set_text(f"hour {fine_t[f]:.0f} of the unseen future:   " +
                       "    ".join(f"{SHORT.get(k, k)} {e:,.0f} km off" if e >= 10 else f"{SHORT.get(k, k)} {e:.1f} km off"
                                  for k, e in err.items()))
        ax.view_init(elev=20, azim=30 + 0.25 * f * 6 / per)
        return list(lines.values()) + list(dots.values())
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    a = anim.FuncAnimation(fig, upd, frames=len(fine_t), interval=40)
    a.save(path, writer=anim.FFMpegWriter(fps=25, bitrate=3000))
    plt.close(fig)


def parametrize_floats(rhs):
    """Replace every decimal constant in a law (not exponents like **1.5, not integers) by p0, p1, ... (equal values
    share one parameter). Returns (rhs_with_params, initial values)."""
    import re
    vals, out = [], {}
    pat = re.compile(r"(?<![\w.*])(\d+\.\d*(?:[eE][-+]?\d+)?|\d+[eE][-+]?\d+)")

    def sub_(m, s):
        if re.search(r"\*\*\(?-?$", s[max(0, m.start() - 4):m.start()]):      # exponent: **1.5, **(-1.5)
            return m.group(0)
        v = float(m.group(0))
        if v not in vals:
            vals.append(v)
        return f"p{vals.index(v)}"
    for k, e in rhs.items():
        out[k] = pat.sub(lambda m: sub_(m, e), e)
    return out, vals


def lageos_refit(path, sub, train_days=365, units=None, max_pairs=1500):
    """Protocol refit of the agent's ODE law: keep its structure, re-estimate its constants by flow-map shooting on the
    2017 training data (in the agent's own units), so typed/rounded numbers are never scored."""
    from .flow import fit_flow
    t, X, _ = load_lageos(path)
    sc = np.array([units[0]] * 3 + [units[0] / units[1]] * 3) if units else np.ones(6)
    ts = units[1] if units else 1.0
    n_tr = 24 * train_days
    names = [f"u{i}" for i in range(1, 7)]
    meta = {"kind": "ode", "variables": names, "dt": float(t[1] - t[0]) * ts}
    withp, init = parametrize_floats(sub)
    f = fit_flow(meta, {"U": (X[:n_tr] * sc)[None], "t": t[:n_tr] * ts}, withp, max_pairs=max_pairs, init=init)
    return f["rhs"], {"submitted_constants": init, "refit_constants": list(f["params"].values()),
                      "sigma": list(f["param_sigma"].values()), "one_step_rel_err_heldout": f["one_step_rel_err_heldout"]}


def lageos_agent_forecast(path, sub, train_days=365, test_days=30, units=None):
    """Integrate the agent's submitted law (in its own, possibly blinded, units) from the last training state over the
    unseen window. Returns (tt, states) in the standard nondimensional units of load_lageos."""
    from scipy.integrate import solve_ivp
    from .flow import make_flow_fn
    t, X, _ = load_lageos(path)
    sc = np.array([units[0]] * 3 + [units[0] / units[1]] * 3) if units else np.ones(6)
    ts = units[1] if units else 1.0
    n_tr, n_te = 24 * train_days, 24 * test_days
    rhs = make_flow_fn([f"u{i}" for i in range(1, 7)], sub, [])
    tt = t[n_tr:n_tr + n_te + 1] - t[n_tr]
    sol = solve_ivp(lambda s_, y: rhs(y[None], [])[0], (0, tt[-1] * ts), X[n_tr] * sc, t_eval=tt * ts,
                    rtol=1e-11, atol=1e-13, method="DOP853")
    return tt, sol.y.T / sc


def lageos_agent(path, out="runs/oos_lageos_dataonly", train_days=365, test_days=30, max_tools=16, units=None):
    """Agent arm for LAGEOS: the agent sees only the 2017 training year (nondimensional, inertial frame) and must find
    the law itself; its submitted model is then forecast out of sample exactly like the scripted fits."""
    from scipy.integrate import solve_ivp
    from .agent import run_agent
    from .flow import fit_flow, make_flow_fn
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t, X, dates = load_lageos(path)
    if units:      # blind the units: lengths x Ls, time x Ts (so mu and the Earth radius are no longer 1)
        Ls, Ts = units
        X = X * np.array([Ls] * 3 + [Ls / Ts] * 3)
        t = t * Ts
    n_tr = 24 * train_days
    # data only: neutral variable names (no position/velocity labels), no context, no domain guidance
    names = [f"u{i}" for i in range(1, 7)]
    meta = {"name": "dataset", "kind": "ode", "variables": names, "dt": float(t[1] - t[0]),
            "allowed_symbols": names + ["t"], "n_traj": 1, "shape": [1, n_tr, 6], "system": None}
    d = write_ode_dataset(out / "agent_dataset", meta, {"U": X[None, :n_tr], "t": t[:n_tr]})
    r = run_agent(d, max_tools=max_tools, verbose=True, out_dir=out / "agent", final_assessment=False,
                  context=None)
    sub = (r.get("submitted") or {}).get("rhs")
    res = {"submitted": sub, "cost_usd": r["cost_usd"], "n_tool_calls": r["n_tool_calls"], "units": units,
           "tools": [ev["name"] for ev in json.loads((out / "agent" / "transcript.json").read_text()) if ev["type"] == "tool"]}
    if sub:
        refit, res["refit"] = lageos_refit(path, sub, train_days, units)
        res["refit_rhs"] = refit
        tt, S = lageos_agent_forecast(path, refit, train_days, test_days, units)
        km = np.linalg.norm(S[:, :3] - load_lageos(path)[1][n_tr:n_tr + len(tt), :3], axis=1) * RE_E / 1e3
        res["position_error_km"] = {f"{dd}d": float(km[min(24 * dd, len(tt) - 1)]) for dd in (1, 7, 30)}
        np.savez_compressed(out / "agent_forecast.npz", tt=tt, S=S)
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
    gs_video(out / "gs_forecast.mp4", t, rows, f"Gray–Scott reaction–diffusion ({int(noise * 100)}% noise): forecasting a held-out run",
             labels={"truth (held-out trajectory)": "truth (what really happened)", "weak SINDy (no LLM)": "sparse regression, no LLM",
                     "eqdisc agent (refit)": "eqdisc: equation from the data", "FNO (same noisy data)": "neural operator (FNO), same data"})
    return res


def gs_video(path, t, rows, title, field=1, fps=6, labels=None):
    """Top row: field B for the held-out truth and each forecast. Bottom row: |forecast - truth| as filled contours."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.animation as anim
    import matplotlib.pyplot as plt
    labels = labels or {}
    show = [r for r in rows if not r[0].startswith("true PDE")]
    U = show[0][1]
    nc = len(show)
    fig, axes = plt.subplots(2, nc, figsize=(3.3 * nc, 7.0))
    vmin, vmax = float(np.nanmin(U[..., field])), float(np.nanmax(U[..., field]))
    emax = 0.6 * (vmax - vmin)
    le = np.linspace(0, emax, 16)
    ims = []
    for c, (lab, Y) in enumerate(show):
        ax = axes[0, c]
        ims.append(ax.imshow(Y[0, ..., field].T, origin="lower", cmap="magma", vmin=vmin, vmax=vmax))
        ax.set_title(labels.get(lab, lab), fontsize=14, fontweight="bold")
        ax.set_axis_off()
        axes[1, c].set_axis_off()
    axes[1, 0].text(0.5, 0.5, "errors\n(dark = right)", ha="center", va="center", fontsize=16, fontweight="bold",
                    transform=axes[1, 0].transAxes)
    sm_ = plt.cm.ScalarMappable(cmap="inferno", norm=plt.Normalize(0, emax))
    fig.colorbar(sm_, cax=fig.add_axes([0.915, 0.05, 0.012, 0.36]), label="|error|")
    sup = fig.suptitle(f"{title}\nstep 0", fontsize=16, fontweight="bold")

    def upd(i):
        for im, (_, Y) in zip(ims, show):
            im.set_data(np.where(np.isfinite(Y[i, ..., field]), Y[i, ..., field], np.nan).T)
        for c, (lab, Y) in enumerate(show[1:], 1):
            ax = axes[1, c]
            for coll in list(ax.collections):
                coll.remove()
            e = np.abs(np.nan_to_num(Y[i, ..., field], nan=vmax) - U[i, ..., field])
            ax.contourf(np.minimum(e, emax).T, levels=le, cmap="inferno", origin="lower")
            ax.set_aspect("equal")
            ax.set_title(f"error: {labels.get(lab, lab)}", fontsize=13)
        sup.set_text(f"{title}\nstep {i} of the forecast")
        return ims
    fig.subplots_adjust(left=0.02, right=0.9, top=0.86, bottom=0.03, wspace=0.12, hspace=0.18)
    a = anim.FuncAnimation(fig, upd, frames=len(t), interval=1000 / fps)
    a.save(path, writer=anim.FFMpegWriter(fps=fps, bitrate=2400))
    plt.close(fig)


# ----------------------------------------------------------------------------- synthetic orbit with a big J2
ORBIT_TRUTH = {"vx": "-x/r**3 - 0.75*x/r**5*(1 - 5*z**2/r**2)", "vy": "-y/r**3 - 0.75*y/r**5*(1 - 5*z**2/r**2)",
               "vz": "-z/r**3 - 0.75*z/r**5*(3 - 5*z**2/r**2)"}        # generator: mu = 1, Re = 1, J2 = 0.5


def refit_ode(meta, data, sub, stride=1, max_pairs=1500):
    """Protocol refit of an agent's ODE law: keep the structure, re-estimate every decimal constant by flow-map
    shooting on the training data. Returns (refit rhs, info)."""
    from .flow import fit_flow
    withp, init = parametrize_floats(sub)
    if not init:
        return sub, {"submitted_constants": []}
    f = fit_flow(meta, data, withp, max_pairs=max_pairs, init=init, stride=stride)
    return f["rhs"], {"submitted_constants": init, "refit_constants": list(f["params"].values()),
                      "sigma": list(f["param_sigma"].values()), "one_step_rel_err_heldout": f["one_step_rel_err_heldout"]}


def _std_rhs(names, rhs, sc=None, ts=1.0):
    """Callable f(y) in standard units for a law written in (possibly blinded) units x_b = sc * x, t_b = ts * t."""
    from .flow import make_flow_fn
    f = make_flow_fn(names, rhs, [])
    sc = np.ones(len(names)) if sc is None else np.asarray(sc, float)
    return lambda y: f((np.asarray(y) * sc)[None], [])[0] * ts / sc


def _shoot_state(f, t, Y, rtol=1e-9):
    """Estimate the state at t[-1] from noisy observations Y on t (backward shooting, least squares)."""
    from scipy.integrate import solve_ivp
    from scipy.optimize import least_squares
    tb = t[::-1]

    def run(x):            # x is the state at time 0 (the end of the training window); t <= 0
        return solve_ivp(lambda s_, y: f(y), (0.0, tb[-1]), x, t_eval=tb, rtol=rtol, atol=rtol * 1e-2,
                         method="DOP853").y.T[::-1]
    r = least_squares(lambda x: ((run(x) - Y) / Y.std(0)).ravel(), Y[-1], method="lm", max_nfev=200, x_scale="jac")
    return r.x


def orbit_case(path, out="runs/oos_orbit", noise=0.01, train_frac=0.5, units=(0.53, 1.7), agent=True, max_tools=20,
               mlp_stride=10, shoot_orbits=2.0):
    """Synthetic satellite with an exaggerated Earth bulge (orbit_discover Challenge1: mu = Re = 1, J2 = 0.5, 30 s
    samples over 6 days). 1% noise. The agent gets the first 3 days only, as six unnamed columns in random units, and
    no context. Every model forecasts the unseen last 3 days from a state estimated (by shooting) on the last two
    training orbits (each law its own estimate); the MLP gets the true law's estimate, the most generous start."""
    import pandas as pd
    from scipy.integrate import solve_ivp
    from .agent import run_agent
    from .flow import fit_flow
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(path)
    X = np.hstack([df[["rx", "ry", "rz"]].values / RE_E, df[["vx", "vy", "vz"]].values / (RE_E / T_E)])
    t = np.arange(len(X)) * 30.0 / T_E
    rng = np.random.default_rng(0)
    Xn = X + noise * X.std(0) * rng.standard_normal(X.shape)
    n_tr = int(len(X) * train_frac)
    std_names = ["x", "y", "z", "vx", "vy", "vz"]
    res = {"noise": noise, "train_samples": n_tr, "test_samples": len(X) - n_tr, "units": units}
    # --- laws in standard units
    with_r = lambda rhs: {k: v.replace("r**", "sqrt(x**2+y**2+z**2)**") for k, v in rhs.items()}
    truth = {"x": "vx", "y": "vy", "z": "vz", **with_r(ORBIT_TRUTH)}
    meta_std = {"kind": "ode", "variables": std_names, "dt": float(t[1] - t[0])}
    kep = fit_flow(meta_std, {"U": Xn[None, :n_tr], "t": t[:n_tr]},
                   {"x": "vx", "y": "vy", "z": "vz", **with_r({"vx": "-p0*x/r**3", "vy": "-p0*y/r**3", "vz": "-p0*z/r**3"})},
                   stride=20, init=[1.0])
    res["Kepler_mu"] = kep["params"]
    laws = {"Kepler + J2": _std_rhs(std_names, truth), "Kepler": _std_rhs(std_names, kep["rhs"])}
    # --- agent (data only, blinded)
    if agent:
        Ls, Ts = units
        sc = np.array([Ls] * 3 + [Ls / Ts] * 3)
        names = [f"u{i}" for i in range(1, 7)]
        meta = {"name": "dataset", "kind": "ode", "variables": names, "dt": float(t[1] - t[0]) * Ts,
                "allowed_symbols": names + ["t"], "n_traj": 1, "shape": [1, n_tr, 6], "system": None}
        data = {"U": (Xn[:n_tr] * sc)[None], "t": t[:n_tr] * Ts}
        d = write_ode_dataset(out / "agent_dataset", meta, data)
        r = run_agent(d, max_tools=max_tools, verbose=False, out_dir=out / "agent", final_assessment=False,
                      context=None, use_memory=False)
        sub = (r.get("submitted") or {}).get("rhs")
        ag = {"submitted": sub, "cost_usd": r["cost_usd"], "n_tool_calls": r["n_tool_calls"],
              "tools": [ev["name"] for ev in json.loads((out / "agent" / "transcript.json").read_text())
                        if ev["type"] == "tool"]}
        rat = [ev.get("input", {}).get("rationale") for ev in json.loads((out / "agent" / "transcript.json").read_text())
               if ev.get("name") == "submit"]
        ag["rationale"] = rat[-1] if rat else None
        if sub:
            refit, ag["refit"] = refit_ode(meta, data, sub, stride=20)
            ag["refit_rhs"] = refit
            laws["data-only agent"] = _std_rhs(names, refit, sc, Ts)
        res["agent"] = ag
    # --- forecasts of the unseen half, from shooting-estimated states
    k_sh = int(shoot_orbits * 18.4 / (t[1] - t[0]))
    sl = np.arange(n_tr - 1, n_tr - 1 - k_sh, -5)[::-1]       # ends exactly on the last training sample
    tt = t[n_tr - 1:] - t[n_tr - 1]
    ev = np.arange(0, len(tt), 10)                     # store every 5 min
    preds = {}
    for k, f in laws.items():
        x0 = _shoot_state(f, t[sl] - t[n_tr - 1], Xn[sl])
        preds[k] = solve_ivp(lambda s_, y: f(y), (0, tt[-1]), x0, t_eval=tt[ev], rtol=1e-10, atol=1e-12,
                             method="DOP853").y.T
        if k == "Kepler + J2":
            x0_best = x0         # oracle state estimate (true law): the most generous start for the neural net
    roll = train_mlp_step(Xn[:n_tr:mlp_stride], max_minutes=4)
    M = roll(x0_best, (len(tt) - 1) // mlp_stride)
    preds["neural step model (MLP)"] = M[np.minimum(ev // mlp_stride, len(M) - 1)]
    truth_tr = X[n_tr - 1:][ev]
    hrs = tt[ev] * T_E / 3600
    err = {k: np.linalg.norm(P[:, :3] - truth_tr[:, :3], axis=1) * RE_E / 1e3 for k, P in preds.items()}
    res["position_error_km"] = {k: {f"{h}h": float(np.interp(h, hrs, e)) for h in (1, 12, 24, 72)} for k, e in err.items()}
    np.savez_compressed(out / "forecasts.npz", tt=tt[ev], truth=truth_tr, t_all=t, X_noisy=Xn[:, :3].astype(np.float32),
                        n_tr=n_tr, **{f"P_{k}": P for k, P in preds.items()})
    (out / "results.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res["position_error_km"], indent=1), flush=True)
    lageos_video(out / "orbit_forecast.mp4", tt[ev], truth_tr, {k: P for k, P in preds.items() if k != "Kepler + J2"},
                 days=1, name="Satellite with a large Earth bulge")
    return res
