"""Figures for docs/robustness.md (run from the repo root after `python -m eqdisc.robustness run`)."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
LOGFMT = FuncFormatter(lambda v, _: f"{v:g}")
from scipy.integrate import solve_ivp
from eqdisc import solvers
plt.rcParams.update({"font.family": "cmb10", "mathtext.fontset": "cm", "axes.unicode_minus": False, "font.size": 13,
                     "axes.titlesize": 14, "axes.labelsize": 13, "legend.fontsize": 11})
R = {r["case"]: r for r in json.load(open("runs/robust2/results.json"))}
C = {"truth": "k", "agent": "#16a34a", "sindy": "#eb6834"}
CONDS = [("clean", "Clean"), ("forcing", "Outside kicks"), ("spikes", "Spikes"), ("sensor", "Lost sensor")]

def ode_roll(vars_, rhs, x0, t):
    f = solvers.make_ode_rhs(vars_, rhs)
    s = solve_ivp(lambda tt, y: np.asarray(f(y[None], np.array([tt]))[0], float), (t[0], t[-1]), x0, t_eval=t,
                  rtol=1e-8, atol=1e-10, method="LSODA")
    Y = np.full((len(t), len(x0)), np.nan); Y[:s.y.shape[1]] = s.y.T
    return Y

# ------------------------------------------------------------------ Lorenz
fig, axes = plt.subplots(4, 3, figsize=(17, 16), gridspec_kw={"width_ratios": [1.1, 1.4, 1]})
for row, (cond, title) in enumerate(CONDS):
    d = f"datasets/robust2/lorenz_{cond}"; m = json.load(open(d + "/meta.json")); sc = json.load(open(d + "/hidden/score.json"))
    D = np.load(d + "/data.npz"); U, t = D["U"], D["t"]; te = np.load(d + "/hidden/test.npz")["U"]
    a0 = axes[row, 0]
    a0.plot(t, U[0, :, 0], color="0.35", lw=0.8)
    if cond == "forcing":
        for ev in sc["events"][0]:
            a0.axvspan(ev[2] * sc["scales"]["time"], ev[3] * sc["scales"]["time"], color="#eb6834", alpha=0.25)
    if cond == "spikes":
        clean_like = np.load("datasets/robust2/lorenz_clean/data.npz")["U"][0, :, 0]
        sp_ = np.abs(U[0, :, 0] - clean_like) > 4 * U[0, :, 0].std() * 0.5
        a0.plot(t[sp_], U[0, sp_, 0], "o", color="#dc2626", ms=4)
    a0.set_title(f"{title}: training data (x)", loc="left"); a0.set_xlabel("time")
    ag, sb = R[f"lorenz_{cond}"]["agent"]["rhs"], R[f"lorenz_{cond}"]["baseline"]["rhs"]
    a1, a2 = axes[row, 1], axes[row, 2]
    if cond != "sensor":
        n = 400; tt = t[:n] - t[0]; X = te[0, :n]
        Ya = ode_roll(m["variables"], ag, X[0], tt); Ys = ode_roll(m["variables"], sb, X[0], tt)
        a1.plot(tt, X[:, 0], color=C["truth"], lw=2.2, label="Reality")
        a1.plot(tt, Ya[:, 0], color=C["agent"], lw=1.6, ls="--", label="Discovered law")
        a1.plot(tt, Ys[:, 0], color=C["sindy"], lw=1.2, ls=":", label="Sparse regression")
        a1.set_title("Forecast of a new run (x)", loc="left"); a1.set_xlabel("time"); a1.legend(loc="lower left")
        sd = X.std(0)
        for k, Y in (("agent", Ya), ("sindy", Ys)):
            e = np.sqrt(np.mean(((Y - X) / sd) ** 2, axis=1))
            a2.semilogy(tt, np.maximum(e, 1e-5), color=C[k], lw=1.8, label="Discovered law" if k == "agent" else "Sparse regression")
        a2.set_title("Forecast error", loc="left"); a2.set_xlabel("time"); a2.set_ylim(1e-4, 10); a2.legend(loc="lower right"); a2.yaxis.set_major_formatter(LOGFMT)
    else:
        X = te.reshape(-1, 2)[::7]
        full = sc["full_law"]; Xt = np.load("datasets/robust2/lorenz_clean/hidden/test.npz")["U"].reshape(-1, 3)[::7]
        tx = solvers.make_ode_rhs(["x", "y", "z"], full)(Xt, np.zeros(len(Xt)))[:, 0]
        fa = solvers.make_ode_rhs(["x", "y"], {"x": ag["x"], "y": "0"})(Xt[:, :2], np.zeros(len(Xt)))[:, 0]
        a1.plot(tx, fa, ".", color=C["agent"], ms=3, alpha=0.6)
        lim = np.abs(tx).max(); a1.plot([-lim, lim], [-lim, lim], "k-", lw=1)
        a1.set_title("x-equation: discovered vs true dx/dt", loc="left"); a1.set_xlabel("true dx/dt"); a1.set_ylabel("discovered")
        a2.axis("off")
        a2.text(0.0, 0.85, "z was never measured.\n\nx' recovered: 0.2% error.\n\ny' cannot close without z:\n"
                "the agent says 'a hidden third\nvariable must be present',\nso its verdict is: inconclusive.", fontsize=14,
                va="top", transform=a2.transAxes)
fig.suptitle("Lorenz (blinded, 2% noise): recovering the law under outside kicks, spikes and a lost sensor", fontsize=16, y=0.995)
fig.tight_layout()
fig.savefig("docs/figures/robust_lorenz.png", dpi=110)
plt.close(fig)

# ------------------------------------------------------------------ KdV
fig, axes = plt.subplots(4, 4, figsize=(19, 15))
for row, (cond, title) in enumerate(CONDS):
    d = f"datasets/robust2/kdv_{cond}"; m = json.load(open(d + "/meta.json")); sc = json.load(open(d + "/hidden/score.json"))
    D = np.load(d + "/data.npz"); U, t, x = D["U"], D["t"], D["x"]; te = np.load(d + "/hidden/test.npz")["U"][0, ..., 0]
    lay = solvers.pde_layout(m)
    ag, sb = R[f"kdv_{cond}"]["agent"]["rhs"], R[f"kdv_{cond}"]["baseline"]["rhs"]
    tt = t - t[0]
    Ya = solvers.integrate_pde_general(["u"], ag, lay, te[0][:, None], tt, dt_sim=tt[1] / 100)[..., 0]
    Ys = solvers.integrate_pde_general(["u"], sb, lay, te[0][:, None], tt, dt_sim=tt[1] / 100)[..., 0]
    vmax = np.abs(te).max()
    kw = dict(aspect="auto", origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax, extent=[x[0], x[-1], t[0], t[-1]])
    axes[row, 0].imshow(U[0, ..., 0], **kw); axes[row, 0].set_title(f"{title}: training data", loc="left")
    if cond == "forcing":
        for ev in sc["events"][0]:
            a_ = sc["scales"]["space"]; s_ = sc["scales"]["time"]
            axes[row, 0].add_patch(plt.Rectangle((ev[1] * a_, ev[3] * s_), (ev[2] - ev[1]) * a_, (ev[4] - ev[3]) * s_,
                                                 fill=False, ec="#eb6834", lw=2.5))
    axes[row, 1].imshow(te, **kw); axes[row, 1].set_title("Reality (new run)", loc="left")
    axes[row, 2].imshow(Ya, **kw); axes[row, 2].set_title("Discovered law forecast", loc="left")
    sd = te.std()
    for k, Y in (("agent", Ya), ("sindy", Ys)):
        e = np.sqrt(np.mean((Y - te) ** 2, axis=1)) / sd
        axes[row, 3].semilogy(tt, np.maximum(e, 1e-5), color=C[k], lw=2, label="Discovered law" if k == "agent" else "Sparse regression")
    axes[row, 3].set_title("Forecast error", loc="left"); axes[row, 3].set_ylim(1e-4, 3); axes[row, 3].legend(loc="lower right"); axes[row, 3].yaxis.set_major_formatter(LOGFMT)
    axes[row, 3].set_xlabel("time")
    for j in range(3):
        axes[row, j].set_xlabel("x"); axes[row, j].set_ylabel("time")
fig.suptitle("KdV (blinded, 2% noise): recovering the law under outside kicks, spikes and stuck sensors", fontsize=16, y=0.995)
fig.tight_layout()
fig.savefig("docs/figures/robust_kdv.png", dpi=110)
print("ok")
