"""PySINDy-tutorial-style Lorenz figures for the robustness cases: in-sample (a training run) vs out-of-sample (a new run).
Data in colour (x blue, y red, z green), the discovered law dashed black, plus the 3-D attractor. Run from the repo root."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from eqdisc import solvers

plt.rcParams.update({"font.family": "cmb10", "mathtext.fontset": "cm", "axes.unicode_minus": False, "font.size": 12,
                     "axes.titlesize": 13, "legend.fontsize": 10})
R = {r["case"]: r for r in json.load(open("runs/robust2/results.json"))}
COL = ["b", "r", "g"]
TITLES = {"clean": "Clean data", "forcing": "Outside kicks in the training data", "spikes": "Spikes in the training data",
          "sensor": "Lost sensor (z never measured)"}


def simulate(names, rhs, x0, t, drive=None):
    """Integrate the law from x0. drive: {var: (t, values)} feeds a measured variable in as a known input."""
    f = solvers.make_ode_rhs(names, rhs)
    if drive:
        (dv, (td, vd)), = drive.items()
        i = names.index(dv)

        def g(tt, y):
            full = np.array(y, float)
            full[i] = np.interp(tt, td, vd)
            d = np.asarray(f(full[None], np.array([tt]))[0], float)
            d[i] = 0.0
            return d
    else:
        g = lambda tt, y: np.asarray(f(np.asarray(y)[None], np.array([tt]))[0], float)
    s = solve_ivp(g, (t[0], t[-1]), x0, t_eval=t, rtol=1e-8, atol=1e-10, method="LSODA")
    Y = np.full((len(t), len(x0)), np.nan)
    Y[:s.y.shape[1]] = s.y.T
    return Y


def panel(fig, gs_col, t, X, Y, title, labels, three_d=True, lyap_mark=None):
    n = X.shape[1]
    axs = []
    for i in range(n):
        ax = fig.add_subplot(gs_col[i])
        ax.plot(t, X[:, i], COL[i], lw=1.3, label=f"Data {labels[i]}")
        if Y is not None:
            ax.plot(t, Y[:, i], "k--", lw=1.1, label=f"Discovered law {labels[i]}")
        ax.set_ylabel(labels[i])
        ax.legend(loc="upper right")
        if lyap_mark:
            ax.axvline(lyap_mark, color="0.5", ls=":", lw=1)
        axs.append(ax)
    axs[0].set_title(title, loc="left")
    axs[-1].set_xlabel("time")
    if three_d and n == 3:
        ax = fig.add_subplot(gs_col[n], projection="3d")
        ax.plot(*X.T, COL[0], lw=0.5, label="Data")
        if Y is not None:
            ax.plot(*Y.T, "k--", lw=0.5, label="Discovered law")
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.legend(loc="upper left")


for cond in ("clean", "forcing", "spikes", "sensor"):
    d = f"datasets/robust2/lorenz_{cond}"
    m = json.load(open(d + "/meta.json"))
    D = np.load(d + "/data.npz"); U, t = D["U"], D["t"] - D["t"][0]
    te = np.load(d + "/hidden/test.npz")["U"]
    law = R[f"lorenz_{cond}"]["agent"]["rhs"]
    names = m["variables"]
    fig = plt.figure(figsize=(15, 13))
    gs = fig.add_gridspec(5, 2, hspace=0.45, wspace=0.18)
    X_in, X_out = U[0], te[0]
    if cond != "sensor":
        Y_in = simulate(names, law, X_in[0], t)
        Y_out = simulate(names, law, X_out[0], t)
        panel(fig, [gs[i, 0] for i in range(3)] + [gs[3:, 0]], t, X_in, Y_in, "In-sample: a training run (noisy, as given)", names)
        panel(fig, [gs[i, 1] for i in range(3)] + [gs[3:, 1]], t, X_out, Y_out, "Out-of-sample: a new run it never saw", names)
    else:
        # z is missing, so the law cannot be simulated on its own: drive the x-equation with the measured y
        Y_in = simulate(names, {"x": law["x"], "y": "0"}, X_in[0], t, drive={"y": (t, X_in[:, 1])})
        Y_out = simulate(names, {"x": law["x"], "y": "0"}, X_out[0], t, drive={"y": (t, X_out[:, 1])})
        for col, (X, Y, ttl) in enumerate(((X_in, Y_in, "In-sample: x rebuilt from measured y"),
                                           (X_out, Y_out, "Out-of-sample: x rebuilt from measured y (new run)"))):
            ax = fig.add_subplot(gs[0:2, col])
            ax.plot(t, X[:, 0], "b", lw=1.3, label="Data x")
            ax.plot(t, Y[:, 0], "k--", lw=1.1, label="Discovered x-law, driven by y")
            ax.set_title(ttl, loc="left"); ax.set_ylabel("x"); ax.legend(loc="upper right")
            ax = fig.add_subplot(gs[2, col])
            ax.plot(t, X[:, 1], "r", lw=1.3, label="Data y (input)"); ax.set_ylabel("y"); ax.legend(loc="upper right")
            ax.set_xlabel("time")
            ax = fig.add_subplot(gs[3:, col])
            ax.plot(X[:, 0], X[:, 1], "b", lw=0.4)
            ax.set_xlabel("x"); ax.set_ylabel("y")
            ax.set_title("Measured (x, y): the curves cross, so a hidden variable must exist", loc="left", fontsize=11)
    fig.suptitle(f"Lorenz (blinded, 2% noise): {TITLES[cond]}", fontsize=16, y=0.995)
    fig.savefig(f"docs/figures/lorenz_tutorial_{cond}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print("saved", cond, flush=True)
