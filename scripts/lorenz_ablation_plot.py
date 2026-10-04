"""Tutorial-style Lorenz figure with ablations: data vs discovered law vs SINDy vs FNO, in sample and out of sample."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
src = open("scripts/lorenz_tutorial_plots.py").read().split("\nfor cond in")[0]
exec(src)                                    # simulate(), rcParams, R
cond = sys.argv[1] if len(sys.argv) > 1 else "spikes"
d = f"datasets/robust2/lorenz_{cond}"
names = json.load(open(d + "/meta.json"))["variables"]
D = np.load(d + "/data.npz"); U, t = D["U"], D["t"] - D["t"][0]
T = np.load(d + "/hidden/test.npz")["U"]
F = np.load(f"runs/lorenz_showcase/fno_{cond}.npz")
r = R[f"lorenz_{cond}"]
sd = T.reshape(-1, T.shape[-1]).std(0)
STY = {"Discovered law": ("k", "--", 1.4), "SINDy": ("#1baf7a", ":", 1.6), "FNO": ("#eb6834", "-.", 1.4)}
COLS = ["b", "r", "g"]


def despike(X):
    """Score the in-sample run against a median-filtered copy, so isolated spikes do not count as forecast errors."""
    from scipy.signal import medfilt
    return np.stack([medfilt(X[:, i], 5) for i in range(X.shape[1])], 1)


def vt(Y, X):
    e = np.sqrt(np.mean(((Y - X) / sd) ** 2, 1)); e = np.where(np.isfinite(e), e, 9)
    b = np.nonzero(e > 0.5)[0]
    return t[b[0]] if len(b) else t[-1]


fig = plt.figure(figsize=(16, 14))
gs = fig.add_gridspec(5, 2, hspace=0.5, wspace=0.18)
for col, (tag, X, fno) in enumerate([("In sample: a training run (as given)", U[0], F["in"]),
                                     ("Out of sample: a new run it never saw", T[0], F["oos"][0])]):
    Xs = despike(X) if col == 0 else X
    preds = {"Discovered law": simulate(names, r["agent"]["rhs"], X[0], t),
             "SINDy": simulate(names, r["baseline"]["rhs"], X[0], t), "FNO": fno}
    for i, v in enumerate(names):
        ax = fig.add_subplot(gs[i, col])
        ax.plot(t, X[:, i], COLS[i], lw=1.3, label=f"Data {v}")
        for k, Y in preds.items():
            c, ls, lw = STY[k]
            ax.plot(t, Y[:, i], color=c, ls=ls, lw=lw, label=k)
        ax.set_ylabel(v); lo, hi = np.percentile(Xs[:, i], [0, 100]); pad = 0.25 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        if i == 0:
            ax.set_title(tag + "\non track until  " + ",  ".join(f"{k}: t={vt(Y, Xs):.2f}" for k, Y in preds.items()),
                         loc="left", fontsize=11)
            ax.legend(loc="upper right", ncol=4, fontsize=9)
    ax.set_xlabel("time")
    ax = fig.add_subplot(gs[3:, col], projection="3d")
    ax.plot(*Xs.T, "b", lw=0.6, label="Data" + (" (spikes removed for display)" if col == 0 else ""))
    for k, Y in preds.items():
        c, ls, lw = STY[k]
        ok = np.all(np.isfinite(Y), 1) & np.all(np.abs(Y) < 5 * np.abs(X).max(0), 1)
        ax.plot(*Y[ok].T, color=c, ls=ls, lw=0.7, label=k)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z"); ax.legend(loc="upper left", fontsize=9)
fig.savefig(f"docs/figures/lorenz_tutorial_{cond}_ablation.png", dpi=110, bbox_inches="tight")
print("ok")
