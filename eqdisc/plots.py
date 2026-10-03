"""Diagnostic figures for the agent (returned to Claude as images) and for reports.

    plot_data(meta, data, path)        overview: time series + phase portrait / space-time + spectrum
    plot_model(meta, data, rhs, path)  model vs held-out public trajectory: rollout, error, derivative residual
"""
import numpy as np
from matplotlib.figure import Figure  # backend-free: works headless and leaves notebook backends alone

from . import solvers
from . import toolbox as tb


def _subplots(nrows=1, ncols=1, figsize=None, squeeze=True, **kw):
    fig = Figure(figsize=figsize)
    return fig, fig.subplots(nrows, ncols, squeeze=squeeze, **kw)


def _spatial_ndim(meta, data):
    return data["U"].ndim - 3


def plot_data(meta, data, path, max_traj=4):
    U, t = data["U"], data["t"]
    names = meta["variables"]
    if meta["kind"] == "ode":
        n = len(names)
        fig = Figure(figsize=(11, 1.9 * n + 0.5))
        gs = fig.add_gridspec(n, 2, width_ratios=[2.2, 1])
        for i, v in enumerate(names):
            ax = fig.add_subplot(gs[i, 0])
            for j in range(min(max_traj, U.shape[0])):
                ax.plot(t, U[j, :, i], lw=0.8)
            ax.set_ylabel(v)
        ax.set_xlabel("t")
        ax = fig.add_subplot(gs[:, 1])
        if n >= 2:
            for j in range(min(max_traj, U.shape[0])):
                ax.plot(U[j, :, 0], U[j, :, 1], lw=0.6)
                ax.plot(U[j, 0, 0], U[j, 0, 1], "k.", ms=4)
            ax.set(xlabel=names[0], ylabel=names[1], title="phase portrait")
        else:
            Us, dU = tb.smooth_and_differentiate(meta, U)
            ax.plot(Us[..., 0].ravel(), dU[..., 0].ravel(), ",", alpha=0.5)
            ax.set(xlabel=names[0], ylabel=f"d{names[0]}/dt (smoothed)", title="derivative vs state")
    elif _spatial_ndim(meta, data) == 1:
        nf = len(names)
        fig, axes = _subplots(1, nf + 1, figsize=(4.5 * (nf + 1), 3.6))
        x = data.get("x", np.arange(U.shape[2]))
        for i, f in enumerate(names):
            im = axes[i].pcolormesh(x, t, U[0, :, :, i], shading="auto", cmap="RdBu_r")
            axes[i].set(xlabel="x", ylabel="t", title=f"{f}(x,t), traj 0")
            fig.colorbar(im, ax=axes[i])
        P = (np.abs(np.fft.rfft(U, axis=2)) ** 2).mean(axis=(0, 1))
        for i, f in enumerate(names):
            axes[-1].semilogy(P[:, i] + 1e-30, label=f)
        axes[-1].set(xlabel="mode number", title="mean power spectrum (noise floor = flat tail)")
        axes[-1].legend()
    else:
        nf = len(names)
        idx = [0, U.shape[1] // 2, U.shape[1] - 1]
        fig, axes = _subplots(nf, 3, figsize=(10, 3.2 * nf), squeeze=False)
        for i, f in enumerate(names):
            for k, ti in enumerate(idx):
                im = axes[i, k].imshow(U[0, ti, ..., i].T, origin="lower", cmap="RdBu_r")
                axes[i, k].set_title(f"{f}, t={t[ti]:.2f}")
                fig.colorbar(im, ax=axes[i, k])
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    return str(path)


def _rollout(meta, data, rhs, U0, t):
    if meta["kind"] == "ode":
        return solvers.integrate_ode(meta["variables"], rhs, U0, t)
    general = getattr(solvers, "integrate_pde_general", None)
    if general is not None:
        try:
            return general(meta["variables"], rhs, meta, U0, t)
        except TypeError:
            pass
    h = tb._pde_step(meta, data["U"])
    sub = max(1, int(round(meta["dt"] / h)))
    n = min(len(t), int(6000 / sub) + 1)
    out = np.full((len(t),) + U0.shape, np.nan)
    out[:n] = solvers.integrate_pde(meta["variables"], rhs, meta["L"], U0, t[:n], meta["dt"] / sub)
    return out


def plot_model(meta, data, rhs, path):
    """Compare a model with the LAST public trajectory (the one toolbox.validate holds out)."""
    U, t = data["U"], data["t"]
    names = meta["variables"]
    val = U[-1:]
    Us, dU = tb.smooth_and_differentiate(meta, val, lowpass_frac=0.3 if meta["kind"] == "pde" else None)
    roll = _rollout(meta, data, rhs, Us[0, 0], t)
    if meta["kind"] == "ode":
        f = solvers.make_ode_rhs(names, rhs)(Us, t[None, :])
        n = len(names)
        fig, axes = _subplots(n, 2, figsize=(12, 2.0 * n + 0.5), squeeze=False)
        for i, v in enumerate(names):
            axes[i, 0].plot(t, val[0, :, i], color="0.6", lw=1, label="data (held out)")
            axes[i, 0].plot(t, roll[:, i], "C3--", lw=1.2, label="model rollout")
            axes[i, 0].set_ylabel(v)
            axes[i, 1].plot(t, dU[0, :, i] - f[0, :, i], "C0", lw=0.8)
            axes[i, 1].axhline(0, color="k", lw=0.5)
            axes[i, 1].set_ylabel(f"resid d{v}/dt")
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].set_title("rollout from held-out initial state")
        axes[0, 1].set_title("derivative residual (smoothed data - model)")
        axes[-1, 0].set_xlabel("t")
        axes[-1, 1].set_xlabel("t")
    elif U.ndim == 4:
        x = data.get("x", np.arange(U.shape[2]))
        nf = len(names)
        fig, axes = _subplots(nf, 3, figsize=(13, 3.4 * nf), squeeze=False)
        for i, fld in enumerate(names):
            lim = np.nanmax(np.abs(val[0, :, :, i]))
            for k, (A, ttl) in enumerate([(val[0, :, :, i], "data (held out)"), (roll[:, :, i], "model rollout"),
                                          (roll[:, :, i] - val[0, :, :, i], "rollout - data")]):
                im = axes[i, k].pcolormesh(x, t, A, shading="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
                axes[i, k].set(title=f"{fld}: {ttl}", xlabel="x", ylabel="t")
                fig.colorbar(im, ax=axes[i, k])
    else:
        nf = len(names)
        ti = min(len(t) - 1, max(1, len(t) // 3))
        fig, axes = _subplots(nf, 3, figsize=(11, 3.2 * nf), squeeze=False)
        for i, fld in enumerate(names):
            for k, (A, ttl) in enumerate([(val[0, ti, ..., i], "data"), (roll[ti, ..., i], "model"),
                                          (roll[ti, ..., i] - val[0, ti, ..., i], "model - data")]):
                im = axes[i, k].imshow(A.T, origin="lower", cmap="RdBu_r")
                axes[i, k].set_title(f"{fld} {ttl}, t={t[ti]:.2f}")
                fig.colorbar(im, ax=axes[i, k])
    fig.suptitle("  |  ".join(f"d{k}/dt = {v[:70]}" for k, v in rhs.items()), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    return str(path)


def plot_recommendations(meta, data, assessment, path, top=3):
    """ODE: data in state space (first two variables) with the recommended new starting points, and the spread of
    plausible-model predictions from the best one. PDE: the per-coefficient information gain of each experiment."""
    ex = (assessment.get("experiments") or {}).get("ranked", [])
    if not ex:
        return None
    names = meta["variables"]
    fig, axes = _subplots(1, 2, figsize=(12, 4.2))
    if meta["kind"] == "ode" and len(names) >= 2 and "initial_condition" in ex[0]:
        U = data["U"]
        for j in range(U.shape[0]):
            axes[0].plot(U[j, :, 0], U[j, :, 1], color="0.75", lw=0.8)
        for i, e in enumerate(ex[:top]):
            ic = e["initial_condition"]
            axes[0].plot(ic[0], ic[1], "*", ms=15, color=f"C{i}", label=f"#{i + 1} (score {e['score']})")
        axes[0].set(xlabel=names[0], ylabel=names[1], title="existing data (grey) and\nrecommended new starting points")
        axes[0].legend(fontsize=8)
    else:
        labels = [str(e.get("description") or e.get("initial_condition"))[:38] for e in ex[:top + 2]]
        axes[0].barh(range(len(labels)), [e["score"] or 0 for e in ex[:top + 2]], color="C0")
        axes[0].set_yticks(range(len(labels)))
        axes[0].set_yticklabels(labels, fontsize=8)
        axes[0].invert_yaxis()
        axes[0].set(xlabel="discrimination score (model spread / noise)^2", title="candidate experiments")
    info = ex[0].get("informs_coefficients", [])
    if info:
        axes[1].barh([c["coefficient"] for c in info], [c["info_gain_vs_existing"] or 0 for c in info], color="C2")
        axes[1].invert_yaxis()
        axes[1].set(xlabel="information vs. re-measuring existing conditions (x)",
                    title="what experiment #1 would pin down")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    return str(path)
