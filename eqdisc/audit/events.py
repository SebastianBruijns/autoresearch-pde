"""Model check for real events (WS8): can the fitted model produce the large events the data audit kept?

    audit(meta, data, rhs) -> list[Finding]      (stage "model")

The data audit (audit/data.py: detect_events) keeps extreme events and external shocks as real dynamics. Here the
model is rolled forward from a few samples before each event (the observed, lightly smoothed state) to a few after,
and the predicted excursion is compared with the observed one relative to noise:
    event_explained    (info)  the model reproduces the event: evidence for the model
    event_unexplained  (info by default; warn once "events_unexplained_warn" is set in thresholds.json) the model
                       cannot produce the event: missing term or external forcing
No events -> one non-fired info finding. Never raises (wrapped by audit._safe), NaN-safe, capped at MAX_EVENTS.
"""
import time

import numpy as np

from . import finding, threshold
from .data import detect_events

STAGE = "model"
MAX_EVENTS = 5
PRE, POST = 5, 5        # samples rolled before / after the event window
TIME_BUDGET = 8.0       # seconds for all rollouts


def _smooth_state(meta, U, j, i):
    """Observed state at row i of trajectory j, lightly smoothed in time (3-point mean; PDE also in space)."""
    a, b = max(i - 1, 0), min(i + 2, U.shape[1])
    x0 = np.nanmean(U[j, a:b], axis=0)
    if meta.get("kind") == "pde":
        from ..solvers import smooth_space
        x0 = smooth_space(x0[None], meta, frac=0.5)[0]
    return x0


def _roll(meta, rhs, x0, tt, U):
    if meta.get("kind") == "pde":
        from ..solvers import integrate_pde_general, pde_layout
        from ..toolbox import _pde_step
        h = _pde_step(meta, U)
        sub = max(1, int(round(meta["dt"] / h)))
        return integrate_pde_general(meta["variables"], rhs, pde_layout(meta), x0, tt, meta["dt"] / sub,
                                     max_seconds=3.0)
    from ..solvers import integrate_ode
    return integrate_ode(meta["variables"], rhs, x0, tt, max_seconds=2.0)


def audit(meta, data, rhs):
    U = np.asarray(data["U"], float)
    t = np.asarray(data["t"], float)
    det = detect_events(meta, U)
    evs = sorted([e for e in det["events"] if e["kind"] in ("extreme_event", "external_shock")],
                 key=lambda e: -e["peak_z"])[:MAX_EVENTS]
    if not evs:
        return [finding("event_explained", STAGE, None, None, False, "info",
                        message="no large events in the data to check the model against")]
    tol = threshold("events_explained_tol", 3.0)       # rms(pred - obs) / noise inside the event window
    warn = bool(threshold("events_unexplained_warn", 0))
    t0 = time.time()
    checked = []
    for e in evs:
        if time.time() - t0 > TIME_BUDGET:
            break
        j, v = e["traj"], e["var"]
        a, b = max(e["i_start"] - PRE, 0), min(e["i_end"] + POST, U.shape[1] - 1)
        if b - a < 3:
            continue
        sigma = float(det["noise_sigma"][v]) if v < len(det["noise_sigma"]) else np.nan
        if not (np.isfinite(sigma) and sigma > 0):
            continue
        try:
            pred = _roll(meta, rhs, _smooth_state(meta, U, j, a), t[a:b + 1], U)
        except Exception:  # noqa: BLE001
            continue
        obs = U[j, a:b + 1]
        if meta.get("kind") == "pde" and "x_index" in e:
            (x0, x1), = e["x_index"][:1] or [(0, U.shape[2] - 1)]
            sl = (slice(None), slice(max(x0 - 2, 0), x1 + 3), v)
        else:
            sl = (slice(None), v)
        with np.errstate(all="ignore"):
            d = pred[sl] - obs[sl]
            rms = float(np.sqrt(np.nanmean(d ** 2)) / sigma) if np.isfinite(pred[sl]).all() else float("inf")
            base = obs[sl][0]
            peak_obs = float(np.nanmax(np.abs(obs[sl] - base)) / sigma)
            peak_pred = float(np.nanmax(np.abs(pred[sl] - base)) / sigma) if np.isfinite(pred[sl]).all() \
                else float("inf")
        checked.append({"traj": j, "variable": e["variable"], "kind": e["kind"], "t_start": e["t_start"],
                        "t_end": e["t_end"], "rms_err_sd": rms, "peak_obs_sd": peak_obs, "peak_pred_sd": peak_pred,
                        "explained": bool(rms <= tol)})
    if not checked:
        return [finding("event_explained", STAGE, None, tol, False, "info",
                        message="large events found but none could be checked (rollout failed or too short)",
                        details={"n_events": len(evs)})]
    bad = [c for c in checked if not c["explained"]]
    worst = max(c["rms_err_sd"] for c in checked)
    det_ = {"events": checked, "tolerance_sd": tol, "seconds": round(time.time() - t0, 2)}
    if bad:
        when = ", ".join(f"{c['t_start']:g}" for c in bad[:3])
        return [finding("event_unexplained", STAGE, worst, tol, True, "warn" if warn else "info", None,
                        message=f"the model cannot produce the large event at t={when}: missing term or external "
                                f"forcing", details=det_)]
    when = ", ".join(f"{c['t_start']:g}" for c in checked[:3])
    return [finding("event_explained", STAGE, worst, tol, True, "info", None,
                    message=f"the model reproduces the large event at t={when}: evidence for it", details=det_)]
