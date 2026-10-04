"""Revise: turn fired model findings into challenger models that must win the tournament to be adopted.

    challengers(meta, data, rhs, findings) -> {name: rhs}     one structural proposal per fired finding
    revise(meta, data, rhs, findings, tournament) -> (rhs, log)

Each finding proposes structure only; coefficients are refitted by a fixed-structure weak-form least squares
(slices.weak_system + lstsq), so no new integrator or library code. The tournament (orchestrate.tournament, passed
in to avoid an import cycle) decides: a challenger replaces the incumbent only if it wins on CV error / BIC /
rollouts. Mapping:
    residual_time_only   (fix.args.profile [[t, r]]) -> + a*sin(w*t) + b*cos(w*t), w from a least-squares sinusoid
                         scan of the profile (skipped when time was reset by a gap split, meta["segment_t0"])
    residual_space_only  (fix.args.profile [[x, r]]) -> + a*sin(k*x) + b*cos(k*x), k = 2*pi*j/L (periodic grids:
                         best integer mode j; otherwise a continuous scan)
    residual_amplitude   (variable v)                -> + c*v**3, and separately + c*v**2 (when absent)
slice_trajectory stays report-only (per-trajectory coefficients are not one rhs).
"""
import traceback

import numpy as np

from ..solvers import pde_layout
from ..uq import _lstsq, _rhs_from, _structure
from . import fired as fired_findings
from .slices import weak_system

KINDS = ("residual_time_only", "residual_space_only", "residual_amplitude")


def _g(x):
    return f"{float(x):.6g}"


def _sinusoid_scan(s, r, ws):
    """Fraction of variance of r(s) explained by a*sin(ws) + b*cos(ws) + c, for each w."""
    r = r - r.mean()
    tot = float(r @ r) + 1e-300
    ev = np.zeros(len(ws))
    for i, w in enumerate(ws):
        A = np.stack([np.sin(w * s), np.cos(w * s), np.ones_like(s)], 1)
        c = np.linalg.lstsq(A, r, rcond=None)[0]
        e = r - A @ c
        ev[i] = 1.0 - float(e @ e) / tot
    return ev


def dominant_frequency(profile, w_min=None, w_max=None, n_grid=2000):
    """Angular frequency of the best-fitting sinusoid of a profile [[s, r], ...]: dense scan over [2*pi/T, pi/ds)
    (or the given bounds), then a local refinement. Returns (w, explained_variance) or (None, 0)."""
    P = np.asarray(profile, float)
    if P.ndim != 2 or len(P) < 6:
        return None, 0.0
    s, r = P[:, 0], P[:, 1]
    T, ds = float(s.max() - s.min()), float(np.median(np.diff(np.sort(s))))
    if T <= 0 or ds <= 0:
        return None, 0.0
    lo, hi = w_min or 2 * np.pi / T, w_max or 0.999 * np.pi / ds
    if hi <= lo:
        return None, 0.0
    ws = np.linspace(lo, hi, n_grid)
    ev = _sinusoid_scan(s, r, ws)
    i = int(np.argmax(ev))
    fine = np.linspace(ws[max(i - 1, 0)], ws[min(i + 1, n_grid - 1)], 201)
    evf = _sinusoid_scan(s, r, fine)
    j = int(np.argmax(evf))
    return float(fine[j]), float(evf[j])


def dominant_wavenumber(meta, profile):
    """Wavenumber of the dominant spatial mode of a profile [[x, r], ...]: integer modes 2*pi*j/L on periodic
    grids (least squares, the FFT of an evenly sampled profile), a continuous scan otherwise."""
    lay = pde_layout(meta)
    g = lay["grid"][lay["spatial_dims"][0]]
    if lay["boundary"] != "periodic":
        return dominant_frequency(profile)
    P = np.asarray(profile, float)
    if P.ndim != 2 or len(P) < 6:
        return None, 0.0
    ds = float(np.median(np.diff(np.sort(P[:, 0]))))
    jmax = max(1, int(g["L"] / (2 * ds)))
    ks = 2 * np.pi * np.arange(1, jmax + 1) / g["L"]
    ev = _sinusoid_scan(P[:, 0] - g["x0"], P[:, 1], ks)
    i = int(np.argmax(ev))
    return float(ks[i]), float(ev[i])


def _variable(f, meta):
    v = (f.get("details") or {}).get("variable") or ((f.get("fix") or {}).get("args") or {}).get("variable")
    return v if v in meta["variables"] else None


def refit(meta, data, rhs, variables):
    """Fixed-structure weak-form least squares of rhs[v] for v in `variables` (other components unchanged)."""
    struct = _structure(meta, rhs)
    W = weak_system(meta, data, struct)
    coefs = {v: [c for _, c in terms] for v, terms in struct.items()}
    for v in variables:
        A, y = W["cols"][v], W["lhs"][:, meta["variables"].index(v)]
        if A.shape[1]:
            coefs[v] = list(_lstsq(A, y))
    return _rhs_from(struct, coefs)


def _proposals(meta, rhs, findings):
    """[(name, from_finding, variable, [new term strings])] from fired findings."""
    out = []
    pde = meta["kind"] == "pde"
    for f in fired_findings(findings or [], "info"):
        fid, v = f["id"], _variable(f, meta)
        if fid not in KINDS or v is None:
            continue
        prof = ((f.get("fix") or {}).get("args") or {}).get("profile") or (f.get("details") or {}).get("profile")
        if fid == "residual_time_only" and prof and not meta.get("segment_t0"):
            w, _ = dominant_frequency(prof)
            if w:
                out.append((f"forcing_t_{v}", fid, v, [f"sin({_g(w)}*t)", f"cos({_g(w)}*t)"]))
        elif fid == "residual_space_only" and prof and pde:
            k, _ = dominant_wavenumber(meta, prof)
            if k:
                x = pde_layout(meta)["spatial_dims"][0]
                out.append((f"source_{x}_{v}", fid, v, [f"sin({_g(k)}*{x})", f"cos({_g(k)}*{x})"]))
        elif fid == "residual_amplitude":
            out += [(f"amp_{v}{p}", fid, v, [f"{v}**{p}"]) for p in (3, 2)]
    return out


def challengers(meta, data, rhs, findings, with_source=False):
    """{name: rhs} (refitted) from fired findings; a proposal that cannot be built or fitted is skipped.
    with_source=True returns {name: (rhs, from_finding)}."""
    out = {}
    try:
        have = {v: {tm for tm, _ in terms} for v, terms in _structure(meta, rhs).items()}
    except Exception:  # noqa: BLE001
        return out
    for name, fid, v, terms in _proposals(meta, rhs, findings):
        try:
            new = [tm for tm in terms if str(_structure(meta, {v: tm})[v][0][0]) not in have.get(v, set())]
            if not new or name in out:
                continue
            cand = dict(rhs)
            cand[v] = f"{rhs.get(v, '0')} " + " ".join(f"+ 1.0*{tm}" for tm in new)
            fitted = refit(meta, data, cand, [v])
            if not all(np.isfinite(c) for _, cs in _structure(meta, fitted).items() for _, c in cs):
                continue
            out[name] = (fitted, fid) if with_source else fitted
        except Exception:  # noqa: BLE001  (e.g. `t` not yet evaluable for this kind of model)
            continue
    return out


def _prune(meta, data, rhs, tournament, log, ledger, rounds=3):
    """After a revision: terms the incumbent fit used to absorb the missing physics may now be spurious. Offer every
    one-term-smaller model (refitted); the tournament keeps the simpler one when it is indistinguishable."""
    for rnd in range(rounds):
        struct = _structure(meta, rhs)
        ch = {}
        for v, terms in struct.items():
            if len(terms) < 2:
                continue
            for i, (tm, _) in enumerate(terms):
                r = {**rhs, v: " + ".join(f"1.0*({t})" for j, (t, _) in enumerate(terms) if j != i)}
                try:
                    ch[f"drop_{v}_{tm}"] = refit(meta, data, r, [v])
                except Exception:  # noqa: BLE001
                    continue
        res = tournament(meta, data, {"incumbent": rhs, **ch}) or {}
        win = res.get("winner")
        if win in ch:
            entry = {"round": f"prune {rnd + 1}", "name": win, "from_finding": "parsimony", "rhs": ch[win],
                     "adopted": True, "verdict": res.get("verdict")}
            log.append(entry)
            if ledger is not None:
                ledger.append("revision", **entry)
            rhs = ch[win]
        else:
            break
    return rhs


def revise(meta, data, rhs, findings, tournament, rounds=2, ledger=None):
    """Up to `rounds` rounds of propose -> tournament -> adopt-if-winner -> re-audit. Never raises.
    Returns (rhs_final, log); log entries: {round, name, from_finding, rhs, adopted, verdict}."""
    from . import audit_model
    log, cur, fnd = [], dict(rhs), findings
    try:
        for rnd in range(1, rounds + 1):
            ch = challengers(meta, data, cur, fnd, with_source=True)
            if not ch:
                break
            res = tournament(meta, data, {"incumbent": cur, **{k: r for k, (r, _) in ch.items()}}) or {}
            win = res.get("winner")
            for name, (r, fid) in ch.items():
                entry = {"round": rnd, "name": name, "from_finding": fid, "rhs": r, "adopted": name == win,
                         "verdict": res.get("verdict"), "details": (res.get("details") or {}).get(name)}
                log.append(entry)
                if ledger is not None:
                    ledger.append("revision", **entry)
            if win not in ch:
                break
            cur = ch[win][0]
            fnd = audit_model(meta, data, cur)
        if cur != rhs:
            cur = _prune(meta, data, cur, tournament, log, ledger)
    except Exception as e:  # noqa: BLE001
        entry = {"round": None, "name": "error", "error": f"{type(e).__name__}: {e}",
                 "traceback": traceback.format_exc()[-1500:], "adopted": False}
        log.append(entry)
        if ledger is not None:
            try:
                ledger.append("revision", **entry)
            except Exception:  # noqa: BLE001
                pass
    return cur, log
